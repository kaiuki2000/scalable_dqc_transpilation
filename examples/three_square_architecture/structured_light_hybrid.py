# --- 0. Locate the shared example helpers (examples/) on sys.path ---
# Deviation from the original script: it imported these from the private
# `qiskit_dev.custom_targets` package tree. See examples/README.md.
import os
import sys

_EXAMPLES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLES_ROOT)

# --- 0b. Repository-relative data locations ---
# Deviation from the original script, which used absolute paths on the
# author's machine. Both are overridable for cluster runs. See examples/README.md.
BENCHMARKS_DIR = os.environ.get("DQC_BENCHMARKS_DIR", os.path.join(_EXAMPLES_ROOT, "benchmarks"))
RESULTS_DIR = os.environ.get(
    "DQC_RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- 1. Standard Library & Frameworks ---
# [KEEP ALL THE SAME IMPORTS AS THE CZ SCRIPT ABOVE]
import logging, os, math, json, warnings, time, gc
from time import perf_counter
import concurrent.futures
import multiprocessing as mp
import numpy as np
import networkx as nx
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.compiler import transpile
from qiskit.qasm2 import load as load_qasm2, LEGACY_CUSTOM_INSTRUCTIONS
from pytket.extensions.qiskit import tk_to_qiskit, qiskit_to_tk
from pytket_dqc.distributors import PartitioningHeterogeneous
from pytket_dqc.networks.nisq_network import NISQNetwork
from pytket_dqc.utils import DQCPass
from mqpu_utils import rectangular_backend, generate_multi_qpu_backend_from_monolithic_backend_with_links, sanitize_qiskit_labels
from experiment_utils import *

logging.getLogger("qiskit").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# HARDWARE & HELPERS
# ==========================================
# [SAME AS CZ SCRIPT]
n_qpus, l = 3, 4
square_backend = rectangular_backend(l, l, basis_gates=['cz', 'id', 'rz', 'sx', 'x'], seed=42)
square_remote_links = [(3, 31), (7, 27), (15, 35), (11, 39), (23, 43), (19, 47)]
square_inter_qpu_edges = square_remote_links + [(v, u) for u, v in square_remote_links]
square_mqpu_backend = generate_multi_qpu_backend_from_monolithic_backend_with_links(n_qpus=n_qpus, monolithic_backend=square_backend, inter_qpu_links=square_inter_qpu_edges, backend_name=f'{n_qpus}_Square_r1')
network = NISQNetwork(server_coupling=[[0, 1], [0, 2], [1, 2]], server_qubits={0: range(0, 16), 1: range(16, 32), 2: range(32, 48)}, server_ebit_mem=None, server_link_capacities={(0, 1): 2, (0, 2): 2, (1, 2): 2})
qubit_qpu_map = [i // square_backend.num_qubits for i in range(square_mqpu_backend.num_qubits)]
inter_qpu_coupling_map = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
def generate_custom_distance_matrix(mqpu_backend, edges):
    G = nx.DiGraph()
    G.add_weighted_edges_from([(e[0], e[1], 1) if e in edges else (e[0], e[1], 1.0) for e in mqpu_backend.coupling_map.get_edges()])
    return nx.floyd_warshall_numpy(G, range(len(G.nodes)))
S_matrix_default = generate_custom_distance_matrix(square_mqpu_backend, square_inter_qpu_edges)

class InjectStartingLayout(AnalysisPass):
    def __init__(self, layouts): super().__init__(); self.custom_layouts = layouts
    def run(self, dag): self.property_set["sabre_starting_layouts"] = self.custom_layouts

def generate_core_respecting_layout(mapping, qc, seed):
    np.random.seed(seed)
    phys = {0: list(range(0, 16)), 1: list(range(16, 32)), 2: list(range(32, 48))}
    layout = {}
    for core in range(3):
        v_in_core = sorted([v for v, c in mapping.items() if int(str(c).split('_')[-1][0]) == core], key=lambda v: v.index[0])
        avail = phys[core].copy()
        for v_tk in v_in_core:
            q = qc.qubits[v_tk.index[0]]
            assigned = int(np.random.choice(avail))
            layout[q] = assigned
            avail.remove(assigned)
    return Layout(layout)

def load_and_prepare_benchmark(file_path):
    qc = load_qasm2(file_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    qc.data = [i for i in qc.data if i.operation.name not in ['measure', 'reset', 'barrier', 'delay']]
    qc.cregs.clear(); qc.clbits.clear()
    qc = transpile(qc, basis_gates=['cz', 'id', 'rz', 'sx', 'x'], optimization_level=2)
    circ = qiskit_to_tk(qc) 
    DQCPass().apply(circ)
    return circ, sanitize_qiskit_labels(tk_to_qiskit(circ))

def run_custom_dqc_experiment(circ_tk, qc, seed, num_mapping_seeds=5, num_layout_seeds=5, f_weight=10):
    # (Note: In the cz_frac script, you pass `cz_frac` and generate the circ/qc inside the function, 
    # but the inner loop logic below is exactly the same!)

    orig_intra_swaps, orig_inter_swaps = 0, 0
    for instr in qc.data:
        if instr.operation.name == "swap":
            q0_idx = qc.find_bit(instr.qubits[0]).index
            q1_idx = qc.find_bit(instr.qubits[1]).index
            if (q0_idx, q1_idx) in square_inter_qpu_edges or (q1_idx, q0_idx) in square_inter_qpu_edges:
                orig_inter_swaps += 1
            else:
                orig_intra_swaps += 1

    best_cost = float('inf')
    best_metrics = {}
    best_time = 0.0

    routing_metrics = {}
    inter_edges_set = {tuple(sorted(e)) for e in square_inter_qpu_edges}

    def audit_sabreswap(pass_, dag, time, property_set, count):
        if pass_.name() == "SabreLayout":
            intra, inter, czs = 0, 0, 0
            for n in dag.op_nodes():
                if len(n.qargs) == 2 and n.op.name not in ["barrier", "routing_placeholder"]:
                    edge = tuple(sorted((dag.find_bit(n.qargs[0]).index, dag.find_bit(n.qargs[1]).index)))
                    is_inter_qpu = edge in inter_edges_set
                    if n.op.name == "swap":
                        if is_inter_qpu: inter += 1
                        else: intra += 1
                    elif is_inter_qpu:
                        if n.op.name in ["cz", "cx", "cu1", "cp"]: czs += 1
                        else: czs += 1 # Non-standard gates
            routing_metrics['final_intra_swaps'] = intra
            routing_metrics['final_inter_swaps'] = inter
            routing_metrics['final_inter_czs'] = czs

    # --- THE FIX: Exact same RNG logic as your previous scripts ---
    rng = np.random.default_rng(seed)
    mapping_seeds = rng.integers(0, 2**31 - 1, size=num_mapping_seeds)
    layout_seeds = rng.integers(0, 2**31 - 1, size=num_layout_seeds)

    for m_seed in mapping_seeds:
        m_seed = int(m_seed) # Ensure standard python int for PyTket
        t_dist_0 = perf_counter()
        mapping = PartitioningHeterogeneous().distribute(circ_tk, network=network, seed=m_seed).get_qubit_mapping()
        dt_distribution = perf_counter() - t_dist_0
        
        for l_seed in layout_seeds:
            l_seed = int(l_seed) # Ensure standard python int for Qiskit
            
            pm = PassManager([
                InjectStartingLayout([generate_core_respecting_layout(mapping, qc, seed=l_seed)]),
                SabreLayout(
                    coupling_map=square_mqpu_backend.coupling_map, 
                    penalized_swaps=square_inter_qpu_edges, 
                    qubit_qpu_map=qubit_qpu_map, 
                    inter_qpu_coupling_map=inter_qpu_coupling_map, 
                    alpha=9.0, 
                    beta=3.0, 
                    seed=l_seed, 
                    layout_trials=0, 
                    custom_distance_matrix=S_matrix_default.tolist(), 
                    extended_set_length=20
                )
            ])
            routing_metrics.clear()
            
            t_route_0 = perf_counter()
            _ = pm.run(qc, callback=audit_sabreswap)
            trial_time = perf_counter() - t_route_0
            
            total_eprs = (routing_metrics.get('final_inter_swaps', 0) * 3) + routing_metrics.get('final_inter_czs', 0)
            cost = (total_eprs * f_weight) + routing_metrics.get('final_intra_swaps', 0)
            
            if cost < best_cost: 
                best_cost = cost
                best_metrics = routing_metrics.copy()
                best_time = dt_distribution + trial_time
                
            del pm
            gc.collect()

    final_total_ebits = (best_metrics.get('final_inter_swaps', 0) * 3) + best_metrics.get('final_inter_czs', 0)
    final_added_swaps = best_metrics.get('final_intra_swaps', 0) - orig_intra_swaps

    return final_total_ebits, final_added_swaps, best_time

def run_single_seed_task(circ_tk, qc, seed):
    try: return seed, run_custom_dqc_experiment(circ_tk, qc, seed)
    except Exception as e: return seed, None

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    light_dir = os.path.join(BENCHMARKS_DIR, 'light')
    CIRCUITS_TO_RUN = [os.path.join(light_dir, f) for f in os.listdir(light_dir) if f.endswith('.qasm')]
    
    RESULTS_FILE = os.path.join(RESULTS_DIR, "three_square_structured_light_hybrid.json")
    db = load_experiment_database(RESULTS_FILE)
    METHOD_IDX = get_or_create_method_index(db, "PartitioningHeterogeneous + Custom Lookahead SABRE", "#8338ec")

    for file_path in CIRCUITS_TO_RUN:
        c_name = os.path.basename(file_path).replace('.qasm', '')
        if c_name in db.get("circuit_list", []): circ_idx = db["circuit_list"].index(c_name)
        else: circ_idx = len(db.setdefault("circuit_list", [])); db["circuit_list"].append(c_name)
            
        m = db["methods_data"][METHOD_IDX]
        for k in ["ebits", "swaps", "time"]:
            while len(m.setdefault(k, [])) <= circ_idx: m[k].append(None)
        for k in ["raw_ebits", "raw_swaps", "raw_time"]:
            while len(m.setdefault(k, [])) <= circ_idx: m[k].append([])

        if m["ebits"][circ_idx] is not None and len(m["raw_ebits"][circ_idx]) >= 5: continue

        print(f"\n--- Circuit: {c_name} ---")
        circ_tk, qc = load_and_prepare_benchmark(file_path)
        
        successful_runs, current_seed = [], 42
        with concurrent.futures.ProcessPoolExecutor() as exe:
            f2s = {exe.submit(run_single_seed_task, circ_tk, qc, current_seed + i): current_seed + i for i in range(5)}
            current_seed += 5
            for f in concurrent.futures.as_completed(f2s):
                seed, res = f.result()
                if res: successful_runs.append(res); print(f" -> Seed {seed} done. EPRs: {res[0]}")
                else: f2s[exe.submit(run_single_seed_task, circ_tk, qc, current_seed)] = current_seed; current_seed += 1
                if len(successful_runs) == 5: break
                
        t_ebits, t_swaps, t_time = [r[0] for r in successful_runs], [r[1] for r in successful_runs], [r[2] for r in successful_runs]
        m["ebits"][circ_idx], m["swaps"][circ_idx], m["time"][circ_idx] = compute_statistics(t_ebits), compute_statistics(t_swaps), compute_statistics(t_time)
        m["raw_ebits"][circ_idx], m["raw_swaps"][circ_idx], m["raw_time"][circ_idx] = t_ebits, t_swaps, t_time
        save_experiment_database(db, RESULTS_FILE)