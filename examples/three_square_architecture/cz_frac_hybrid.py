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


# --- 1. Standard Library ---
import logging
import os
import math
import json
import warnings
import time
import gc
from time import perf_counter
import concurrent.futures

# --- 2. Third-Party Libraries ---
import numpy as np
import networkx as nx

# --- 3. Quantum Frameworks ---
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.basepasses import AnalysisPass
from pytket.extensions.qiskit import tk_to_qiskit, qiskit_to_tk
from qiskit.compiler import transpile

# Pytket DQC
from pytket_dqc.distributors import PartitioningHeterogeneous
from pytket_dqc.networks.nisq_network import NISQNetwork
from pytket_dqc.utils import DQCPass

# Custom Project Modules
from mqpu_utils import (
    build_cz_fraction_circuit,
    generate_multi_qpu_backend_from_monolithic_backend_with_links,
    rectangular_backend_with_links,
    rectangular_backend,
    sanitize_qiskit_labels,
)
from experiment_utils import *

# Suppress logs
logging.getLogger("qiskit").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 1. HARDWARE & TOPOLOGY SETUP
# ==========================================
n_qpus, l = 3, 4
basis_gates = ['cz', 'id', 'rz', 'sx', 'x']

square_backend = rectangular_backend(l, l, basis_gates=basis_gates, seed=42)
square_remote_links = [
    (3, 31), (7, 27),    # 0 to 1
    (15, 35), (11, 39),  # 0 to 2
    (23, 43), (19, 47),  # 1 to 2
]
square_inter_qpu_edges = square_remote_links + [(v, u) for u, v in square_remote_links]
square_mqpu_backend = generate_multi_qpu_backend_from_monolithic_backend_with_links(
    n_qpus=n_qpus, monolithic_backend=square_backend,
    inter_qpu_links=square_inter_qpu_edges, backend_name=f'{n_qpus}_Square_r1'
)

server_coupling = [[0, 1], [0, 2], [1, 2]]
server_qubits = {0: range(0, 16), 1: range(16, 32), 2: range(32, 48)}
server_link_capacities = {(0, 1): 2, (0, 2): 2, (1, 2): 2}

network = NISQNetwork(
    server_coupling=server_coupling, server_qubits=server_qubits,
    server_ebit_mem=None, server_link_capacities=server_link_capacities
)

# Helpers for SABRE
qubit_qpu_map = [i // square_backend.num_qubits for i in range(square_mqpu_backend.num_qubits)]
inter_qpu_coupling_map = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]

def generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=1.0):
    weighted_edges = [(e[0], e[1], 1) if e in inter_qpu_edges else (e[0], e[1], factor) for e in mqpu_backend.coupling_map.get_edges()]
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    return nx.floyd_warshall_numpy(G, range(len(G.nodes)))

S_matrix_default = generate_custom_distance_matrix(square_mqpu_backend, square_inter_qpu_edges, factor=1.0)

# ==========================================
# 2. CUSTOM PASSES & ROUTING LOGIC
# ==========================================
class InjectStartingLayout(AnalysisPass):
    def __init__(self, custom_layouts):
        super().__init__()
        self.custom_layouts = custom_layouts
    def run(self, dag):
        self.property_set["sabre_starting_layouts"] = self.custom_layouts

def generate_core_respecting_layout(mapping, qc, seed):
    np.random.seed(seed)
    physical_qubits_per_core = {0: list(range(0, 16)), 1: list(range(16, 32)), 2: list(range(32, 48))}
    physical_layout = {}
    for core in range(3):
        virtuals_in_core = sorted([v for v, c in mapping.items() if int(str(c).split('_')[-1][0]) == core], key=lambda v: v.index[0])
        available = physical_qubits_per_core[core].copy()
        for v_tk in virtuals_in_core:
            qiskit_qubit = qc.qubits[v_tk.index[0]]
            assigned_physical = int(np.random.choice(available))
            physical_layout[qiskit_qubit] = assigned_physical
            available.remove(assigned_physical)
    return Layout(physical_layout)

def _generate_base_circuit(cz_frac, seed):
    qc = build_cz_fraction_circuit(n=36, d=36, p=cz_frac, seed=seed)
    clean_data = [inst for inst in qc.data if inst.operation.name not in ['measure', 'reset', 'barrier', 'delay']]
    qc.data = clean_data
    qc.cregs.clear(); qc.clbits.clear()
    qc = transpile(qc, basis_gates=['cz', 'id', 'rz', 'sx', 'x'], optimization_level=2)
    circ = qiskit_to_tk(qc) 
    DQCPass().apply(circ)
    return circ, sanitize_qiskit_labels(tk_to_qiskit(circ))

def run_custom_dqc_experiment(cz_frac, seed, num_mapping_seeds=5, num_layout_seeds=5, f_weight=10):
    # 1. Generate the circuit specifically for this fraction and seed!
    circ_tk, qc = _generate_base_circuit(cz_frac, seed)

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

    # --- MATCHING RNG LOGIC ---
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

    del circ_tk, qc
    gc.collect()

    return final_total_ebits, final_added_swaps, best_time

def run_single_seed_task(frac, seed):
    try:
        e, s, t = run_custom_dqc_experiment(frac, seed)
        return seed, (e, s, t)
    except Exception as e:
        print(f"Seed {seed} failed: {e}")
        return seed, None

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    CZ_FRACS_TO_RUN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    TARGET_SUCCESSFUL_SEEDS = 5
    STARTING_SEED = 42

    RESULTS_FILE = os.path.join(RESULTS_DIR, "three_square_cz_frac_hybrid.json")
    db = load_experiment_database(RESULTS_FILE)
    
    METHOD_LABEL = "PartitioningHeterogeneous + Custom Lookahead SABRE"
    METHOD_COLOR = "#8338ec"
    METHOD_IDX = get_or_create_method_index(db, METHOD_LABEL, METHOD_COLOR)

    for frac in CZ_FRACS_TO_RUN:
        if frac in db.get("cz_frac_list", []):
            frac_idx = db["cz_frac_list"].index(frac)
        else:
            frac_idx = len(db.setdefault("cz_frac_list", []))
            db["cz_frac_list"].append(frac)
            
        m = db["methods_data"][METHOD_IDX]
        while len(m["ebits"]) <= frac_idx:
            m["ebits"].append(None); m["swaps"].append(None); m["time"].append(None)
            m.setdefault("raw_ebits", []).append([]); m.setdefault("raw_swaps", []).append([]); m.setdefault("raw_time", []).append([])

        if m["ebits"][frac_idx] is not None and len(m["raw_ebits"][frac_idx]) >= TARGET_SUCCESSFUL_SEEDS:
            print(f"Skipping CZ Fraction {frac}: Already computed.")
            continue

        print(f"\n--- Evaluating CZ Fraction: {frac} ---")
        successful_runs = []
        current_seed = STARTING_SEED

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_seed = {}
            for _ in range(TARGET_SUCCESSFUL_SEEDS):
                future_to_seed[executor.submit(run_single_seed_task, frac, current_seed)] = current_seed
                current_seed += 1

            for future in concurrent.futures.as_completed(future_to_seed):
                seed, res = future.result()
                if res is not None:
                    successful_runs.append(res)
                    print(f" -> Seed {seed} completed. Cost: {res[0]} EPRs")
                else:
                    future_to_seed[executor.submit(run_single_seed_task, frac, current_seed)] = current_seed
                    current_seed += 1

        if len(successful_runs) == TARGET_SUCCESSFUL_SEEDS:
            t_ebits = [r[0] for r in successful_runs]
            t_swaps = [r[1] for r in successful_runs]
            t_time  = [r[2] for r in successful_runs]
            
            m["ebits"][frac_idx] = compute_statistics(t_ebits)
            m["swaps"][frac_idx] = compute_statistics(t_swaps)
            m["time"][frac_idx] = compute_statistics(t_time)
            m["raw_ebits"][frac_idx] = t_ebits
            m["raw_swaps"][frac_idx] = t_swaps
            m["raw_time"][frac_idx] = t_time

            save_experiment_database(db, RESULTS_FILE)
            print(f"Saved results for fraction {frac}.")