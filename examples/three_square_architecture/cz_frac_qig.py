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
import gc
import time
from time import perf_counter
import concurrent.futures

# --- 2. Third-Party Libraries ---
import numpy as np
import networkx as nx

# --- 3. Quantum Frameworks ---
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    SabreLayout,
)
from qiskit.compiler import transpile
from qiskit.transpiler import AnalysisPass
from qiskit_ibm_runtime import QiskitRuntimeService

# Pytket-dqc
from pytket_dqc.utils import DQCPass
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

# Custom Project Modules
from mqpu_utils import (
    build_cz_fraction_circuit,
    generate_multi_qpu_backend_from_monolithic_backend_with_links,
    sanitize_qiskit_labels,
    rectangular_backend
)
from experiment_utils import *
# QIG partitioning now comes from this repository's own `qig-partitioning`
# package rather than an inline copy. See examples/README.md.
from qig_partitioning import get_heterogeneous_core_assignment
LF_FLAG = True # For consistency with other baselines; doesn't affect anything in the KaHyPar partitioning itself.

# Suppress logs
logging.getLogger("qiskit").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 0. KAHYPAR CONFIGURATION
# ==========================================
# IMPORTANT: Point this to your KaHyPar configuration file
# `None` selects the km1_kKaHyPar_sea20.ini bundled with qig-partitioning.
KAHYPAR_CONFIG_PATH = os.environ.get("DQC_KAHYPAR_CONFIG") or None

# ==========================================
# 1. HARDWARE & TOPOLOGY SETUP (SQUARE)
# ==========================================
n_qpus, l = 3, 4
basis_gates = ['cz', 'id', 'rz', 'sx', 'x']
square_backend = rectangular_backend(l, l, basis_gates=basis_gates, seed=42) # For visualization and layout generation purposes
remote_links = [
    (3, 31), (7, 27),    # 0 to 1
    (15, 35), (11, 39),  # 0 to 2
    (23, 43), (19, 47),  # 1 to 2
]
inter_qpu_edges = remote_links + [(v, u) for u, v in remote_links]
mqpu_backend = generate_multi_qpu_backend_from_monolithic_backend_with_links( # Multi-QPU for the square topology
    n_qpus=n_qpus,
    monolithic_backend=square_backend,
    inter_qpu_links=inter_qpu_edges,
    backend_name=f'{n_qpus}_Square_r1'
)

# For Layout Initialization
server_qubits = {
    0: range(0, l**2), 
    1: range(l**2, 2 * l**2), 
    2: range(2 * l**2, 3 * l**2)
}

# Cost matrix for Flamingo (Uniform triangle)
core_cost_matrix = {
    0: {0: 0.0, 1: 1.0, 2: 1.0},
    1: {0: 1.0, 1: 0.0, 2: 1.0},
    2: {0: 1.0, 1: 1.0, 2: 0.0}
}

# Helpers for Custom SABRE Lookahead (Used in Baselines)
qubit_qpu_map = [i // (l**2) for i in range(mqpu_backend.num_qubits)]
inter_qpu_coupling_map = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]

def generate_custom_distance_matrix(mqpu_backend, inter_edges, factor=10):
    weighted_edges = [(e[0], e[1], 1) if e in inter_edges else (e[0], e[1], factor) for e in mqpu_backend.coupling_map.get_edges()]
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    return nx.floyd_warshall_numpy(G, range(len(G.nodes)))

S_matrix_default = generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=1.0)
S_matrix_1_10 = generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=10.0)

# ==========================================
# 2. CIRCUIT GENERATION
# ==========================================
def _generate_base_circuit(cz_frac, seed):
    # Scale exactly to Heron's size
    qc = build_cz_fraction_circuit(n=36, d=36, p=cz_frac, seed=seed)
    
    clean_data = [inst for inst in qc.data if inst.operation.name not in ['measure', 'reset', 'barrier', 'delay']]
    qc.data = clean_data
    qc.cregs.clear(); qc.clbits.clear()
    
    qc = transpile(qc, basis_gates=['cz', 'id', 'rz', 'sx', 'x'], optimization_level=2)
    
    # --- Re-add DQCPass for scientific consistency with other baselines ---
    circ_tk = qiskit_to_tk(qc) 
    DQCPass().apply(circ_tk)
    qc_dqc = tk_to_qiskit(circ_tk)
    
    # Sanitize the labels to ensure SABRE accepts the register format
    return sanitize_qiskit_labels(qc_dqc)

# ==========================================
# 3. KAHYPAR QIG PARTITIONING LOGIC
# ==========================================
class InjectStartingLayout(AnalysisPass):
    def __init__(self, layouts):
        super().__init__()
        self.custom_layouts = layouts
    def run(self, dag):
        self.property_set["sabre_starting_layouts"] = self.custom_layouts

def generate_core_respecting_layout(mapping, qc, seed):
    np.random.seed(seed)
    physical_layout = {}
    for core in range(3):
        # Extract Qiskit qubits mapped to this core
        qubits_in_core = sorted([q for q, c in mapping.items() if c == core], key=lambda q: qc.find_bit(q).index)
        available = list(server_qubits[core]).copy()
        
        for q in qubits_in_core:
            assigned_physical = int(np.random.choice(available))
            physical_layout[q] = assigned_physical
            available.remove(assigned_physical)
            
    return Layout(physical_layout)

def run_kahypar_qig_experiment(qc, orig_intra_swaps, orig_inter_swaps, seed, seed_start_time, timeout_limit, f_weight=10, S_matrix=S_matrix_default, extended_set_length=20, cla=False, name="KaHyPar QIG + Default SABRE"):
    pid = os.getpid()
    print(f"[Worker {pid} | Seed {seed}] Starting {name}...")
    
    best_cost = float('inf')
    best_metrics = {}
    best_time = 0.0

    routing_metrics = {}
    inter_edges_set = {tuple(sorted(e)) for e in inter_qpu_edges}

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
                    elif is_inter_qpu: czs += 1
            routing_metrics['final_intra_swaps'] = intra
            routing_metrics['final_inter_swaps'] = inter
            routing_metrics['final_inter_czs'] = czs

    rng = np.random.default_rng(seed)
    layout_seeds = rng.integers(0, 2**31 - 1, size=5)

    print(f"  -> [Worker {pid} | Seed {seed}] Running KaHyPar Partitioning...")
    t_dist_0 = perf_counter()
    mapping = get_heterogeneous_core_assignment(qc, core_cost_matrix, KAHYPAR_CONFIG_PATH, max_capacity=(l**2))
    dt_distribution = perf_counter() - t_dist_0
    print(f"  -> [Worker {pid} | Seed {seed}] Partitioning finished in {dt_distribution:.2f}s. Running 5 Layout trials...")
    
    for l_idx, l_seed in enumerate(layout_seeds):
        if time.time() - seed_start_time > timeout_limit:
            raise TimeoutError(f"Seed {seed} exceeded the {timeout_limit}s limit.")

        l_seed = int(l_seed)
        
        if cla: # Custom Lookahead enabled
            pm = PassManager([
                InjectStartingLayout([generate_core_respecting_layout(mapping, qc, seed=l_seed)]),
                SabreLayout(
                    coupling_map=mqpu_backend.coupling_map, 
                    routing_pass=None,
                    seed=l_seed, 
                    layout_trials=0, # Force it to use injected layout
                    max_iterations=3,
                    swap_trials=5,
                    # Custom Lookahead SABRE parameters
                    penalized_swaps=inter_qpu_edges,
                    qubit_qpu_map=qubit_qpu_map,
                    inter_qpu_coupling_map=inter_qpu_coupling_map,
                    alpha=9.0,
                    beta=3.0,
                    extended_set_length=extended_set_length,
                    custom_distance_matrix=S_matrix.tolist() if hasattr(S_matrix, 'tolist') else S_matrix, 
                )
            ])
            routing_metrics.clear()
        else: # No Custom Lookahead
            pm = PassManager([
                InjectStartingLayout([generate_core_respecting_layout(mapping, qc, seed=l_seed)]),
                SabreLayout(
                    coupling_map=mqpu_backend.coupling_map, 
                    routing_pass=None,
                    seed=l_seed, 
                    layout_trials=0, # Force it to use injected layout
                    max_iterations=3,
                    swap_trials=5,
                    extended_set_length=extended_set_length,
                    custom_distance_matrix=S_matrix.tolist() if hasattr(S_matrix, 'tolist') else S_matrix, 
                )
            ])
            routing_metrics.clear()
            
        t_route_0 = perf_counter()
        _ = pm.run(qc, callback=audit_sabreswap)
        trial_time = perf_counter() - t_route_0
        
        total_eprs = (routing_metrics.get('final_inter_swaps', 0) * 3) + routing_metrics.get('final_inter_czs', 0)
        cost = (total_eprs * f_weight) + routing_metrics.get('final_intra_swaps', 0)
        
        print(f"    -> [Worker {pid} | Seed {seed}] SABRE Layout Trial {l_idx+1}/5 finished in {trial_time:.2f}s (Cost: {cost})")
        
        if cost < best_cost: 
            best_cost = cost
            best_metrics = routing_metrics.copy()
            best_time = dt_distribution + trial_time
            
        del pm; gc.collect()

    final_total_ebits = (best_metrics.get('final_inter_swaps', 0) * 3) + best_metrics.get('final_inter_czs', 0)
    final_added_swaps = best_metrics.get('final_intra_swaps', 0) - orig_intra_swaps

    print(f"[Worker {pid} | Seed {seed}] Finished KaHyPar method. Best EPRs: {final_total_ebits}")
    return final_total_ebits, final_added_swaps, best_time


def get_baseline_pass_manager(mqpu_backend, S_matrix, seed, penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20):
    pm = generate_preset_pass_manager(optimization_level=0, backend=mqpu_backend, seed_transpiler=seed)
    
    sl = SabreLayout(
        coupling_map=mqpu_backend.coupling_map,
        routing_pass=None,        
        seed=seed,
        max_iterations=3,
        swap_trials=5,            
        layout_trials=1,          
        penalized_swaps=penalized_swaps,
        qubit_qpu_map=qubit_qpu_map,
        inter_qpu_coupling_map=inter_qpu_coupling_map,
        alpha=9.0,
        beta=3.0,
        extended_set_length=extended_set_length,
        custom_distance_matrix=S_matrix.tolist() if hasattr(S_matrix, 'tolist') else S_matrix, 
    )
    
    pm.layout.replace(index=2, passes=[BarrierBeforeFinalMeasurements(), sl])
    pm.routing = PassManager() 
    return pm

def run_baseline_experiment(qc, orig_intra_swaps, orig_inter_swaps, seed, mqpu_backend, inter_qpu_edges, S_matrix, seed_start_time, timeout_limit, penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20, strategy_name="Baseline"):
    pid = os.getpid()
    print(f"[Worker {pid} | Seed {seed}] Starting {strategy_name} (5 trials)...")
    
    num_trials = 5
    f_weight = 10
    best_cost = float('inf')
    best_metrics = {}
    best_time = 0.0

    routing_metrics = {}
    inter_qpu_edges_set = {tuple(sorted(edge)) for edge in inter_qpu_edges}

    def audit_sabreswap(pass_, dag, time, property_set, count):
        if pass_.name() == "SabreLayout":
            intra_swaps, inter_swaps, inter_czs = 0, 0, 0
            for node in dag.op_nodes():
                if len(node.qargs) == 2 and node.op.name not in ["barrier", "routing_placeholder"]:
                    q0_idx, q1_idx = dag.find_bit(node.qargs[0]).index, dag.find_bit(node.qargs[1]).index
                    is_inter_qpu = tuple(sorted((q0_idx, q1_idx))) in inter_qpu_edges_set
                    if node.op.name == "swap":
                        if is_inter_qpu: inter_swaps += 1
                        else: intra_swaps += 1
                    elif is_inter_qpu:
                        inter_czs += 1
            routing_metrics['final_intra_swaps'] = intra_swaps 
            routing_metrics['final_inter_swaps'] = inter_swaps
            routing_metrics['final_inter_czs'] = inter_czs

    for trial_idx in range(num_trials):
        if time.time() - seed_start_time > timeout_limit:
            raise TimeoutError(f"Seed {seed} exceeded the {timeout_limit}s limit.")

        current_seed = seed + trial_idx 
        pm = get_baseline_pass_manager(mqpu_backend, S_matrix, current_seed, penalized_swaps, qubit_qpu_map, inter_qpu_coupling_map, extended_set_length)
        routing_metrics.clear()
        
        t_0 = perf_counter()
        _ = pm.run(qc, callback=audit_sabreswap)
        trial_time = perf_counter() - t_0
        
        intra_swaps = routing_metrics.get('final_intra_swaps', 0)
        total_eprs = (routing_metrics.get('final_inter_swaps', 0) * 3) + routing_metrics.get('final_inter_czs', 0)
        aggregated_cost = (total_eprs * f_weight) + intra_swaps
        
        print(f"  -> [Worker {pid} | Seed {seed}] {strategy_name} Trial {trial_idx+1}/{num_trials} finished in {trial_time:.2f}s (Cost: {aggregated_cost})")

        if aggregated_cost < best_cost:
            best_cost = aggregated_cost
            best_metrics = routing_metrics.copy()
            best_time = trial_time

        del pm; gc.collect()

    added_intra_swaps = best_metrics['final_intra_swaps'] - orig_intra_swaps
    total_ebits = (best_metrics['final_inter_swaps'] * 3) + best_metrics['final_inter_czs']
    
    print(f"[Worker {pid} | Seed {seed}] Finished {strategy_name}. Best EPRs: {total_ebits}")
    return total_ebits, added_intra_swaps, best_time

# ==========================================
# 5. UNIFIED PARALLEL TASK
# ==========================================
def run_single_seed_task(frac, seed, run_dqc1=True, run_dqc2=True, run_dqc3=True, run_dqc4=True, run_base=True, run_default=True, run_custom=True, run_custom_e100=True, timeout_limit=7200):
    pid = os.getpid()
    seed_start_time = time.time()
    
    dqc_res1, dqc_res2, dqc_res3, dqc_res4, base_res, default_res, custom_res, custom_res_e100 = None, None, None, None, None, None, None, None
    try:
        print(f"\n[Worker {pid} | Seed {seed}] --- Generating Base Circuit for CZ Frac {frac} ---")
        t_gen_0 = perf_counter()
        qc = _generate_base_circuit(frac, seed)
        print(f"[Worker {pid} | Seed {seed}] Circuit generated in {perf_counter() - t_gen_0:.2f}s.")

        orig_intra_swaps, orig_inter_swaps = 0, 0
        for instr in qc.data:
            if instr.operation.name == "swap":
                q0, q1 = qc.find_bit(instr.qubits[0]).index, qc.find_bit(instr.qubits[1]).index
                if (q0, q1) in inter_qpu_edges or (q1, q0) in inter_qpu_edges: orig_inter_swaps += 1
                else: orig_intra_swaps += 1

        # 1. New Method: KaHyPar QIG + Default SABRE
        if run_dqc1:
            dqc_res1 = run_kahypar_qig_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, seed_start_time, timeout_limit,
                S_matrix=S_matrix_default, cla=False,
                extended_set_length=20,
                name="KaHyPar QIG + Default SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + Default SABRE as per configuration.")
    
        # 2. New Method: KaHyPar QIG + (1,10) SABRE
        if run_dqc2:
            dqc_res2 = run_kahypar_qig_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, seed_start_time, timeout_limit,
                S_matrix=S_matrix_1_10, cla=False,
                extended_set_length=20,
                name="KaHyPar QIG + (1,10) SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + (1,10) SABRE as per configuration.")

        # 3. New Method: KaHyPar QIG + Custom Lookahead SABRE
        if run_dqc3:
            dqc_res3 = run_kahypar_qig_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, seed_start_time, timeout_limit,
                S_matrix=S_matrix_default, cla=True,
                extended_set_length=20,
                name="KaHyPar QIG + Custom Lookahead SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + Custom Lookahead SABRE as per configuration.")

        # 4. New Method: KaHyPar QIG + Custom Lookahead SABRE (e=100)
        if run_dqc4:
            dqc_res4 = run_kahypar_qig_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, seed_start_time, timeout_limit,
                S_matrix=S_matrix_default, cla=True,
                extended_set_length=100,
                name="KaHyPar QIG + Custom Lookahead SABRE (e=100)"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + Custom Lookahead SABRE (e=100) as per configuration.")

        # 5. (1, 10) SABRE
        if run_base:
            base_res = run_baseline_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, mqpu_backend, inter_qpu_edges,
                S_matrix_1_10, seed_start_time,
                timeout_limit, strategy_name="(1, 10) SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping (1, 10) SABRE as per configuration.")

        # 6. Default SABRE Baseline
        if run_default:
            default_res = run_baseline_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, mqpu_backend, inter_qpu_edges,
                S_matrix_default, seed_start_time,
                timeout_limit, strategy_name="Default SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping Default SABRE as per configuration.")

        # 7. Custom Lookahead SABRE
        if run_custom:
            custom_res = run_baseline_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, mqpu_backend, inter_qpu_edges,
                S_matrix_default, seed_start_time,
                timeout_limit, penalized_swaps=inter_qpu_edges,
                qubit_qpu_map=qubit_qpu_map,
                inter_qpu_coupling_map=inter_qpu_coupling_map,
                extended_set_length=20,
                strategy_name="Custom Lookahead SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping Custom Lookahead SABRE as per configuration.")

        # 8. Custom Lookahead SABRE (e=100)
        if run_custom_e100:
            custom_res_e100 = run_baseline_experiment(
                qc, orig_intra_swaps, orig_inter_swaps,
                seed, mqpu_backend, inter_qpu_edges,
                S_matrix_default, seed_start_time,
                timeout_limit, penalized_swaps=inter_qpu_edges,
                qubit_qpu_map=qubit_qpu_map,
                inter_qpu_coupling_map=inter_qpu_coupling_map,
                extended_set_length=100,
                strategy_name="Custom Lookahead SABRE (e=100)"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping Custom Lookahead SABRE (e=100) as per configuration.")


        del qc; gc.collect()
        print(f"[Worker {pid} | Seed {seed}] --- ALL TASKS COMPLETED FOR SEED ---")
        return seed, dqc_res1, dqc_res2, dqc_res3, dqc_res4, base_res, default_res, custom_res, custom_res_e100

    except TimeoutError as e:
        print(f"[Worker {pid} | Seed {seed}] -> ABORTED: {e}")
        return seed, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"[Worker {pid} | Seed {seed}] -> FAILED with unexpected error: {e}")
        return seed, None, None, None, None, None, None, None, None

# ==========================================
# 6. PARALLELIZED EXPERIMENT SETUP
# ==========================================
if __name__ == "__main__":
    CZ_FRACS_TO_RUN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    TARGET_SUCCESSFUL_SEEDS = 5
    STARTING_SEED = 42
    
    PER_SEED_TIMEOUT = 1.5 * 3600 

    RESULTS_FILE = os.path.join(RESULTS_DIR, f"three_square_cz_frac_lf={LF_FLAG}.json")
    db = load_experiment_database(RESULTS_FILE)
    
    WORKERS = 5

    # Dynamic Index Mapping updated to reflect new algorithm!
    DQC_LABEL_1 = "KaHyPar QIG + Default SABRE"
    DQC_LABEL_2 = "KaHyPar QIG + (1,10) SABRE"
    DQC_LABEL_3 = "KaHyPar QIG + Custom Lookahead SABRE"
    DQC_LABEL_4 = "KaHyPar QIG + Custom Lookahead SABRE (e=100)"

    DQC1_IDX = get_or_create_method_index(db, DQC_LABEL_1, "#384d83") 
    DQC2_IDX = get_or_create_method_index(db, DQC_LABEL_2, "#384d83")
    DQC3_IDX = get_or_create_method_index(db, DQC_LABEL_3, "#384d83")
    DQC4_IDX = get_or_create_method_index(db, DQC_LABEL_4, "#384d83")

    BASE_IDX = get_or_create_method_index(db, "(1,10) SABRE", "#00b4d8")
    DEFAULT_IDX = get_or_create_method_index(db, "Default SABRE Baseline", "#90e0ef")
    CUSTOM_IDX = get_or_create_method_index(db, "Custom Lookahead SABRE", "#0077b6")
    CUSTOM_E100_IDX = get_or_create_method_index(db, "Custom Lookahead SABRE (e=100)", "#0077b6")

    for frac in CZ_FRACS_TO_RUN:
        if frac in db.get("cz_frac_list", []): frac_idx = db["cz_frac_list"].index(frac)
        else:
            frac_idx = len(db.setdefault("cz_frac_list", []))
            db["cz_frac_list"].append(frac)
            
        for m in db["methods_data"]:
            while len(m["ebits"]) <= frac_idx:
                m["ebits"].append(None); m["swaps"].append(None); m["time"].append(None)
                m.setdefault("raw_ebits", []).append([]); m.setdefault("raw_swaps", []).append([]); m.setdefault("raw_time", []).append([])

        def has_data(db, m_idx, frac_idx):
            if m_idx >= len(db["methods_data"]): return False
            method = db["methods_data"][m_idx]
            if "ebits" not in method or frac_idx >= len(method["ebits"]): return False
            val = method["ebits"][frac_idx]
            if val is None or val == []: return False
            if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], float) and math.isnan(val[0]): return True
            return True

        # Check all 8 methods!
        needs_dqc1 = not has_data(db, DQC1_IDX, frac_idx)
        needs_dqc2 = not has_data(db, DQC2_IDX, frac_idx)
        needs_dqc3 = not has_data(db, DQC3_IDX, frac_idx)
        needs_dqc4 = not has_data(db, DQC4_IDX, frac_idx)

        needs_base = not has_data(db, BASE_IDX, frac_idx)
        needs_default = not has_data(db, DEFAULT_IDX, frac_idx)
        needs_custom = not has_data(db, CUSTOM_IDX, frac_idx)
        needs_custom_e100 = not has_data(db, CUSTOM_E100_IDX, frac_idx)

        if not (needs_dqc1 or needs_dqc2 or needs_dqc3 or needs_dqc4 or needs_base or needs_default or needs_custom or needs_custom_e100):
            print(f"Skipping CZ Fraction {frac}: All requested variants already computed.")
            continue
            
        print(f"\n--- Evaluating CZ Fraction: {frac} ---")
        
        successful_runs = []
        current_seed = STARTING_SEED

        historical_seeds = []
        if frac_idx < len(db.get("seed_history", [])) and db["seed_history"][frac_idx]:
            historical_seeds = list(db["seed_history"][frac_idx])

        enforce_history = bool(historical_seeds and not needs_dqc1 and not needs_dqc2 and not needs_dqc3 and not needs_dqc4)

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)
        future_to_seed = {}
        
        for i in range(TARGET_SUCCESSFUL_SEEDS):
            seed_val = historical_seeds[i] if enforce_history else current_seed
            if not enforce_history: current_seed += 1
            fut = executor.submit(run_single_seed_task, frac, seed_val, needs_dqc1, needs_dqc2, needs_dqc3, needs_dqc4, needs_base, needs_default, needs_custom, needs_custom_e100, PER_SEED_TIMEOUT)
            future_to_seed[fut] = seed_val

        MAX_ALLOWED_FAILURES = 20 
        total_failures = 0

        try:
            while len(successful_runs) < TARGET_SUCCESSFUL_SEEDS:
                done, not_done = concurrent.futures.wait(
                    future_to_seed.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    finished_seed = future_to_seed.pop(future)
                    try:
                        seed_res, dqc_res1, dqc_res2, dqc_res3, dqc_res4, base_res, default_res, custom_res, custom_res_e100 = future.result()
                        
                        dqc1_ok = (not needs_dqc1) or (dqc_res1 is not None)
                        dqc2_ok = (not needs_dqc2) or (dqc_res2 is not None)
                        dqc3_ok = (not needs_dqc3) or (dqc_res3 is not None)
                        dqc4_ok = (not needs_dqc4) or (dqc_res4 is not None)

                        base_ok = (not needs_base) or (base_res is not None)
                        def_ok = (not needs_default) or (default_res is not None)
                        cust_ok = (not needs_custom) or (custom_res is not None)
                        cust_e100_ok = (not needs_custom_e100) or (custom_res_e100 is not None)
                        
                        if dqc1_ok and dqc2_ok and dqc3_ok and dqc4_ok and base_ok and def_ok and cust_ok and cust_e100_ok:
                            successful_runs.append((finished_seed, dqc_res1, dqc_res2, dqc_res3, dqc_res4, base_res, default_res, custom_res, custom_res_e100))
                            print(f"  -> Seed {finished_seed} completed successfully! ({len(successful_runs)}/{TARGET_SUCCESSFUL_SEEDS})")
                        else:
                            print(f"  -> Seed {finished_seed} failed or timed out. Spinning up a replacement seed.")
                            total_failures += 1
                            if total_failures < MAX_ALLOWED_FAILURES:
                                fut = executor.submit(run_single_seed_task, frac, current_seed, needs_dqc1, needs_dqc2, needs_dqc3, needs_dqc4, needs_base, needs_default, needs_custom, needs_custom_e100, PER_SEED_TIMEOUT)
                                future_to_seed[fut] = current_seed
                                current_seed += 1
                    except Exception as exc:
                        total_failures += 1
                        if total_failures < MAX_ALLOWED_FAILURES:
                            fut = executor.submit(run_single_seed_task, frac, current_seed, needs_dqc1, needs_dqc2, needs_dqc3, needs_dqc4, needs_base, needs_default, needs_custom, needs_custom_e100, PER_SEED_TIMEOUT)
                            future_to_seed[fut] = current_seed
                            current_seed += 1
                            
                if total_failures >= MAX_ALLOWED_FAILURES:
                    print(f"Reached max allowed failures ({MAX_ALLOWED_FAILURES}). Moving to save.")
                    break

        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if len(successful_runs) == 0:
            print(f"WARNING: CZ Frac {frac} yielded 0 successful runs (likely timed out). Skipping save.")
            continue
        
        def update_db(m_idx, needs_run, res_idx):
            if not needs_run or len(successful_runs) == 0: return
            t_ebits, t_swaps, t_time = [], [], []
            for run_data in successful_runs:
                res = run_data[res_idx] 
                if res is not None:
                    t_ebits.append(res[0]); t_swaps.append(res[1]); t_time.append(res[2])
                
            if len(t_ebits) == 0: return 
                
            db["methods_data"][m_idx]["ebits"][frac_idx] = compute_statistics(t_ebits)
            db["methods_data"][m_idx]["swaps"][frac_idx] = compute_statistics(t_swaps)
            db["methods_data"][m_idx]["time"][frac_idx] = compute_statistics(t_time)
            db["methods_data"][m_idx]["raw_ebits"][frac_idx] = t_ebits
            db["methods_data"][m_idx]["raw_swaps"][frac_idx] = t_swaps
            db["methods_data"][m_idx]["raw_time"][frac_idx] = t_time

        if needs_dqc1: update_db(DQC1_IDX, True, 1)
        if needs_dqc2: update_db(DQC2_IDX, True, 2)
        if needs_dqc3: update_db(DQC3_IDX, True, 3)
        if needs_dqc4: update_db(DQC4_IDX, True, 4)

        update_db(BASE_IDX, needs_base, 5)
        update_db(DEFAULT_IDX, needs_default, 6)
        update_db(CUSTOM_IDX, needs_custom, 7)
        update_db(CUSTOM_E100_IDX, needs_custom_e100, 8)

        while len(db.setdefault("seed_history", [])) <= frac_idx: db["seed_history"].append([])
        db["seed_history"][frac_idx] = [r[0] for r in successful_runs]

        save_experiment_database(db, RESULTS_FILE)
        print(f"Results for CZ Fraction {frac} successfully updated and saved to disk.")