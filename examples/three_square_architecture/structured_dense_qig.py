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
import re
import math
import json 
import warnings 
import time
import gc
from time import perf_counter
import concurrent.futures 
import multiprocessing as mp

# --- 2. Third-Party Libraries ---
import numpy as np
import networkx as nx

# --- 3. Quantum Frameworks ---
from qiskit.transpiler import CouplingMap, Target, PassManager, Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    BarrierBeforeFinalMeasurements,
    SabreLayout,
)
from qiskit.compiler import transpile
from qiskit.transpiler import AnalysisPass
from qiskit.qasm2 import load as load_qasm2, LEGACY_CUSTOM_INSTRUCTIONS
from qiskit_ibm_runtime import QiskitRuntimeService

# Pytket & Qiskit Extensions
from pytket.extensions.qiskit import tk_to_qiskit, qiskit_to_tk
from pytket_dqc.utils import DQCPass

# --- 4. Local/Custom Project Modules ---
from mqpu_utils import (
    generate_multi_qpu_backend_from_monolithic_backend_with_links,
    sanitize_qiskit_labels,
    rectangular_backend
)
from experiment_utils import *
# QIG partitioning now comes from this repository's own `qig-partitioning`
# package rather than an inline copy. See examples/README.md.
from qig_partitioning import get_heterogeneous_core_assignment
LF_FLAG = True # For consistency with other baselines; doesn't affect anything in the KaHyPar partitioning itself.

# Suppress excessive logging from the transpiler
logging.getLogger("qiskit").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 0. KAHYPAR CONFIGURATION
# ==========================================
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

qubit_qpu_map = [i // (l**2) for i in range(mqpu_backend.num_qubits)]
inter_qpu_coupling_map = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]

def generate_custom_distance_matrix(mqpu_backend, inter_edges, factor=10):
    weighted_edges = [(e[0], e[1], 1) if e in inter_edges else (e[0], e[1], factor) for e in mqpu_backend.coupling_map.get_edges()]
    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    return nx.floyd_warshall_numpy(G, range(len(G.nodes)))

S_matrix_1_10 = generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=10.0)
S_matrix_default = generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=1.0)

# ==========================================
# 2. CIRCUIT GENERATION & PREPARATION
# ==========================================
def load_and_prepare_benchmark(file_path):
    t_0 = perf_counter()
    qc = load_qasm2(file_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS) 
    dt_load = perf_counter() - t_0
    print(f"[load_qasm2] Time taken = {dt_load:.4f}s") 

    clean_data = [inst for inst in qc.data if inst.operation.name not in ['measure', 'reset', 'barrier', 'delay']]
    qc.data = clean_data
    qc.cregs.clear()
    qc.clbits.clear()

    qc = transpile(qc, basis_gates=['cz', 'id', 'rz', 'sx', 'x'], optimization_level=2)
    circ = qiskit_to_tk(qc) 
    
    t_0 = perf_counter()
    DQCPass().apply(circ)
    dt_dqc_pass = perf_counter() - t_0
    print(f"[DQCPass] Time taken = {dt_dqc_pass:.4f}s") 

    cz_qiskit_sanitized = sanitize_qiskit_labels(tk_to_qiskit(circ))

    del qc, circ; gc.collect() 
    return cz_qiskit_sanitized

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
        qubits_in_core = sorted([q for q, c in mapping.items() if c == core], key=lambda q: qc.find_bit(q).index)
        available = list(server_qubits[core]).copy()
        for q in qubits_in_core:
            assigned_physical = int(np.random.choice(available))
            physical_layout[q] = assigned_physical
            available.remove(assigned_physical)
    return Layout(physical_layout)

def run_kahypar_qig_experiment(qc, seed, seed_start_time, timeout_limit, f_weight=10, S_matrix=S_matrix_default, extended_set_length=20, cla=False, name="KaHyPar QIG + Default SABRE"):
    pid = os.getpid()
    print(f"[Worker {pid} | Seed {seed}] Starting {name} (5 trials)...")
    
    local_qc = qc.copy()
    orig_intra_swaps, orig_inter_swaps = 0, 0
    for instr in local_qc.data:
        if instr.operation.name == "swap":
            q0_idx = local_qc.find_bit(instr.qubits[0]).index
            q1_idx = local_qc.find_bit(instr.qubits[1]).index
            if (q0_idx, q1_idx) in inter_qpu_edges or (q1_idx, q0_idx) in inter_qpu_edges: orig_inter_swaps += 1
            else: orig_intra_swaps += 1

    best_cost = float('inf')
    best_metrics = {}
    best_time = 0.0

    routing_metrics = {}
    inter_qpu_edges_set = {tuple(sorted(edge)) for edge in inter_qpu_edges}

    def audit_sabreswap(pass_, dag, time, property_set, count):
        if pass_.name() == "SabreLayout":
            intra_swaps, inter_swaps, inter_standard_ops, inter_other_ops = 0, 0, 0, 0 
            for node in dag.op_nodes():
                if len(node.qargs) == 2:
                    op_name = node.op.name
                    if op_name in ["barrier", "routing_placeholder"]: continue
                    q0_idx = dag.find_bit(node.qargs[0]).index
                    q1_idx = dag.find_bit(node.qargs[1]).index
                    is_inter_qpu = tuple(sorted((q0_idx, q1_idx))) in inter_qpu_edges_set
                    
                    if op_name == "swap":
                        if is_inter_qpu: inter_swaps += 1
                        else: intra_swaps += 1
                    elif is_inter_qpu:
                        if op_name in ["cz", "cx", "cu1", "cp"]: inter_standard_ops += 1
                        else: inter_other_ops += 1
                            
            routing_metrics['final_intra_swaps'] = intra_swaps 
            routing_metrics['final_inter_swaps'] = inter_swaps
            routing_metrics['final_inter_czs'] = inter_standard_ops + inter_other_ops

    rng = np.random.default_rng(seed)
    layout_seeds = rng.integers(0, 2**31 - 1, size=5)

    print(f"  -> [Worker {pid} | Seed {seed}] Running KaHyPar Partitioning...")
    t_dist_0 = perf_counter()
    mapping = get_heterogeneous_core_assignment(local_qc, core_cost_matrix, KAHYPAR_CONFIG_PATH, max_capacity=(l**2))
    dt_distribution = perf_counter() - t_dist_0
    print(f"  -> [Worker {pid} | Seed {seed}] Partitioning finished in {dt_distribution:.4f}s.")
    
    for trial_idx, l_seed in enumerate(layout_seeds):
        if time.time() - seed_start_time > timeout_limit:
            raise TimeoutError(f"Seed {seed} exceeded the {timeout_limit}s limit.")

        l_seed = int(l_seed)
        
        if cla: # Custom Lookahead enabled
            pm = PassManager([
                InjectStartingLayout([generate_core_respecting_layout(mapping, local_qc, seed=l_seed)]),
                SabreLayout(
                    coupling_map=mqpu_backend.coupling_map, 
                    routing_pass=None,
                    seed=l_seed, 
                    layout_trials=0, 
                    max_iterations=3,
                    swap_trials=5,
                    penalized_swaps=inter_qpu_edges,
                    qubit_qpu_map=qubit_qpu_map,
                    inter_qpu_coupling_map=inter_qpu_coupling_map,
                    alpha=9.0,
                    beta=3.0,
                    extended_set_length=extended_set_length,
                    custom_distance_matrix=S_matrix.tolist() if hasattr(S_matrix, 'tolist') else S_matrix, 
                )
            ])
        else: # Default SABRE behavior
            pm = PassManager([
                InjectStartingLayout([generate_core_respecting_layout(mapping, local_qc, seed=l_seed)]),
                SabreLayout(
                    coupling_map=mqpu_backend.coupling_map, 
                    routing_pass=None,
                    seed=l_seed, 
                    layout_trials=0, 
                    max_iterations=3,
                    swap_trials=5,
                    extended_set_length=extended_set_length,
                    custom_distance_matrix=S_matrix.tolist() if hasattr(S_matrix, 'tolist') else S_matrix, 
                )
            ])

        routing_metrics.clear()
        
        t_route_0 = perf_counter()
        _ = pm.run(local_qc, callback=audit_sabreswap)
        trial_time = perf_counter() - t_route_0
        
        intra_swaps = routing_metrics.get('final_intra_swaps', 0)
        inter_swaps = routing_metrics.get('final_inter_swaps', 0)
        inter_gates = routing_metrics.get('final_inter_czs', 0)
        
        total_eprs = (inter_swaps * 3) + inter_gates
        cost = (total_eprs * f_weight) + intra_swaps
        
        if cost < best_cost: 
            best_cost = cost
            best_metrics = routing_metrics.copy()
            best_time = dt_distribution + trial_time
            
        del pm; gc.collect()

    added_intra_swaps = best_metrics['final_intra_swaps'] - orig_intra_swaps
    total_ebits = (best_metrics['final_inter_swaps'] * 3) + (best_metrics['final_inter_czs'] * 1)

    print(f"[Worker {pid} | Seed {seed}] [{name}] BEST Routing -> Time: {best_time:.4f}s | Inter SWAPs: {best_metrics['final_inter_swaps']} | Inter CZs: {best_metrics['final_inter_czs']} | Intra SWAPs Added: {added_intra_swaps}")
    del local_qc; gc.collect()
    return total_ebits, added_intra_swaps, best_time


def get_baseline_pass_manager(
    mqpu_backend, S_matrix, seed, penalized_swaps=None, qubit_qpu_map=None, 
    inter_qpu_coupling_map=None, alpha=9.0, beta=3.0, extended_set_length=20
):
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
        alpha=alpha,
        beta=beta,
        extended_set_length=extended_set_length,
        custom_distance_matrix=S_matrix.tolist() if hasattr(S_matrix, 'tolist') else S_matrix, 
    )
    pm.layout.replace(index=2, passes=[BarrierBeforeFinalMeasurements(), sl])
    pm.routing = PassManager() 
    return pm

def run_baseline_experiment(cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix, seed_start_time, timeout_limit, penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20, strategy_name="Baseline"):
    pid = os.getpid()
    local_qc = cz_qiskit_sanitized.copy()

    orig_intra_swaps, orig_inter_swaps = 0, 0
    for instr in local_qc.data:
        if instr.operation.name == "swap":
            q0_idx = local_qc.find_bit(instr.qubits[0]).index
            q1_idx = local_qc.find_bit(instr.qubits[1]).index
            if (q0_idx, q1_idx) in inter_qpu_edges or (q1_idx, q0_idx) in inter_qpu_edges: orig_inter_swaps += 1
            else: orig_intra_swaps += 1

    num_trials = 5
    f_weight = 10
    best_cost = float('inf')
    best_metrics = {}
    best_time = 0.0

    routing_metrics = {}
    inter_qpu_edges_set = {tuple(sorted(edge)) for edge in inter_qpu_edges}

    def audit_sabreswap(pass_, dag, time, property_set, count):
        if pass_.name() == "SabreLayout":
            intra_swaps, inter_swaps, inter_standard_ops, inter_other_ops = 0, 0, 0, 0 
            for node in dag.op_nodes():
                if len(node.qargs) == 2:
                    op_name = node.op.name
                    if op_name in ["barrier", "routing_placeholder"]: continue
                    q0_idx = dag.find_bit(node.qargs[0]).index
                    q1_idx = dag.find_bit(node.qargs[1]).index
                    is_inter_qpu = tuple(sorted((q0_idx, q1_idx))) in inter_qpu_edges_set
                    
                    if op_name == "swap":
                        if is_inter_qpu: inter_swaps += 1
                        else: intra_swaps += 1
                    elif is_inter_qpu:
                        if op_name in ["cz", "cx", "cu1", "cp"]: inter_standard_ops += 1
                        else: inter_other_ops += 1
                            
            routing_metrics['final_intra_swaps'] = intra_swaps 
            routing_metrics['final_inter_swaps'] = inter_swaps
            routing_metrics['final_inter_czs'] = inter_standard_ops + inter_other_ops

    print(f"[Worker {pid} | Seed {seed}] [{strategy_name}] Testing {num_trials} baseline random layouts...")

    for trial_idx in range(num_trials):
        if time.time() - seed_start_time > timeout_limit:
            raise TimeoutError(f"Seed {seed} exceeded the {timeout_limit}s limit.")

        current_seed = seed + trial_idx 
        pm = get_baseline_pass_manager(mqpu_backend, S_matrix, current_seed, penalized_swaps, qubit_qpu_map, inter_qpu_coupling_map, extended_set_length=extended_set_length)
        routing_metrics.clear()
        
        t_0 = perf_counter()
        _ = pm.run(local_qc, callback=audit_sabreswap)
        trial_time = perf_counter() - t_0
        
        intra_swaps = routing_metrics.get('final_intra_swaps', 0)
        inter_swaps = routing_metrics.get('final_inter_swaps', 0)
        inter_gates = routing_metrics.get('final_inter_czs', 0)
        
        total_eprs = (inter_swaps * 3) + inter_gates
        aggregated_cost = (total_eprs * f_weight) + intra_swaps
        
        if aggregated_cost < best_cost:
            best_cost = aggregated_cost
            best_metrics = routing_metrics.copy()
            best_time = trial_time 

        del pm; gc.collect()

    added_intra_swaps = best_metrics['final_intra_swaps'] - orig_intra_swaps
    total_ebits = (best_metrics['final_inter_swaps'] * 3) + (best_metrics['final_inter_czs'] * 1)

    print(f"[Worker {pid} | Seed {seed}] [{strategy_name}] BEST Routing -> Time: {best_time:.4f}s | Inter SWAPs: {best_metrics['final_inter_swaps']} | Inter CZs: {best_metrics['final_inter_czs']} | Intra SWAPs Added: {added_intra_swaps}")
    del local_qc; gc.collect()

    return total_ebits, added_intra_swaps, best_time


# ============================================
# 5. UNIFIED PARALLEL TASK
# ============================================
def run_single_seed_task(
    cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix_1_10, S_matrix_default, 
    penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20,
    run_dqc1=True, run_dqc2=True, run_dqc3=True, run_dqc4=True, 
    run_base=True, run_default=True, run_custom=True, run_custom_e100=True,
    timeout_limit=7200
):
    pid = os.getpid()
    seed_start_time = time.time()
    
    dqc_res1, dqc_res2, dqc_res3, dqc_res4 = None, None, None, None
    base_res, default_res, custom_res, custom_res_e100 = None, None, None, None
    
    try:
        # 1. KaHyPar QIG + Default SABRE
        if run_dqc1:
            dqc_res1 = run_kahypar_qig_experiment(
                cz_qiskit_sanitized, seed, seed_start_time, timeout_limit, 
                S_matrix=S_matrix_default, cla=False, extended_set_length=20, 
                name="KaHyPar QIG + Default SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + Default SABRE.")

        # 2. KaHyPar QIG + (1,10) SABRE
        if run_dqc2:
            dqc_res2 = run_kahypar_qig_experiment(
                cz_qiskit_sanitized, seed, seed_start_time, timeout_limit, 
                S_matrix=S_matrix_1_10, cla=False, extended_set_length=20, 
                name="KaHyPar QIG + (1,10) SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + (1,10) SABRE.")

        # 3. KaHyPar QIG + Custom Lookahead SABRE
        if run_dqc3:
            dqc_res3 = run_kahypar_qig_experiment(
                cz_qiskit_sanitized, seed, seed_start_time, timeout_limit, 
                S_matrix=S_matrix_default, cla=True, extended_set_length=20, 
                name="KaHyPar QIG + Custom Lookahead SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + Custom Lookahead SABRE.")

        # 4. KaHyPar QIG + Custom Lookahead SABRE (e=100)
        if run_dqc4:
            dqc_res4 = run_kahypar_qig_experiment(
                cz_qiskit_sanitized, seed, seed_start_time, timeout_limit, 
                S_matrix=S_matrix_default, cla=True, extended_set_length=100, 
                name="KaHyPar QIG + Custom Lookahead SABRE (e=100)"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping KaHyPar QIG + Custom Lookahead SABRE (e=100).")

        # 5. Baseline (1, 10) SABRE
        if run_base:
            base_res = run_baseline_experiment(
                cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix_1_10, 
                seed_start_time, timeout_limit, strategy_name="(1, 10) SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping (1, 10) SABRE baseline.")

        # 6. Default SABRE Baseline
        if run_default:
            default_res = run_baseline_experiment(
                cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix_default, 
                seed_start_time, timeout_limit, strategy_name="Default SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping Default SABRE baseline.")
            
        # 7. Custom Lookahead SABRE
        if run_custom:
            custom_res = run_baseline_experiment(
                cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix_default, 
                seed_start_time, timeout_limit, penalized_swaps=penalized_swaps, 
                qubit_qpu_map=qubit_qpu_map, inter_qpu_coupling_map=inter_qpu_coupling_map, 
                extended_set_length=20, strategy_name="Custom Lookahead SABRE"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping Custom Lookahead SABRE baseline.")

        # 8. Custom Lookahead SABRE (e=100)
        if run_custom_e100:
            custom_res_e100 = run_baseline_experiment(
                cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix_default, 
                seed_start_time, timeout_limit, penalized_swaps=penalized_swaps, 
                qubit_qpu_map=qubit_qpu_map, inter_qpu_coupling_map=inter_qpu_coupling_map, 
                extended_set_length=100, strategy_name="Custom Lookahead SABRE (e=100)"
            )
        else:
            print(f"[Worker {pid} | Seed {seed}] Skipping Custom Lookahead SABRE (e=100) baseline.")

        return seed, dqc_res1, dqc_res2, dqc_res3, dqc_res4, base_res, default_res, custom_res, custom_res_e100
        
    except TimeoutError as e:
        print(f"[Worker {pid} | Seed {seed}] -> ABORTED: {e}")
        return seed, None, None, None, None, None, None, None, None
    except Exception as e:
        print(f"[Worker {pid} | Seed {seed}] -> FAILED with unexpected error: {e}")
        return seed, None, None, None, None, None, None, None, None
    finally:
        gc.collect()


# ==========================================
# 6. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    dense_dir = os.path.join(BENCHMARKS_DIR, 'dense')
    CIRCUITS_TO_RUN = [os.path.join(dense_dir, f) for f in os.listdir(dense_dir) if f.endswith('.qasm')]

    benchmark_dir = BENCHMARKS_DIR
    temp_circuits = []
    qreg_pattern = re.compile(r"qreg\s+[a-zA-Z0-9_]+\s*\[\s*(\d+)\s*\]\s*;")

    def get_qubits_fast(filepath):
        qubits = 0
        with open(filepath, 'r') as f:
            for line in f:
                if line.lstrip().startswith(('cx', 'cz', 'x', 'h', 'rz', 'sx', 'measure', 'barrier')): break
                match = qreg_pattern.search(line)
                if match: qubits += int(match.group(1))
        return qubits

    for root, dirs, files in os.walk(benchmark_dir):
        for file in files:
            if file.endswith('.qasm') and (os.path.basename(root) == 'hamiltonians'): 
                file_path = os.path.join(root, file)
                try:
                    num_qubits = get_qubits_fast(file_path)
                    if 32 < num_qubits <= 48:
                        print(f"Adding {file_path.split('/')[-1]} with {num_qubits} qubits to the experiment list.")
                        temp_circuits.append((file_path, num_qubits))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    temp_circuits.sort(key=lambda x: x[1])
    BENCHPRESS_TO_RUN = [filepath for filepath, num_qubits in temp_circuits]
    CIRCUITS_TO_RUN = CIRCUITS_TO_RUN + BENCHPRESS_TO_RUN

    for circuit in CIRCUITS_TO_RUN:
        print(f"Planned circuit for experiment: {circuit.split('/')[-1]}")

    TARGET_SUCCESSFUL_SEEDS = 5  
    STARTING_SEED = 42           
    PER_SEED_TIMEOUT = 1.5 * 3600 

    RESULTS_FILE = os.path.join(RESULTS_DIR, f"three_square_structured_dense_lf={LF_FLAG}.json")
    db = load_experiment_database(RESULTS_FILE)
    WORKERS = 5

    penalized_swaps = inter_qpu_edges
    extended_set_length = 20

    def has_data(db, m_idx, circ_idx):
        if m_idx >= len(db["methods_data"]): return False
        method = db["methods_data"][m_idx]
        if "ebits" not in method or circ_idx >= len(method["ebits"]): return False
        val = method["ebits"][circ_idx]
        if val is None or val == []: return False
        if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], float) and math.isnan(val[0]):
            return True
        return True

    # Setup the 8 database indices
    DQC1_IDX = get_or_create_method_index(db, "KaHyPar QIG + Default SABRE", "#384d83")
    DQC2_IDX = get_or_create_method_index(db, "KaHyPar QIG + (1,10) SABRE", "#384d83")
    DQC3_IDX = get_or_create_method_index(db, "KaHyPar QIG + Custom Lookahead SABRE", "#384d83")
    DQC4_IDX = get_or_create_method_index(db, "KaHyPar QIG + Custom Lookahead SABRE (e=100)", "#384d83")

    BASE_IDX = get_or_create_method_index(db, "(1,10) SABRE", "#f78c6b")
    DEFAULT_IDX = get_or_create_method_index(db, "Default SABRE Baseline", "#ffd166")
    CUSTOM_IDX = get_or_create_method_index(db, "Custom Lookahead SABRE", "#06d6a0")
    CUSTOM_E100_IDX = get_or_create_method_index(db, "Custom Lookahead SABRE (e=100)", "#06d6a0")

    for file_path in CIRCUITS_TO_RUN:
        circuit_name = os.path.basename(file_path).replace('.qasm', '')

        if circuit_name in db.get("circuit_list", []): 
            circ_idx = db["circuit_list"].index(circuit_name)
        else:
            circ_idx = len(db.setdefault("circuit_list", []))
            db["circuit_list"].append(circuit_name)
            
        for m in db["methods_data"]:
            for key in ["ebits", "swaps", "time"]:
                while len(m.setdefault(key, [])) <= circ_idx: m[key].append(None)
            for key in ["raw_ebits", "raw_swaps", "raw_time"]:
                while len(m.setdefault(key, [])) <= circ_idx: m[key].append([])

        # Evaluate needs for the current benchmark circuit across all 8 variants
        needs_dqc1 = not has_data(db, DQC1_IDX, circ_idx)
        needs_dqc2 = not has_data(db, DQC2_IDX, circ_idx)
        needs_dqc3 = not has_data(db, DQC3_IDX, circ_idx)
        needs_dqc4 = not has_data(db, DQC4_IDX, circ_idx)

        needs_base = not has_data(db, BASE_IDX, circ_idx)
        needs_default = not has_data(db, DEFAULT_IDX, circ_idx)
        needs_custom = not has_data(db, CUSTOM_IDX, circ_idx)
        needs_custom_e100 = not has_data(db, CUSTOM_E100_IDX, circ_idx)
        
        if not (needs_dqc1 or needs_dqc2 or needs_dqc3 or needs_dqc4 or needs_base or needs_default or needs_custom or needs_custom_e100):
            print(f"Skipping Circuit {circuit_name}: All requested variants already computed.")
            continue
            
        print(f"\n--- Evaluating Circuit: {circuit_name} (Target: {TARGET_SUCCESSFUL_SEEDS} successes) ---")
        
        print(f"Loading and pre-processing circuit {circuit_name} once to save RAM...")
        master_cz_qiskit_sanitized = load_and_prepare_benchmark(file_path)

        successful_runs = []
        current_seed = STARTING_SEED

        historical_seeds = []
        if circ_idx < len(db.get("seed_history", [])) and db["seed_history"][circ_idx]:
            historical_seeds = list(db["seed_history"][circ_idx])

        enforce_history = bool(historical_seeds and not needs_dqc1 and not needs_dqc2 and not needs_dqc3 and not needs_dqc4)
        if enforce_history:
            print(f"Synchronizing with previously successful seeds: {historical_seeds}")

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)
        future_to_seed = {}
        
        def submit_job(executor, seed_val):
            return executor.submit(
                run_single_seed_task, master_cz_qiskit_sanitized, seed_val, 
                mqpu_backend, inter_qpu_edges, S_matrix_1_10, S_matrix_default, penalized_swaps=penalized_swaps, 
                qubit_qpu_map=qubit_qpu_map, inter_qpu_coupling_map=inter_qpu_coupling_map, 
                extended_set_length=extended_set_length,
                run_dqc1=needs_dqc1, run_dqc2=needs_dqc2, run_dqc3=needs_dqc3, run_dqc4=needs_dqc4,
                run_base=needs_base, run_default=needs_default, run_custom=needs_custom, run_custom_e100=needs_custom_e100,
                timeout_limit=PER_SEED_TIMEOUT
            )

        for i in range(TARGET_SUCCESSFUL_SEEDS):
            seed_val = historical_seeds[i] if enforce_history else current_seed
            if not enforce_history: current_seed += 1
            fut = submit_job(executor, seed_val)
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
                        # Unpack 9 items now (seed + 8 algorithm results)
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
                                fut = submit_job(executor, current_seed)
                                future_to_seed[fut] = current_seed
                                current_seed += 1
                    except Exception as exc:
                        print(f"  -> Critical Failure on seed {finished_seed}: {exc}")
                        total_failures += 1
                        if total_failures < MAX_ALLOWED_FAILURES:
                            fut = submit_job(executor, current_seed)
                            future_to_seed[fut] = current_seed
                            current_seed += 1
                        
                if total_failures >= MAX_ALLOWED_FAILURES:
                    print(f"FATAL: Reached {MAX_ALLOWED_FAILURES} failures. Aborting Circuit {circuit_name}.")
                    break

        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if len(successful_runs) == 0:
            print(f"WARNING: Circuit {circuit_name} yielded 0 successful runs (likely timed out). Skipping save.")
            continue

        def update_db(m_idx, needs_run, res_idx):
            if not needs_run: return
            t_ebits, t_swaps, t_time = [], [], []
            for run_data in successful_runs:
                res = run_data[res_idx] 
                if res is not None:
                    t_ebits.append(res[0]); t_swaps.append(res[1]); t_time.append(res[2])
                
            if len(t_ebits) == 0: return 
                
            db["methods_data"][m_idx]["ebits"][circ_idx] = compute_statistics(t_ebits)
            db["methods_data"][m_idx]["swaps"][circ_idx] = compute_statistics(t_swaps)
            db["methods_data"][m_idx]["time"][circ_idx] = compute_statistics(t_time)
            db["methods_data"][m_idx]["raw_ebits"][circ_idx] = t_ebits
            db["methods_data"][m_idx]["raw_swaps"][circ_idx] = t_swaps
            db["methods_data"][m_idx]["raw_time"][circ_idx] = t_time

        update_db(DQC1_IDX, needs_dqc1, 1)
        update_db(DQC2_IDX, needs_dqc2, 2)
        update_db(DQC3_IDX, needs_dqc3, 3)
        update_db(DQC4_IDX, needs_dqc4, 4)

        update_db(BASE_IDX, needs_base, 5)
        update_db(DEFAULT_IDX, needs_default, 6)
        update_db(CUSTOM_IDX, needs_custom, 7)
        update_db(CUSTOM_E100_IDX, needs_custom_e100, 8)

        while len(db.setdefault("seed_history", [])) <= circ_idx: db["seed_history"].append([])
        db["seed_history"][circ_idx] = [r[0] for r in successful_runs]

        save_experiment_database(db, RESULTS_FILE)
        print(f"Results for Circuit {circuit_name} successfully updated and saved to disk.")