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
import multiprocessing as mp

# --- 2. Third-Party Libraries ---
import numpy as np
import networkx as nx

# --- 3. Quantum Frameworks ---

# Qiskit Core & Transpiler
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    BarrierBeforeFinalMeasurements,
    SabreLayout,
    SabreSwap,
)

# Pytket & Qiskit Extensions
from pytket import Qubit
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

# Pytket DQC (Distributed Quantum Computing)
from pytket_dqc.circuits.distribution import remove_barriers
from pytket_dqc.distributors import CoverEmbedding, PartitioningAnnealing, PartitioningHeterogeneous
from pytket_dqc.networks.nisq_network import NISQNetwork
from pytket_dqc.utils import DQCPass, check_equivalence
from pytket_dqc.utils.circuit_analysis import ebit_cost

from pytket_dqc.refiners import (
    IntertwinedDTypeMerge,
    NeighbouringDTypeMerge,
    RepeatRefiner,
    SequenceRefiner,
)

# --- 4. Local/Custom Project Modules ---
from mqpu_utils import (
    build_distributed_subcircuits,
    check_violations,
    create_hardware_layout,
    generate_multi_qpu_backend_from_monolithic_backend_with_links,
    get_optimized_locked_layout,
    optimize_circuit_barriers,
    rectangular_backend_with_links,
    rectangular_backend,
    remove_routing_placeholders,
    sanitize_qiskit_labels,
    squash_placeholders_per_pair,
    TranspilerConfig,
    verify_fixed_qubits_layout,
    RoutingPlaceholder,
    MakePlaceholdersOpaque
)

# Suppress excessive logging from the transpiler
logging.getLogger("qiskit.compiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logging.getLogger("qiskit.transpiler.passes.layout.sabre_layout").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Define one-way connections (Source QPU -> Target QPU)
raw_links = [
    (16, 39), (17, 38),  # 0 to 1
    (18, 57), (19, 56),  # 0 to 2
    (36, 59), (37, 58)   # 1 to 2
]

# 2. Mirror them automatically to ensure bidirectional connectivity
inter_qpu_edges = raw_links + [(v, u) for u, v in raw_links]

# 3. Backend Parameters 
n_qpus, l = 3, 4
monolithic_n_qubits = l * (l + 1)
basis_gates = ['cz', 'id', 'rz', 'sx', 'x']

# 4. Generate `Backend`s
backend = rectangular_backend_with_links(l, l, basis_gates=basis_gates, seed=42) # Monolithic
mqpu_backend = generate_multi_qpu_backend_from_monolithic_backend_with_links( # Multi-QPU
    n_qpus=n_qpus,
    monolithic_backend=backend,
    inter_qpu_links=inter_qpu_edges,
    backend_name=f'{n_qpus}_Square_r1'
)
square_backend = rectangular_backend(l, l, basis_gates=basis_gates, seed=42) 
square_remote_links = [
    (3, 31), (7, 27),    # 0 to 1
    (15, 35), (11, 39),  # 0 to 2
    (23, 43), (19, 47),  # 1 to 2
]
square_inter_qpu_edges = square_remote_links + [(v, u) for u, v in square_remote_links]
square_mqpu_backend = generate_multi_qpu_backend_from_monolithic_backend_with_links( 
    n_qpus=n_qpus,
    monolithic_backend=square_backend,
    inter_qpu_links=square_inter_qpu_edges,
    backend_name=f'{n_qpus}_Square_r1'
)

from qiskit.compiler import transpile
from qiskit.qasm2 import load as load_qasm2, LEGACY_CUSTOM_INSTRUCTIONS

def load_and_prepare_benchmark(file_path):
    """Loads a QASM file and strictly transpiles it to a PyZX-safe hardware basis."""
    t_0 = perf_counter()
    qc = load_qasm2(file_path, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS) 
    dt_load = perf_counter() - t_0
    print(f"[load_qasm2] Time taken = {dt_load:.4f}s") 

    clean_data = []
    for inst in qc.data:
        if inst.operation.name not in ['measure', 'reset', 'barrier', 'delay']:
            clean_data.append(inst)
    qc.data = clean_data
    qc.cregs.clear()
    qc.clbits.clear()

    qc = transpile(qc, basis_gates=['cz', 'id', 'rz', 'sx', 'x'], optimization_level=2)
    circ = qiskit_to_tk(qc) 
    
    t_0 = perf_counter()
    DQCPass().apply(circ)
    dt_dqc_pass = perf_counter() - t_0
    print(f"[DQCPass] Time taken = {dt_dqc_pass:.4f}s") 

    cz_qiskit = tk_to_qiskit(circ)
    cz_qiskit_sanitized = sanitize_qiskit_labels(cz_qiskit)
    
    return circ, cz_qiskit_sanitized

# 1. Start with physical edges
monolithic_edges = list(backend.coupling_map.get_edges())

# 2. Define Virtual Connections (Virtual Qubit, Physical Qubit)
virtual_pairs = [
    (20, 15), (21, 11), (22, 7), (23, 3),   # Virtual Comp
    (24, 19), (25, 18), (26, 17), (27, 16)  # Virtual Link
]
virtual_edges = []
for v, p in virtual_pairs:
    virtual_edges.extend([(v, p), (p, v)])

# 3. Create Map and Backend
monolithic_virtual_map = CouplingMap(monolithic_edges + virtual_edges)
server_coupling = [[0, 1], [0, 2], [1, 2]] 
server_qubits = {0: range(0, 16), 1: range(16, 32), 2: range(32, 48)} 
server_link_capacities = {(0, 1): 2, (0, 2): 2, (1, 2): 2} 

network = NISQNetwork(
    server_coupling=server_coupling,
    server_qubits=server_qubits,
    server_ebit_mem=None, 
    server_link_capacities=server_link_capacities
)
N_SERVERS = len(network.get_server_list())
N_LINKS = 4 

# --- Auxiliary Data Structures ---
link_to_virtual_mapping = { 
    f"server_{s}_link_register[{i}]": Qubit(f"server_{s}_virtual_sink", 3 - i)
    for s in range(N_SERVERS) for i in range(N_LINKS)
}
link_to_virtual_return = { 
    f"server_{s}_link_register[{i}]": Qubit(f"server_{s}_virtual_sink", 7 - i)
    for s in range(N_SERVERS) for i in range(N_LINKS)
}

# --- Link Tracking ---
PHYSICAL_LINKS = {
    (0, 1): [(0, 3), (1, 2)],
    (0, 2): [(2, 1), (3, 0)],
    (1, 2): [(0, 3), (1, 2)]
}

comp_qubits, link_qubits, virt_qubits = range(16), range(16, 20), range(20, 28)
type_map = {**{i: 'C' for i in comp_qubits}, **{i: 'L' for i in link_qubits}, **{i: 'V' for i in virt_qubits}}

weight_lookup = {
    ('C', 'C'): 10**9, 
    ('C', 'L'): 10**6,
    ('C', 'V'): 10**3,
    ('L', 'V'): 10**0
}

G = nx.Graph()
for u, v in monolithic_virtual_map.get_edges():
    pair = tuple(sorted((type_map.get(u), type_map.get(v))))
    G.add_edge(u, v, weight=weight_lookup.get(pair, 10**0)) 

S = nx.floyd_warshall_numpy(G, nodelist=range(len(G.nodes)))


def _distribute_and_optimize(cz_circ, seed, distributor_class):
    t_0 = perf_counter()
    distributor_instance = distributor_class()
    distribution = distributor_instance.distribute(cz_circ, network=network, seed=seed)

    # refiner_list = [NeighbouringDTypeMerge(), IntertwinedDTypeMerge()]
    # RepeatRefiner(SequenceRefiner(refiner_list)).refine(distribution)
    # assert distribution.detached_gate_count() == 0, f"There exist {distribution.detached_gate_count()} detached gates!"
    print(f"[Seed {seed}] There exist {distribution.detached_gate_count()} detached gates!") 

    t_1 = perf_counter()
    dt_distribution = t_1 - t_0
    print(f"[Distribution] Distribution and refinement time = {dt_distribution:.4f}s") 

    final_circuit = distribution.to_pytket_circuit(allow_update=True, verify_equivalence=False)
    dt_generation = perf_counter() - t_1
    print(f"[Distribution] Distributed circuit generation time = {dt_generation:.4f}s") 

    final_circuit_without_barriers = remove_barriers(final_circuit)
    assert check_equivalence(cz_circ, final_circuit_without_barriers, distribution.get_qubit_mapping()), "Equivalence check failed!"
    assert not check_violations(final_circuit, server_link_capacities, verbose=True), "Network constraints violated!"

    routing_time = dt_distribution + dt_generation
    routing_cost = ebit_cost(final_circuit)
    
    return final_circuit, routing_time, routing_cost, distribution

def _generate_sanitized_subcircuits(cz_circ, pytket_dqc_circuit, distribution): 
    busy_links = {pair: [] for pair in PHYSICAL_LINKS}
    available_links = {pair: list(links) for pair, links in PHYSICAL_LINKS.items()}

    t_0 = perf_counter()
    subcircuits, distributed_circ = build_distributed_subcircuits(
        pytket_dqc_circuit, network, available_links, busy_links,
        link_to_virtual_mapping, link_to_virtual_return, 
        n_servers=N_SERVERS, n_links=N_LINKS
    )
    dt_subcircuit_generation = perf_counter() - t_0
    print(f"[Subcircuit Generation] Time taken = {dt_subcircuit_generation:.4f}s") 

    distributed_circ_no_barriers = remove_barriers(distributed_circ)
    distributed_circ_no_barriers.remove_blank_wires() 
    assert check_equivalence(cz_circ, distributed_circ_no_barriers, distribution.get_qubit_mapping()), "The reconstructed circuit from subcircuits is not equivalent to the original circuit! This indicates a potential issue in the subcircuit generation process. Please investigate the transformations applied during subcircuit generation to identify where the discrepancy arises."
    assert not check_violations(distributed_circ, server_link_capacities, verbose=True), "The reconstructed circuit violates the network constraints!"

    final_optimized_circuit_distributed, removed_barrier_indices = optimize_circuit_barriers(
        distributed_circ, 
        server_link_capacities, 
        barrier_index_to_check=0
    )
    assert check_equivalence(remove_barriers(pytket_dqc_circuit), distributed_circ_no_barriers, distribution.get_qubit_mapping(), distributed_comparison=True) # Re-run with this addition.
    assert not check_violations(final_optimized_circuit_distributed, server_link_capacities, verbose=True), "The optimized (distributed) circuit violates the network constraints!"

    subcircuits = sync_subcircuit_barriers(subcircuits, removed_barrier_indices)

    sanitized_qiskit_circuits = {}
    for qpu_id, circ in subcircuits.items():
        squashed = squash_placeholders_per_pair(circ)
        qiskit_circ = tk_to_qiskit(squashed)
        sanitized = sanitize_qiskit_labels(qiskit_circ)

        # Identity Verification
        # Skipped!

        sanitized_qiskit_circuits[qpu_id] = sanitized

    return sanitized_qiskit_circuits

# --- NEW: Safe Custom Pass Manager Builder for Subcircuits ---
def create_custom_pm(backend,
                     monolithic_virtual_map,
                     initial_layout,
                     config,
                     distance_matrix):
    
    try:
        base_gates = backend.configuration().basis_gates
    except AttributeError:
        base_gates = ['id', 'rz', 'sx', 'x', 'cz', 'measure']

    custom_mapping = {"routing_placeholder": RoutingPlaceholder()}
    
    translation_target = Target.from_configuration(
        basis_gates=base_gates + ["routing_placeholder"],
        num_qubits=monolithic_virtual_map.size(),
        coupling_map=monolithic_virtual_map, 
        custom_name_mapping=custom_mapping
    )

    pm = generate_preset_pass_manager(
        optimization_level=0, 
        initial_layout=initial_layout, 
        target=translation_target,
        seed_transpiler=config.seed
    )
    pm.init.append(MakePlaceholdersOpaque())

    real_qubit_limit = backend.num_qubits 
    virtual_edges = [
        (src, dst) for src, dst in monolithic_virtual_map.get_edges() 
        if src >= real_qubit_limit or dst >= real_qubit_limit
    ]
    qubit_qpu_map = [0 for _ in range(monolithic_virtual_map.size())]

    # Use SabreSwap to prevent backwards sweeps from breaking the locks!
    routing_pass = SabreSwap(
        coupling_map=monolithic_virtual_map,
        heuristic=config.heuristic if hasattr(config, 'heuristic') else 'lookahead',
        seed=config.seed,
        trials=5, 
        penalized_swaps=virtual_edges,
        qubit_qpu_map=qubit_qpu_map,
        inter_qpu_coupling_map=[],
        alpha=100000.0,
        beta=0.0
    )
    routing_pass._routing_target.set_distance_matrix(distance_matrix)

    pm.routing.replace(
        index=1, 
        passes=(BarrierBeforeFinalMeasurements(), routing_pass)
    )
    
    return pm

# --- NEW: Locked Trial Evaluator ---
def _transpile_subcircuits(subcircuits_dict, lf, base_seed, num_trials=5):
    transpiled_subcircuits = {}
    exact_routing_costs = {}
    intra_qpu_routing_time_costs = {}

    rng = np.random.default_rng(base_seed)

    for qpu_id, qc in subcircuits_dict.items():
        original_swap_count = qc.count_ops().get('swap', 0)
        
        best_qc = None
        best_swaps = float('inf')
        best_layout_time = 0.0
        best_routing_time = 0.0

        trial_seeds = rng.integers(0, 2**31 - 1, size=num_trials)
        print(f"[{qpu_id}] Optimizing over {num_trials} locked layout trials...")

        for trial_seed in trial_seeds:
            trial_seed = int(trial_seed)
            config = TranspilerConfig(seed=trial_seed) 
            
            layout_t0 = perf_counter()
            if lf:
                init_layout = get_optimized_locked_layout(qc, monolithic_virtual_map, seed=config.seed)
            else:
                init_layout = create_hardware_layout(qc=qc, seed=config.seed, verbose=False)
            layout_time = perf_counter() - layout_t0

            pm = create_custom_pm(backend, monolithic_virtual_map, init_layout, config, S)

            trial_metrics = {'added_swaps': 0}

            def audit_sabre(pass_, dag, time, property_set, count):
                if pass_.name() == "SabreSwap": 
                    total_swaps = dag.count_ops().get('swap', 0)
                    trial_metrics['added_swaps'] = total_swaps - original_swap_count

            routing_t0 = perf_counter()
            transpiled_qc = pm.run(qc, callback=audit_sabre)
            routing_time = perf_counter() - routing_t0

            if not verify_fixed_qubits_layout(transpiled_qc, qc, verbose=False):
                print(f"  -> Trial (Seed {trial_seed}) violated layout constraints! Skipping.")
                continue

            added_swaps = trial_metrics['added_swaps']
            
            if added_swaps < best_swaps:
                best_swaps = added_swaps
                best_qc = transpiled_qc
                best_layout_time = layout_time
                best_routing_time = routing_time

        if best_qc is None:
            raise RuntimeError(f"CRITICAL: All {num_trials} layout trials on QPU {qpu_id} violated fixed constraints!")

        print(f"  -> Best [QPU {qpu_id}] Cost: {best_swaps} SWAPs (Layout: {best_layout_time:.4f}s | Route: {best_routing_time:.4f}s)")

        transpiled_subcircuits[qpu_id] = remove_routing_placeholders(best_qc)
        exact_routing_costs[qpu_id] = best_swaps
        intra_qpu_routing_time_costs[qpu_id] = best_layout_time + best_routing_time

    intra_time = sum(intra_qpu_routing_time_costs.values())
    intra_cost = sum(exact_routing_costs.values())

    return transpiled_subcircuits, intra_time, intra_cost

def run_experiment(circ_tk, lf, seed, distributor_class):
    print(f"\n--- Running Experiment (Seed: {seed}) ---")
    
    distributed_circuit, inter_time, inter_cost, distribution = _distribute_and_optimize(circ_tk, seed, distributor_class)
    sanitized_subcircuits = _generate_sanitized_subcircuits(circ_tk, distributed_circuit, distribution)
    
    # Pass seed down to become base_seed
    final_subcircuits, intra_time, intra_cost = _transpile_subcircuits(sanitized_subcircuits, lf, seed)

    print(f"Inter-QPU Routing -> Time: {inter_time:.4f}s | Cost: {inter_cost} EPR pairs")
    print(f"Intra-QPU Routing -> Time: {intra_time:.4f}s | Cost: {intra_cost} SWAPs")

    return final_subcircuits, inter_time, inter_cost, intra_time, intra_cost

# ============================================
# Helper Functions Custom SABRE Baseline
# ============================================

def generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=10):
    weighted_edges = []
    for edge in mqpu_backend.coupling_map.get_edges():
        if edge in inter_qpu_edges:
            weighted_edges.append((edge[0], edge[1], 1)) 
        else:
            weighted_edges.append((edge[0], edge[1], factor)) 

    G = nx.DiGraph()
    G.add_weighted_edges_from(weighted_edges)
    return nx.floyd_warshall_numpy(G, range(len(G.nodes)))

from qiskit.transpiler import PassManager

def get_baseline_pass_manager(
    mqpu_backend, 
    S_matrix, 
    seed, 
    penalized_swaps=None, 
    qubit_qpu_map=None, 
    inter_qpu_coupling_map=None, 
    alpha=9.0, 
    beta=3.0,
    extended_set_length=20
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
    
    pm.layout.replace(index=2, passes=[
        BarrierBeforeFinalMeasurements(), sl, 
    ])
    pm.routing = PassManager() 
    
    return pm

def run_baseline_experiment(cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix, penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20, strategy_name="Baseline"):
    local_qc = cz_qiskit_sanitized.copy()

    orig_intra_swaps = 0
    orig_inter_swaps = 0
    for instr in local_qc.data:
        if instr.operation.name == "swap":
            q0_idx = local_qc.find_bit(instr.qubits[0]).index
            q1_idx = local_qc.find_bit(instr.qubits[1]).index
            if (q0_idx, q1_idx) in inter_qpu_edges or (q1_idx, q0_idx) in inter_qpu_edges:
                orig_inter_swaps += 1
            else:
                orig_intra_swaps += 1

    original_swap_count = local_qc.count_ops().get('swap', 0)
    assert original_swap_count == 0, (
        f"CRITICAL: Expected 0 original SWAPs after DQC_Pass, but found {original_swap_count}."
    )

    num_trials = 5
    f_weight = 10
    best_cost = float('inf')
    best_metrics = {}
    best_time = 0.0

    routing_metrics = {}
    inter_qpu_edges_set = {tuple(sorted(edge)) for edge in inter_qpu_edges}

    def audit_sabreswap(pass_, dag, time, property_set, count):
        if pass_.name() == "SabreLayout":
            intra_swaps, inter_swaps = 0, 0
            inter_standard_ops = 0  
            inter_other_ops = 0     
            
            for node in dag.op_nodes():
                if len(node.qargs) == 2:
                    op_name = node.op.name
                    if op_name in ["barrier", "routing_placeholder"]:
                        continue
                        
                    q0_idx = dag.find_bit(node.qargs[0]).index
                    q1_idx = dag.find_bit(node.qargs[1]).index
                    # Fixed Newer Code Audit
                    is_inter_qpu = tuple(sorted((q0_idx, q1_idx))) in inter_qpu_edges_set
                    
                    if op_name == "swap":
                        if is_inter_qpu: inter_swaps += 1
                        else: intra_swaps += 1
                    elif is_inter_qpu:
                        if op_name in ["cz", "cx", "cu1", "cp"]:
                            inter_standard_ops += 1
                        else:
                            inter_other_ops += 1
                            
            routing_metrics['final_intra_swaps'] = intra_swaps 
            routing_metrics['final_inter_swaps'] = inter_swaps
            routing_metrics['final_inter_czs'] = inter_standard_ops + inter_other_ops

    print(f"[{strategy_name}] Testing {num_trials} baseline random layouts...")

    for trial_idx in range(num_trials):
        current_seed = seed + trial_idx 
        
        pm = get_baseline_pass_manager(
            mqpu_backend, S_matrix, current_seed, penalized_swaps, qubit_qpu_map, inter_qpu_coupling_map, extended_set_length=extended_set_length
        )
        
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

        try:
            del pm
        except NameError:
            pass
        gc.collect()

    added_intra_swaps = best_metrics['final_intra_swaps'] - orig_intra_swaps
    total_ebits = (best_metrics['final_inter_swaps'] * 3) + (best_metrics['final_inter_czs'] * 1)

    print(f"[{strategy_name}] BEST Routing -> Time: {best_time:.4f}s | Inter SWAPs: {best_metrics['final_inter_swaps']} | Inter CZs: {best_metrics['final_inter_czs']} | Intra SWAPs Added: {added_intra_swaps}")

    try:
        del local_qc
    except NameError:
        pass 
        
    gc.collect()

    return total_ebits, added_intra_swaps, best_time

# ============================================
# Helper Functions for Data Handling & Experiment Management
# ============================================

from experiment_utils import *

def run_single_seed_task(
    circ_tk, cz_qiskit_sanitized, lf, seed, mqpu_backend, inter_qpu_edges, S_matrix, S_matrix_default,
    penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20,
    run_dqc=True, run_base=True, run_default=True, run_custom=True, verbose=False, distributor_class=CoverEmbedding):
    
    dqc_results, base_results, default_results, custom_results = None, None, None, None
    
    try:
        if run_dqc:
            local_circ_tk = circ_tk.copy() 
            _, inter_t, inter_c, intra_t, intra_c = run_experiment(local_circ_tk, lf, seed, distributor_class)
            dqc_results = (inter_c, intra_c, inter_t + intra_t)

        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed}: Pytket-DQC finished.")
            
        if run_base:
            base_ebits, base_swaps, base_time = run_baseline_experiment(
                cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix, 
                strategy_name="(1, 10) SABRE"
            )
            base_results = (base_ebits, base_swaps, base_time)

        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed}: (1, 10) SABRE finished.")
            
        if run_default:
            default_ebits, default_swaps, default_time = run_baseline_experiment(
                cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix_default, 
                strategy_name="Default SABRE"
            )
            default_results = (default_ebits, default_swaps, default_time)

        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed}: Default SABRE finished.")
            
        if run_custom:
            custom_ebits, custom_swaps, custom_time = run_baseline_experiment(
                cz_qiskit_sanitized, seed, mqpu_backend, inter_qpu_edges, S_matrix, 
                penalized_swaps=penalized_swaps, qubit_qpu_map=qubit_qpu_map, 
                inter_qpu_coupling_map=inter_qpu_coupling_map, extended_set_length=extended_set_length,
                strategy_name="Custom Lookahead SABRE"
            )
            custom_results = (custom_ebits, custom_swaps, custom_time)

        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed}: Custom Lookahead SABRE finished.")
            
        return seed, dqc_results, base_results, default_results, custom_results
        
    except KeyError as e:
        if str(e) == "'0'" or str(e) == "0":
            print(f"  -> Seed {seed} failed: pytket-dqc hyperedge split bug. Discarding.")
            return seed, None, None, None, None
        raise e
        
    except Exception as e:
        if "ConstraintException" in str(e) or "recursion" in str(e).lower():
            print(f"  -> Seed {seed} failed: pytket-dqc routing heuristic aborted. Discarding.")
            return seed, None, None, None, None
            
        print(f"  -> Seed {seed} failed with an unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return seed, None, None, None, None
    
    finally:
        gc.collect()


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    light_dir = os.path.join(BENCHMARKS_DIR, 'light')
    CIRCUITS_TO_RUN = [os.path.join(light_dir, f) for f in os.listdir(light_dir) if f.endswith('.qasm')]
    for circuit in CIRCUITS_TO_RUN:
        print(f"Planned circuit for experiment: {circuit.split('/')[-1]}")

    from pytket_dqc.distributors import CoverEmbedding, PartitioningHeterogeneous, PartitioningHeterogeneousEmbedding 

    CURRENT_DISTRIBUTOR = PartitioningHeterogeneous 
    DISTRIBUTOR_NAME = "PartitioningHeterogeneous"
    DQC_COLOR = "#118ab2"

    TARGET_SUCCESSFUL_SEEDS = 5  
    STARTING_SEED = 42           
    TIMEOUT_SECONDS = 0          
    LF_FLAG = True               

    RESULTS_FILE = os.path.join(RESULTS_DIR, f"three_square_structured_light_lf={LF_FLAG}.json")
    db = load_experiment_database(RESULTS_FILE)
    WORKERS = min(TARGET_SUCCESSFUL_SEEDS, os.cpu_count() or 1)

    qubit_qpu_map = [i // backend.num_qubits for i in range(mqpu_backend.num_qubits)] 
    inter_qpu_coupling_map = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]

    S_matrix = generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges) 
    S_matrix_default = generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=1) 

    penalized_swaps = inter_qpu_edges
    extended_set_length = 20

    def has_data(db, m_idx, frac_idx):
        if m_idx >= len(db["methods_data"]): return False
        method = db["methods_data"][m_idx]
        if "ebits" not in method or frac_idx >= len(method["ebits"]): return False
        val = method["ebits"][frac_idx]
        if val is None or val == []: return False
        if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], float) and math.isnan(val[0]):
            return True
        return True

    DQC_LABEL = f"Pytket-DQC + SABRE ({DISTRIBUTOR_NAME}, lf={LF_FLAG})"

    DQC_IDX = get_or_create_method_index(db, DQC_LABEL, DQC_COLOR)
    BASE_IDX = get_or_create_method_index(db, "(1,10) SABRE", "#f78c6b")
    DEFAULT_IDX = get_or_create_method_index(db, "Default SABRE Baseline", "#ffd166")
    CUSTOM_IDX = get_or_create_method_index(db, "Custom Lookahead SABRE", "#06d6a0")

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

        needs_dqc = not has_data(db, DQC_IDX, circ_idx)
        needs_base = not has_data(db, BASE_IDX, circ_idx)
        needs_default = not has_data(db, DEFAULT_IDX, circ_idx)
        needs_custom = not has_data(db, CUSTOM_IDX, circ_idx)
        
        if not (needs_dqc or needs_base or needs_default or needs_custom):
            print(f"Skipping Circuit {circuit_name}: All requested variants already computed.")
            continue
            
        print(f"\n--- Evaluating Circuit: {circuit_name} (Target: {TARGET_SUCCESSFUL_SEEDS} successes) ---")
        
        print(f"Loading and pre-processing circuit {circuit_name} once to save RAM...")
        master_circ_tk, master_cz_qiskit_sanitized = load_and_prepare_benchmark(file_path)
        
        start_time = time.time()
        successful_runs = []
        current_seed = STARTING_SEED
        dqc_timed_out = False

        def submit_job(executor, seed_val, run_dqc_flag):
            return executor.submit(
                run_single_seed_task, master_circ_tk, master_cz_qiskit_sanitized, LF_FLAG, seed_val, 
                mqpu_backend, inter_qpu_edges, S_matrix, S_matrix_default, penalized_swaps=penalized_swaps, 
                qubit_qpu_map=qubit_qpu_map, inter_qpu_coupling_map=inter_qpu_coupling_map, 
                extended_set_length=extended_set_length, 
                run_dqc=run_dqc_flag, run_base=needs_base, run_default=needs_default, run_custom=needs_custom,
                verbose=True, distributor_class=CURRENT_DISTRIBUTOR
            )

        historical_seeds = []
        if circ_idx < len(db.get("seed_history", [])) and db["seed_history"][circ_idx]:
            historical_seeds = list(db["seed_history"][circ_idx])

        enforce_history = bool(historical_seeds and not needs_dqc)
        
        if enforce_history:
            print(f"Synchronizing baselines with previously successful seeds: {historical_seeds}")

        circuit_qubits = file_path.split('/')[-1].split('_')[-1].replace('.qasm', '')
        try:
            circuit_qubits = int(circuit_qubits)
        except ValueError:
            print(f"Warning: Could not parse qubit count from filename for {circuit_name}. Raising an error to prevent proceeding without RAM management.")
            raise ValueError(f"Filename {file_path} does not contain a valid qubit count. Please ensure the filename ends with '_<num_qubits>.qasm' or adjust the parsing logic accordingly.")
        
        if circuit_qubits > 48:
            print(f"Error: Circuit {circuit_name} has {circuit_qubits} qubits, which exceeds the maximum supported by our 3x16 architecture. Skipping this circuit.")
            raise ValueError(f"Circuit {circuit_name} has {circuit_qubits} qubits, exceeding the 48-qubit limit. Please remove it from the experiment list or adjust the architecture accordingly.")

        TIMEOUT_SECONDS = 1.10 * np.ceil(TARGET_SUCCESSFUL_SEEDS/WORKERS) * 3600
        print(f"Setting timeout to {TIMEOUT_SECONDS/3600:.2f} hours based on worker count and target seeds.")

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)
        future_to_seed = {}
        
        for i in range(TARGET_SUCCESSFUL_SEEDS):
            seed_val = historical_seeds[i] if enforce_history else current_seed
            if not enforce_history: current_seed += 1
                
            fut = submit_job(executor, seed_val, needs_dqc)
            future_to_seed[fut] = seed_val

        MAX_ALLOWED_FAILURES = 20 
        total_failures = 0

        try:
            while len(successful_runs) < TARGET_SUCCESSFUL_SEEDS:
                time_left = TIMEOUT_SECONDS - (time.time() - start_time)
                
                if needs_dqc and time_left <= 0:
                    print(f"TIMEOUT: Circuit {circuit_name} exceeded time limit! Aborting DQC.")
                    dqc_timed_out = True
                    break

                done, not_done = concurrent.futures.wait(
                    future_to_seed.keys(), 
                    timeout=time_left if needs_dqc else None,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                if not done and needs_dqc:
                    print(f"TIMEOUT: Circuit {circuit_name} exceeded time limit! Aborting DQC.")
                    dqc_timed_out = True
                    break
                    
                for future in done:
                    finished_seed = future_to_seed.pop(future)
                    try:
                        seed_res, dqc_res, base_res, default_res, custom_res = future.result()
                        
                        dqc_ok = (not needs_dqc) or (dqc_res is not None)
                        base_ok = (not needs_base) or (base_res is not None)
                        def_ok = (not needs_default) or (default_res is not None)
                        cust_ok = (not needs_custom) or (custom_res is not None)
                        
                        if dqc_ok and base_ok and def_ok and cust_ok:
                            successful_runs.append((finished_seed, dqc_res, base_res, default_res, custom_res))
                            print(f"  -> Seed {finished_seed} completed successfully! ({len(successful_runs)}/{TARGET_SUCCESSFUL_SEEDS})")

                            with open(f"./results/CHECKPOINT_{circuit_name}.json", 'w') as ckpt:
                                json.dump(successful_runs, ckpt)
                        else:
                            if enforce_history:
                                print(f"FATAL: A baseline failed on historical seed {finished_seed}! Cannot ensure fair comparison.")
                                total_failures = MAX_ALLOWED_FAILURES 
                                break
                            else:
                                total_failures += 1
                                if total_failures < MAX_ALLOWED_FAILURES:
                                    fut = submit_job(executor, current_seed, needs_dqc)
                                    future_to_seed[fut] = current_seed
                                    current_seed += 1

                    except Exception as exc:
                        if enforce_history:
                            print(f"FATAL: A baseline crashed on historical seed {finished_seed}! Cannot ensure fair comparison.")
                            total_failures = MAX_ALLOWED_FAILURES
                            break
                        else:
                            total_failures += 1
                            if total_failures < MAX_ALLOWED_FAILURES:
                                fut = submit_job(executor, current_seed, needs_dqc)
                                future_to_seed[fut] = current_seed
                                current_seed += 1
                        
                if total_failures >= MAX_ALLOWED_FAILURES:
                    print(f"FATAL: Reached {MAX_ALLOWED_FAILURES} consecutive failures. Aborting Circuit {circuit_name}.")
                    break

        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if dqc_timed_out and (needs_base or needs_default or needs_custom):
            print(f"--- Running Fallback: Gathering Baseline-only data for Circuit {circuit_name} ---")
            successful_runs = [] 
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as fallback_executor:
                future_to_seed_fb = {}
                for _ in range(TARGET_SUCCESSFUL_SEEDS):
                    fut = submit_job(fallback_executor, current_seed, False) 
                    future_to_seed_fb[fut] = current_seed
                    current_seed += 1
                    
                for future in concurrent.futures.as_completed(future_to_seed_fb):
                    finished_seed = future_to_seed_fb[future]
                    _, _, base_res, default_res, custom_res = future.result()
                    successful_runs.append((finished_seed, None, base_res, default_res, custom_res))
                    print(f"  -> Baseline Seed {finished_seed} completed! ({len(successful_runs)}/{TARGET_SUCCESSFUL_SEEDS})")

        if len(successful_runs) < TARGET_SUCCESSFUL_SEEDS and not dqc_timed_out:
            print(f"WARNING: Circuit {circuit_name} did not complete successfully. Skipping save.")
            continue

        def update_db(m_idx, needs_run, res_idx):
            if not needs_run: return
            t_ebits, t_swaps, t_time = [], [], []
            for run_data in successful_runs:
                res = run_data[res_idx] 
                t_ebits.append(res[0]); t_swaps.append(res[1]); t_time.append(res[2])
                
            db["methods_data"][m_idx]["ebits"][circ_idx] = compute_statistics(t_ebits)
            db["methods_data"][m_idx]["swaps"][circ_idx] = compute_statistics(t_swaps)
            db["methods_data"][m_idx]["time"][circ_idx] = compute_statistics(t_time)
            db["methods_data"][m_idx]["raw_ebits"][circ_idx] = t_ebits
            db["methods_data"][m_idx]["raw_swaps"][circ_idx] = t_swaps
            db["methods_data"][m_idx]["raw_time"][circ_idx] = t_time

        if needs_dqc:
            if dqc_timed_out:
                nan_tup = (np.nan, np.nan, np.nan)
                db["methods_data"][DQC_IDX]["ebits"][circ_idx] = nan_tup
                db["methods_data"][DQC_IDX]["swaps"][circ_idx] = nan_tup
                db["methods_data"][DQC_IDX]["time"][circ_idx] = nan_tup
            else:
                update_db(DQC_IDX, True, 1)

        update_db(BASE_IDX, needs_base, 2)
        update_db(DEFAULT_IDX, needs_default, 3)
        update_db(CUSTOM_IDX, needs_custom, 4)

        while len(db.setdefault("seed_history", [])) <= circ_idx: db["seed_history"].append([])
        db["seed_history"][circ_idx] = [r[0] for r in successful_runs]

        save_experiment_database(db, RESULTS_FILE)
        print(f"Results for Circuit {circuit_name} successfully updated and saved to disk.")