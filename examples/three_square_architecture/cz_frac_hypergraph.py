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
from time import perf_counter
import concurrent.futures # Added for parallel execution of experiments

# --- 2. Third-Party Libraries ---
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- 3. Quantum Frameworks ---

# Qiskit Core & Transpiler
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    ApplyLayout,
    BarrierBeforeFinalMeasurements,
    EnlargeWithAncilla,
    FullAncillaAllocation,
    SabreLayout,
    SabreSwap,
)

# Qiskit Aer
from qiskit_aer import AerSimulator

# Pytket & Qiskit Extensions
from pytket import Qubit
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk

# Pytket DQC (Distributed Quantum Computing)
from pytket_dqc.circuits.distribution import remove_barriers
from pytket_dqc.distributors import CoverEmbedding
from pytket_dqc.networks.nisq_network import NISQNetwork
from pytket_dqc.refiners import (
    IntertwinedDTypeMerge,
    NeighbouringDTypeMerge,
    RepeatRefiner,
    SequenceRefiner,
)
from pytket_dqc.utils import DQCPass, check_equivalence
from pytket_dqc.utils.circuit_analysis import ebit_cost

# --- 4. Local/Custom Project Modules ---
from mqpu_utils import (
    build_cz_fraction_circuit,
    build_distributed_subcircuits,
    check_violations,
    create_custom_pm,
    create_hardware_layout,
    fast_identity_test,
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
)

# Suppress excessive logging from the transpiler
logging.getLogger("qiskit.passmanager").setLevel(logging.WARNING)
logging.getLogger("qiskit.compiler.transpiler").setLevel(logging.WARNING)
logging.getLogger("qiskit.transpiler.passes.layout.sabre_layout").setLevel(logging.WARNING)

# Suppress specific Qiskit/stevedore deprecation warnings to keep logs clean
warnings.filterwarnings("ignore", category=DeprecationWarning)

raw_links = [
    (16, 39), (17, 38),  # 0 to 1
    (18, 57), (19, 56),  # 0 to 2
    (36, 59), (37, 58)   # 1 to 2
]

# 2. Mirror them automatically to ensure bidirectional connectivity
inter_qpu_edges = raw_links + [(v, u) for u, v in raw_links]

# 3. Backend Parameters (Keep this clean and grouped)
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
square_backend = rectangular_backend(l, l, basis_gates=basis_gates, seed=42) # For visualization and layout generation purposes
square_remote_links = [
    (3, 31), (7, 27),    # 0 to 1
    (15, 35), (11, 39),  # 0 to 2
    (23, 43), (19, 47),  # 1 to 2
]
square_inter_qpu_edges = square_remote_links + [(v, u) for u, v in square_remote_links]
square_mqpu_backend = generate_multi_qpu_backend_from_monolithic_backend_with_links( # Multi-QPU for the square topology
    n_qpus=n_qpus,
    monolithic_backend=square_backend,
    inter_qpu_links=square_inter_qpu_edges,
    backend_name=f'{n_qpus}_Square_r1'
)

# 1. Start with physical edges
monolithic_edges = list(backend.coupling_map.get_edges())

# 2. Define Virtual Connections (Virtual Qubit, Physical Qubit)
# Grouped by Comp Virtuals (20-23) and Link Virtuals (24-27)
virtual_pairs = [
    (20, 15), (21, 11), (22, 7), (23, 3),   # Virtual Comp
    (24, 19), (25, 18), (26, 17), (27, 16)  # Virtual Link
]

# Flatten into bidirectional edges
virtual_edges = []
for v, p in virtual_pairs:
    virtual_edges.extend([(v, p), (p, v)])

# 3. Create Map and Backend
monolithic_virtual_map = CouplingMap(monolithic_edges + virtual_edges)
server_coupling = [[0, 1], [0, 2], [1, 2]] # All-to-all 3-cores
server_qubits = {0: range(0, 16), 1: range(16, 32), 2: range(32, 48)} # Comp. Qubits
server_link_capacities = {(0, 1): 2, (0, 2): 2, (1, 2): 2} # Bidirectional links

network = NISQNetwork(
    server_coupling=server_coupling,
    server_qubits=server_qubits,
    server_ebit_mem=None,
    server_link_capacities=server_link_capacities
)
N_SERVERS = len(network.get_server_list())
N_LINKS = 4

# --- Auxiliary Data Structures ---
link_to_virtual_mapping = { # 1. Map: link_register[i] -> virtual_sink[3 - i]
    f"server_{s}_link_register[{i}]": Qubit(f"server_{s}_virtual_sink", 3 - i)
    for s in range(N_SERVERS) for i in range(N_LINKS)
}
link_to_virtual_return = { # 3. Map: link_register[i] -> virtual_sink[7 - i]
    f"server_{s}_link_register[{i}]": Qubit(f"server_{s}_virtual_sink", 7 - i)
    for s in range(N_SERVERS) for i in range(N_LINKS)
}

# --- Link Tracking ---
PHYSICAL_LINKS = {
    (0, 1): [(0, 3), (1, 2)],
    (0, 2): [(2, 1), (3, 0)],
    (1, 2): [(0, 3), (1, 2)]
}
# Define qubit categories
comp_qubits, link_qubits, virt_qubits = range(16), range(16, 20), range(20, 28)
type_map = {**{i: 'C' for i in comp_qubits}, **{i: 'L' for i in link_qubits}, **{i: 'V' for i in virt_qubits}}

# Weight rules based on sorted pairs of types (Before: 1e{0,2,4,6}. Now with increased separation to ensure Sabre prioritizes Comp-Comp over others.)
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

# ============================================
# Helper Function for Barrier Synchronization Across Subcircuits
# ============================================


# ============================================
# Helper Functions Pytket DQC + SABRE
# ============================================

from qiskit.compiler import transpile

def _generate_base_circuit(cz_frac, seed):
    """Generates and prepares the initial logical circuit."""
    qc = build_cz_fraction_circuit(n=36, d=36, p=cz_frac, seed=seed)
    
    # 1. Aggressive Sanitization (Strip classical logic, resets, measurements)
    clean_data = []
    for inst in qc.data:
        if inst.operation.name not in ['measure', 'reset', 'barrier', 'delay']:
            clean_data.append(inst)
    qc.data = clean_data
    qc.cregs.clear()
    qc.clbits.clear()

    qc = transpile(qc, basis_gates=['cz', 'id', 'rz', 'sx', 'x'], optimization_level=2)

    circ = qiskit_to_tk(qc) 
    
    # Pytket DQC Setup
    t_0 = perf_counter()
    DQCPass().apply(circ)
    dt_dqc_pass = perf_counter() - t_0
    print(f"[DQCPass] Time taken = {dt_dqc_pass:.4f}s") # Debug print

    # Generate the sanitized Qiskit version for the baselines
    cz_qiskit = tk_to_qiskit(circ)
    cz_qiskit_sanitized = sanitize_qiskit_labels(cz_qiskit)
    
    return circ, cz_qiskit_sanitized

def _distribute_and_optimize(cz_circ, seed, distributor_class):
    """Handles network embedding, refinement, and barrier optimization."""
    t_0 = perf_counter()
    distributor_instance = distributor_class()
    distribution = distributor_instance.distribute(cz_circ, network=network, seed=seed)
    
    # refiner_list = [NeighbouringDTypeMerge(), IntertwinedDTypeMerge()]
    # RepeatRefiner(SequenceRefiner(refiner_list)).refine(distribution)
    # assert distribution.detached_gate_count() == 0, f"There exist {distribution.detached_gate_count()} detached gates!"
    print(f"[Seed {seed}] There exist {distribution.detached_gate_count()} detached gates!") # Debug print; Use when considering `PartitioningHeterogeneous` which may produce detached gates before refinement.

    t_1 = perf_counter()
    dt_distribution = t_1 - t_0
    print(f"[Distribution] Distribution and refinement time = {dt_distribution:.4f}s") # Debug print

    final_circuit = distribution.to_pytket_circuit(allow_update=True, verify_equivalence=False)
    dt_generation = perf_counter() - t_1
    print(f"[Distribution] Distributed circuit generation time = {dt_generation:.4f}s") # Debug print

    # Equivalence & Constraint Checks
    final_circuit_without_barriers = remove_barriers(final_circuit)
    assert check_equivalence(cz_circ, final_circuit_without_barriers, distribution.get_qubit_mapping()), "Equivalence check failed!"
    assert not check_violations(final_circuit, server_link_capacities, verbose=True), "Network constraints violated!"

    routing_time = dt_distribution + dt_generation
    routing_cost = ebit_cost(final_circuit)
    
    return final_circuit, routing_time, routing_cost, distribution

def _generate_sanitized_subcircuits(cz_circ, pytket_dqc_circuit, distribution):
    """Builds subcircuits, squashes them, converts to Qiskit, and sanitizes."""
    busy_links = {pair: [] for pair in PHYSICAL_LINKS}
    available_links = {pair: list(links) for pair, links in PHYSICAL_LINKS.items()}

    t_0 = perf_counter()
    subcircuits, distributed_circ = build_distributed_subcircuits(
        pytket_dqc_circuit, network, available_links, busy_links,
        link_to_virtual_mapping, link_to_virtual_return, # virtual_to_comp_mapping, # --- IGNORE ---
        n_servers=N_SERVERS, n_links=N_LINKS
    )
    dt_subcircuit_generation = perf_counter() - t_0

    print(f"[Subcircuit Generation] Time taken = {dt_subcircuit_generation:.4f}s") # Debug print

    # Re-build the global circuit from the generated subcircuits to verify that the transformations applied during subcircuit generation (like placeholder insertion) do not alter the overall functionality of the distributed circuit. This is crucial for validating the correctness of our subcircuit generation process before we proceed to transpilation and routing.
    distributed_circ_no_barriers = remove_barriers(distributed_circ)
    distributed_circ_no_barriers.remove_blank_wires() # Strip out all auto-padded Ghost qubits generated by PyTket's register sizing

    # #####################################################
    # # Peek into distributed circuit's registers and sizes
    # print("\n--- Distributed Circuit Registers (pytket_dqc_circuit) ---")
    # for reg in pytket_dqc_circuit.q_registers:
    #     print(f"Register: {reg.name}, Size: {reg.size}")
    # for cmd in pytket_dqc_circuit.get_commands():
    #     if cmd.op.type == OpType.CustomGate:
    #         print(f"{cmd}")

    # print("\n--- distributed_circ_no_barriers Registers ---")
    # for reg in distributed_circ_no_barriers.q_registers:
    #     print(f"Register: {reg.name}, Size: {reg.size}")
    # for q in distributed_circ_no_barriers.qubits:
    #     print(f"Qubit: {q}")
    
    # print("\n--- `distribution.get_qubit_mapping()` ---")
    # print(F"Mapping: {distribution.get_qubit_mapping()}")
    # #####################################################

    assert check_equivalence(cz_circ, distributed_circ_no_barriers, distribution.get_qubit_mapping()), "The reconstructed circuit from subcircuits is not equivalent to the original circuit! This indicates a potential issue in the subcircuit generation process. Please investigate the transformations applied during subcircuit generation to identify where the discrepancy arises."
    assert not check_violations(distributed_circ, server_link_capacities, verbose=True), "The reconstructed circuit violates the network constraints! This indicates a potential issue in the subcircuit generation process. Please investigate the transformations applied during subcircuit generation to identify where the discrepancy arises."

    # 1. Get the optimized master circuit and the list of safe-to-remove indices
    final_optimized_circuit_distributed, removed_barrier_indices = optimize_circuit_barriers(
        distributed_circ, 
        server_link_capacities, 
        barrier_index_to_check=0
    )
    assert check_equivalence(remove_barriers(pytket_dqc_circuit), distributed_circ_no_barriers, distribution.get_qubit_mapping(), distributed_comparison=True) # Re-run with this addition.
    assert not check_violations(final_optimized_circuit_distributed, server_link_capacities, verbose=True), "The optimized (distributed) circuit violates the network constraints!"

    # 2. Sync the subcircuits by stripping out those exact same indices
    subcircuits = sync_subcircuit_barriers(subcircuits, removed_barrier_indices)

    sanitized_qiskit_circuits = {}
    for qpu_id, circ in subcircuits.items():
        # Process pipeline
        squashed = squash_placeholders_per_pair(circ)
        qiskit_circ = tk_to_qiskit(squashed)
        sanitized = sanitize_qiskit_labels(qiskit_circ)
        
        # Identity Verification
        id_test = fast_identity_test(qiskit_circ, sanitized, sim=AerSimulator(method='matrix_product_state'))
        assert id_test, f"Identity test failed for subcircuit {qpu_id} after sanitization!"
        
        sanitized_qiskit_circuits[qpu_id] = sanitized

    return sanitized_qiskit_circuits

import numpy as np

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

            # --- Change this back to SabreSwap! ---
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

        # --- SAFETY NET ---
        if best_qc is None:
            raise RuntimeError(f"CRITICAL: All {num_trials} layout trials on QPU {qpu_id} violated fixed constraints!")

        print(f"  -> Best [QPU {qpu_id}] Cost: {best_swaps} SWAPs (Layout: {best_layout_time:.4f}s | Route: {best_routing_time:.4f}s)")

        transpiled_subcircuits[qpu_id] = remove_routing_placeholders(best_qc)
        exact_routing_costs[qpu_id] = best_swaps
        intra_qpu_routing_time_costs[qpu_id] = best_layout_time + best_routing_time

    intra_time = sum(intra_qpu_routing_time_costs.values())
    intra_cost = sum(exact_routing_costs.values())

    return transpiled_subcircuits, intra_time, intra_cost

def run_experiment(cz_frac, lf, seed, distributor_class):
    print(f"\n--- Running Experiment (CZ Frac: {cz_frac}, Seed: {seed}) ---")
    
    # 1. Generation
    cz_circ, _ = _generate_base_circuit(cz_frac, seed)
    
    # 2. Distribution
    distributed_circuit, inter_time, inter_cost, distribution = _distribute_and_optimize(cz_circ, seed, distributor_class)
    
    # 3. Subcircuit processing
    sanitized_subcircuits = _generate_sanitized_subcircuits(cz_circ, distributed_circuit, distribution)
    
    # 4. Transpilation
    final_subcircuits, intra_time, intra_cost = _transpile_subcircuits(sanitized_subcircuits, lf, seed)

    # Debug prints
    print(f"Inter-QPU Routing -> Time: {inter_time:.4f}s | Cost: {inter_cost} EPR pairs")
    print(f"Intra-QPU Routing -> Time: {intra_time:.4f}s | Cost: {intra_cost} SWAPs")

    return final_subcircuits, inter_time, inter_cost, intra_time, intra_cost

# ============================================
# Helper Functions Custom SABRE Baseline
# ============================================

import gc

def generate_custom_distance_matrix(mqpu_backend, inter_qpu_edges, factor=10):
    """Calculates the (1, 10) weighted distance matrix S once."""
    weighted_edges = []
    for edge in mqpu_backend.coupling_map.get_edges():
        if edge in inter_qpu_edges:
            weighted_edges.append((edge[0], edge[1], 1)) # Penalized
        else:
            weighted_edges.append((edge[0], edge[1], factor)) # Prioritized

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
        layout_trials=1,           # <--- CRITICAL: Force Rust to only output this exact seed's result!
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
        # FullAncillaAllocation(mqpu_backend.coupling_map), EnlargeWithAncilla(), ApplyLayout()
    ])
    pm.routing = PassManager() # Mute default routing
    
    return pm

import gc
from time import perf_counter

def run_baseline_experiment(cz_frac, seed, mqpu_backend, inter_qpu_edges, S_matrix, penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20, strategy_name="Baseline"):
    """Runs multiple baseline trials, picks the best by EPR cost, and manages memory."""
    # 1. Circuit Generation
    _, cz_qiskit_sanitized = _generate_base_circuit(cz_frac, seed)

    # 2. Pre-evaluate original swaps (Should be 0, but good to keep the check)
    orig_intra_swaps = 0
    orig_inter_swaps = 0
    for instr in cz_qiskit_sanitized.data:
        if instr.operation.name == "swap":
            q0_idx = cz_qiskit_sanitized.find_bit(instr.qubits[0]).index
            q1_idx = cz_qiskit_sanitized.find_bit(instr.qubits[1]).index
            if (q0_idx, q1_idx) in inter_qpu_edges or (q1_idx, q0_idx) in inter_qpu_edges:
                orig_inter_swaps += 1
            else:
                orig_intra_swaps += 1

    original_swap_count = cz_qiskit_sanitized.count_ops().get('swap', 0)
    assert original_swap_count == 0, (
        f"CRITICAL: Expected 0 original SWAPs after DQC_Pass, but found {original_swap_count}."
    )

    # Variables for the multi-trial loop
    num_trials = 5
    f_weight = 10
    best_cost = float('inf')
    best_metrics = {}
    best_time = 0.0

    routing_metrics = {}
    inter_qpu_edges_set = {tuple(sorted(edge)) for edge in inter_qpu_edges}

    # Define the robust callback
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

    # --- 3. The 5-Trial Evaluation Loop ---
    for trial_idx in range(num_trials):
        current_seed = seed + trial_idx 
        
        # Build PassManager with layout_trials=1 so it respects this specific seed
        pm = get_baseline_pass_manager(
            mqpu_backend, S_matrix, current_seed, penalized_swaps, qubit_qpu_map, inter_qpu_coupling_map, extended_set_length=extended_set_length
        )
        
        routing_metrics.clear()
        
        t_0 = perf_counter()
        _ = pm.run(cz_qiskit_sanitized, callback=audit_sabreswap)
        trial_time = perf_counter() - t_0
        
        intra_swaps = routing_metrics.get('final_intra_swaps', 0)
        inter_swaps = routing_metrics.get('final_inter_swaps', 0)
        inter_gates = routing_metrics.get('final_inter_czs', 0)
        
        # Calculate custom cost
        total_eprs = (inter_swaps * 3) + inter_gates
        aggregated_cost = (total_eprs * f_weight) + intra_swaps
        
        # Keep the best one
        if aggregated_cost < best_cost:
            best_cost = aggregated_cost
            best_metrics = routing_metrics.copy()
            best_time = trial_time  # Save the time of the winning trial

        # Aggressive memory scrubbing per trial!
        try:
            del pm
        except NameError:
            pass
        gc.collect()

    # --- 4. Math and Returns (Using the Best Metrics) ---
    added_intra_swaps = best_metrics['final_intra_swaps'] - orig_intra_swaps
    total_ebits = (best_metrics['final_inter_swaps'] * 3) + (best_metrics['final_inter_czs'] * 1)

    print(f"[{strategy_name}] BEST Routing -> Time: {best_time:.4f}s | Inter SWAPs: {best_metrics['final_inter_swaps']} | Inter CZs: {best_metrics['final_inter_czs']} | Intra SWAPs Added: {added_intra_swaps}")

    # Scrub the circuit before returning
    try:
        del cz_qiskit_sanitized
    except NameError:
        pass 
    gc.collect()

    return total_ebits, added_intra_swaps, best_time

# ============================================
# Helper Functions for Data Handling & Experiment Management
# ============================================

from datetime import time
from experiment_utils import *

def run_single_seed_task(
    frac, lf, seed, mqpu_backend, inter_qpu_edges, S_matrix, S_matrix_default,
    penalized_swaps=None, qubit_qpu_map=None, inter_qpu_coupling_map=None, extended_set_length=20,
    run_dqc=True, run_base=True, run_default=True, run_custom=True, verbose=False, distributor_class=CoverEmbedding):
    
    dqc_results, base_results, default_results, custom_results = None, None, None, None
    
    try:
        # ---- 1. Pytket-DQC Method ----
        if run_dqc:
            _, inter_t, inter_c, intra_t, intra_c = run_experiment(frac, lf, seed, distributor_class)
            dqc_results = (inter_c, intra_c, inter_t + intra_t)

        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed}: Pytket-DQC finished. Memory should spike NOW.")

        # ---- 2. (1, 10) SABRE ----
        if run_base:
            base_ebits, base_swaps, base_time = run_baseline_experiment(
                frac, seed, mqpu_backend, inter_qpu_edges, S_matrix, 
                strategy_name="(1, 10) SABRE"
            )
            base_results = (base_ebits, base_swaps, base_time)

        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed}: (1, 10) SABRE finished.")
            
        # ---- 3. Default SABRE Baseline ----
        if run_default:
            default_ebits, default_swaps, default_time = run_baseline_experiment(
                frac, seed, mqpu_backend, inter_qpu_edges, S_matrix_default, 
                strategy_name="Default SABRE"
            )
            default_results = (default_ebits, default_swaps, default_time)
            
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed}: Default SABRE finished.")

        # ---- 4. Custom Lookahead SABRE ----
        if run_custom:
            custom_ebits, custom_swaps, custom_time = run_baseline_experiment(
                frac, seed, mqpu_backend, inter_qpu_edges, S_matrix_default, 
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
        # Force garbage collection to mitigate memory issues in long runs
        gc.collect()

# ==========================================
# Parallelized Experiment Setup & Configuration
# ==========================================

from pytket_dqc.distributors import CoverEmbedding, PartitioningHeterogeneous, PartitioningHeterogeneousEmbedding # (Add your imports)

CURRENT_DISTRIBUTOR = PartitioningHeterogeneous 
DISTRIBUTOR_NAME = "PartitioningHeterogeneous"
DQC_COLOR = "#118ab2"

# Colour palette for plotting (Add more colors if you add more methods!)
# "Summer Sunset Paradise": https://coolors.co/ef476f-f78c6b-ffd166-06d6a0-118ab2-073b4c
# "#ef476f": Pytket-DQC + SABRE (CoverEmbedding, lf=True)
# "#f78c6b": (1,10) SABRE
# "#ffd166": Default SABRE Baseline
# "#06d6a0": Custom Lookahead SABRE
# "#118ab2": Pytket-DQC + SABRE (PartitioningHeterogeneous, lf=True)
# "#073b4c": DMapS MultiModeRouting

# ---------------------------------------------------------

# ------------------------------------------
# 1. Experiment Hyperparameters
# ------------------------------------------
CZ_FRACS_TO_RUN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9] # List of CZ fractions to test
TARGET_SUCCESSFUL_SEEDS = 5  # Exact number of valid runs we want per fraction
STARTING_SEED = 42           # Incremented by 1 for every new attempt
TIMEOUT_SECONDS = 1.0 * 3600 # 1 hour(s) max execution time per cz_frac
LF_FLAG = True               # Flag for Layout initialization (lf)      

# ------------------------------------------
# 2. File I/O & Multiprocessing Setup
# ------------------------------------------
RESULTS_FILE = os.path.join(RESULTS_DIR, f"three_square_cz_frac_lf={LF_FLAG}.json")
db = load_experiment_database(RESULTS_FILE)
WORKERS = min(TARGET_SUCCESSFUL_SEEDS, os.cpu_count() or 1)

# ------------------------------------------
# 3. Hardware & Topology Maps
# ------------------------------------------
# PhysicalQubit i -> QPU number it belongs to.
qubit_qpu_map = [i // square_backend.num_qubits for i in range(square_mqpu_backend.num_qubits)] 
# Abstract map of how the QPUs connect to each other
inter_qpu_coupling_map = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]

# ------------------------------------------
# 4. Algorithm-Specific Parameters
# ------------------------------------------
# A) Distance Matrices for Distance-Penalty SABRE
# (10, 1) weighted matrix (prioritizes keeping qubits in the same QPU)
S_matrix = generate_custom_distance_matrix(square_mqpu_backend, square_inter_qpu_edges) 
# Uniform distance matrix without prioritization (for default SABRE baseline)
S_matrix_default = generate_custom_distance_matrix(square_mqpu_backend, square_inter_qpu_edges, factor=1) 

# B) Parameters for Custom Lookahead SABRE
penalized_swaps = square_inter_qpu_edges
extended_set_length = 20

# ==========================================
# Parallelized Experiment Execution
# ==========================================

import time

def has_data(db, m_idx, frac_idx):
    # (Keep your existing has_data function exactly as is)
    if m_idx >= len(db["methods_data"]): return False
    method = db["methods_data"][m_idx]
    if "ebits" not in method or frac_idx >= len(method["ebits"]): return False
    val = method["ebits"][frac_idx]
    if val is None or val == []: return False
    if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], float) and math.isnan(val[0]):
        return True
    return True

# --- DYNAMIC INDEX MAPPING ---
# Define the label for the current run based on the configuration above
DQC_LABEL = f"Pytket-DQC + SABRE ({DISTRIBUTOR_NAME}, lf={LF_FLAG})"

# Look up (or create) the indices dynamically!
DQC_IDX = get_or_create_method_index(db, DQC_LABEL, DQC_COLOR)
BASE_IDX = get_or_create_method_index(db, "(1,10) SABRE", "#f78c6b")
DEFAULT_IDX = get_or_create_method_index(db, "Default SABRE Baseline", "#ffd166")
CUSTOM_IDX = get_or_create_method_index(db, "Custom Lookahead SABRE", "#06d6a0")

for frac in CZ_FRACS_TO_RUN:
    # 1. Register Fraction and Get Index
    if frac in db["cz_frac_list"]:
        frac_idx = db["cz_frac_list"].index(frac)
    else:
        frac_idx = len(db["cz_frac_list"])
        db["cz_frac_list"].append(frac)
        
    # 2. Pad arrays to ensure we can safely overwrite at frac_idx
    for m in db["methods_data"]:
        while len(m["ebits"]) <= frac_idx:
            m["ebits"].append(None); m["swaps"].append(None); m["time"].append(None)
            m.setdefault("raw_ebits", []).append([])
            m.setdefault("raw_swaps", []).append([])
            m.setdefault("raw_time", []).append([])

    # 3. Assess missing data
    needs_dqc = not has_data(db, DQC_IDX, frac_idx)
    needs_base = not has_data(db, BASE_IDX, frac_idx)
    needs_default = not has_data(db, DEFAULT_IDX, frac_idx)
    needs_custom = not has_data(db, CUSTOM_IDX, frac_idx)
    
    if not (needs_dqc or needs_base or needs_default or needs_custom):
        print(f"Skipping CZ Fraction {frac}: All requested variants already computed.")
        continue
        
    print(f"\n--- Evaluating CZ Fraction: {frac} (Target: {TARGET_SUCCESSFUL_SEEDS} successes) ---")
    print(f"Tasks needed -> DQC: {needs_dqc}, Base: {needs_base}, Default: {needs_default}, Custom: {needs_custom}")
    
    start_time = time.time()
    successful_runs = []
    current_seed = STARTING_SEED
    dqc_timed_out = False

    def submit_job(executor, seed_val, run_dqc_flag):
        return executor.submit(
            # Altered to use `square_` variables for the homogeneous topology (to match DMapS behaviour);
            # This assumes every "link" is associated with a pair of link qubits,
            # hence we can only perform quantum gate teleportation, not quantum state teleportation.
            run_single_seed_task, frac, LF_FLAG, seed_val, square_mqpu_backend, square_inter_qpu_edges,
            S_matrix, S_matrix_default, penalized_swaps=penalized_swaps, 
            qubit_qpu_map=qubit_qpu_map, inter_qpu_coupling_map=inter_qpu_coupling_map, 
            extended_set_length=extended_set_length, 
            run_dqc=run_dqc_flag, run_base=needs_base, run_default=needs_default, run_custom=needs_custom,
            verbose=True, distributor_class=CURRENT_DISTRIBUTOR # Enable verbose logging for each seed's progress and memory spike points
        )

    # ---------------------------------------------------------
    # PHASE 1: Attempt needed evaluations
    # ---------------------------------------------------------
    
    # Check if we have a history we MUST synchronize with
    historical_seeds = []
    if frac_idx < len(db.get("seed_history", [])) and db["seed_history"][frac_idx]:
        historical_seeds = list(db["seed_history"][frac_idx])

    # If we are filling in baselines, we strictly enforce the historical seeds
    enforce_history = bool(historical_seeds and not needs_dqc)
    
    if enforce_history:
        print(f"Synchronizing baselines with previously successful seeds: {historical_seeds}")
        
    # Status print about timeout configuration
    print(f"Setting timeout to {TIMEOUT_SECONDS/3600:.2f} hours based on worker count and target seeds.")

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)
    future_to_seed = {}
    
    # Initial job submission
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
            
            # Only enforce the strict timer if DQC is actively running
            if needs_dqc and time_left <= 0:
                print(f"TIMEOUT: CZ fraction {frac} exceeded time limit! Aborting DQC.")
                dqc_timed_out = True
                break

            done, not_done = concurrent.futures.wait(
                future_to_seed.keys(), 
                timeout=time_left if needs_dqc else None,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            if not done and needs_dqc:
                print(f"TIMEOUT: CZ fraction {frac} exceeded time limit! Aborting DQC.")
                dqc_timed_out = True
                break
                
            for future in done:
                finished_seed = future_to_seed.pop(future)
                try:
                    seed_res, dqc_res, base_res, default_res, custom_res = future.result()
                    
                    # Ensure the runs we *asked for* actually returned successfully
                    dqc_ok = (not needs_dqc) or (dqc_res is not None)
                    base_ok = (not needs_base) or (base_res is not None)
                    def_ok = (not needs_default) or (default_res is not None)
                    cust_ok = (not needs_custom) or (custom_res is not None)
                    
                    if dqc_ok and base_ok and def_ok and cust_ok:
                        successful_runs.append((finished_seed, dqc_res, base_res, default_res, custom_res))
                        print(f"  -> Seed {finished_seed} completed successfully! ({len(successful_runs)}/{TARGET_SUCCESSFUL_SEEDS})")

                        # --- NEW: Checkpoint Save ---
                        with open(f"./results/CHECKPOINT_{frac}.json", 'w') as ckpt:
                            json.dump(successful_runs, ckpt)
                    else:
                        if enforce_history:
                            print(f"FATAL: A baseline failed on historical seed {finished_seed}! Cannot ensure fair comparison.")
                            total_failures = MAX_ALLOWED_FAILURES # Force abort
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
                print(f"FATAL: Reached {MAX_ALLOWED_FAILURES} consecutive failures. Aborting CZ Frac {frac}.")
                break

    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------
    # PHASE 2: Fallback (Gathering missing baselines if DQC timed out)
    # ---------------------------------------------------------
    if dqc_timed_out and (needs_base or needs_default or needs_custom):
        print(f"--- Running Fallback: Gathering Baseline-only data for CZ Fraction {frac} ---")
        successful_runs = [] # Clear incomplete data to perfectly align target seeds
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as fallback_executor:
            future_to_seed_fb = {}
            for _ in range(TARGET_SUCCESSFUL_SEEDS):
                fut = submit_job(fallback_executor, current_seed, False) # Force DQC flag to False
                future_to_seed_fb[fut] = current_seed
                current_seed += 1
                
            for future in concurrent.futures.as_completed(future_to_seed_fb):
                finished_seed = future_to_seed_fb[future]
                _, _, base_res, default_res, custom_res = future.result()
                successful_runs.append((finished_seed, None, base_res, default_res, custom_res))
                print(f"  -> Baseline Seed {finished_seed} completed! ({len(successful_runs)}/{TARGET_SUCCESSFUL_SEEDS})")

    # ---------------------------------------------------------
    # PHASE 3: Data Unpacking & Direct Index Overwrite
    # ---------------------------------------------------------
    if len(successful_runs) < TARGET_SUCCESSFUL_SEEDS and not dqc_timed_out:
        print(f"WARNING: CZ Frac {frac} did not complete successfully. Skipping save.")
        continue

    def update_db(m_idx, needs_run, res_idx):
        """Helper to unpack successfully gathered metrics and write to DB."""
        if not needs_run: return
        t_ebits, t_swaps, t_time = [], [], []
        for run_data in successful_runs:
            res = run_data[res_idx] 
            t_ebits.append(res[0]); t_swaps.append(res[1]); t_time.append(res[2])
            
        db["methods_data"][m_idx]["ebits"][frac_idx] = compute_statistics(t_ebits)
        db["methods_data"][m_idx]["swaps"][frac_idx] = compute_statistics(t_swaps)
        db["methods_data"][m_idx]["time"][frac_idx] = compute_statistics(t_time)
        db["methods_data"][m_idx]["raw_ebits"][frac_idx] = t_ebits
        db["methods_data"][m_idx]["raw_swaps"][frac_idx] = t_swaps
        db["methods_data"][m_idx]["raw_time"][frac_idx] = t_time

    # Evaluate DQC (Index 1 in run_data tuple)
    if needs_dqc:
        if dqc_timed_out:
            nan_tup = (np.nan, np.nan, np.nan)
            db["methods_data"][DQC_IDX]["ebits"][frac_idx] = nan_tup
            db["methods_data"][DQC_IDX]["swaps"][frac_idx] = nan_tup
            db["methods_data"][DQC_IDX]["time"][frac_idx] = nan_tup
        else:
            update_db(DQC_IDX, True, 1)

    # Evaluate Baselines
    update_db(BASE_IDX, needs_base, 2)
    update_db(DEFAULT_IDX, needs_default, 3)
    update_db(CUSTOM_IDX, needs_custom, 4)

    # Save exact seeds used for this run
    while len(db.setdefault("seed_history", [])) <= frac_idx: db["seed_history"].append([])
    db["seed_history"][frac_idx] = [r[0] for r in successful_runs]

    save_experiment_database(db, RESULTS_FILE)
    print(f"Results for CZ Fraction {frac} successfully updated and saved to disk.")