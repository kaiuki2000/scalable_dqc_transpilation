"""Quantum Interaction Graph (QIG) partitioning.

Implements the QIG pre-processing step described in Section V-C of
"Scalable Transpilation for Overcoming Restricted Connectivity in
Distributed Superconducting Quantum Architectures" (Azenha, Polian &
Brandhofer): rather than the time-sliced, per-gate hypergraphs used by
pytket-dqc, a single global interaction graph is built over the circuit's
virtual qubits (vertices) and pairwise two-qubit-gate counts (edge
weights), then partitioned into one block per physical core with KaHyPar.

This is independent of both patched forks in this repo -- it only needs
Qiskit (to read the circuit), NumPy, and KaHyPar's Python bindings -- so
it ships here as an ordinary importable package rather than as a patch.

Cores are assumed to be equally sized, as they are in the paper's
architectures: one `max_capacity` applies to all of them, and the KaHyPar
call asks for near-equal blocks.

Two post-processing steps are then applied, matching the paper's own
description (Section V-A) of adapting KaHyPar's output to a fixed,
per-core hardware capacity:

1. Strict Capacity Enforcement: iteratively eject the qubit whose removal
   incurs the smallest increase in communication cost from any core that
   KaHyPar over-allocated, until every core is within its hardware limit.
2. Boundary Reallocation: iteratively move qubits at partition boundaries
   to a neighboring, under-allocated core whenever doing so strictly
   reduces total communication cost.
"""

from __future__ import annotations

import itertools
from importlib import resources
from typing import Any

import kahypar
from qiskit.circuit import QuantumCircuit, Qubit

__all__ = [
    "build_interaction_graph",
    "partition_with_kahypar",
    "match_partitions_to_cores",
    "enforce_strict_capacity",
    "boundary_reallocation",
    "get_heterogeneous_core_assignment",
    "default_kahypar_config_path",
]


def default_kahypar_config_path() -> str:
    """Path to the bundled KaHyPar SEA20 config (`km1_kKaHyPar_sea20.ini`).

    Reproduced from pytket-dqc (Apache License 2.0) so this package does
    not itself require pytket-dqc to be installed.
    """
    with resources.as_file(
        resources.files("qig_partitioning") / "config" / "km1_kKaHyPar_sea20.ini"
    ) as path:
        return str(path)


def build_interaction_graph(
    qc: QuantumCircuit,
) -> tuple[dict[tuple[int, int], int], dict[Qubit, int], dict[int, Qubit]]:
    """Build the QIG: one vertex per virtual qubit, edge weight = number of
    two-qubit gates between that pair anywhere in the circuit.

    Returns `(edges, qubit_to_idx, idx_to_qubit)`, where `edges` maps a
    sorted `(u, v)` index pair to its gate count.
    """
    qubit_to_idx = {q: i for i, q in enumerate(qc.qubits)}
    idx_to_qubit = {i: q for i, q in enumerate(qc.qubits)}

    edges: dict[tuple[int, int], int] = {}
    for instruction in qc.data:
        qargs = instruction.qubits
        if len(qargs) == 2:
            u, v = qubit_to_idx[qargs[0]], qubit_to_idx[qargs[1]]
            if u > v:
                u, v = v, u
            edges[(u, v)] = edges.get((u, v), 0) + 1

    return edges, qubit_to_idx, idx_to_qubit


def partition_with_kahypar(
    num_nodes: int,
    edges: dict[tuple[int, int], int],
    k: int,
    kahypar_config_path: str | None = None,
    epsilon: float = 0.01,
) -> dict[int, int]:
    """Partition the QIG into `k` abstract blocks with KaHyPar.

    Returns a mapping from qubit index to abstract block id (0..k-1) --
    *not* yet matched to physical cores; see `match_partitions_to_cores`.
    """
    hyperedge_indices = [0]
    hyperedges: list[int] = []
    edge_weights: list[int] = []
    for (u, v), w in edges.items():
        hyperedges.extend([u, v])
        hyperedge_indices.append(len(hyperedges))
        edge_weights.append(w)

    node_weights = [1] * num_nodes

    hypergraph = kahypar.Hypergraph(
        num_nodes, len(edges), hyperedge_indices, hyperedges, k, edge_weights, node_weights
    )

    context = kahypar.Context()
    context.loadINIconfiguration(kahypar_config_path or default_kahypar_config_path())
    context.setK(k)
    context.setEpsilon(epsilon)
    context.suppressOutput(True)

    kahypar.partition(hypergraph, context)
    return {i: hypergraph.blockID(i) for i in range(num_nodes)}


def match_partitions_to_cores(
    abstract_assignment: dict[int, int],
    edges: dict[tuple[int, int], int],
    core_cost_matrix: dict[Any, dict[Any, float]],
) -> dict[int, Any]:
    """Match KaHyPar's abstract blocks (0..k-1) to physical cores.

    Evaluates every block-to-core permutation and picks the one that
    minimizes total communication cost against `core_cost_matrix`, so
    heterogeneous inter-core costs (e.g. a restricted, non-all-to-all
    topology) are taken into account even though KaHyPar itself optimizes
    a topology-agnostic cut metric.
    """
    cores = list(core_cost_matrix.keys())
    best_cost = float("inf")
    best_mapping: dict[int, Any] = {}

    for perm in itertools.permutations(cores):
        mapping = dict(enumerate(perm))
        cost = sum(
            w * core_cost_matrix[mapping[abstract_assignment[u]]][mapping[abstract_assignment[v]]]
            for (u, v), w in edges.items()
        )
        if cost < best_cost:
            best_cost = cost
            best_mapping = mapping

    return {node: best_mapping[block] for node, block in abstract_assignment.items()}


def _neighbors(node: int, edges: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    result = []
    for (u, v), w in edges.items():
        if u == node:
            result.append((v, w))
        elif v == node:
            result.append((u, w))
    return result


def enforce_strict_capacity(
    assignment: dict[int, Any],
    edges: dict[tuple[int, int], int],
    core_cost_matrix: dict[Any, dict[Any, float]],
    max_capacity: int,
) -> dict[int, Any]:
    """Force any core holding more than `max_capacity` qubits to eject
    qubits to under-full cores, always choosing the qubit/target pair with
    the smallest resulting increase in communication cost. `max_capacity`
    is a per-core qubit count. Mutates and returns `assignment`.
    """
    core_counts = {core: sum(1 for c in assignment.values() if c == core) for core in core_cost_matrix}

    for core in core_cost_matrix:
        while core_counts[core] > max_capacity:
            nodes_in_core = [n for n, c in assignment.items() if c == core]

            best_node, best_target, min_penalty = None, None, float("inf")
            for node in nodes_in_core:
                neighbors = _neighbors(node, edges)
                current_cost = sum(w * core_cost_matrix[core][assignment[v]] for v, w in neighbors)
                for candidate in core_cost_matrix:
                    if candidate == core or core_counts[candidate] >= max_capacity:
                        continue
                    cand_cost = sum(w * core_cost_matrix[candidate][assignment[v]] for v, w in neighbors)
                    penalty = cand_cost - current_cost
                    if penalty < min_penalty:
                        min_penalty, best_node, best_target = penalty, node, candidate

            if best_node is not None and best_target is not None:
                assignment[best_node] = best_target
                core_counts[core] -= 1
                core_counts[best_target] += 1
            else:
                # Failsafe for isolated qubits with no edges to weigh a choice by.
                moved = False
                for n in nodes_in_core:
                    for c in core_cost_matrix:
                        if core_counts[c] < max_capacity:
                            assignment[n] = c
                            core_counts[core] -= 1
                            core_counts[c] += 1
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    break  # Every core is at capacity; nothing left to do.

    return assignment


def boundary_reallocation(
    assignment: dict[int, Any],
    edges: dict[tuple[int, int], int],
    core_cost_matrix: dict[Any, dict[Any, float]],
    max_capacity: int,
    max_iters: int = 100,
) -> dict[int, Any]:
    """Iteratively move boundary qubits to a neighboring core that is under
    its per-core `max_capacity`, whenever doing so strictly reduces
    communication cost. Mutates and returns `assignment`.
    """
    core_counts = {core: sum(1 for c in assignment.values() if c == core) for core in core_cost_matrix}

    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        for node in list(assignment.keys()):
            current_core = assignment[node]
            neighbors = _neighbors(node, edges)
            best_cost = sum(w * core_cost_matrix[current_core][assignment[v]] for v, w in neighbors)
            best_core = current_core

            for candidate in {assignment[v] for v, _ in neighbors}:
                if candidate == current_core or core_counts[candidate] >= max_capacity:
                    continue
                cand_cost = sum(w * core_cost_matrix[candidate][assignment[v]] for v, w in neighbors)
                if cand_cost < best_cost:
                    best_cost, best_core = cand_cost, candidate

            if best_core != current_core:
                assignment[node] = best_core
                core_counts[current_core] -= 1
                core_counts[best_core] += 1
                improved = True
        iters += 1

    return assignment


def get_heterogeneous_core_assignment(
    qc: QuantumCircuit,
    core_cost_matrix: dict[Any, dict[Any, float]],
    kahypar_config_path: str | None = None,
    max_capacity: int | None = None,
) -> dict[Qubit, Any]:
    """Run the full QIG pre-processing pipeline: build the interaction
    graph, partition it with KaHyPar, match blocks to physical cores by
    communication cost, then enforce hardware capacity limits.

    `core_cost_matrix` is a `{core: {other_core: cost}}` mapping (cost 0
    on the diagonal); it need not be uniform, so restricted/non-all-to-all
    inter-core topologies are supported directly.

    `max_capacity` is the maximum number of qubits **per core**, not a
    total across all cores, and the same limit applies to every core --
    cores of differing sizes are not currently supported. If it is `None`,
    cores are treated as having unlimited capacity and no capacity
    post-processing is applied.

    "Heterogeneous" here refers to the inter-core *topology*, not to core
    sizes.

    Returns a mapping from each circuit qubit to its assigned core.
    """
    edges, qubit_to_idx, idx_to_qubit = build_interaction_graph(qc)
    num_cores = len(core_cost_matrix)

    abstract_assignment = partition_with_kahypar(
        len(qubit_to_idx), edges, num_cores, kahypar_config_path
    )
    assignment = match_partitions_to_cores(abstract_assignment, edges, core_cost_matrix)

    if max_capacity is not None:
        assignment = enforce_strict_capacity(assignment, edges, core_cost_matrix, max_capacity)
        assignment = boundary_reallocation(assignment, edges, core_cost_matrix, max_capacity)

    return {idx_to_qubit[idx]: core for idx, core in assignment.items()}
