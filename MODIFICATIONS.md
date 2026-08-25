# Key changes

This document walks through the hunks that carry the paper's contribution,
pulled out of the full patches in `patches/` and annotated against the
paper itself (`Paper_draft_portrait.pdf` in this repo — *"Scalable
Transpilation for Overcoming Restricted Connectivity in Distributed
Superconducting Quantum Architectures,"* Azenha, Polian & Brandhofer). It's
meant to be read, not applied — see `README.md` for how to apply the real
patches to a working checkout.

## Qiskit: three DQC-aware SABRE variants (paper §IV)

The paper introduces three named SABRE variants, each building on the
last. All three are implemented in
`patches/qiskit/0001-dqc-aware-sabre-variants.patch`, diffed against
[`c62ce998`](https://github.com/kaiuki2000/qiskit/tree/c62ce9982a7bf7764072ea89296077db4583b679)
on the fork's `exponential_decay` branch — the code as it stood when the
paper's results were produced. (Later commits on that branch continue
development past the paper, e.g. an exponential-decay-weighted variant —
not part of what's reproduced here.)

### Default SABRE — the DQC-adapted baseline

The paper's central architectural idea (§IV-A) is the **conjoined
coupling map**: instead of routing each core separately, stitch every
core's coupling graph together with extra edges for the inter-core links
(Fig. 1 in the paper), and run ordinary SABRE over the whole thing. That
idea needs no patch by itself — it's just a `CouplingMap` that happens to
span multiple cores. What *does* need patching is making that scale to a
DQC setting:

- `SabreLayout`/`SabreSwap` gain an `extended_set_length` argument (paper:
  they use the Qiskit default `|E|=20` but also test `|E|=100` for a
  deeper lookahead window), instead of the hardcoded `20`.
- SABRE's layout search normally seeds itself with several random initial
  layouts to pick the best from. When you already have a fixed, externally
  constrained core assignment (as most DQC workflows do), that's wasted
  work — so heuristic random layouts are now only added when
  `num_random_trials > 1`:

  ```rust
  if num_random_trials > 1 {
      add_heuristic_layouts(&mut starting_layouts, problem, allow_parallel);
  }
  ```

- `PyRoutingTarget` (`route.rs`) gains a `set_distance_matrix` method,
  letting a distance matrix be swapped onto an already-constructed
  routing target directly from Python.

The paper calls the variant that uses *only* this infrastructure — no
custom distance weights, no cost-function changes — "Default SABRE": *"all
evaluated SABRE variants are DQC-adapted implementations... they leverage
our conjoined coupling map and modified layout selection criteria...
'Default SABRE' simply denotes the variant that utilizes these two
underlying DQC adaptations but otherwise retains standard SABRE routing
logic."*

### (1,10) SABRE — distance-matrix customization (paper §IV-B, Table I)

Rather than let SABRE's cost function derive distances purely from the
conjoined coupling map's native (all-edges-equal) topology, the distance
matrix itself can now be overridden before a routing trial runs, letting
you assign different costs to intra- vs. inter-core edges:

```rust
// crates/transpiler/src/passes/sabre/layout.rs
let mut target = RoutingTarget::from_neighbors(neighbors);
if let Some(custom_mat) = &parsed_custom_matrix {
    if target.num_qubits() == custom_mat.nrows() {
        target.distance = custom_mat.clone();
    }
}
```

exposed on the Python side as `SabreLayout`/`SabreSwap`'s
`custom_distance_matrix` argument. The paper's own example, and the
variant it names **(1,10) SABRE**, assigns intra-QPU edges weight 10 and
inter-QPU edges weight 1 — "assigning a higher weight to the 'cheaper'
local edges seems counter-intuitive, but it fundamentally exploits
SABRE's greedy logic": because a local SWAP now looks numerically
*expensive*, SABRE prefers to exhaust local routing pathways before
resorting to a remote link, without needing any change to the cost
function itself — the same greedy heuristic, fed different numbers.

### CLA-SABRE — Custom Lookahead SABRE (paper §IV-C, Eq. 3–5)

The most involved variant actively modifies the lookahead heuristic to
recognize and reward multi-gate-teleportation-like structure, rather than
just reshaping the distance matrix. Two pieces, both gated on a swap being
in the `penalized_swaps` set (canonicalized, i.e. order-independent):

**An upfront penalty, independent of QPU topology**, applied in the basic
(front-layer) term — the paper's *"upfront penalty α scaled by the
inverse of the front layer size"* (Eq. 4's `α/|F|` term):

```rust
// crates/transpiler/src/passes/sabre/route.rs — basic-term scoring
if self.penalized_swaps.contains(&sorted_swap) {
    let penalty_factor = weight * self.alpha;
    s += penalty_factor;
}
```

(`weight` here is the basic heuristic's own weight, which becomes `1/|F|`
when the pass is configured with `SetScaling::Size` rather than
`SetScaling::Constant` — that's what makes this literally `α/|F|`.)

**A reward/penalty term based on inter-core distance**, added in the
lookahead term via `mqpu_lookahead_score` — this is the code-level version
of Eq. 5's `φ(p, p̄, k_gp)`:

```rust
// crates/transpiler/src/passes/sabre/layer.rs
// Original gate [a, other] is on QPUs [qpu_a, qpu_other]; after the swap
// it becomes [b, other] on QPUs [qpu_b, qpu_other].
let original_distance = dist_qpus[qpu_a][qpu_other];
let new_distance = dist_qpus[qpu_b][qpu_other];
let delta_distance = new_distance - original_distance;

if new_distance == 0.0 && original_distance != 0.0 {
    total -= alpha / 3.0;                                   // fully localized: reward
} else {
    total -= (-delta_distance * 1.0 / beta) * (alpha / 3.0); // partial move: scaled by β
}
```

`dist_qpus` is `inter_qpu_distance_matrix` — computed on the Python side
from `inter_qpu_coupling_map` via `networkx.floyd_warshall_numpy` — and
`alpha`/`beta` are the paper's tunable hyperparameters, empirically set to
9.0 and 3.0.

Finally, the two penalty modes are switched between depending on whether a
real multi-QPU distance matrix is available:

```rust
let is_real_mqpu = !self.inter_qpu_distance_matrix.is_empty() && self.beta > 0.0;
if self.penalized_swaps.contains(&sorted_swap) {
    if is_real_mqpu {
        s += self.extended_set.mqpu_lookahead_score(*swap, &self.qubit_qpu_map, &self.inter_qpu_distance_matrix, self.alpha, self.beta);
    } else {
        // Used when driving SABRE from pytket-dqc's virtual-sink model,
        // where there's no real multi-QPU distance matrix to score against.
        s += weight * self.alpha;
    }
}
```

The `is_real_mqpu == false` branch is what lets the same modified SABRE
also serve as the underlying router pytket-dqc's subcircuit routing calls
into (§V-A) — a simpler flat-penalty "virtual sink" mode, distinct from
CLA-SABRE's full distance-aware scoring used on the Qiskit side.

## pytket-dqc: superconducting hardware adaptation (paper §V-A)

`patches/pytket-dqc/0001-superconducting-topology-awareness.patch`, base
pytket-dqc `bfa0b4e` (tip of `origin/main`).

The paper is explicit that pytket-dqc *"assumes all-to-all intra-core
connectivity, a model invalid for near-term superconducting hardware,"*
and that adapting it required *"major modifications to its tracking of
remote physical link utilization and subcircuit generation logic"* —
specifically, rewriting the distributed circuit generation logic *"to
strictly track physical link usage"* instead of link-qubit usage in the
abstract. That's exactly what this patch does:

Upstream `NISQNetwork` bounds communication capacity per *server*
(`server_ebit_mem`) — how many ebits a module can hold, with no regard for
which other module they're shared with. Superconducting hardware instead
has a fixed, dedicated physical link (and physical link qubit) per pair of
connected modules, so the meaningful constraint is per-*link*:

```python
server_link_capacities: Optional[dict[tuple[int, int], int]] = None
# ...
self.server_link_capacities = server_link_capacities or {}
```

`Distribution.to_pytket_circuit` (in `distribution.py`) enforces this
per-edge capacity when allocating link qubits, replacing the old
per-server accounting:

```python
edge_key = self._get_edge_key(server, peer_server)
current_usage = self.active_links.get(edge_key, 0)
capacity = canonical_capacities.get(edge_key, float("inf"))
if current_usage >= capacity:
    raise ConstraintException(
        f"Link capacity between {server} and {peer_server} exceeded "
        f"(Current: {current_usage}, Max: {capacity}).",
        server,
    )
```

## Also bundled in (not the headline contribution, but part of the same patch)

- **Steiner-tree and shortest-path memoization** in `Distribution`
  (`_get_steiner_tree`, `_get_tree_shortest_path`) — these recomputations
  are NP-hard and were previously redone on every call; caching them by a
  sorted-server key was a meaningful speedup on larger circuits.
- **A determinism fix**: candidate servers are now iterated in sorted
  order (`for c_server in sorted(connected_servers)`) rather than in
  whatever order a Python `set` happens to produce, so two runs on the
  same input no longer silently pick different (but equal-cost) routing
  paths.
- `Distribution.to_pytket_circuit` gained a `verify_equivalence: bool =
  False` flag. Upstream always ran a `check_equivalence` assertion against
  the original circuit; that's now opt-in, since it's expensive and was a
  significant fraction of total runtime on larger circuits during
  experiments. Worth turning back on (`verify_equivalence=True`) when
  correctness, not throughput, is what you're checking.

## QIG partitioning (paper §V-C): `qig-partitioning/`

Unlike the two items above, this isn't a patch — the QIG (Quantum
Interaction Graph) pre-processing step doesn't modify either fork. It
only needs a circuit, NumPy, and KaHyPar's Python bindings, so it ships as
its own small importable package, `qig-partitioning/` (import name
`qig_partitioning`), rather than being bundled into either patch or
re-derived by every consumer of this repo.

Where pytket-dqc builds a hypergraph over gate-level "packets," the QIG
is a single global graph: *"vertices represent virtual qubits, and edge
weights denote the total frequency of two-qubit gates between any pair
across the entire circuit."* `qig_partitioning.build_interaction_graph`
builds exactly that, and `partition_with_kahypar` partitions it into one
block per core using the same KaHyPar library pytket-dqc uses (bundling
its own copy of the `km1_kKaHyPar_sea20.ini` config so pytket-dqc doesn't
need to be installed just to get that file).

Because KaHyPar's own objective is topology-agnostic (it minimizes cut
size, implicitly assuming an all-to-all inter-core topology), two
post-processing steps adapt its output to the real, possibly
heterogeneous hardware — matching the paper's own description verbatim:

- **`match_partitions_to_cores`** — *"we evaluate all block-to-core
  permutations, selecting the assignment that minimizes actual
  communication costs based on the target hardware topology."*
- **`enforce_strict_capacity` + `boundary_reallocation`** — the paper's
  two-stage refinement: *"1) Strict Capacity Enforcement: if a core is
  over-allocated, we evaluate all assigned virtual qubits, iteratively
  ejecting the candidate that incurs the minimum communication penalty...
  and moving it to an under-allocated core... 2) Boundary Reallocation:
  we iteratively evaluate virtual qubits on partition boundaries, shifting
  them to neighboring under-allocated cores if the move strictly reduces
  global communication costs."*

`get_heterogeneous_core_assignment` wraps all of the above into one call,
returning a `{qubit: core}` mapping meant to seed a fixed initial layout
for one of the three SABRE variants (see `notebooks/usage_demo.ipynb`).

## Not part of these patches or this repo

The paper's evaluation also covers a **hybrid** approach that uses
pytket-dqc's existing `PartitioningHeterogeneous`/`CoverEmbedding`
allocators purely for initial mapping before handing off to CLA-SABRE
(§V-B). That's orchestration built on top of the pytket-dqc patch, using
pytket-dqc functionality that already exists upstream — it doesn't live
inside the patch, and isn't reproduced by this repo.
