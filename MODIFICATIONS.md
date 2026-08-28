# Key changes

This document walks through the hunks that carry the paper's contribution,
pulled out of the full patches in `patches/` and annotated against the
paper itself ([`paper.pdf`](paper.pdf) in this repo — *"Scalable
Transpilation for Overcoming Restricted Connectivity in Distributed
Superconducting Quantum Architectures,"* Azenha, Polian & Brandhofer). It's
meant to be read, not applied — see [`docs/INSTALL.md`](docs/INSTALL.md) for
how to apply the real patches to a working checkout, or
[`README.md`](README.md) for the Docker route.

Each section describes what its patch does, in the paper's own terms and with
the code that implements it. Two closing sections complete the map: **Where the
rest of the paper's method lives** covers the parts implemented in the
experiment scripts rather than in a fork, and **Limitations and deviations**
records the rough edges in the shipped code. Both exist so the mapping from
paper to code can be checked line by line — not because the contribution is
qualified.

## Contents

- [Qiskit: three DQC-aware SABRE variants (§IV)](#qiskit-three-dqc-aware-sabre-variants-paper-iv)
  - [Default SABRE — the DQC-adapted baseline](#default-sabre--the-dqc-adapted-baseline)
  - [(1,10) SABRE — distance-matrix customization (§IV-B)](#110-sabre--distance-matrix-customization-paper-iv-b)
  - [CLA-SABRE — Custom Lookahead SABRE (§IV-C)](#cla-sabre--custom-lookahead-sabre-paper-iv-c-eq-35)
- [pytket-dqc: superconducting hardware adaptation (§V-A)](#pytket-dqc-superconducting-hardware-adaptation-paper-v-a)
  - [`check_equivalence`'s `distributed_comparison` flag](#check_equivalences-distributed_comparison-flag)
  - [Supporting changes in the same patch](#supporting-changes-in-the-same-patch)
- [QIG partitioning (§V-C)](#qig-partitioning-paper-v-c-qig-partitioning)
- [Where the rest of the paper's method lives (`examples/`)](#where-the-rest-of-the-papers-method-lives-examples)
- [Limitations and deviations from the paper](#limitations-and-deviations-from-the-paper)

The results these changes produce are in [`docs/RESULTS.md`](docs/RESULTS.md);
the patches themselves are in [`patches/qiskit/`](patches/qiskit/) and
[`patches/pytket-dqc/`](patches/pytket-dqc/).

## Qiskit: three DQC-aware SABRE variants (paper §IV)

The paper introduces three named SABRE variants, each building on the
last. All three are implemented in
`patches/qiskit/0001-dqc-aware-sabre-variants.patch`, diffed against commit
`c62ce998` of a private development fork of Qiskit — the code as it stood when
the paper's results were produced. The patch is self-contained: it carries
everything needed to reproduce that state on top of the pinned upstream
commit, so the fork itself is not required.

### Default SABRE — the DQC-adapted baseline

The starting point (§IV-A) is the **conjoined coupling map**, a
straightforward extension of the usual coupling-map picture that gives SABRE
direct DQC support. Instead of routing each core separately, every core's
coupling graph is stitched together with extra edges representing the
inter-core links (Fig. 1 in the paper), and ordinary SABRE runs over the whole
thing. The map itself needs no patch — it is just a `CouplingMap` that happens
to span multiple cores.

**Default SABRE** is the variant that uses only the two DQC adaptations below,
with no custom distance weights and no cost-function changes: *"all evaluated
SABRE variants are DQC-adapted implementations... they leverage our conjoined
coupling map and modified layout selection criteria... 'Default SABRE' simply
denotes the variant that utilizes these two underlying DQC adaptations but
otherwise retains standard SABRE routing logic."*

**The two adaptations that define it:**

1. **The conjoined coupling map** (§IV-A), as above.
2. **Layout selection by aggregated cost** (§IV-B). SABRE evaluates several
   initial layouts in parallel and keeps one of them. Qiskit's stock rule is to
   keep whichever trial inserted the fewest SWAP gates; we instead keep the one
   that produced the lowest aggregated cost \(C_{agg}\) (Eq. 2), which prices
   an inter-core SWAP at three EPR pairs instead of counting it the same as a
   local one. This selection is **not part of the Qiskit patch** — it lives in
   the experiment scripts' routing loop; see
   [Aggregated-cost trial selection](#where-the-rest-of-the-papers-method-lives-examples)
   below.

**Supporting changes in the patch.** These are enabling knobs rather than
headline contributions — they exist so the variants above can be configured and
compared, and are not claimed to affect SABRE's scalability:

- `SabreLayout`/`SabreSwap` gain an `extended_set_length` argument in place of
  the hardcoded `20`, making the lookahead window selectable. We use the Qiskit
  default `|E|=20` throughout, and additionally test `|E|=100` for a deeper
  lookahead window.
- **Random layout seeding becomes optional.** Stock Qiskit always adds its
  heuristic random layouts to the trial set, whether or not an external initial
  layout was supplied — so an externally fixed core assignment would still have
  to compete against randomly generated ones. Gating them on
  `num_random_trials > 1` gives us the option to route from exactly the layout
  we provide, which is what the QIG and hypergraph pipelines need:

  ```rust
  if num_random_trials > 1 {
      add_heuristic_layouts(&mut starting_layouts, problem, allow_parallel);
  }
  ```

- `PyRoutingTarget` (`route.rs`) gains a `set_distance_matrix` method, letting
  a distance matrix be swapped onto an already-constructed routing target
  directly from Python.
- **`SabreLayout`'s decay term is disabled.** Its hardcoded heuristic loses the
  `.with_decay(0.001, 5)` stage upstream applies:

  ```python
  # qiskit/transpiler/passes/layout/sabre_layout.py
  .with_lookahead(0.5, self.extended_set_length, SetScaling.Size)
  # .with_decay(0.001, 5)
  ```

  Decay is SABRE's mechanism for discouraging repeated use of the same qubits.
  `SabreLayout` performs both mapping and routing internally, and applies decay
  unconditionally as a hardcoded assumption; since decay is not part of this
  work's routing model, it is disabled there explicitly. `SabreSwap` needs no
  equivalent change — there decay is opt-in through the `"decay"` heuristic,
  which these experiments do not select.

### (1,10) SABRE — distance-matrix customization (paper §IV-B)

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
CLA-SABRE's full distance-aware scoring used on the Qiskit side. In the
paper's terms, those "virtual sinks" are the **pseudo-sink qubits** of
Fig. 1 and §V-A, and this branch is the *"explicit heuristic penalty
against SWAPs involving pseudo-sinks"* described there. The *penalty* is the
half that lives in the fork; the machinery that creates the sinks is
orchestration, and lives in
[`examples/`](#where-the-rest-of-the-papers-method-lives-examples).

## pytket-dqc: superconducting hardware adaptation (paper §V-A)

`patches/pytket-dqc/0001-superconducting-topology-awareness.patch`, base
pytket-dqc `bfa0b4e` (tip of `origin/main`), diffed against commit `5a28ad6`
of a private development fork, restricted to `src/pytket_dqc/`.

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

That capacity check is the visible half. The mechanism that actually
implements the paper's *"locking the link until the multi-gate
teleportation primitive is fully resolved"* is a rewrite of how link
qubits are booked and released, and it is worth spelling out, because it
is where the per-link model differs from upstream in substance rather than
in bookkeeping.

**Link qubits are now booked against an edge, and the edge stays booked
until the primitive unwinds.** `_request_link_qubit` gains a
`peer_server` argument, records `qubit_peer_map[qubit] = peer_server`, and
increments `active_links[edge_key]`. `_release_link_qubit` looks the peer
back up and decrements the same counter. So occupancy is held on the
specific physical link for the whole lifetime of the link qubit, rather
than being a count of how many link qubits a module happens to hold.

**Ending processes now unwind hop by hop.** Upstream `end_links` ends
every target directly against the hyperedge's home link qubit, in whatever
order `targets` arrived in:

```python
# upstream
for target in targets:
    target_link = self.get_link_qubit(target)
    ending_actions.append(EjppAction(from_qubit=target_link, to_qubit=home_link))
```

The patched version ends each target against **its own peer** — the
neighbour it was entangled from — and sorts targets by their depth along
the entanglement-swapping chain, deepest first:

```python
valid_targets.sort(key=lambda t: (get_depth(t), t), reverse=True)
for target in valid_targets:
    peer_server = self.qubit_peer_map.get(target_link, <home server>)
    peer_link = self.get_link_qubit(peer_server)
    ending_actions.append(EjppAction(target_link, peer_link))
```

Under all-to-all intra-core connectivity the two are equivalent, since any
link qubit can talk to any other. Under the restricted connectivity the
paper targets, they are not: a multi-hop chain has to be torn down from
the far end inward, releasing each physical link only once the hop beyond
it is finished. Correspondingly, `start_link` now advances
`source = next_server` as it walks the path, so each hop's `peer_server`
is the previous hop rather than the original source.

### `check_equivalence`'s `distributed_comparison` flag

`src/pytket_dqc/utils/verification.py` gains a fourth parameter:

```python
def check_equivalence(
    circ1: Circuit, circ2: Circuit, qubit_mapping: dict[Qubit, Qubit],
    distributed_comparison: bool = False
) -> bool:
    ...
    if distributed_comparison:
        zx1 = to_pyzx(circ1, qubits2)   # circ1 masked with the *distributed* qubit set
        zx2 = to_pyzx(circ2, qubits2)
    else:
        zx1 = to_pyzx(circ1, qubits1)
        zx2 = to_pyzx(circ2, qubits2)
```

The default path assumes `circ1` is the *original, monolithic* circuit and
`circ2` its distributed counterpart, so it masks them with
`qubit_mapping.keys()` and `.values()` respectively. `to_pyzx` projects
anything outside the mask to |0>.

That assumption breaks when both arguments are distributed circuits — when
checking one distributed circuit against another, as the experiment scripts
do after rebuilding a distributed circuit from per-core subcircuits. There
`circ1` carries link qubits and EJPP start/end processes that are absent from
`qubit_mapping.keys()`, so they would fall outside the mask and be projected
to |0>, which `to_pyzx`'s own docstring notes is *not* equivalent to an EJPP
ending process. Setting `distributed_comparison=True` masks `circ1` with the
distributed-side qubit set instead, making the comparison meaningful.

This is what `examples/three_square_architecture/*_hypergraph.py` pass, and
the flag is required for those scripts to run.

### Supporting changes in the same patch

- **Steiner-tree and shortest-path memoization** in `Distribution`
  (`_get_steiner_tree`, `_get_tree_shortest_path`) — these recomputations
  are NP-hard and were previously redone on every call; caching them by a
  sorted-server key is a meaningful speedup on larger circuits.
  `_get_tree_shortest_path` returns `list(...)`, a copy, because
  `start_link` calls `.pop(0)` on the result and would otherwise corrupt
  the cached entry. Its key is `(id(tree), source, target)`, which is safe
  for the trees `_get_steiner_tree` produces (see
  [Limitations](#limitations-and-deviations-from-the-paper) for the one
  case where it isn't).
- **A determinism fix**: candidate servers are now iterated in sorted
  order (`for c_server in sorted(connected_servers)`) rather than in
  whatever order a Python `set` happens to produce, so two runs on the
  same input no longer silently pick different (but equal-cost) routing
  paths.
- `Distribution.to_pytket_circuit` gained a `verify_equivalence: bool =
  False` flag. Upstream always ran a `check_equivalence` assertion against
  the original circuit; that's now opt-in, since it's expensive and was a
  significant fraction of total runtime on larger circuits during
  experiments. Turn it back on (`verify_equivalence=True`) when
  correctness, not throughput, is what you're checking — the `all_cu1_local`
  assertion beside it stays unconditional either way.
- **`fast_get_server_id`**, an `lru_cache`d wrapper around
  `get_server_id`, plus `is_robust_start_proc`/`is_robust_end_proc` and
  `clean_circuit`. The two `is_robust_*` helpers identify EJPP start and
  end processes so they can be counted and stripped.
- **Debug instrumentation in three files**, containing *no* algorithmic
  change at all:
  - `allocators/hypergraph_partitioning.py` — five unconditional `print()`
    calls and `perf_counter` timings around `HypergraphCircuit` construction,
    `initial_distribute` and `make_valid`.
  - `distributors/partitioning_heterogeneous.py` — four prints timing the
    initial partitioning and the boundary reallocation separately.
  - `distributors/cover_embedding.py` — two prints timing the `VertexCover`
    refinement.

  These were added while developing the modifications, as a way to see which
  stages of the pytket-dqc pipeline dominated its runtime. The paper's reported
  runtimes are end-to-end figures taken at the end of the whole compilation
  chain, so no result depends on these prints; they survive here only because
  both patches reproduce the fork as it stood rather than a tidied version of
  it. See
  [Limitations](#limitations-and-deviations-from-the-paper).

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

## Where the rest of the paper's method lives (`examples/`)

Three pieces of the paper's method are orchestration rather than fork
changes, and so live in the experiment scripts under
[`examples/`](examples/) — they *call* the patched code rather than
modifying it. [`examples/README.md`](examples/README.md) maps each to the
function implementing it. This section completes the paper-to-code map.

**The pseudo-sink subcircuit-generation machinery (§V-A, Table I,
Fig. 1).** The paper describes *two* modifications to pytket-dqc. The
first — strict physical link tracking — is in the patch, above. The
second is not: *"the second modification introduces explicit subcircuit
generation for each core,"* using auxiliary **pseudo-sink qubits** and
**routing placeholders** (dummy two-qubit gates) to force virtual qubits
into the communication zone, with the hierarchical coupling-map edge
weights of Table I (`10^9` computational↔computational down to `10^0`
pseudo-sink↔link) applied before the distance matrix is computed. Nothing
in `patches/pytket-dqc/` mentions sinks, placeholders or those weights —
the string "sink" does not appear in the patch at all; the only patched
half is the SABRE-side *penalty*, the flat-`α` branch described earlier,
which penalises SWAPs involving sinks once something else has created them.
Sink insertion, placeholder insertion, edge-weight assignment and per-core
subcircuit generation live in the experiment orchestration, outside both
forks — specifically in `examples/mqpu_utils.py` (`RoutingPlaceholder`,
`MakePlaceholdersOpaque`, `create_subcircuit`,
`build_distributed_subcircuits`, `squash_placeholders_per_pair`,
`remove_routing_placeholders`), with Table I's
weights as the `weight_lookup` dict in each
`examples/three_square_architecture/*_hypergraph.py`. The Big-M sink mode is
selected by passing `alpha=100000.0, beta=0.0`, since `beta = 0.0` is what
makes `is_real_mqpu` false.

**Aggregated-cost trial selection (§IV-B, Eq. 2).** The paper prioritizes
*"the lowest aggregated cost"* over *"the fewest SWAP gates inserted"* when
choosing among parallel SABRE trials. The fork does not: `swap_map` still
selects by raw SWAP count, unchanged from upstream —

```rust
// crates/transpiler/src/passes/sabre/route.rs — unmodified by this patch
.min_by_key(|(index, result)| (result.swap_count(), *index))
```

— and no aggregated-cost computation exists anywhere in its Rust core or
transpiler Python, at the pinned commit or at the fork's current HEAD.
Choosing by \(C_{agg} = 10 \times N_{EPR} + N_{local\,SWAP}\) is done
outside the fork, in each experiment script's routing loop:

```python
# examples/**/*.py
total_eprs = (routing_metrics["final_inter_swaps"] * 3) + routing_metrics["final_inter_czs"]
aggregated_cost = (total_eprs * f_weight) + intra_swaps
if aggregated_cost < best_cost:
    ...
```

Note the EPR accounting this implies: an inter-QPU SWAP costs three EPR pairs,
an inter-QPU CZ costs one.

**The §V-B hybrid orchestration.** The paper's hybrid approach uses
pytket-dqc's existing `PartitioningHeterogeneous`/`CoverEmbedding`
allocators purely for initial mapping before handing off to CLA-SABRE.
That's orchestration built on top of the pytket-dqc patch, using pytket-dqc
functionality that already exists upstream — it doesn't live inside the patch.
It lives in the three `examples/three_square_architecture/*_hybrid.py`
scripts, which run `PartitioningHeterogeneous` only far enough to obtain a
qubit-to-core mapping, convert that into a core-respecting Qiskit layout, and
inject it into CLA-SABRE.

The division is consistent: the patches provide the primitives the paper
needs — per-link capacity, a pluggable distance matrix, the CLA cost terms,
the sink penalty — and `examples/` composes them into the paper's evaluated
pipelines.

The one thing not reproduced anywhere in this repository is **DMapS**, the
external state-of-the-art transpiler used as the baseline in Table II (it also
appears as a row in Table III, but is not the baseline there); it is a
[third-party tool](https://github.com/RoccoLoter/DMapS).

## Limitations and deviations from the paper

Collected here rather than interleaved above, so each patch can be read for
what it does in one pass and audited in another.

None of these are fixed in place, deliberately. Both patches exist to reproduce
the exact state the paper's results were produced from; correcting a rough edge
here would mean shipping code that no longer matches the numbers in
[`docs/RESULTS.md`](docs/RESULTS.md). They are recorded instead. The scope
limits under "QIG partitioning" are a different kind of entry: assumptions the
method is built on, not defects in it.

### Qiskit patch

- **`SabreSwap(coupling_map=None)` now raises at construction.** Upstream
  deliberately sets `self.target = None` there, with the comment *"this is
  an invalid state, but we defer the error to runtime to match historical
  behaviour of Qiskit."* The patch moves routing-target construction into
  `__init__` and calls `RoutingTarget.from_target(self.target)`
  unconditionally, and that PyO3 signature takes a non-optional `&Target`
  — so the deferred error becomes an immediate `TypeError`. Read from the
  source rather than executed, since no built environment was to hand.
- **Two docstring inaccuracies.** The `alpha`/`beta` docstrings added to both
  passes describe `beta` as a *"fractional penalty per inter-QPU SWAP
  distance"*; it is actually a dampening divisor on the partial-move reward
  (`α/(3β)` per unit of distance change). `inter_qpu_coupling_map` is
  annotated `// Implementation pending`, though it is implemented directly
  below the annotation.
- **Inert debug scaffolding.** Both Rust files carry a large volume of
  commented-out `eprintln!` debug scaffolding, left in place for the same
  reproduce-the-original-state reason.
- **`SabreLayout`'s decay term is disabled**, as described under "Default
  SABRE" above. The paper does not discuss it, so it is best read as an
  undocumented experimental choice rather than a described contribution.
  `SabreSwap`'s `"decay"` heuristic is unaffected.
- **Aggregated-cost trial selection (Eq. 2) is not in the patch** — the fork
  still picks the best trial by raw SWAP count. See "Where the rest of the
  paper's method lives" above.

### QIG partitioning

- **Cores are assumed to hold the same number of qubits.** This is a scope
  limit rather than a rough edge: every architecture the paper evaluates has
  equally sized cores (3×16 and 3×133), so a single per-core capacity is all
  the evaluated configurations need. Two places encode it —
  `get_heterogeneous_core_assignment` takes one scalar `max_capacity` applied
  to every core, and `partition_with_kahypar` passes `epsilon=0.01` with unit
  node weights, which asks KaHyPar for blocks within 1% of equal size.
  Supporting cores of differing sizes means changing both together: a
  `{core: capacity}` mapping through `enforce_strict_capacity` and
  `boundary_reallocation`, and per-block target weights in the KaHyPar call.
  Changing only the first leaves the partitioner still aiming for balance, so
  the capacity pass would spend its effort undoing that.

  Note that "heterogeneous" in `get_heterogeneous_core_assignment` refers to
  the inter-core *topology* — `core_cost_matrix` need not be uniform, which is
  what makes the line-topology results possible — and not to core sizes.

### pytket-dqc patch

The first two follow from the per-server → per-link switch, and neither fails
loudly.

- **`server_ebit_mem` is no longer enforced anywhere in
  `to_pytket_circuit`.** The old per-server check
  (`server_ebit_mem[server] <= len(self.occupied[server])`) is gone, fully
  replaced by the per-edge check. Because unknown edges default to
  `float("inf")`, a network built the upstream way — `server_ebit_mem` set,
  `server_link_capacities` omitted — now runs with **no communication
  bound at all**, silently, rather than raising `ConstraintException`. If
  you are porting an upstream script, you must supply
  `server_link_capacities`; the field is still accepted, still stored, and
  still checked by `can_implement`, which makes the omission easy to miss.
- **`NISQNetwork` does not round-trip through `to_dict`/`from_dict`.**
  `to_dict` serialises the new field with stringified tuple keys, but
  `from_dict` never parses it back — it carries only a comment saying that
  parsing *"would be needed here"*. Since `__eq__` was also extended to
  compare `server_link_capacities`, `NISQNetwork.from_dict(n.to_dict())`
  both loses every capacity and compares unequal to `n` whenever any
  capacity is set.
- **`verify_equivalence` now defaults to `False`**, as noted under "Supporting
  changes" — the only one of those changes that weakens a correctness
  guarantee. Equivalence to the original circuit is not checked unless asked
  for.
- **Debug `print()` instrumentation ships in three files**, also noted under
  "Supporting changes". It prints to stdout from library code on every call,
  and is the one category of repo-local noise that survives into a shipped
  patch — it is those files' only content here. Stripping the debug-only hunks
  while keeping `verification.py`'s functional change would be straightforward
  if wanted, at the cost of no longer matching the state the paper's runtime
  breakdown came from.
- **The Steiner path cache keys on `id(tree)`.** That is safe for trees
  `_get_steiner_tree` produced, since `_steiner_cache` keeps them alive for the
  life of the `Distribution` and their ids stay unique. It is *not* safe for a
  `tree` supplied through the optional `tree` argument, which nothing retains:
  a garbage-collected tree's address can be reused and return another tree's
  cached path. Nothing in this repo exercises that path, but it is a real
  hazard if the caller-supplied `tree` argument is ever used in a loop.
- **`is_robust_start_proc`/`is_robust_end_proc` identify EJPP processes by
  substring-matching `str(op)`/`repr(op)`** inside a bare `except: pass`. If
  pytket ever changes how these custom gates render, they answer `False` rather
  than failing — worth knowing if start/end process accounting ever looks
  wrong.
