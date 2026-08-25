# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a companion artifact for *"Scalable Transpilation for Overcoming
Restricted Connectivity in Distributed Superconducting Quantum
Architectures"* (Azenha, Polian & Brandhofer — QCE26 submission; full draft
in `Paper_draft_portrait.pdf`). It does **not** contain a working codebase
to build/lint/test — it ships small, self-contained patches against pinned
upstream commits of two forked projects (Qiskit and pytket-dqc), so the
changes under study can be reviewed in isolation without vendoring the full
forks.

The paper introduces three DQC-aware SABRE routing variants in Qiskit
(**Default SABRE**, **(1,10) SABRE**, **CLA-SABRE**) and a
superconducting-hardware-aware adaptation of pytket-dqc's distribution
model. When describing or modifying this repo's content, use those names —
they're the paper's own terminology, not something invented for this repo.
See `MODIFICATIONS.md` for the mapping from paper section/equation to code.

There is no build system, linter, or test suite in this repo itself. The patched
code only builds/runs once applied to a checkout of the relevant upstream project
(see "Applying the patches" below); any build/lint/test workflow belongs to that
upstream project, not to this repo.

## Layout

```
patches/
  qiskit/
    0001-dqc-aware-sabre-variants.patch
    BASE_COMMIT.txt
  pytket-dqc/
    0001-superconducting-topology-awareness.patch
    basic_usage_demo.ipynb
    BASE_COMMIT.txt
env/
  qiskit/
    requirements.sh            # packages installed on top of Qiskit's own deps
    requirements-freeze.txt    # pip freeze of the environment used for experiments
    requirements-list.txt
```

Each `patches/<project>/BASE_COMMIT.txt` records the exact pinned upstream commit
the patch was generated against and the reproduction steps.

## What each patch does

### `patches/qiskit/0001-dqc-aware-sabre-variants.patch`

Base: Qiskit `848178940d2def0dbdee578d20ce5e6b3451f4c2` (tag `2.2.0rc1` — **not**
the same commit as the final `2.2.0` release tag; the patch will not apply cleanly
on top of `2.2.0`), diffed against
[`c62ce9982a7bf7764072ea89296077db4583b679`](https://github.com/kaiuki2000/qiskit/tree/c62ce9982a7bf7764072ea89296077db4583b679)
on the fork's `exponential_decay` branch — the state of the code when this
paper's results were produced. That branch continues past this commit with
further, post-paper work; see `patches/qiskit/BASE_COMMIT.txt`.

Touches SABRE's layout and routing passes: the Rust core
(`crates/transpiler/src/passes/sabre/{layer,layout,route}.rs`,
`crates/transpiler/src/transpiler.rs`), the C extension entry points in
`crates/cext/`, and the Python-facing `SabreLayout`/`SabreSwap` classes.
Implements the paper's three DQC-aware SABRE variants (§IV):

- **Default SABRE** — the DQC-adapted baseline: infrastructure to route
  over a conjoined (multi-core) coupling map at scale — a tunable
  `extended_set_length`, and skipping random-layout search when the core
  assignment is already fixed (`num_random_trials > 1` gate).
- **(1,10) SABRE** — a pluggable `custom_distance_matrix` for
  `SabreLayout`/routing, letting intra- vs. inter-core edges carry
  different costs instead of assuming a homogeneous coupling map.
- **CLA-SABRE** (Custom Lookahead SABRE) — explicit, configurable-weight
  penalization of SWAPs marked inter-QPU (`penalized_swaps`), in both the
  basic (front-layer) term (a flat `alpha` penalty) and the lookahead term
  (`mqpu_lookahead_score`, which rewards a swap for making a gate fully
  local to one QPU and penalizes it proportionally to inter-QPU distance
  otherwise; hyperparameters `alpha`, `beta`, defaults 9.0, 3.0).

### `patches/pytket-dqc/0001-superconducting-topology-awareness.patch`

Base: pytket-dqc `bfa0b4eff5b77d0a9b3c260c7842e57e75091796` (tip of `origin/main`
at the time it was pinned).

Adds superconducting-hardware-specific topology awareness to the
distribution/placement logic
(`src/pytket_dqc/allocators/hypergraph_partitioning.py`,
`src/pytket_dqc/circuits/distribution.py`,
`src/pytket_dqc/networks/nisq_network.py`,
`src/pytket_dqc/utils/circuit_analysis.py`). Also bundled in, since it was part of
the same working state used to produce the paper's results:

- Memoization of Steiner-tree and in-tree shortest-path computations in
  `Distribution` (`_get_steiner_tree`, `_get_tree_shortest_path`).
- A determinism fix: connected-server candidates are iterated in sorted order
  when picking the shortest connection path, removing a dependency on Python set
  iteration order.

`patches/pytket-dqc/basic_usage_demo.ipynb` is `examples/basic_usage.ipynb` with
cell outputs stripped and one demonstration cell added
(`CoverEmbeddingSteinerDetached`), shipped as a full file rather than a diff
because notebook JSON diffs are not reliably hand-applicable.

Deliberately excluded from this patch: a CI-only change (removal of the fork's own
GitHub Pages docs-deploy workflow) with no bearing on the algorithm.

### Not part of these patches

The paper's evaluation also covers a Quantum Interaction Graph (QIG)
partitioning scheme and a hybrid approach pairing pytket-dqc's existing
allocators (`PartitioningHeterogeneous`, `CoverEmbedding`) with CLA-SABRE
for initial mapping (§V-B, §V-C). Both are orchestration on top of what's
in these two patches, using pytket-dqc functionality that already exists
upstream — neither lives inside either patch, and neither is reproduced by
this repo.

## Applying the patches

```bash
# Qiskit
git clone https://github.com/Qiskit/qiskit.git
cd qiskit
git checkout 848178940d2def0dbdee578d20ce5e6b3451f4c2
git apply /path/to/scalable_dqc_transpilation/patches/qiskit/0001-dqc-aware-sabre-variants.patch
# build as usual (maturin develop / pip install -e ., see env/qiskit/requirements.sh)

# pytket-dqc
git clone https://github.com/Quantinuum/pytket-dqc.git
cd pytket-dqc
git checkout bfa0b4eff5b77d0a9b3c260c7842e57e75091796
git apply /path/to/scalable_dqc_transpilation/patches/pytket-dqc/0001-superconducting-topology-awareness.patch
cp /path/to/scalable_dqc_transpilation/patches/pytket-dqc/basic_usage_demo.ipynb examples/basic_usage.ipynb
```

Both patches have been verified to apply cleanly (`git apply --check`) against a
fresh worktree at their pinned base commit.

## Working on this repo

- When either patch needs to change, regenerate it from the corresponding fork
  rather than hand-editing the `.patch` file: diff the fork's pinned base commit
  against the relevant files at the fork's current HEAD, and re-verify with
  `git apply --check` against a fresh checkout/worktree of the pinned base before
  committing the updated patch.
- Keep patches scoped to the algorithmic contribution. Repo-local noise from the
  forks (personal env/tooling files, CI workflow edits, notebook execution
  outputs) should stay out of the shipped patches, matching the exclusions above.
- The full forks this repo's patches are derived from live outside this
  repository (not part of this working tree); do not assume their source files
  are available locally when reasoning about this repo in isolation.
