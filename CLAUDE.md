# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a companion artifact for *"Scalable Transpilation for Overcoming
Restricted Connectivity in Distributed Superconducting Quantum
Architectures"* (Azenha, Polian & Brandhofer — QCE26 submission; full draft
in `Paper_draft_portrait.pdf`). It ships small, self-contained patches
against pinned upstream commits of two forked projects (Qiskit and
pytket-dqc), so the changes under study can be reviewed in isolation
without vendoring the full forks, plus one standalone Python package
(`qig-partitioning/`) for the one piece of the paper's method that doesn't
modify either fork.

The paper introduces three DQC-aware SABRE routing variants in Qiskit
(**Default SABRE**, **(1,10) SABRE**, **CLA-SABRE**), a
superconducting-hardware-aware adaptation of pytket-dqc's distribution
model, and a QIG (Quantum Interaction Graph) partitioning pre-processing
step. When describing or modifying this repo's content, use those names —
they're the paper's own terminology, not something invented for this repo.
See `MODIFICATIONS.md` for the mapping from paper section/equation to code.

There is no build system, linter, or test suite for the two patches
themselves — that patched code only builds/runs once applied to a checkout
of the relevant upstream project (see "Applying the patches" below); any
build/lint/test workflow belongs to that upstream project, not to this
repo. `qig-partitioning/` is the exception: it's ordinary, self-contained
Python with its own `pyproject.toml` (see "QIG partitioning" below).

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
qig-partitioning/               # standalone package, no upstream to diff against
  pyproject.toml
  qig_partitioning/
    partitioning.py
    config/km1_kKaHyPar_sea20.ini
env/
  requirements.sh                  # extra packages on top of Qiskit's/pytket-dqc's own deps
  requirements-freeze.txt          # pip freeze of the combined environment (both forks) used
                                    # for the paper's experiments
  requirements-list.txt
  requirements-freeze-slurm.txt    # pip freeze from a later, post-submission environment; not
  requirements-list-slurm.txt      # the one the paper's results were produced on, but this pair
                                    # is what "Reproducing the environment" in README.md and
                                    # docker/Dockerfile actually build from now, since it's the
                                    # one that includes pytket-qiskit (see README's env/
                                    # description) -- requirements-freeze.txt above remains the
                                    # historical record of the paper's own results environment
docker/
  Dockerfile                       # builds both patches + qig-partitioning into one environment
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

### `qig-partitioning/` (package, not a patch)

Package name `qig-partitioning` (note the hyphen — it's deliberate, see
below), import name `qig_partitioning`. Implements the paper's QIG
pre-processing step (§V-C): one global interaction graph over a circuit's
virtual qubits, partitioned into per-core blocks with KaHyPar, then
adapted to real hardware capacity via the paper's own two-stage
refinement (`enforce_strict_capacity`, `boundary_reallocation`). Only
depends on `qiskit`, `numpy`, and `kahypar` — not on either patched fork —
so it's ordinary installable Python (`pip install -e qig-partitioning/`),
not a patch. Bundles a copy of pytket-dqc's `km1_kKaHyPar_sea20.ini`
KaHyPar config (Apache-2.0, attributed in the file) as a default, so
pytket-dqc doesn't need to be installed just to get it.

The outer directory is named `qig-partitioning` (hyphen) while the
importable package is `qig_partitioning` (underscore) *on purpose*: naming
both identically causes Python to resolve `import qig_partitioning` to an
empty implicit namespace package when the current working directory is
the repo root (`''` in `sys.path` shadows the real, `pip install -e`-registered
package) — this was caught by actually testing `import qig_partitioning`
from the repo root, not by inspection. Don't rename the outer directory to
match the package name; it will silently reintroduce this bug.

### Not part of these patches or this repo

The paper's evaluation also covers a hybrid approach that pairs pytket-dqc's
existing allocators (`PartitioningHeterogeneous`, `CoverEmbedding`) with
CLA-SABRE for initial mapping (§V-B), which needs `pytket-qiskit` to bridge
the two — the environment setup below installs it as part of the frozen
environment either way. What's *not* here is the orchestration itself: the
actual script/notebook calling pytket-dqc's allocators and CLA-SABRE
together for that experiment. That's built on functionality that already
exists upstream in both forks — it doesn't live inside either patch, and
isn't reproduced by this repo.

## Applying the patches

Base commits and patch paths:

- Qiskit: base `848178940d2def0dbdee578d20ce5e6b3451f4c2` (tag `2.2.0rc1`,
  **not** the final `2.2.0` tag — the patch won't apply cleanly there),
  patch `patches/qiskit/0001-dqc-aware-sabre-variants.patch`.
- pytket-dqc: base `bfa0b4eff5b77d0a9b3c260c7842e57e75091796` (clean pin at
  `origin/main`'s tip, no upstream drift), patch
  `patches/pytket-dqc/0001-superconducting-topology-awareness.patch`, plus
  `patches/pytket-dqc/basic_usage_demo.ipynb` copied over
  `examples/basic_usage.ipynb`.

Both have been verified to apply cleanly (`git apply --check`) against a
fresh worktree at their pinned base commit — but applying the patch is only
one step of a longer, order-dependent sequence (build KaHyPar from source
first, including a required pybind11 swap for Python ≥3.11; bulk-install a
frozen environment; *then* build+patch Qiskit; *then* pytket-dqc — each step
exists to avoid a specific, previously-hit failure, not arbitrary ordering).
Don't reconstruct that sequence from memory or from an older version of this
file: `README.md`'s "Reproducing the environment" section is the up-to-date,
step-by-step version of it, verified against a real build; `docker/Dockerfile`
runs the same sequence in a container and has been build-tested end to end.
If you're setting this up (or advising someone through it), follow one of
those two, not a reconstruction from what's summarized here.

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
