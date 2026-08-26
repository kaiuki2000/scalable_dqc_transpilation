# scalable_dqc_transpilation

Companion artifact for *"Scalable Transpilation for Overcoming Restricted
Connectivity in Distributed Superconducting Quantum Architectures"*
(Afonso Azenha, Ilia Polian, Sebastian Brandhofer — QCE26 submission). Full
draft: [`Paper_draft_portrait.pdf`](Paper_draft_portrait.pdf) (this repo).
TODO: arXiv / DOI link once published.

The paper tackles qubit mapping and routing for Distributed Quantum
Computing (DQC) on near-term superconducting hardware, where inter-QPU
links are scarce, high-latency, and topologically restricted compared to
intra-QPU couplings. It ships three things:

- Three **DQC-aware SABRE variants** in Qiskit (Default SABRE, (1,10)
  SABRE, and CLA-SABRE), which route with explicit awareness of which
  SWAPs cross a QPU boundary — as a patch against a pinned upstream commit.
- A **superconducting-hardware-aware distribution model** in pytket-dqc,
  used as one of the paper's baselines/building blocks, adapted to track
  per-link (not per-server) communication capacity — also as a patch.
- **QIG partitioning** (`qig-partitioning/`), the pre-processing step that
  produces an initial per-core qubit assignment for the SABRE variants
  above — a small, standalone Python package, since it doesn't modify
  either fork.

## What's actually different

- **Default SABRE** — the DQC-adapted baseline all other variants build
  on: SABRE routes over a *conjoined coupling map* (every core's coupling
  graph stitched together with edges for the inter-core links), with fixes
  so that scales (skipping unneeded random-layout search when the core
  assignment is already fixed) and a tunable lookahead window
  (`extended_set_length`).
- **(1,10) SABRE** — the conjoined map's distance matrix can be
  overridden (`custom_distance_matrix`) so intra- and inter-core edges
  carry different costs — e.g. the paper's namesake scheme, weight 10 for
  local edges and 1 for remote ones, which (counter-intuitively) makes
  SABRE's greedy logic prefer to exhaust local routing before crossing a
  QPU boundary.
- **CLA-SABRE** (Custom Lookahead SABRE) — actively rewrites SABRE's cost
  function: an upfront penalty for any SWAP marked as inter-QPU, plus a
  lookahead-term reward/penalty based on whether that SWAP moves a gate's
  qubits onto the same QPU or further apart.
- **pytket-dqc's placement/distribution model** gets per-link (rather than
  per-server) communication capacity, matching how real superconducting
  hardware connects separate modules with links of differing capacity —
  the paper's adaptation of pytket-dqc for restricted (non-all-to-all)
  intra-core connectivity.
- **QIG partitioning** — a lightweight alternative to pytket-dqc's
  hypergraph model: one global interaction graph over the whole circuit's
  virtual qubits, partitioned per-core with KaHyPar, then adapted to real
  hardware capacity limits by matching abstract blocks to physical cores
  and reallocating boundary qubits — see `qig-partitioning/`.

For the actual diffs, hand-picked, annotated, and mapped to the paper's
equations and section numbers: see [`MODIFICATIONS.md`](MODIFICATIONS.md).

### Headline results (from the paper; see Tables II–III for full data)

Against a DMapS baseline on a 48-qubit (3×16, all-to-all) architecture:
up to 24.72% (unstructured) / 24.01% (structured) aggregated cost
reduction, with >250× runtime speed-ups on structured workloads where
DMapS scales poorly. On a 399-qubit IBM Flamingo architecture (3×133,
where DMapS and pytket-dqc largely time out), CLA-SABRE reduces structured
circuit routing cost by up to 51.94% (all-to-all) / 51.96% (restricted
line topology) against the DQC-adapted "Default SABRE" baseline, while
remaining practical to run.

TODO: pull a standalone results figure out of the paper for a quick visual
(see Figs. 2–3 in the PDF for now).

## Repository layout

```
Paper_draft_portrait.pdf      # full paper draft
MODIFICATIONS.md              # curated, annotated walkthrough of the key changes
notebooks/
  usage_demo.ipynb            # illustrative usage example (draft, partially verified)
patches/
  qiskit/
    0001-dqc-aware-sabre-variants.patch
    BASE_COMMIT.txt
  pytket-dqc/
    0001-superconducting-topology-awareness.patch
    basic_usage_demo.ipynb
    BASE_COMMIT.txt
qig-partitioning/              # standalone package: QIG partitioning pre-processing (paper §V-C)
  pyproject.toml
  qig_partitioning/
    partitioning.py
    config/km1_kKaHyPar_sea20.ini
env/
  requirements.sh                  # extra packages on top of Qiskit's/pytket-dqc's own deps
  requirements-freeze.txt          # pip freeze of the combined environment (both forks) used
                                    # for the paper's experiments
  requirements-list.txt
  requirements-freeze-slurm.txt    # pip freeze from a later, post-submission environment;
  requirements-list-slurm.txt      # reference only, not the one the results were produced on
docker/
  Dockerfile                       # builds both patches + qig-partitioning into one environment
```

## Reproducing the environment

This is the reproducibility layer: small patches against pinned upstream
commits, rather than full forks. Each `patches/<project>/BASE_COMMIT.txt`
records the exact pinned commit and reproduction steps.

Two ways to set this up: `docker/Dockerfile` (self-contained, and the one
that's actually been build-tested end to end — see "Docker" at the bottom of
this section), or the manual, step-by-step walkthrough below, for anyone who'd
rather set it up directly on their own machine. The manual steps are the same
sequence `docker/Dockerfile` runs, translated out of a container — **order
matters** here (each step notes why), so don't skip around. Where a step
needs more justification than fits here, the matching comment in
`docker/Dockerfile` goes deeper.

Everything below assumes one working directory holding a single Python 3.12
virtual environment and all the checkouts side by side (`~/dqc-workspace`
here — pick your own), matching the layout `docker/Dockerfile` uses under
`/opt`. `/path/to/scalable_dqc_transpilation` below means wherever you've
cloned *this* repo (not the working directory).

### 0. Prerequisites

System packages (Debian/Ubuntu names shown — see the linked build
instructions in step 1 for other distros/macOS):

- **git**, **build-essential** (a C/C++ toolchain), **cmake**, **curl**.
- **Graphviz**, including its development headers (`libgraphviz-dev` /
  `graphviz-devel`) — required by `pygraphviz`, one of pytket-dqc's own deps.
- **Boost.Program_options** (`libboost-program-options-dev` / `boost-devel` +
  `boost-program-options-devel`) — required to build KaHyPar.
- **Python 3.12** — matches the environment `env/requirements-freeze-slurm.txt`
  below was frozen from; other 3.10+ interpreters likely work too, but aren't
  the one this sequence has actually been verified against.
- **Rust**, via [rustup](https://rustup.rs/) — Qiskit's routing/layout core
  (what the SABRE patch touches) is written in Rust.

```bash
mkdir ~/dqc-workspace && cd ~/dqc-workspace
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
```

### 1. Build KaHyPar from source

pytket-dqc needs KaHyPar's Python interface, and only versions ≥1.3.5 are
available as a prebuilt PyPI wheel — this pytket-dqc version's partitioning
code is incompatible with that wheel (`Vertices with weight 0 are not
supported`, confirmed by running pytket-dqc's own unmodified test suite, not
just this patch). Building `1.3.2` from source is required, not optional.
[pytket-dqc's own build instructions](https://github.com/Quantinuum/pytket-dqc#kahypar-with-python-interface)
cover the base recipe; one addition below is needed on top of it for a
modern Python:

```bash
git clone --filter=blob:none https://github.com/kahypar/kahypar.git kahypar-src
cd kahypar-src
git checkout efa12a90acae05469520d43cff122ca8620fab48   # pinned commit, tag 1.3.2
git submodule update --init --recursive

# Swap the vendored pybind11 (pinned at commit a6355b00, ~v2.4.dev4, 2019) for
# a version that supports Python >=3.11. CPython 3.11 made PyFrameObject's
# internals opaque; that old pybind11 dereferences them directly and
# unconditionally, so the build fails under ANY Python >=3.11 with "invalid
# use of incomplete type 'PyFrameObject'" (verified empirically, not just
# inferred). This is not a Python-version-vs-KaHyPar compatibility ceiling --
# it's fixable, and this is how. Do NOT instead bump KaHyPar itself past
# 1.3.2 to pick up a newer pybind11: 1.3.5+ is exactly where the incompatible-
# PyPI-wheel regression above lives, so that would trade one problem for the
# other. This swaps only the binding layer, keeping KaHyPar's own C++ at the
# version pytket-dqc needs.
rm -rf python/pybind11
git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git python/pybind11
# v2.13.6 specifically: last of the pybind11 2.x line, supports Python
# 3.7-3.13, and keeps the CMake behavior KaHyPar 1.3.2's own
# python/CMakeLists.txt expects (pybind11 3.x changed it).

mkdir build && cd build
# If your system CMake is >=4 (Ubuntu 24.04+, current Homebrew), add
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to the line below: the vendored
# googletest submodule declares cmake_minimum_required(VERSION 2.6.2), and
# CMake 4 refuses that outright (not just a warning), in an error pointing at
# googletest that has nothing to do with anything you're actually building.
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1
cd python && make -j"$(nproc)"
cp kahypar*.so "$(python3 -c 'import site; print(site.getsitepackages()[0])')"
cd ~/dqc-workspace && rm -rf kahypar-src
```

### 2. Bulk-install the frozen environment

`env/requirements-freeze-slurm.txt` is a `pip freeze` of a later,
post-submission environment (`env/requirements-freeze.txt` is the one that
actually produced the paper's results). Used here because it's the one that
includes `pytket-qiskit`, needed for the Section V-B hybrid approach. This is
a real install (`pip install -r`), not just version pinning:

```bash
grep -v -e '^-e ' -e '@ file://' \
    /path/to/scalable_dqc_transpilation/env/requirements-freeze-slurm.txt > /tmp/frozen.txt
pip install -r /tmp/frozen.txt
```

The excluded lines are qiskit's own and pytket-dqc's own local checkouts —
both get built from the pinned patch commits below instead, not from
whatever arbitrary state that environment happened to record. This step also
pulls in an ordinary, unpatched qiskit release (needed to satisfy
`pytket-qiskit==0.73.0`'s own `qiskit>=2.2.3` requirement, checked against
PyPI metadata — our patched fork's `2.2.0rc1` fails that as a plain version
comparison). That's expected, and gets replaced in the next step: pip's
explicit "install this exact local package" (step 3's `pip install -e .`)
unconditionally replaces whatever's currently installed under that name, so
doing the bulk install *first* and the patched qiskit *last* is what makes
this safe rather than a race. Nothing installed after qiskit in this sequence
(pytket-dqc, qig-partitioning) declares a `qiskit` or `pytket-qiskit`
requirement of its own, so nothing later can undo this again — see
`docker/Dockerfile`'s comments on this exact step for the fuller reasoning.

### 3. Build and patch Qiskit

```bash
cd ~/dqc-workspace
git clone https://github.com/Qiskit/qiskit.git qiskit-src   # not `qiskit` -- see below
cd qiskit-src
git checkout 848178940d2def0dbdee578d20ce5e6b3451f4c2
git apply /path/to/scalable_dqc_transpilation/patches/qiskit/0001-dqc-aware-sabre-variants.patch
pip install -e . -c /tmp/frozen.txt
# A plain `pip install -e .` builds the Rust core in *debug* mode for an
# editable install; a release build (needed for meaningful routing-time
# numbers) needs this explicit second step, per Qiskit's own CONTRIBUTING.md:
python setup.py build_rust --release --inplace
cd ~/dqc-workspace
```

`qiskit-src`, not `qiskit`: if this checkout sits next to a directory you
later run Python from (a shared workspace root — exactly the layout this
repo's own Docker image used to have), a `qiskit` directory with no
`__init__.py` gets picked up by Python's `PathFinder` as an implicit
namespace package and silently wins over the real, editable-installed
`qiskit` — `import qiskit` appears to succeed but `qiskit.__file__` is `None`
and `from qiskit import QuantumCircuit` fails with a confusing
`AttributeError`. Caught by actually hitting it while building
`docker/Dockerfile`, not by inspection.

Qiskit's fork branched off `848178940` (tag `2.2.0rc1`), **not** the final
`2.2.0` tag (`2155673bc325026f85dbc3fafe9b7b0207eb615b`) — the patch will not
apply cleanly on top of `2.2.0`.

Verify the patched, release-built install actually took (the same two checks
`docker/Dockerfile` runs at build time):

```bash
python -c "
import qiskit
assert qiskit.__file__ is not None, 'qiskit imported as a namespace package -- the editable install is not active'
assert '/qiskit-src/' in qiskit.__file__, f'qiskit resolved to {qiskit.__file__}, not the patched checkout'
print('qiskit OK:', qiskit.__file__, qiskit.__version__)
"
cd qiskit-src \
    && git apply --reverse --check /path/to/scalable_dqc_transpilation/patches/qiskit/0001-dqc-aware-sabre-variants.patch \
    && echo "SABRE patch OK: applied in the tree the editable install points at" \
    && cd ~/dqc-workspace
```

The second check matters because the first only proves *location* — a stock
checkout at the right commit would pass it too. `git apply --reverse --check`
only succeeds if the patch is currently applied, which is the actual
question.

### 4. Build and patch pytket-dqc

pytket-dqc's base (`bfa0b4e`) is a clean pin at `origin/main`'s tip with no
upstream drift. Its own checkout doesn't need the `qiskit-src`-style rename:
its import name is `pytket_dqc`, underscored, which can't collide with a
`pytket-dqc` directory the way `qiskit` collides with `qiskit`.

```bash
git clone https://github.com/Quantinuum/pytket-dqc.git
cd pytket-dqc
git checkout bfa0b4eff5b77d0a9b3c260c7842e57e75091796
git apply /path/to/scalable_dqc_transpilation/patches/pytket-dqc/0001-superconducting-topology-awareness.patch
cp /path/to/scalable_dqc_transpilation/patches/pytket-dqc/basic_usage_demo.ipynb examples/basic_usage.ipynb
pip install -c /tmp/frozen.txt -e ".[tests]"

git apply --reverse --check /path/to/scalable_dqc_transpilation/patches/pytket-dqc/0001-superconducting-topology-awareness.patch \
    && echo "pytket-dqc patch OK: applied in the tree the editable install points at"
cd ~/dqc-workspace
```

**pytket-dqc: installs and imports correctly; distribution not verified
end-to-end.** `server_link_capacities` is present on `NISQNetwork` as
documented, but actually distributing a circuit (which exercises KaHyPar)
hasn't been checked against this exact manual sequence — see "Docker" below
for what *has* been verified end to end.

### 5. Install QIG partitioning

```bash
pip install --no-deps -e /path/to/scalable_dqc_transpilation/qig-partitioning
```

`--no-deps`: qig-partitioning's own `pyproject.toml` lists a plain `kahypar`
PyPI dependency, which would otherwise pull in a prebuilt wheel here and
silently overwrite the source-built KaHyPar 1.3.2 `.so` from step 1 with an
incompatible version. Its other declared deps (`qiskit`, `numpy`) are already
installed by this point.

### Verification status

Qiskit's SABRE patch was run for real, beyond the checks above: on a toy
2-QPU coupling map with one inter-QPU link, baseline `SabreSwap` crossed that
link once; with CLA-SABRE's
`penalized_swaps`/`qubit_qpu_map`/`inter_qpu_coupling_map` set, it crossed
zero times (at the cost of one extra local SWAP) — the intended effect. Both
patches apply cleanly (`git apply --check`) against a fresh worktree at their
pinned base commit, independent of any of the above.

**Docker: builds and runs end-to-end.** `docker/Dockerfile` runs this same
sequence and has been built successfully against a real Docker daemon,
including the KaHyPar-from-source step. It bakes in the same qiskit-location
and patch-still-applied checks shown above as part of the build itself, so a
broken build fails loudly there rather than shipping a silently-unpatched
image. See the comments at the top of that file for build/run instructions,
and throughout it for what each check catches and why. Still worth doing
yourself, either way: exercising a real KaHyPar partitioning call (e.g.
`qig_partitioning.partition_with_kahypar(...)`) — a successful `import
kahypar` alone doesn't rule out the broken-PyPI-wheel failure mode described
in step 1, since that only surfaces at partition time, and none of the
checks above call into KaHyPar itself.

## Using QIG partitioning

Unlike the two patches above, `qig-partitioning/` doesn't modify either
fork — it's ordinary installable Python, so it's just:

```bash
pip install -e /path/to/scalable_dqc_transpilation/qig-partitioning
```

(If you've also gone through "Reproducing the environment" above, this is
already installed — step 5 there uses `--no-deps` specifically to avoid
clobbering the from-source KaHyPar build with a PyPI wheel. The plain install
shown here is for using `qig-partitioning` entirely on its own, where letting
it pull its own `kahypar` wheel is the simpler, correct default.)

```python
from qig_partitioning import get_heterogeneous_core_assignment

core_cost_matrix = {0: {0: 0.0, 1: 1.0}, 1: {1: 0.0, 0: 1.0}}  # 2 cores, uniform cost
assignment = get_heterogeneous_core_assignment(qc, core_cost_matrix, max_capacity=64)
```

**Verified.** Tested end-to-end on a synthetic circuit with three
tightly-coupled qubit clusters and a hard per-core capacity: it correctly
placed each cluster on its own core and respected the capacity limit. See
`notebooks/usage_demo.ipynb` for the full pipeline — QIG partitioning
feeding a fixed initial layout into the three SABRE variants.

## Not part of these patches or this repo

The paper's evaluation also covers a hybrid approach that pairs
pytket-dqc's existing `PartitioningHeterogeneous`/`CoverEmbedding`
allocators with CLA-SABRE for initial mapping (§V-B), which needs
`pytket-qiskit` to bridge the two — "Reproducing the environment" above
installs it (step 2), since it's part of the frozen environment either way.
What's *not* here is the orchestration itself: the actual script/notebook
that calls pytket-dqc's allocators and CLA-SABRE together for that
experiment. That's built on top of functionality that already exists
upstream in both forks — it doesn't live inside either patch, and isn't
reproduced by this repo. See [`MODIFICATIONS.md`](MODIFICATIONS.md) for the
full picture of what is and isn't reproduced here.

## License

Apache License 2.0, see `LICENSE`. This matches the license of both upstream
projects the patches are derived from (Qiskit and pytket-dqc), which are
themselves Apache-2.0.
