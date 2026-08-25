# scalable_dqc_transpilation

Companion artifact for *"Scalable Transpilation for Overcoming Restricted
Connectivity in Distributed Superconducting Quantum Architectures"*
(Afonso Azenha, Ilia Polian, Sebastian Brandhofer — QCE26 submission). Full
draft: [`Paper_draft_portrait.pdf`](Paper_draft_portrait.pdf) (this repo).
TODO: arXiv / DOI link once published.

The paper tackles qubit mapping and routing for Distributed Quantum
Computing (DQC) on near-term superconducting hardware, where inter-QPU
links are scarce, high-latency, and topologically restricted compared to
intra-QPU couplings. It ships two things, as small patches against pinned
upstream commits rather than full forks:

- Three **DQC-aware SABRE variants** in Qiskit (Default SABRE, (1,10)
  SABRE, and CLA-SABRE), which route with explicit awareness of which
  SWAPs cross a QPU boundary.
- A **superconducting-hardware-aware distribution model** in pytket-dqc,
  used as one of the paper's baselines/building blocks, adapted to track
  per-link (not per-server) communication capacity.

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
env/
  qiskit/
    requirements.sh            # packages installed on top of Qiskit's own deps
    requirements-freeze.txt    # pip freeze of the environment used for experiments
    requirements-list.txt
```

## Reproducing the environment

This is the reproducibility layer: small patches against pinned upstream
commits, rather than full forks. Each `patches/<project>/BASE_COMMIT.txt`
records the exact pinned commit and reproduction steps.

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

Qiskit's fork branched off `848178940` (tag `2.2.0rc1`), **not** the final
`2.2.0` tag (`2155673bc325026f85dbc3fafe9b7b0207eb615b`) — the patch will not
apply cleanly on top of `2.2.0`. pytket-dqc's base (`bfa0b4e`) is a clean pin
at `origin/main`'s tip with no upstream drift.

### Verification status

Both patches apply cleanly (`git apply --check`) against a fresh worktree at
their pinned base commit.

**Qiskit: fully verified.** `pip install -e .` builds successfully (Rust
core via maturin). One thing not obvious from the patch alone: it adds a
hard `from networkx import ...` import (used to compute the inter-QPU
distance matrix) that isn't declared in Qiskit's own `pyproject.toml` —
install it explicitly (`env/qiskit/requirements.sh` already does this).
Ran an actual routing test: on a toy 2-QPU coupling map with one inter-QPU
link, baseline `SabreSwap` crossed that link once; with CLA-SABRE's
`penalized_swaps`/`qubit_qpu_map`/`inter_qpu_coupling_map` set, it crossed
zero times (at the cost of one extra local SWAP) — the intended effect.

**pytket-dqc: installs and imports correctly; distribution not verified
end-to-end.** `pip install -e .` succeeds and `server_link_capacities` is
present on `NISQNetwork` as documented. Actually distributing a circuit
requires KaHyPar, and only versions ≥1.3.5 are available as a prebuilt pip
wheel — this pytket-dqc version's partitioning code is incompatible with
that wheel (`Vertices with weight 0 are not supported`, confirmed by running
pytket-dqc's own unmodified test suite, not just this patch). A working
KaHyPar means building `1.3.2` from source instead.

#### pytket-dqc prerequisites

Beyond the Python dependencies `pip install -e .` handles, pytket-dqc needs
these installed system-wide:

- **CMake** — for building KaHyPar and `pygraphviz`.
- **Graphviz**, including its development headers (e.g. `graphviz-devel` /
  `libgraphviz-dev`) — required by `pygraphviz`.
- **Boost.Program_options** (e.g. `boost-devel` +
  `boost-program-options-devel` / `libboost-program-options-dev`) — required
  to build KaHyPar.
- **[KaHyPar 1.3.2](https://github.com/kahypar/kahypar/releases/tag/1.3.2)**,
  built from source and its Python interface `.so` copied into your
  venv's site-packages — see
  [pytket-dqc's own build instructions](https://github.com/Quantinuum/pytket-dqc#kahypar-with-python-interface)
  for the exact steps, which are more involved than a `pip install` and
  specific enough to your OS/compiler that we're not duplicating them here.

## Not part of these patches

The paper's evaluation also covers a Quantum Interaction Graph (QIG)
partitioning scheme and a hybrid approach that pairs pytket-dqc's existing
allocators with CLA-SABRE for initial mapping (§V). Both are orchestration
built on top of what's in these two patches, using pytket-dqc
functionality that already exists upstream — see
[`MODIFICATIONS.md`](MODIFICATIONS.md) for what is and isn't reproduced
here.

## License

MIT, see `LICENSE`.
