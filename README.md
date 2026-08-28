# Scalable transpilation for distributed superconducting quantum architectures

Companion artifact for *"Scalable Transpilation for Overcoming Restricted
Connectivity in Distributed Superconducting Quantum Architectures"* — Afonso
Azenha, Ilia Polian, Sebastian Brandhofer (QCE26 submission). Full draft:
[`paper.pdf`](paper.pdf); one-page overview:
[`poster-iqst.pdf`](poster-iqst.pdf).
*arXiv/DOI link to follow.*

Qubit mapping and routing for Distributed Quantum Computing (DQC) on near-term
superconducting hardware, where inter-QPU links are scarce and roughly an
order of magnitude noisier than intra-QPU couplings. Three contributions,
shipped as two patches against pinned commits of the *upstream* projects they
modify — the unmodified Qiskit and pytket-dqc releases — plus one standalone
package:

| | Contribution | Where |
| --- | --- | --- |
| **DQC-aware SABRE variants** | A fast routing heuristic for the restricted connectivity of distributed architectures — three variants that price a SWAP by whether it crosses a QPU boundary | [`patches/qiskit/`](patches/qiskit/) |
| **Superconducting-aware pytket-dqc distribution** | pytket-dqc's hypergraph mapping and distributed-circuit generation, re-modelled on per-link rather than per-server capacity, with per-core subcircuit routing as the subsequent step | [`patches/pytket-dqc/`](patches/pytket-dqc/) |
| **QIG partitioning** | Partitioning a Quantum Interaction Graph (QIG) — one node per virtual qubit — to produce the qubit-to-core mapping | [`qig-partitioning/`](qig-partitioning/) |

## What's different

- **Default SABRE** — the DQC-adapted baseline that (1,10) SABRE and CLA-SABRE
  build on. SABRE routes over a *conjoined coupling map* (every core's coupling
  graph stitched together with inter-core link edges), and the best parallel
  trial is chosen by lowest aggregated cost rather than fewest SWAPs. Two
  further knobs come with it: a tunable lookahead window
  (`extended_set_length`, the paper's `|E|`) and the option to skip the
  random-layout search when the core assignment is already fixed.
- **(1,10) SABRE** — the conjoined map's distance matrix becomes overridable
  (`custom_distance_matrix`), so intra- and inter-core edges can carry different
  costs. The paper's namesake scheme weights local edges 10 and remote ones 1,
  which — counter-intuitively — makes SABRE's greedy logic exhaust local routing
  before crossing a QPU boundary.
- **CLA-SABRE** (Custom Lookahead SABRE) — rewrites SABRE's cost function. A
  SWAP that stays inside a QPU costs nothing extra; one that crosses a link
  takes an upfront penalty, which each pending gate can then offset: a full
  reward when the SWAP makes a previously non-local gate local to one QPU, and
  otherwise a term proportional to the *change* in inter-QPU distance — a
  reward when that distance shrinks, a penalty only when it grows.
- **pytket-dqc adaptation** — per-link communication capacity, matching how real
  superconducting modules connect over dedicated links of differing capacity,
  rather than the per-server capacity upstream pytket-dqc assumes.
- **QIG partitioning** — a lightweight alternative to the hypergraph
  partitioning model. The Quantum Interaction Graph has one node per virtual
  qubit, and an edge weight equal to the number of two-qubit gates between that
  pair across the whole circuit. KaHyPar partitions it into one block per core;
  the blocks are then adapted to real hardware by matching them to physical
  cores and reallocating boundary qubits under a per-core capacity limit.

Annotated diffs mapped to the paper's sections and equations:
[`MODIFICATIONS.md`](MODIFICATIONS.md).

## Results

Routing quality is measured as **aggregated cost** — local SWAPs plus ten times
the EPR pairs consumed (the paper's Eq. 2). The two scales are scored against
different baselines, following the paper: at 48 qubits against
[DMapS](https://github.com/RoccoLoter/DMapS), a state-of-the-art DQC
transpiler, and at 399 qubits against our own DQC-adapted Default SABRE,
because DMapS and the hypergraph partitioning methods time out on most of that
suite.

On a **48-qubit** architecture (3×16 square-lattice cores, all-to-all
*inter-core* connectivity), QIG + CLA-SABRE cuts aggregated cost by **24.72%**
(unstructured) and **24.01%** (structured) while running **140–253× faster**
than DMapS — and completes the structured suite that the hypergraph
partitioning methods time out on.

On a **399-qubit** IBM Flamingo architecture (3×133 IBM Heron r1 cores, tested
with both all-to-all and line inter-core connectivity), where DMapS and
pytket-dqc largely time out entirely, QIG + CLA-SABRE reduces
structured-circuit routing cost by up to **51.94%** (all-to-all) and **51.96%**
(line) against the Default SABRE baseline, at practical runtimes.

Full tables, per-circuit figures, and where each method loses as well as wins:
[`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start

The Docker image builds both patched forks and `qig-partitioning` into one
environment, which is what the experiments need — Qiskit and pytket-dqc
importable side by side. It is also the path with the most verification behind
it.

```bash
git clone <this repo> && cd scalable_dqc_transpilation
docker build -f docker/Dockerfile -t dqc-env .    # ~10-20 min: compiles Rust and KaHyPar
docker run -it --rm dqc-env                       # opens a shell with everything installed
```

Run `docker build` from the repository root, as shown — it needs to read
`patches/`, `env/` and `qig-partitioning/`. The long first build is compilation,
not a hang; later builds reuse the layer cache.

Inside the container:

```bash
python -c "import qiskit, pytket_dqc, kahypar, qig_partitioning; print('OK')"
```

The image also carries the paper's experiment scripts and benchmark circuits at
`/opt/examples` — see [`examples/README.md`](examples/README.md).

Needs Docker installed and its daemon running
([get Docker](https://docs.docker.com/get-started/get-docker/)). Files written
inside the container are discarded on exit unless you mount a directory in —
[`examples/README.md`](examples/README.md) has the bind mount that gets results
back out.

Prefer to build on your own machine instead? See
**[`docs/INSTALL.md`](docs/INSTALL.md)** — the same sequence, outside a
container.

## Using QIG partitioning on its own

It doesn't touch either fork, so it needs none of the above:

```bash
pip install -e qig-partitioning/
```

```python
from qig_partitioning import get_heterogeneous_core_assignment

core_cost_matrix = {0: {0: 0.0, 1: 1.0}, 1: {1: 0.0, 0: 1.0}}   # 2 cores, uniform cost
assignment = get_heterogeneous_core_assignment(
    qc, core_cost_matrix, max_capacity=64      # qubits per core, not across all cores
)
```

[`notebooks/usage_demo.ipynb`](notebooks/usage_demo.ipynb) runs the full
pipeline — QIG partitioning feeding a fixed initial layout into the three SABRE
variants.

## Where to go next

| If you want to… | Read |
| --- | --- |
| See all the numbers, not just the headline | [`docs/RESULTS.md`](docs/RESULTS.md) |
| Understand what the patches actually change | [`MODIFICATIONS.md`](MODIFICATIONS.md) |
| Re-run the paper's experiments | [`examples/README.md`](examples/README.md) |
| Build without Docker | [`docs/INSTALL.md`](docs/INSTALL.md) |
| Know why a build step is the way it is | [`docs/BUILD-NOTES.md`](docs/BUILD-NOTES.md) |
| Read the paper itself | [`paper.pdf`](paper.pdf) |
| Get the whole thing on one page | [`poster-iqst.pdf`](poster-iqst.pdf) |

The rest of the tree:

```
patches/
  qiskit/                  DQC-aware SABRE variants + pinned base commit
  pytket-dqc/              superconducting topology awareness + pinned base commit
qig-partitioning/          standalone package: QIG pre-processing (paper §V-C)
examples/                  the experiment scripts that produced the results, plus
                           the benchmark circuits they run on
docker/Dockerfile          builds both patches + qig-partitioning into one environment
env/                       pip freezes of the reproduction environment
notebooks/usage_demo.ipynb illustrative end-to-end example
figures/                   result figures from the paper
```

Each `patches/<project>/BASE_COMMIT.txt` records the exact pinned upstream
commit and how the patch was generated.

## Reproducing the paper's experiments

The patches provide the primitives; a good deal of the paper's method lives in
orchestration code that sits in neither fork — the pseudo-sink subcircuit
generation of §V-A, the aggregated-cost trial selection of §IV-B, the
conjoined-backend construction, the §V-B hybrid bridge. That code is in
[`examples/`](examples/), as the scripts were run:

```
examples/three_square_architecture/   48-qubit runs   (Table II, Fig. 2)
examples/flamingo_architecture/       399-qubit runs  (Table III, Fig. 3)
examples/benchmarks/                  structured benchmark circuits (QASM)
```

[`examples/README.md`](examples/README.md) maps each paper feature to the
function implementing it, says which script produces which table rows, and
lists the deviations from the originals.

Two things are not in this repository: DMapS itself, which is
[third-party](https://github.com/RoccoLoter/DMapS), and two oversized benchmark
circuits — `examples/README.md` says which, and how to regenerate them.

## How this repository was assembled

The research contributions — the three SABRE variants, the pytket-dqc
adaptation, the QIG partitioning method, and the experiments behind every
number reported here — are the authors' own work, developed and run before this
repository existed.

This packaging around them was assembled with the help of generative AI
([Claude Code](https://claude.com/claude-code)): extracting the patches from
the development forks, the Docker and native build sequences, and the
documentation you are reading. Every patch is verified to apply against its
pinned upstream commit, and [`docs/BUILD-NOTES.md`](docs/BUILD-NOTES.md#verification-status)
records what has and has not been exercised.

## License

Apache License 2.0, see [`LICENSE`](LICENSE) — matching both upstream projects
the patches derive from.
