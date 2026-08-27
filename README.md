# Scalable transpilation for distributed superconducting quantum architectures

Companion artifact for *"Scalable Transpilation for Overcoming Restricted
Connectivity in Distributed Superconducting Quantum Architectures"* — Afonso
Azenha, Ilia Polian, Sebastian Brandhofer (QCE26 submission). Full draft:
[`Paper_draft_portrait.pdf`](Paper_draft_portrait.pdf). *arXiv/DOI link to
follow.*

Qubit mapping and routing for Distributed Quantum Computing (DQC) on near-term
superconducting hardware, where inter-QPU links are scarce and roughly an
order of magnitude noisier than intra-QPU couplings. Three contributions,
shipped as two patches against pinned upstream commits plus one standalone
package:

| | What it is | Where |
| --- | --- | --- |
| **DQC-aware SABRE variants** | Three routing variants in Qiskit that account for which SWAPs cross a QPU boundary: *Default SABRE*, *(1,10) SABRE*, *CLA-SABRE* | [`patches/qiskit/`](patches/qiskit/) |
| **Superconducting-aware distribution** | pytket-dqc's placement model, adapted to per-link (not per-server) communication capacity and restricted intra-core connectivity | [`patches/pytket-dqc/`](patches/pytket-dqc/) |
| **QIG partitioning** | The pre-processing step producing the initial per-core qubit assignment — a plain installable package, since it modifies neither fork | [`qig-partitioning/`](qig-partitioning/) |

## What's different

- **Default SABRE** — the DQC-adapted baseline the other two build on. SABRE
  routes over a *conjoined coupling map* (every core's coupling graph stitched
  together with inter-core link edges), with the changes needed to make that
  scale: a tunable lookahead window (`extended_set_length`) and skipping the
  random-layout search when the core assignment is already fixed.
- **(1,10) SABRE** — the conjoined map's distance matrix becomes overridable
  (`custom_distance_matrix`), so intra- and inter-core edges can carry different
  costs. The paper's namesake scheme weights local edges 10 and remote ones 1,
  which — counter-intuitively — makes SABRE's greedy logic exhaust local routing
  before crossing a QPU boundary.
- **CLA-SABRE** (Custom Lookahead SABRE) — rewrites SABRE's cost function: an
  upfront penalty on any SWAP marked inter-QPU, plus a lookahead term that
  rewards a SWAP for making a gate local to one QPU and penalizes it in
  proportion to inter-QPU distance otherwise.
- **pytket-dqc adaptation** — per-link communication capacity, matching how real
  superconducting modules connect over links of differing capacity, rather than
  the per-server capacity the upstream model assumes.
- **QIG partitioning** — a lightweight alternative to the hypergraph model: one
  global interaction graph over the circuit's virtual qubits, partitioned with
  KaHyPar, then adapted to real hardware capacity by matching abstract blocks to
  physical cores and reallocating boundary qubits.

Annotated diffs mapped to the paper's sections and equations:
[`MODIFICATIONS.md`](MODIFICATIONS.md).

## Results

Against a DMapS baseline on a **48-qubit** (3×16, all-to-all) architecture,
QIG + CLA-SABRE cuts aggregated routing cost by **24.72%** (unstructured) and
**24.01%** (structured) while running **140–253× faster** — and completes the
structured suite that the hypergraph methods time out on.

On a **399-qubit** IBM Flamingo architecture (3×133), where DMapS and pytket-dqc
largely time out entirely, QIG + CLA-SABRE reduces structured-circuit routing
cost by up to **51.94%** (all-to-all) and **51.96%** (restricted line topology)
against the DQC-adapted Default SABRE baseline, at practical runtimes.

Full tables and per-circuit figures: [`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start

The Docker image builds both patched forks and `qig-partitioning` into one
environment, which is what the experiments need — Qiskit and pytket-dqc
importable side by side. It is also the path with the most verification behind
it.

```bash
git clone <this repo> && cd scalable_dqc_transpilation
docker build -f docker/Dockerfile -t dqc-env .    # ~20-30 min: compiles Rust and KaHyPar
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

**New to Docker?** It runs the whole environment in an isolated container, so
nothing is installed on your machine and you can delete it with
`docker rmi dqc-env`. You need Docker installed and its daemon running
([Docker Desktop](https://docs.docker.com/get-started/get-docker/) on
macOS/Windows, `docker.io` or Docker Engine on Linux). The first command builds
the image once; the second opens a shell inside it, and `exit` leaves. Files
you create in the container are discarded on exit unless you mount a directory
into it (`docker run -it --rm -v "$PWD":/work dqc-env`).

Prefer to build on your own machine instead? See
**[`docs/INSTALL.md`](docs/INSTALL.md)** — same sequence, outside a container.
Either way, [`docs/BUILD-NOTES.md`](docs/BUILD-NOTES.md) explains why the steps
are what they are, and records what has and hasn't been verified.

## Using QIG partitioning on its own

It doesn't touch either fork, so it needs none of the above:

```bash
pip install -e qig-partitioning/
```

```python
from qig_partitioning import get_heterogeneous_core_assignment

core_cost_matrix = {0: {0: 0.0, 1: 1.0}, 1: {1: 0.0, 0: 1.0}}   # 2 cores, uniform cost
assignment = get_heterogeneous_core_assignment(qc, core_cost_matrix, max_capacity=64)
```

[`notebooks/usage_demo.ipynb`](notebooks/usage_demo.ipynb) runs the full
pipeline — QIG partitioning feeding a fixed initial layout into the three SABRE
variants.

## Layout

```
Paper_draft_portrait.pdf   full paper draft
MODIFICATIONS.md           annotated walkthrough of the key changes, mapped to the paper
docs/
  RESULTS.md               full result tables and per-circuit figures
  INSTALL.md               native (non-Docker) setup
  BUILD-NOTES.md           why the build is shaped this way; verification status
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
orchestration code that sits in neither fork. That code is in
[`examples/`](examples/) — the experiment scripts as they were run, together
with the structured benchmark circuits:

```
examples/three_square_architecture/   48-qubit runs   (Table II, Fig. 2)
examples/flamingo_architecture/       399-qubit runs  (Table III, Fig. 3)
examples/benchmarks/                  structured benchmark circuits (QASM)
```

That is where the pseudo-sink subcircuit generation of §V-A, the
aggregated-cost trial selection of §IV-B, the conjoined-backend construction,
and the §V-B hybrid bridge actually live. [`examples/README.md`](examples/README.md)
maps each paper feature to the function implementing it, says which script
produces which table rows, and lists the deviations from the originals.

Not included: DMapS, the external baseline both tables compare against, and two
oversized benchmark circuits (noted in `examples/README.md`, with instructions
to regenerate them).

## License

Apache License 2.0, see [`LICENSE`](LICENSE) — matching both upstream projects
the patches derive from.
