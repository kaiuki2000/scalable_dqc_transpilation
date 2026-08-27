# Example scripts

The experiment scripts that produced the paper's results. They are included so
the method is legible end to end: a good deal of what the paper describes lives
in orchestration code rather than in either patch, and this is where that code
is.

**These are reproduction artifacts, not a library.** They are the scripts as
they were run, with a short list of recorded deviations (below) and nothing
else. They are long, repetitive in places, and carry dead imports and
commented-out debugging. That is deliberate — see
[`../CLAUDE.md`](../CLAUDE.md).

All scripts come from
[`kaiuki2000/mqpu-ustutt-ibm`](https://github.com/kaiuki2000/mqpu-ustutt-ibm)
at commit `48dbc57f2ff97b55898f217e004de322e1a3ca3e`, the state matching the
paper's reported results (α = 9.0, β = 3.0, no post-submission variants).

## Layout

```
mqpu_utils.py             # the shared machinery: conjoined backends, pseudo-sinks,
                          # circuit generation, transpiler configs (1935 lines)
experiment_utils.py       # results bookkeeping + subcircuit barrier sync
three_square_architecture/  # 48-qubit, 3x16 square cores (paper Table II, Fig. 2)
flamingo_architecture/      # 399-qubit, 3x133 IBM Flamingo (Table III, Fig. 3)
benchmarks/                 # the structured benchmark circuits, as QASM
```

## Which script produces which table rows

Each architecture's rows are split across two scripts by method family; both
scripts of a pair **append to the same results file**, which is how the paper's
tables were assembled. Run both before reading the results.

### 48-qubit — Table II

| Script | Table II rows |
| --- | --- |
| `cz_frac_hypergraph.py` | Unstructured column: Default SABRE, (1,10) SABRE, CLA-SABRE, pytket (CE), pytket (PH), HGP + CLA |
| `cz_frac_qig.py` | Unstructured column: QIG + Default / (1,10) / CLA (\|E\|=20) / CLA (\|E\|=100) |
| `structured_light_hypergraph.py`, `structured_dense_hypergraph.py` | Structured column, same method set as `cz_frac_hypergraph.py` |
| `structured_light_qig.py`, `structured_dense_qig.py` | Structured column, QIG family |

The structured suite is split "light"/"dense" purely by how long the circuits
take; together they are the paper's 20 structured circuits.

### 399-qubit — Table III

| Script | Table III rows |
| --- | --- |
| `cz_frac.py` | Unstructured column, all seven non-DMapS rows |
| `structured.py` | Structured column, all seven non-DMapS rows |

Both cover the SABRE and QIG families in one file. Select the inter-core
topology with `TOPOLOGY` (see below) — the paper reports both.

DMapS, the external baseline both tables compare against, is not part of this
repository.

## Where the paper's "missing" pieces actually live

Several things the paper describes are in these scripts rather than in either
patch. This is the map:

| Paper feature | Location |
| --- | --- |
| Conjoined coupling map (§IV-A) | `mqpu_utils.generate_multi_qpu_backend_from_monolithic_backend_with_links`, `generate_inter_qpu_links`, `custom_mqpu_backend` |
| (1,10) distance weighting (§IV-B) | `generate_custom_distance_matrix` in each script; `mqpu_utils.compute_mixed_distance_matrix` |
| Aggregated-cost trial selection (§IV-B, Eq. 1) | `aggregated_cost = (total_eprs * f_weight) + intra_swaps` in each script's routing loop, with `total_eprs = 3 * inter_swaps + inter_czs` |
| Pseudo-sink qubits and routing placeholders (§V-A) | `mqpu_utils.RoutingPlaceholder`, `MakePlaceholdersOpaque`, `create_subcircuit`, `build_distributed_subcircuits`, `squash_placeholders_per_pair`, `remove_routing_placeholders` |
| Table I hierarchical edge weights | the `weight_lookup` dict in each `*_hypergraph.py` |
| Big-M sink penalty | `alpha=100000.0, beta=0.0` passed to SABRE — `beta=0.0` selects the flat-penalty branch of the Qiskit patch |
| Per-link capacity checking | `mqpu_utils.check_violations` |
| §V-B hybrid bridge | `mqpu_utils.generate_pytket_dqc_init_layout`, `pytket_dqc_init_layout_to_qiskit_initial_layout` |
| Unstructured (CZ-fraction) circuits | `mqpu_utils.build_cz_fraction_circuit` — generated in-process, no QASM needed |
| QIG partitioning (§V-C) | imported from this repo's [`qig-partitioning/`](../qig-partitioning/) package |

## Benchmark circuits

`benchmarks/` holds the structured suites. The unstructured (CZ-fraction)
circuits are generated at runtime and need no files.

| Directory | Contents |
| --- | --- |
| `light/` (6) | 48-qubit structured, fast |
| `dense/` (7) | 48-qubit structured, slow |
| `large/` (10) | 399-qubit MQT Bench circuits |
| `hamiltonians/` (13) | Benchpress Hamiltonian-simulation circuits; scripts select by qubit count — `32 < n ≤ 48` picks 7 for the 48-qubit runs, `266 < n ≤ 399` picks 6 for the 399-qubit runs |

**Two circuits are deliberately omitted, for size:**

- `quantum_volume_275_indep.qasm` (20.9 MB) — the 399-qubit unstructured
  suite's "+1" circuit
- `shor_alg_42.qasm` (13.6 MB) — a 48-qubit structured circuit which
  **timed out in the paper's own run** (Table II, footnote c: subset 19/20)

Together they were ~85% of the QASM by size. Both are MQT Bench circuits and
can be regenerated at those qubit counts; drop them into `benchmarks/large/`
and `benchmarks/dense/` to restore the full suites.

## Running them

The scripts need the full environment from the repository root README (both
patched forks plus `qig-partitioning`). From an activated environment:

```bash
python examples/three_square_architecture/cz_frac_hypergraph.py
python examples/three_square_architecture/cz_frac_qig.py     # same results file
```

The Docker image carries this directory at `/opt/examples`, and is the quickest
way to get a working environment:

```bash
docker build -f docker/Dockerfile -t dqc-env .
docker run -it --rm -v "$PWD/out":/opt/examples/three_square_architecture/results dqc-env \
    python /opt/examples/three_square_architecture/cz_frac_qig.py
```

The bind mount is what gets the results back out; without it they vanish with
the container.

Results are written to `<script's directory>/results/` and are gitignored.

Environment variables, all optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DQC_TOPOLOGY` | `a2a` | Flamingo scripts only: `a2a` or `line` |
| `DQC_BENCHMARKS_DIR` | `examples/benchmarks` | Where the QASM suites live |
| `DQC_RESULTS_DIR` | `<script dir>/results` | Where results JSON is written |
| `DQC_KAHYPAR_CONFIG` | bundled `.ini` | Override the KaHyPar config |

The flamingo scripts try `QiskitRuntimeService()` for an `ibm_torino` snapshot
and fall back to the local `FakeTorino` backend, so no IBM Quantum credentials
are required.

These are long-running experiments — the paper used 1-hour (48-qubit) and
3-hour (399-qubit) per-circuit timeouts and multiple parallel workers.

## Recorded deviations from the originals

The complete list. Nothing else in these files was changed.

1. **The inline QIG implementation was removed** in favour of
   `from qig_partitioning import get_heterogeneous_core_assignment`. Each of
   the five QIG-using scripts carried its own copy of `build_interaction_graph`,
   `partition_with_kahypar`, `match_partitions_to_cores`,
   `enforce_strict_capacity`, `boundary_reallocation` and
   `get_heterogeneous_core_assignment` — 758 lines across the five, in four
   formatting variants of one algorithm.

   Verified before removing: the three refinement functions were
   differential-tested against the packaged versions on 120 randomized
   partitioning problems with identical output in every case, and
   `build_interaction_graph` and `partition_with_kahypar` are textually
   equivalent (the package adds an optional config-path default and an
   `epsilon` parameter defaulting to the same 0.01). The pipeline in
   `get_heterogeneous_core_assignment` calls the same functions in the same
   order.

2. **Imports.** `from qiskit_dev.custom_targets.mqpu_utils import ...` became
   `from mqpu_utils import ...`, and a `sys.path` bootstrap was added at the top
   of each script so the shared helpers in `examples/` resolve from a
   subdirectory. The original `circuit_utils.py` was merged into
   `experiment_utils.py` (only `sync_subcircuit_barriers` was used from it).

3. **Absolute paths became repository-relative**, with environment-variable
   overrides for cluster runs. This covers the benchmark directories, the
   results files, and `KAHYPAR_CONFIG_PATH`, which now defaults to the
   `km1_kKaHyPar_sea20.ini` bundled with `qig-partitioning`. Results filenames
   were shortened; each pair of scripts still shares one file, as before.

4. **`TOPOLOGY = "a2a" | "line"`** in the flamingo scripts. The paper's two
   399-qubit topologies were originally selected by commenting and uncommenting
   three blocks by hand; one switch now drives all three.

5. **Renamed files**, for legibility: `run_kahypar_simple_arch_*` →
   `*_qig.py`, `cz_frac_benchmarks`/`structured_benchmarks_*` →
   `*_hypergraph.py`, and the flamingo `large_benchmarks.py` → `structured.py`,
   matching the paper's own name for that suite.
