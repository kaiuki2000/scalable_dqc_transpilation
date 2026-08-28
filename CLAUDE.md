# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this repository is

Companion artifact for *"Scalable Transpilation for Overcoming Restricted
Connectivity in Distributed Superconducting Quantum Architectures"* (Azenha,
Polian & Brandhofer — QCE26 submission; full draft in
`paper.pdf`). It ships small, self-contained patches against
pinned upstream commits of two forked projects (Qiskit and pytket-dqc), so the
changes under study can be reviewed in isolation without vendoring the full
forks, plus one standalone Python package (`qig-partitioning/`) for the one
piece of the paper's method that modifies neither fork.

The paper introduces three DQC-aware SABRE routing variants in Qiskit
(**Default SABRE**, **(1,10) SABRE**, **CLA-SABRE**), a
superconducting-hardware-aware adaptation of pytket-dqc's distribution model,
and a QIG (Quantum Interaction Graph) partitioning pre-processing step. Use
those names when describing or modifying this repo's content — they're the
paper's own terminology, not invented here.

There is no build system, linter, or test suite for the two patches: that code
only builds and runs once applied to a checkout of the relevant upstream
project, and any build/lint/test workflow belongs to that upstream project, not
here. `qig-partitioning/` is the exception — ordinary, self-contained Python
with its own `pyproject.toml`.

## Where things are documented

Don't duplicate these explanations back into other files; link to them.

| File | Holds |
| --- | --- |
| `README.md` | The skimmable entry point: what the work is, headline results, Docker quick start. Kept deliberately short — see "Documentation style" below. |
| `MODIFICATIONS.md` | The annotated walkthrough of the patches, mapped to the paper's sections and equations. The authority on *what each patch does*. |
| `docs/RESULTS.md` | Paper Tables II–III as markdown, plus Figs. 2–3. The authority on *the numbers*. |
| `docs/INSTALL.md` | The native (non-Docker) build sequence, commands-first. |
| `docs/BUILD-NOTES.md` | Why each build step is the way it is, and the verification status table. The authority on *why the build is shaped like this*. |
| `docker/Dockerfile` | The same sequence in a container; comments are pointers into `docs/BUILD-NOTES.md`. |
| `examples/README.md` | Which script produces which table rows, the paper-feature → function map, and the recorded deviations from the original scripts. |

## Layout

```
patches/
  qiskit/         0001-dqc-aware-sabre-variants.patch, BASE_COMMIT.txt
  pytket-dqc/     0001-superconducting-topology-awareness.patch,
                  basic_usage_demo.ipynb, BASE_COMMIT.txt
qig-partitioning/ standalone package; no upstream to diff against
docs/             RESULTS.md, INSTALL.md, BUILD-NOTES.md
docker/Dockerfile
env/
  requirements-freeze.txt                   # what the build actually installs
  requirements-freeze-paper-submission.txt  # historical record of the paper's
                                            # own results environment; nothing
                                            # builds from it
figures/          result figures (PNG, referenced by docs/RESULTS.md)
notebooks/        usage_demo.ipynb
examples/
  mqpu_utils.py         shared experiment machinery (conjoined backends,
                        pseudo-sinks, circuit generation) -- 1935 lines
  experiment_utils.py   results bookkeeping
  three_square_architecture/   6 scripts -> Table II
  flamingo_architecture/       2 scripts -> Table III
  benchmarks/                  structured benchmark circuits (QASM)
```

Each `patches/<project>/BASE_COMMIT.txt` records the exact pinned upstream
commit the patch was generated against, and the reproduction steps.

## Pinned bases

- **Qiskit**: `848178940d2def0dbdee578d20ce5e6b3451f4c2` (tag `2.2.0rc1` —
  **not** the same commit as the final `2.2.0` release tag; the patch will not
  apply cleanly on top of `2.2.0`). Diffed against commit `c62ce998` of a
  private development fork — the state of the code when the paper's results
  were produced.
- **pytket-dqc**: `bfa0b4eff5b77d0a9b3c260c7842e57e75091796` (a clean pin at
  `origin/main`'s tip, no upstream drift), diffed against commit `5a28ad6` of
  a private development fork, restricted to `src/pytket_dqc/`.

Both have been verified to apply cleanly (`git apply --check`) against a fresh
worktree at their pinned base commit. But applying a patch is one step of a
longer, order-dependent sequence. **Don't reconstruct that sequence from memory
or from an older version of a doc** — follow `docs/INSTALL.md` or
`docker/Dockerfile`, both of which have been verified against a real build.

## Working on this repo

- When either patch needs to change, **regenerate it from the corresponding
  fork** rather than hand-editing the `.patch` file: diff the fork's pinned base
  commit against the relevant files at the fork's current HEAD, and re-verify
  with `git apply --check` against a fresh checkout/worktree of the pinned base
  before committing.
- Keep patches scoped to the algorithmic contribution. Repo-local noise from the
  forks (personal env/tooling files, CI workflow edits, notebook execution
  outputs) stays out. Two deliberate exclusions already in place: the fork's own
  GitHub Pages docs-deploy workflow removal, and notebook cell outputs. One
  known exception survives: the debug `print()`/`perf_counter` instrumentation in
  three files of the pytket-dqc patch —
  `allocators/hypergraph_partitioning.py`, `distributors/cover_embedding.py`
  and `distributors/partitioning_heterogeneous.py` — which is those files'
  *only* content. It is documented in `MODIFICATIONS.md`; dropping it is the
  obvious cleanup, but it changes the state the paper's results were produced
  from (and it is where the paper's pytket-dqc runtime breakdown came from), so
  it is a decision to raise rather than make.
- **Don't rename `qig-partitioning/` to match its import name
  `qig_partitioning`.** The hyphen/underscore mismatch is deliberate; naming
  them identically silently breaks `import qig_partitioning` from the repo root.
  See `docs/BUILD-NOTES.md`.
- The full forks these patches derive from are private and live outside this
  repository. Don't assume their source files are available locally when
  reasoning about this repo in isolation, and don't add links to them in
  user-facing docs — they resolve for nobody. Bare commit SHAs are fine as
  provenance.

## The `examples/` scripts are reproduction artifacts

They are the experiment scripts as they were run, taken verbatim from a
private experiment repository at `48dbc57`, with exactly five recorded
deviations listed in `examples/README.md`. Treat them the way the patches are
treated: **do not refactor, tidy, deduplicate or lint them.** Their dead
imports, shadowed names, commented-out debugging and near-duplicate structure
are part of what is being reproduced. If a genuine change is needed, add it to
the deviations list in `examples/README.md` in the same commit.

The one substantive edit already made is worth knowing about: each QIG-using
script carried its own inline copy of the QIG algorithm, and all five now
import `qig_partitioning` instead. Don't reintroduce an inline copy — the
package is the single source of truth for that method.

## Documentation style

The README is the one file a passing reader will actually open, so it is
optimized for their time: what this is, what's new, the headline numbers, and a
Docker quick start. Setup instructions in it stay minimal-but-working.

Reasoning, war stories, and "here's the failure this avoids" material are
valuable and are kept — in `docs/BUILD-NOTES.md`, not in the README. When you
learn something new about why a step is necessary, add it there and leave the
README alone.
