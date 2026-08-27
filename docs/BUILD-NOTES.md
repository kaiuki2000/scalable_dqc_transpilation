# Build notes

Why the build is shaped the way it is, and what has actually been verified.

Neither [`docs/INSTALL.md`](INSTALL.md) nor [`docker/Dockerfile`](../docker/Dockerfile)
explains itself in much detail — this file is where that reasoning lives.
Nothing here is needed to *run* the build; it's here so that when a step looks
arbitrary, or you want to change one, you can find out what it was working
around. Every item below is a failure that was actually hit and diagnosed, not
a precaution.

If you are just setting the environment up, you want the README's quick start
or `docs/INSTALL.md`, not this file.

## Contents

- [Why the steps are in that order](#why-the-steps-are-in-that-order)
- [KaHyPar must be built from source at 1.3.2](#kahypar-must-be-built-from-source-at-132)
- [KaHyPar 1.3.2 does not compile under Python 3.11+ as shipped](#kahypar-132-does-not-compile-under-python-311-as-shipped)
- [Two more KaHyPar build hazards: CMake 4 and `-march=native`](#two-more-kahypar-build-hazards-cmake-4-and--marchnative)
- [Why the bulk install runs before the patched Qiskit](#why-the-bulk-install-runs-before-the-patched-qiskit)
- [Why the Qiskit checkout is named `qiskit-src`](#why-the-qiskit-checkout-is-named-qiskit-src)
- [Why Qiskit needs an extra release-build step](#why-qiskit-needs-an-extra-release-build-step)
- [Why `qig-partitioning` installs with `--no-deps`](#why-qig-partitioning-installs-with---no-deps)
- [Why the outer directory is `qig-partitioning` and the package is `qig_partitioning`](#why-the-outer-directory-is-qig-partitioning-and-the-package-is-qig_partitioning)
- [Which environment freeze is used, and why](#which-environment-freeze-is-used-and-why)
- [What the build-time checks actually prove](#what-the-build-time-checks-actually-prove)
- [Verification status](#verification-status)

## Why the steps are in that order

KaHyPar from source → bulk-install the frozen environment → patched Qiskit →
patched pytket-dqc → `qig-partitioning`. The order is load-bearing at three
points, each explained in its own section below: KaHyPar goes first so that
nothing later can overwrite its `.so`; the frozen environment goes before
Qiskit so that `pytket-qiskit` has a satisfiable `qiskit` to install against;
and the patched Qiskit goes after it so that the editable install is the one
left active. Reordering these does not fail loudly — it produces an environment
that imports fine and is quietly wrong.

## KaHyPar must be built from source at 1.3.2

`pip install kahypar` gets you ≥1.3.5, which is **incompatible with this
version of pytket-dqc**: its partitioning code fails at partition time with
`Vertices with weight 0 are not supported`. This was confirmed by running
pytket-dqc's own unmodified test suite, so it's not an artefact of this repo's
patch.

That failure only surfaces when you actually partition something. `import
kahypar` succeeds with the broken wheel, so import-level smoke tests do not
catch it. The only real check is calling into it — see
[Verification status](#verification-status).

The recipe follows
[pytket-dqc's own instructions](https://github.com/Quantinuum/pytket-dqc#kahypar-with-python-interface):
the compiled `.so` is copied straight into `site-packages`, using the system
`libboost-program-options-dev` rather than KaHyPar's `KAHYPAR_USE_MINIMAL_BOOST`
fallback. Because the `.so` is copied by hand it carries no `dist-info`, so
KaHyPar will not appear in `pip freeze` afterwards. That is expected, and is why
neither freeze file in `env/` lists it.

KaHyPar is pinned by commit (`efa12a90`), not by the `1.3.2` tag it currently
points at, matching how Qiskit and pytket-dqc are pinned elsewhere in this repo:
tags can move, commits cannot.

## KaHyPar 1.3.2 does not compile under Python 3.11+ as shipped

KaHyPar vendors pybind11 as a submodule, and the version recorded at each tag
decides whether it compiles against a modern CPython:

| KaHyPar | vendored pybind11 | Python 3.11+ |
| --- | --- | --- |
| 1.3.2 / 1.3.3 | `a6355b00` (~v2.4.dev4, Oct 2019) | ✗ |
| 1.3.5 | `c9638a19` (Sep 2023) | ✓ |
| 1.3.6 / 1.3.7 | `2c9191e9` (Jan 2026) | ✓ |

CPython 3.11 made `PyFrameObject`'s internals opaque. That 2019 pybind11
dereferences them directly and unconditionally in `cast.h`'s `error_string()`
(`frame->f_code->co_filename`), with no version guard — confirmed by reading the
source at that exact pinned commit. The build therefore fails on **any** Python
≥3.11 with `invalid use of incomplete type 'PyFrameObject'` (reproduced under
both 3.11 and 3.12; 3.10 builds clean).

So the Python version is the trigger, not the cause. The fix is to replace the
vendored pybind11 with **v2.13.6** and leave KaHyPar's own C++ alone:

```bash
rm -rf python/pybind11
git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git python/pybind11
```

v2.13.6 specifically: last of the pybind11 2.x line, supports Python 3.7–3.13,
and keeps the CMake behaviour KaHyPar 1.3.2's `python/CMakeLists.txt` expects
(pybind11 3.x switched to `FindPython`). The swap is drop-in because KaHyPar's
`python/module.cpp` uses nothing beyond `PYBIND11_MODULE`, `py::init` and
`py::arg`, all stable across 2.4 → 2.13.

**Do not fix this by bumping KaHyPar past 1.3.2 instead.** 1.3.5+ is exactly
where the `Vertices with weight 0` regression above lives, so that trades one
failure for the other. And do not fix it by downgrading to Python 3.10 — the
frozen environment is a 3.12 one.

The Docker image pins `python:3.12.3-slim-bookworm` rather than a bare
`python:3.12-slim` for a related reason: KaHyPar's pybind11 also relies on
`<cstdint>` being pulled in transitively by another standard header, which
holds on Debian 12 "bookworm" (GCC 12.2) but not from GCC 14 onward (Debian 13
"trixie", now the default for unqualified tags), where `attr.h` fails with
`'uint16_t' in namespace 'std' does not name a type`. Pinning the Debian
release also stops the base image changing compiler versions underneath this
build later.

## Two more KaHyPar build hazards: CMake 4 and `-march=native`

**CMake ≥ 4.** KaHyPar's vendored googletest submodule declares
`cmake_minimum_required(VERSION 2.6.2)`, and CMake 4.0 (March 2025) removed
compatibility with anything below 3.5. On Ubuntu 24.04+, current Homebrew, or
any other CMake ≥ 4 host, the deprecation warnings this build currently prints
become hard configure-time errors — in a message pointing at googletest, which
has nothing to do with anything you are building. Pass
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the `cmake` line rather than editing
submodule files. The Docker image sidesteps this by pinning bookworm, which
ships CMake 3.25.x.

**`-march=native`.** KaHyPar's `RELEASE` flags include `-march=native`, so the
`.so` is tuned to whichever CPU built it. If you build the image on a laptop and
run it on a cluster (or push and pull it across heterogeneous hardware), that can
`SIGILL` at import time. Add
`-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -mtune=generic"` to the `cmake` line to
trade some partitioning speed for a portable build.

## Why the bulk install runs before the patched Qiskit

The frozen environment is installed for real (`pip install -r`), not used
merely as version constraints. That deliberately pulls in an ordinary, unpatched
`qiskit` release at this point, because `pytket-qiskit==0.73.0` requires
`qiskit>=2.2.3` (checked against PyPI metadata) and our patched fork reports
`2.2.0rc1` (from `qiskit/VERSION.txt` at the pinned commit — the patch does not
change it), which fails that as a plain version comparison.

That temporary stock qiskit is then replaced: pip's explicit "install this exact
local package" (`pip install -e .`) unconditionally replaces whatever is
currently installed under that name. So doing the bulk install *first* and the
patched Qiskit *last* is what makes this safe rather than a race —
`pytket-qiskit`, `qiskit-aer` and `qiskit-ibm-runtime` all got a real qiskit to
install successfully against, and the patched editable install ends up the one
actually active.

Nothing installed after Qiskit in the sequence — pytket-dqc, `qig-partitioning`
— declares a `qiskit` or `pytket-qiskit` requirement of its own, so nothing
later can undo this again. The import checks immediately after the Qiskit build
verify that it held.

The two local/editable lines the freeze records (qiskit itself, and pytket-dqc)
are stripped before installing, since both are built from their pinned patch
commits instead:

```bash
grep -v -e '^-e ' -e '@ file://' env/requirements-freeze.txt > /tmp/frozen.txt
```

`/tmp/frozen.txt` is then reused as `-c` for every later install, so nothing
drifts off this pinned set.

## Why the Qiskit checkout is named `qiskit-src`

Not `qiskit`. If the checkout sits next to a directory you later run Python
from — a shared workspace root, exactly the layout this repo's Docker image
originally had under `/opt` — then a `qiskit` directory with no `__init__.py`
is picked up by Python's `PathFinder` as an implicit namespace package, and it
**wins outright**: setuptools registers editable installs through a `meta_path`
finder that it *appends* to `sys.meta_path`, i.e. behind `PathFinder`, so that
finder is never consulted.

The symptom is `import qiskit` appearing to succeed while `qiskit.__file__` is
`None`, and `from qiskit import QuantumCircuit` failing with a confusing
`AttributeError`. This would hit every interactive session and Jupyter kernel
started from that directory, so the rename is a correctness fix for users of
the image, not just for the build-time check. It was found by hitting it, not
by inspection.

pytket-dqc needs no equivalent rename: its import name is `pytket_dqc` with an
underscore, which cannot collide with a `pytket-dqc` directory.

## Why Qiskit needs an extra release-build step

```bash
pip install -e . -c /tmp/frozen.txt
python setup.py build_rust --release --inplace
```

A plain `pip install -e .` always builds the Rust extension in **debug** mode
for editable installs: setuptools-rust decides debug vs. release from whether
the install is editable, not from pip's `-C`/`--config-settings`. There is no
PEP 517 config-settings hook for this, so a maturin-style
`-C build-args="--release"` is silently accepted and silently ignored.

Release matters here — a debug build is dramatically slower at runtime, which
would understate the paper's routing-time results. The explicit second step is
what Qiskit's own `CONTRIBUTING.md` prescribes.

No separate `setuptools-rust` install is needed: `pip install -e .` gets it via
a throwaway PEP 517 build-isolation environment (it is only a `[build-system]`
requirement, not a runtime dependency), but the frozen bulk install has already
put `setuptools-rust==1.12.0` in the main environment before this point.

## Why `qig-partitioning` installs with `--no-deps`

`qig-partitioning`'s `pyproject.toml` declares a plain `kahypar` PyPI
dependency. Installing it normally at this point pulls a prebuilt wheel and
silently overwrites the source-built KaHyPar 1.3.2 `.so`, reintroducing the
`Vertices with weight 0` failure described above.

Skipping it is safe: `qig_partitioning/partitioning.py` only calls KaHyPar's
standard `Hypergraph`/`Context`/`partition` API, which the source build already
provides, and its other declared dependencies (`qiskit`, `numpy`) are installed
by this point anyway.

This applies only when installing into the full reproduction environment. Using
`qig-partitioning` **standalone**, with no patched forks around, a plain
`pip install -e qig-partitioning/` is correct — letting it pull its own
`kahypar` wheel is the right default there.

## Why the outer directory is `qig-partitioning` and the package is `qig_partitioning`

The hyphen/underscore mismatch is deliberate. Naming both identically makes
Python resolve `import qig_partitioning` to an empty implicit namespace package
whenever the current working directory is the repo root, because `''` in
`sys.path` shadows the real, `pip install -e`-registered package — the same
class of failure as the `qiskit-src` rename above. Caught by actually testing
`import qig_partitioning` from the repo root.

Don't rename the outer directory to match the package name; it silently
reintroduces this.

## Which environment freeze is used, and why

`env/` holds two `pip freeze` snapshots:

- **`requirements-freeze.txt`** — what the Docker image and `docs/INSTALL.md`
  actually install. Captured post-submission on the Slurm/Desktop machine.
- **`requirements-freeze-paper-submission.txt`** — the laptop environment the
  paper's reported results were produced on. Historical record; nothing builds
  from it.

The build uses the first because it pins `pytket-qiskit==0.73.0` as a real PyPI
release, whereas the submission freeze records it as a local `@ file://`
checkout — and local paths are stripped by the `grep -v` filter above, so
building from the submission freeze would silently omit `pytket-qiskit`
entirely, which the Section V-B hybrid approach needs.

**Why both files are kept.** They are not near-duplicates: of the 148 packages
they share, **54 differ in version**, and the differences are not confined to
tooling. `networkx` is 3.6.1 in the submission environment and 3.5 in the
build environment — and pytket-dqc's Steiner-tree and shortest-path code, which
this repo's patch modifies, runs on networkx. `scipy` (1.16.3 → 1.16.2),
`pandas` (2.3.3 → 3.0.3) and `scikit-learn` (1.8.0 → 1.9.0) also differ.
`qiskit`, `pytket` (2.10.3), `numpy` (2.2.6) and `rustworkx` (0.17.1) match.
The submission freeze additionally carries a Sphinx docs stack and
`openqasm3`/`qiskit-qasm3-import` that the build environment does not.

So the second file is not redundant reassurance — it is the only record of the
exact dependency set behind the numbers in `docs/RESULTS.md`, and it cannot be
reconstructed from the other. It stays for provenance, is 3.5 KB, and nothing
reads it; the rename and these headers exist so it cannot be mistaken for the
one to install from.

Neither file lists `kahypar`, for the `dist-info` reason given earlier.

## What the build-time checks actually prove

The Docker build runs the same checks `docs/INSTALL.md` shows, and they come in
two kinds. Both are needed; neither is sufficient alone.

**Location checks** prove the active `qiskit` is the editable install pointing
at the patched checkout:

```python
import qiskit
assert qiskit.__file__ is not None      # not shadowed by a namespace package
assert '/qiskit-src/' in qiskit.__file__
import qiskit._accelerate as acc        # compiled Rust loaded from the source tree
```

`PYTHONSAFEPATH=1` (Python 3.11+) keeps the cwd off `sys.path` entirely, so the
check cannot be fooled by namespace shadowing even if some future directory name
collides again. The explicit `is not None` test is there because a shadowed
import yields `qiskit.__file__ == None`, which would otherwise blow up inside
`os.path` handling with a `TypeError` that says nothing about the real cause.
The `_accelerate` check confirms *location* of the compiled extension only — it
does not distinguish the release build from the debug one it overwrote.

**A patch-presence check** answers the question location cannot — a stock
checkout at the right commit would pass every check above:

```bash
git apply --reverse --check <the patch>
```

This succeeds only if the patch is currently applied, and needs no knowledge of
which symbols the patch introduces, so it keeps working when the patch changes.

Together these mean a broken build fails loudly at build time rather than
shipping a silently unpatched image.

## Verification status

Honest accounting of what has and hasn't been exercised.

| Component | Status |
| --- | --- |
| Docker build | **Builds end to end** against a real Docker daemon, including KaHyPar from source, with all checks above passing. |
| Both patches apply | **Verified** — `git apply --check` against a fresh worktree at each pinned base commit. |
| Qiskit SABRE patch, behaviour | **Verified beyond imports.** On a toy 2-QPU coupling map with one inter-QPU link, baseline `SabreSwap` crossed that link once; with CLA-SABRE's `penalized_swaps` / `qubit_qpu_map` / `inter_qpu_coupling_map` set, it crossed zero times, at the cost of one extra local SWAP — the intended effect. |
| QIG partitioning | **Verified end to end** on a synthetic circuit with three tightly-coupled qubit clusters under a hard per-core capacity: each cluster placed on its own core, capacity respected. |
| pytket-dqc patch, manual path | **Installs and imports; distribution not verified end to end.** `server_link_capacities` is present on `NISQNetwork` as documented, but actually distributing a circuit — which is what exercises KaHyPar — has not been checked against the manual sequence specifically. |
| KaHyPar, real partitioning call | **Not covered by the automated checks.** `import kahypar` succeeds with the broken PyPI wheel too; it only fails at partition time. Worth running `qig_partitioning.partition_with_kahypar(...)` or a pytket-dqc distribution yourself after setup. |

The Docker path is the one with the most coverage. If you are choosing between
the two, choose it.
