# Native setup

Building the environment directly on your machine, without Docker.

The Docker image in the README is the recommended path and the one with the
most verification behind it. Use this if you'd rather not run a container, or
want the checkouts editable on your own filesystem.

**Order matters.** Each step exists to avoid a specific failure — the short
reason is inline, the full explanation is in
[`BUILD-NOTES.md`](BUILD-NOTES.md). Don't skip around.

Throughout: `$REPO` is wherever you cloned *this* repository. Everything is
built in one working directory holding a single virtualenv and all checkouts
side by side, mirroring what the Docker image does under `/opt`.

## 0. Prerequisites

System packages (Debian/Ubuntu names):

```bash
sudo apt-get install -y git build-essential cmake curl \
    graphviz libgraphviz-dev libboost-program-options-dev
```

`graphviz` + headers are for `pygraphviz`, one of pytket-dqc's dependencies;
Boost.Program_options is for building KaHyPar.

**Python 3.12** — matches the environment `env/requirements-freeze.txt` was
frozen from. Other 3.10+ interpreters likely work but haven't been verified.
Note that Python ≥3.11 requires the pybind11 swap in step 1;
[3.10 avoids it but doesn't match the freeze](BUILD-NOTES.md#kahypar-132-does-not-compile-under-python-311-as-shipped).

**Rust** — Qiskit's routing/layout core, which the SABRE patch touches, is Rust.

```bash
export REPO=/path/to/scalable_dqc_transpilation

mkdir ~/dqc-workspace && cd ~/dqc-workspace
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
```

## 1. Build KaHyPar 1.3.2 from source

Required, not optional: the PyPI wheel is ≥1.3.5, which
[breaks this version of pytket-dqc at partition time](BUILD-NOTES.md#kahypar-must-be-built-from-source-at-132).

```bash
git clone --filter=blob:none https://github.com/kahypar/kahypar.git kahypar-src
cd kahypar-src
git checkout efa12a90acae05469520d43cff122ca8620fab48   # tag 1.3.2
git submodule update --init --recursive

# The vendored pybind11 (2019) doesn't compile under Python >=3.11. Swap it;
# do NOT bump KaHyPar instead. See BUILD-NOTES.md.
rm -rf python/pybind11
git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git python/pybind11

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1
cd python && make -j"$(nproc)"
cp kahypar*.so "$(python3 -c 'import site; print(site.getsitepackages()[0])')"
cd ~/dqc-workspace && rm -rf kahypar-src
```

On **CMake ≥ 4** (Ubuntu 24.04+, current Homebrew) add
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the `cmake` line, or it fails with an
error pointing at googletest. If the resulting image or venv will run on
different hardware than it was built on, also see the
[`-march=native` note](BUILD-NOTES.md#two-more-kahypar-build-hazards-cmake-4-and--marchnative).

KaHyPar won't show up in `pip freeze` afterwards — the `.so` is copied by hand
and carries no `dist-info`. That's expected.

## 2. Install the frozen environment

Before the patched forks, not after —
[the ordering is what makes step 3 safe](BUILD-NOTES.md#why-the-bulk-install-runs-before-the-patched-qiskit).

```bash
grep -v -e '^-e ' -e '@ file://' "$REPO/env/requirements-freeze.txt" > /tmp/frozen.txt
pip install -r /tmp/frozen.txt
```

The stripped lines are local checkouts of qiskit and pytket-dqc; both get built
from their pinned patch commits below instead. This pulls in a stock qiskit
release along the way — expected, and replaced in the next step.

## 3. Build and patch Qiskit

```bash
cd ~/dqc-workspace
git clone https://github.com/Qiskit/qiskit.git qiskit-src   # `qiskit-src`, not `qiskit`
cd qiskit-src
git checkout 848178940d2def0dbdee578d20ce5e6b3451f4c2       # tag 2.2.0rc1
git apply "$REPO/patches/qiskit/0001-dqc-aware-sabre-variants.patch"

pip install -e . -c /tmp/frozen.txt
python setup.py build_rust --release --inplace   # editable installs build Rust in DEBUG otherwise
```

The base is `848178940` (`2.2.0rc1`), **not** the final `2.2.0` tag — the patch
will not apply cleanly there. The directory must not be called `qiskit`:
[a same-named directory shadows the real package](BUILD-NOTES.md#why-the-qiskit-checkout-is-named-qiskit-src).

Verify — the same two checks the Docker build runs:

```bash
PYTHONSAFEPATH=1 python -c "
import qiskit
assert qiskit.__file__ is not None, 'shadowed by a namespace package'
assert '/qiskit-src/' in qiskit.__file__, f'resolved to {qiskit.__file__}'
print('qiskit OK:', qiskit.__file__, qiskit.__version__)
"

git apply --reverse --check "$REPO/patches/qiskit/0001-dqc-aware-sabre-variants.patch" \
    && echo "SABRE patch OK: applied in the tree the editable install points at"
cd ~/dqc-workspace
```

The first proves *location* only; the second proves the patch is still applied.
[Why both](BUILD-NOTES.md#what-the-build-time-checks-actually-prove).

## 4. Build and patch pytket-dqc

```bash
cd ~/dqc-workspace
git clone https://github.com/Quantinuum/pytket-dqc.git
cd pytket-dqc
git checkout bfa0b4eff5b77d0a9b3c260c7842e57e75091796
git apply "$REPO/patches/pytket-dqc/0001-superconducting-topology-awareness.patch"
cp "$REPO/patches/pytket-dqc/basic_usage_demo.ipynb" examples/basic_usage.ipynb
pip install -c /tmp/frozen.txt -e ".[tests]"

git apply --reverse --check "$REPO/patches/pytket-dqc/0001-superconducting-topology-awareness.patch" \
    && echo "pytket-dqc patch OK"
cd ~/dqc-workspace
```

No rename needed here: the import name `pytket_dqc` can't collide with a
`pytket-dqc` directory.

## 5. Install QIG partitioning

```bash
pip install --no-deps -e "$REPO/qig-partitioning"
```

`--no-deps` matters:
[a plain install would clobber the KaHyPar you built in step 1](BUILD-NOTES.md#why-qig-partitioning-installs-with---no-deps).
Its other dependencies are already present. (Installing `qig-partitioning`
*standalone*, outside this environment, a plain `pip install -e` is correct.)

## 6. Check it works

```bash
python -c "import qiskit, pytket_dqc, kahypar, qig_partitioning; print('imports OK')"
python -c "from qiskit.transpiler.passes.routing import SabreSwap; print('SabreSwap OK')"
```

Then run something that actually calls into KaHyPar — a
`qig_partitioning.partition_with_kahypar(...)` call or a pytket-dqc
distribution. Imports alone don't rule out the broken-wheel failure from step 1,
since it only surfaces at partition time. `notebooks/usage_demo.ipynb` exercises
the full pipeline.

See [`BUILD-NOTES.md`](BUILD-NOTES.md#verification-status) for what has and
hasn't been verified on this path.
