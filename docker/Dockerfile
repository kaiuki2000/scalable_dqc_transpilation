# syntax=docker/dockerfile:1
#
# One environment with BOTH patched forks installed side by side, plus
# qig-partitioning -- what the paper's experiments need (the Section V-B hybrid
# approach uses Qiskit and pytket-dqc together).
#
# Build (from the repo root -- it needs patches/, env/ and qig-partitioning/):
#   docker build -f docker/Dockerfile -t dqc-env .
# Run:
#   docker run -it --rm dqc-env
#
# First build compiles Rust (Qiskit) and KaHyPar from source -- expect 20-30+
# minutes, not a hang.
#
# The step order here is load-bearing and the pins are not arbitrary. Each
# comment below says what a step avoids; docs/BUILD-NOTES.md says why, and
# records what has actually been verified. docs/INSTALL.md is this same
# sequence outside a container.

# 3.12.3 matches env/requirements-freeze.txt. The -bookworm pin is required,
# not cosmetic: GCC 14 (trixie, the default for unqualified tags) breaks
# KaHyPar's pybind11, and CMake >= 4 breaks its vendored googletest.
# BUILD-NOTES.md#kahypar-132-does-not-compile-under-python-311-as-shipped
FROM python:3.12.3-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.cargo/bin:${PATH}"

# graphviz+headers: pygraphviz, a pytket-dqc dependency.
# libboost-program-options-dev: KaHyPar (system Boost, not its minimal fallback).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        curl \
        ca-certificates \
        graphviz \
        libgraphviz-dev \
        pkg-config \
        libboost-program-options-dev \
    && rm -rf /var/lib/apt/lists/*

# Qiskit's routing/layout core -- what the SABRE patch touches -- is Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- KaHyPar 1.3.2, from source ---
# Required, not optional: the PyPI wheel is >=1.3.5, which fails at partition
# time with "Vertices with weight 0 are not supported" against this pytket-dqc.
# Pinned by commit (tag 1.3.2) since tags can move.
#
# The vendored pybind11 (a6355b00, 2019) does not compile under Python >=3.11,
# so it is swapped for v2.13.6. Do NOT "fix" this by bumping KaHyPar instead --
# 1.3.5+ is exactly where the wheel regression above lives.
#
# The .so is copied by hand per pytket-dqc's own README recipe, so KaHyPar will
# not appear in `pip freeze` afterwards. Expected.
#
# Portability: RELEASE flags include -march=native, so this .so is tuned to the
# building CPU. See BUILD-NOTES.md#two-more-kahypar-build-hazards-cmake-4-and--marchnative
# before running this image on different hardware than it was built on.
RUN git clone --filter=blob:none https://github.com/kahypar/kahypar.git /opt/kahypar-src \
    && cd /opt/kahypar-src \
    && git checkout efa12a90acae05469520d43cff122ca8620fab48 \
    && git submodule update --init --recursive \
    && rm -rf python/pybind11 \
    && git clone --depth 1 --branch v2.13.6 https://github.com/pybind/pybind11.git python/pybind11 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1 \
    && cd python && make -j"$(nproc)" \
    && cp kahypar*.so "$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && rm -rf /opt/kahypar-src

WORKDIR /opt

# --- Frozen environment, installed BEFORE the patched forks ---
# A real install, not just constraints. It deliberately pulls in a stock qiskit
# so pytket-qiskit==0.73.0's `qiskit>=2.2.3` is satisfiable (our fork reports
# 2.2.0rc1); the patched editable install below then replaces it by name.
# Reordering these two steps yields an environment that imports fine and is
# quietly wrong. BUILD-NOTES.md#why-the-bulk-install-runs-before-the-patched-qiskit
#
# The grep strips the freeze's local qiskit/pytket-dqc lines -- both are built
# from their pinned patch commits below. /tmp/frozen.txt is reused as -c
# further down so later installs stay on this pinned set.
COPY env/requirements-freeze.txt /tmp/requirements-freeze.txt
RUN grep -v -e '^-e ' -e '@ file://' /tmp/requirements-freeze.txt > /tmp/frozen.txt
RUN pip install --no-cache-dir -r /tmp/frozen.txt

# --- Patched Qiskit ---
# `qiskit-src`, NOT `qiskit`: a cwd-adjacent directory named `qiskit` is picked
# up as an implicit namespace package and silently beats the editable install,
# for every session started from /opt.
# BUILD-NOTES.md#why-the-qiskit-checkout-is-named-qiskit-src
RUN git clone --filter=blob:none https://github.com/Qiskit/qiskit.git qiskit-src \
    && cd qiskit-src \
    && git checkout 848178940d2def0dbdee578d20ce5e6b3451f4c2

COPY patches/qiskit/0001-dqc-aware-sabre-variants.patch /tmp/
RUN cd qiskit-src && git apply /tmp/0001-dqc-aware-sabre-variants.patch

RUN cd qiskit-src && pip install --no-cache-dir -e . -c /tmp/frozen.txt
# Editable installs always build the Rust extension in DEBUG; there is no
# config-settings hook to change that. A debug build would understate the
# paper's routing-time results, so re-build in release per Qiskit's
# CONTRIBUTING.md. BUILD-NOTES.md#why-qiskit-needs-an-extra-release-build-step
RUN cd qiskit-src && python setup.py build_rust --release --inplace

# Location check: proves the active qiskit is the patched editable install and
# that the compiled Rust comes from the in-place build. PYTHONSAFEPATH=1 keeps
# the cwd off sys.path so namespace shadowing cannot fool it.
RUN PYTHONSAFEPATH=1 python -c "\
import qiskit; \
loc = getattr(qiskit, '__file__', None); \
assert loc is not None, f'qiskit imported as a namespace package from {list(qiskit.__path__)} -- the editable install is not active'; \
assert loc.startswith('/opt/qiskit-src/'), f'qiskit resolved to {loc}, not the patched editable install under /opt/qiskit-src'; \
import qiskit._accelerate as acc; \
aloc = getattr(acc, '__file__', None); \
assert aloc is not None and aloc.startswith('/opt/qiskit-src/'), f'qiskit._accelerate loaded from {aloc}, not the in-place build under /opt/qiskit-src'; \
print('qiskit OK:', loc, qiskit.__version__)"

# Patch-presence check: the location check above would pass on a stock checkout
# at the right commit. --reverse --check succeeds only if the patch is applied,
# and needs no knowledge of what the patch contains.
# BUILD-NOTES.md#what-the-build-time-checks-actually-prove
RUN cd qiskit-src && git apply --reverse --check /tmp/0001-dqc-aware-sabre-variants.patch \
    && echo "SABRE patch OK: applied in the tree the editable install points at"

# --- Patched pytket-dqc ---
# No rename needed: import name `pytket_dqc` cannot collide with `pytket-dqc`.
RUN git clone --filter=blob:none https://github.com/Quantinuum/pytket-dqc.git pytket-dqc \
    && cd pytket-dqc \
    && git checkout bfa0b4eff5b77d0a9b3c260c7842e57e75091796

COPY patches/pytket-dqc/0001-superconducting-topology-awareness.patch /tmp/
RUN cd pytket-dqc && git apply /tmp/0001-superconducting-topology-awareness.patch

COPY patches/pytket-dqc/basic_usage_demo.ipynb /opt/pytket-dqc/examples/basic_usage.ipynb

RUN pip install --no-cache-dir -c /tmp/frozen.txt -e "/opt/pytket-dqc[tests]"

RUN cd pytket-dqc && git apply --reverse --check /tmp/0001-superconducting-topology-awareness.patch \
    && echo "pytket-dqc patch OK: applied in the tree the editable install points at"

# --- QIG partitioning (paper Section V-C) ---
# --no-deps: its pyproject.toml declares a plain `kahypar` dependency, which
# would pull a wheel and clobber the source-built .so above. It only uses
# KaHyPar's standard Hypergraph/Context/partition API, and its other deps are
# already installed. BUILD-NOTES.md#why-qig-partitioning-installs-with---no-deps
COPY qig-partitioning /opt/qig-partitioning
RUN pip install --no-cache-dir --no-deps -e /opt/qig-partitioning

COPY notebooks /opt/notebooks
COPY MODIFICATIONS.md /opt/

# Keep /opt free of directories named after importable modules -- see the
# qiskit-src note above. qiskit-src, pytket-dqc and qig-partitioning are all
# safe (import names: qiskit, pytket_dqc, qig_partitioning).
WORKDIR /opt

CMD ["/bin/bash"]
