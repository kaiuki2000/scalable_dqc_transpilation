# syntax=docker/dockerfile:1
#
# One shared environment with BOTH patched forks installed together: the paper's
# experiments need Qiskit and pytket-dqc importable side by side (e.g. the
# Section V-B hybrid approach pairs pytket-dqc's allocators with CLA-SABRE), not
# in separate environments.
#
# Also replicates the KaHyPar install pytket-dqc's own README recommends: KaHyPar
# is built from source at the pinned v1.3.2 tag and its compiled .so is dropped
# straight into site-packages
# (https://github.com/Quantinuum/pytket-dqc#kahypar-with-python-interface).
# This is deliberately NOT `pip install kahypar` -- the PyPI wheel doesn't carry
# the per-block custom capacity feature pytket-dqc's allocator relies on, and a
# manually-copied .so has no dist-info, so it won't show up in `pip freeze`
# either (matching what you saw on your own environment).
#
# Build (from the repo root, so this can COPY patches/, env/, and qig-partitioning/):
#   docker build -f docker/Dockerfile -t dqc-env .
# Run:
#   docker run -it --rm dqc-env
#
# First build compiles Rust (Qiskit) and KaHyPar from source -- expect 20-30+
# minutes, not a hang.
#
# NOT build-tested in this repo yet (the environment this Dockerfile was written
# in has no Docker daemon available) -- treat as a draft to verify before relying
# on it.
#
# Testing (after building, from a shell inside `docker run -it --rm dqc-env`):
#   python -c "import qiskit; print(qiskit.__version__)"
#   python -c "from qiskit.transpiler.passes.routing import SabreSwap; print('ok')"
#   python -c "import pytket_dqc; print('ok')"
#   python -c "import kahypar; print('ok')"   # confirms the source-built .so is
#                                              # picked up, not a PyPI wheel
#   python -c "import qig_partitioning; print('ok')"
# The kahypar import alone doesn't prove the source build is actually correct --
# the broken PyPI wheel (see "pytket-dqc prerequisites" below) imports fine too,
# it just fails at partition time. Actually calling
# qig_partitioning.partition_with_kahypar(...) or a pytket-dqc distribution
# exercises the real code path and would catch that failure mode.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.cargo/bin:${PATH}"

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

# Qiskit's routing/layout core (what the SABRE patch touches) is Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

RUN pip install --no-cache-dir --upgrade pip setuptools wheel maturin

# --- KaHyPar v1.3.2, built from source per pytket-dqc's README ---
# This repo's own README (see "pytket-dqc prerequisites") documents this exact
# recipe as verified: system libboost-program-options-dev (installed above),
# not KaHyPar's KAHYPAR_USE_MINIMAL_BOOST fallback. Also verified there: the
# PyPI kahypar wheel (>=1.3.5) is actually incompatible with this pytket-dqc
# version ("Vertices with weight 0 are not supported"), not just missing a
# feature -- building from source is required, not optional.
RUN git clone --depth=1 --recursive --branch 1.3.2 https://github.com/kahypar/kahypar.git /opt/kahypar-src \
    && mkdir /opt/kahypar-src/build \
    && cd /opt/kahypar-src/build \
    && cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DKAHYPAR_PYTHON_INTERFACE=1 \
    && cd python && make -j"$(nproc)" \
    && cp kahypar*.so "$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && rm -rf /opt/kahypar-src

WORKDIR /opt

# --- Constraints from the environment the paper's results were produced on ---
# env/requirements-freeze.txt is a `pip freeze` of the laptop environment used
# to generate the paper's results, with both forks installed together. Using
# it as constraints (rather than installing it directly) pins third-party
# versions like numpy/scipy to ones already confirmed to work for both forks
# at once, while skipping its local/editable lines for qiskit, pytket-dqc, and
# pytket-qiskit -- those are built from the pinned patch commits below instead
# (the checked-out qiskit commit that file records is a few commits past the
# paper's pinned patch tip, from later work on the same branch -- confirmed by
# diff that this doesn't affect the shipped patch itself, which is byte-for-
# byte the diff between the pinned base commit and the paper's exact tip).
COPY env/requirements-freeze.txt /tmp/requirements-freeze.txt
RUN grep -v -e '^-e ' -e '@ file://' /tmp/requirements-freeze.txt > /tmp/constraints.txt

# --- Patched Qiskit ---
RUN git clone --filter=blob:none https://github.com/Qiskit/qiskit.git qiskit \
    && cd qiskit \
    && git checkout 848178940d2def0dbdee578d20ce5e6b3451f4c2

COPY patches/qiskit/0001-dqc-aware-sabre-variants.patch /tmp/
RUN cd qiskit && git apply /tmp/0001-dqc-aware-sabre-variants.patch

# --release matters here: a debug build of the Rust extension is much slower at
# runtime, which would understate the paper's routing-time results.
RUN cd qiskit && pip install --no-cache-dir -e . -C build-args="--release" -c /tmp/constraints.txt

# Extra packages the paper's experiments were run with, on top of Qiskit's own deps.
COPY env/requirements.sh /tmp/requirements.sh
RUN bash /tmp/requirements.sh

# --- Patched pytket-dqc ---
RUN git clone --filter=blob:none https://github.com/Quantinuum/pytket-dqc.git pytket-dqc \
    && cd pytket-dqc \
    && git checkout bfa0b4eff5b77d0a9b3c260c7842e57e75091796

COPY patches/pytket-dqc/0001-superconducting-topology-awareness.patch /tmp/
RUN cd pytket-dqc && git apply /tmp/0001-superconducting-topology-awareness.patch

COPY patches/pytket-dqc/basic_usage_demo.ipynb /opt/pytket-dqc/examples/basic_usage.ipynb

RUN pip install --no-cache-dir -c /tmp/constraints.txt -e "/opt/pytket-dqc[tests]"

# --- QIG partitioning (Section V-C pre-processing step; standalone package) ---
# --no-deps: qig-partitioning's own pyproject.toml lists a plain `kahypar` PyPI
# dependency, which would otherwise pull in a prebuilt wheel here and silently
# overwrite the source-built KaHyPar v1.3.2 .so above with an unrelated version.
# qig-partitioning only calls KaHyPar's standard Hypergraph/Context/partition API
# (checked against qig_partitioning/partitioning.py), which the source build
# already provides, so skipping the redundant install is safe; its other deps
# (qiskit, numpy) are already installed by this point.
COPY qig-partitioning /opt/qig-partitioning
RUN pip install --no-cache-dir --no-deps -e /opt/qig-partitioning

COPY notebooks /opt/notebooks
COPY MODIFICATIONS.md /opt/

WORKDIR /opt
CMD ["/bin/bash"]
