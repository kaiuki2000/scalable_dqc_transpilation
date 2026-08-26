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
# either (matching what you saw on your own environment). One deviation from
# that README recipe is needed to build on Python 3.12: the vendored pybind11
# submodule is replaced. See the KaHyPar step below.
#
# Build (from the repo root, so this can COPY patches/, env/, and qig-partitioning/):
#   docker build -f docker/Dockerfile -t dqc-env .
# Run:
#   docker run -it --rm dqc-env
#
# First build compiles Rust (Qiskit) and KaHyPar from source -- expect 20-30+
# minutes, not a hang.
#
# Build-tested iteratively against a real Docker daemon (not just inspected),
# and this revision has now completed a full build end to end. History worth
# keeping, because both failures along the way were misleading at first
# glance: an earlier revision failed at the KaHyPar step under Python
# 3.11/3.12 (fixed by the pybind11 swap below, not by downgrading Python), and
# the revision after that failed at the qiskit provenance check with a
# TypeError that had nothing to do with qiskit being installed wrongly (fixed
# by the qiskit-src rename below).
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

# Python 3.12.3, matching the environment the paper's later (SLURM) experiments
# actually ran on. Getting there needs the pybind11 swap in the KaHyPar step
# below: KaHyPar 1.3.2 vendors pybind11 at commit a6355b00 (~v2.4.dev4, Oct
# 2019), which predates pybind11's Python 3.11 support. CPython 3.11 made
# PyFrameObject's internals opaque; that pybind11 version dereferences it
# directly and unconditionally in cast.h's error_string()
# (`frame->f_code->co_filename`), with no version guard at all -- confirmed by
# reading the source at that exact pinned commit, not inferred. Without the
# swap, the KaHyPar step fails to compile on any Python >= 3.11 with "invalid
# use of incomplete type 'PyFrameObject'" (verified empirically under both 3.11
# and 3.12; 3.10 builds clean). Note that the Python version is the trigger, not
# the cause: the cause is the pybind11 vintage that the pinned KaHyPar commit
# happens to record.
#
# -bookworm: KaHyPar's pybind11 separately relies on <cstdint> being pulled in
# transitively by another standard header -- true on older GCC/libstdc++
# (Debian 12 "bookworm", GCC 12.2), no longer true as of GCC 14 (Debian 13
# "trixie", the current default for an unqualified `python:*-slim` tag), where
# attr.h fails with "'uint16_t' in namespace 'std' does not name a type".
# Pinning the Debian release also stops the base image from silently changing
# compiler versions under this Dockerfile in the future.
#
# The Debian pin is load-bearing for CMake too, for a failure this build hasn't
# hit but is one base-image bump away from. Bookworm ships CMake 3.25.x. The
# vendored googletest submodule declares `cmake_minimum_required(VERSION 2.6.2)`
# and CMake 4.0 (March 2025) removed compatibility with anything below 3.5 --
# so on CMake >= 4 the deprecation warnings this build currently prints become
# hard configure-time errors, pointing at a submodule that has nothing to do
# with anything here. If you ever build KaHyPar outside Docker on a machine with
# CMake >= 4 (Ubuntu 24.04+, current Homebrew), pass
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 to restore the old behaviour rather than
# editing submodule files.
FROM python:3.12.3-slim-bookworm

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

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- KaHyPar v1.3.2, built from source per pytket-dqc's README ---
# This repo's own README (see "pytket-dqc prerequisites") documents this recipe
# as verified: system libboost-program-options-dev (installed above), not
# KaHyPar's KAHYPAR_USE_MINIMAL_BOOST fallback. Also verified there: the PyPI
# kahypar wheel (>=1.3.5) is actually incompatible with this pytket-dqc version
# ("Vertices with weight 0 are not supported"), not just missing a feature --
# building from source is required, not optional.
#
# Pinned by commit, not just the 1.3.2 tag (which happens to point here right
# now): tags can be moved, unlike commits, so this matches how Qiskit and
# pytket-dqc are pinned elsewhere in this repo -- see "Working on this repo"
# in CLAUDE.md.
#
# The vendored pybind11 is then replaced with v2.13.6. KaHyPar records its
# pybind11 as a submodule, and the version recorded at each tag is what decides
# whether this compiles under a modern CPython:
#
#   KaHyPar 1.3.2 / v.1.3.3  -> pybind11 a6355b00 (Oct 2019)  -- breaks on 3.11+
#   KaHyPar v1.3.5           -> pybind11 c9638a19 (Sep 2023)
#   KaHyPar v1.3.6 / v1.3.7  -> pybind11 2c9191e9 (Jan 2026)
#
# So bumping KaHyPar itself would also fix the compile -- and must NOT be done
# here, because 1.3.5+ is exactly where the "Vertices with weight 0" regression
# above lives. Replacing only the binding layer keeps KaHyPar's own C++ at the
# version pytket-dqc needs. v2.13.6 specifically: last of the 2.x line, supports
# Python 3.7-3.13, and keeps the classic CMake behaviour that KaHyPar 1.3.2's
# python/CMakeLists.txt expects (pybind11 3.x defaults to FindPython instead).
# The swap is drop-in because python/module.cpp uses nothing beyond
# PYBIND11_MODULE / py::init / py::arg, all stable across 2.4 -> 2.13.
#
# NOTE on portability: KaHyPar's RELEASE flags include -march=native, so the .so
# produced here is tuned to whichever CPU ran the build. If this image is ever
# pushed and pulled onto different hardware (or built on a laptop and run on the
# cluster), that can SIGILL at import time. Add
# -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -mtune=generic" to the cmake line below
# to trade some partitioning speed for a portable image.
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

# --- Bulk-install the frozen SLURM environment (minus what we build ourselves) ---
# env/requirements-freeze-slurm.txt is a `pip freeze` of a later, SLURM-cluster
# environment (post-submission; env/requirements-list-slurm.txt is its `pip
# list` companion). Used here instead of env/requirements-freeze.txt because,
# per the notebook documenting how that environment was set up, it's the one
# that includes pytket-qiskit -- needed for the Section V-B hybrid approach.
#
# This is installed for real (`pip install -r`), not just used as version
# constraints. That's deliberate: it lets this pull in an ordinary, unpatched
# qiskit release here, which is needed to satisfy pytket-qiskit==0.73.0's own
# `qiskit>=2.2.3` requirement (checked against PyPI metadata) -- our patched
# fork reports 2.2.0rc1 (qiskit/VERSION.txt at the pinned commit, unaffected by
# the patch), which fails that check as a plain version comparison. That's fine
# here: qiskit is installed explicitly, by name, from source further below,
# *after* this step -- pip's explicit "install this exact local package"
# unconditionally replaces whatever's currently installed under that name, so
# the patched editable install ends up the one actually active, while
# pytket-qiskit/qiskit-aer/qiskit-ibm-runtime still got a real qiskit to
# install successfully against in the meantime. Checked: nothing installed
# *after* qiskit in this file -- pytket-dqc, qig-partitioning -- declares any
# qiskit or pytket-qiskit requirement of its own, so nothing downstream can
# undo this again; the import checks right after the qiskit build below
# verify it actually held.
#
# Excluded: the two local/editable lines this file records (qiskit itself, and
# pytket-dqc) -- both are built from the pinned patch commits below instead.
# Reused as -c further down, so later installs stay aligned with this same
# pinned set instead of drifting.
COPY env/requirements-freeze-slurm.txt /tmp/requirements-freeze-slurm.txt
RUN grep -v -e '^-e ' -e '@ file://' /tmp/requirements-freeze-slurm.txt > /tmp/frozen.txt
RUN pip install --no-cache-dir -r /tmp/frozen.txt

# --- Patched Qiskit ---
# Cloned as `qiskit-src`, NOT `qiskit`, and the name matters. This lands under
# WORKDIR /opt, and Python puts the cwd at sys.path[0]. A cwd-adjacent directory
# named `qiskit` with no __init__.py inside it gets picked up by PathFinder as an
# implicit namespace package -- and that wins outright, because setuptools
# registers editable installs through a meta_path finder which it *appends* to
# sys.meta_path (i.e. behind PathFinder), so the finder never gets consulted at
# all. The symptom is `import qiskit` appearing to succeed while
# qiskit.__file__ is None and `from qiskit import QuantumCircuit` fails with a
# confusing AttributeError. That would hit every interactive session and Jupyter
# kernel started from /opt, not just the build-time check below -- the rename is
# a correctness fix for users of this image, not just a fix for the check.
RUN git clone --filter=blob:none https://github.com/Qiskit/qiskit.git qiskit-src \
    && cd qiskit-src \
    && git checkout 848178940d2def0dbdee578d20ce5e6b3451f4c2

COPY patches/qiskit/0001-dqc-aware-sabre-variants.patch /tmp/
RUN cd qiskit-src && git apply /tmp/0001-dqc-aware-sabre-variants.patch

# A plain `pip install -e .` always builds the Rust extension in debug mode for
# editable installs (setuptools-rust decides debug vs. release from whether the
# install is editable, not from pip's -C/--config-settings -- there is no PEP 517
# config-settings hook for this, so a maturin-style `-C build-args="--release"`
# is silently accepted and silently ignored). Release matters here: a debug
# build is much slower at runtime, which would understate the paper's
# routing-time results. Per Qiskit's own CONTRIBUTING.md, getting a release
# build with an editable install requires this explicit second step.
RUN cd qiskit-src && pip install --no-cache-dir -e . -c /tmp/frozen.txt
# `pip install -e .` above installs setuptools-rust into a throwaway PEP 517
# build-isolation environment (it's only declared as a [build-system] require in
# pyproject.toml, not a normal dependency) -- that doesn't persist. No extra
# install is needed here, though: the frozen bulk install above already put
# setuptools-rust==1.12.0 in the main environment before we ever got here.
RUN cd qiskit-src && python setup.py build_rust --release --inplace

# Confirms the replacement described above actually happened: fails the build
# loudly, instead of quietly shipping a stock qiskit that still imports fine
# but silently dropped the SABRE patch, if anything above ends up installing
# over the editable source instead of alongside it.
#
# PYTHONSAFEPATH=1 (Python 3.11+) keeps the cwd off sys.path entirely, so this
# check cannot be fooled by the namespace-package shadowing described above even
# if some future directory name under /opt collides again. The explicit
# `is not None` test exists because a shadowed import gives qiskit.__file__ ==
# None, which would otherwise blow up inside os.path/str handling with a
# TypeError that says nothing about what actually went wrong.
#
# The qiskit._accelerate check confirms the compiled Rust extension is being
# loaded from the source tree (where `build_rust --inplace` wrote it), rather
# than from a stray copy in site-packages. It confirms location only -- it does
# not distinguish the release build from the debug one that `pip install -e .`
# produced first and that this overwrote.
RUN PYTHONSAFEPATH=1 python -c "\
import qiskit; \
loc = getattr(qiskit, '__file__', None); \
assert loc is not None, f'qiskit imported as a namespace package from {list(qiskit.__path__)} -- the editable install is not active'; \
assert loc.startswith('/opt/qiskit-src/'), f'qiskit resolved to {loc}, not the patched editable install under /opt/qiskit-src'; \
import qiskit._accelerate as acc; \
aloc = getattr(acc, '__file__', None); \
assert aloc is not None and aloc.startswith('/opt/qiskit-src/'), f'qiskit._accelerate loaded from {aloc}, not the in-place build under /opt/qiskit-src'; \
print('qiskit OK:', loc, qiskit.__version__)"

# The checks above are all about *location*: they prove the active qiskit is the
# editable install pointing at /opt/qiskit-src. They say nothing about whether
# the SABRE patch is still in that tree -- a stock checkout at the right commit
# would pass every one of them. `git apply --reverse --check` succeeds only if
# the patch is currently applied, which is exactly the missing question, and it
# needs no knowledge of which symbols the patch introduces (so it keeps working
# when the patch changes).
RUN cd qiskit-src && git apply --reverse --check /tmp/0001-dqc-aware-sabre-variants.patch \
    && echo "SABRE patch OK: applied in the tree the editable install points at"

# Extra packages the paper's experiments were run with, on top of Qiskit's own
# deps, used to live here via env/requirements.sh -- no longer invoked: every
# package it lists (ipykernel, matplotlib, networkx, pydot, pillow,
# pylatexenc, qiskit-ibm-runtime, setuptools-rust) is already covered by the
# frozen bulk install above. env/requirements.sh itself is unchanged and still
# used by the manual, non-Docker setup path (see README) -- this only drops
# its now-redundant invocation here.

# --- Patched pytket-dqc ---
# Directory name is safe as-is: the import name is `pytket_dqc` (underscore),
# which cannot collide with the `pytket-dqc` directory the way `qiskit` did.
RUN git clone --filter=blob:none https://github.com/Quantinuum/pytket-dqc.git pytket-dqc \
    && cd pytket-dqc \
    && git checkout bfa0b4eff5b77d0a9b3c260c7842e57e75091796

COPY patches/pytket-dqc/0001-superconducting-topology-awareness.patch /tmp/
RUN cd pytket-dqc && git apply /tmp/0001-superconducting-topology-awareness.patch

COPY patches/pytket-dqc/basic_usage_demo.ipynb /opt/pytket-dqc/examples/basic_usage.ipynb

RUN pip install --no-cache-dir -c /tmp/frozen.txt -e "/opt/pytket-dqc[tests]"

RUN cd pytket-dqc && git apply --reverse --check /tmp/0001-superconducting-topology-awareness.patch \
    && echo "pytket-dqc patch OK: applied in the tree the editable install points at"

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

# Keep /opt free of any directory named after an importable module -- see the
# qiskit-src note above for what happens otherwise. Current contents are safe:
# qiskit-src, pytket-dqc and qig-partitioning all differ from their import names
# (`qiskit`, `pytket_dqc`, `qig_partitioning`).
WORKDIR /opt
CMD ["/bin/bash"]