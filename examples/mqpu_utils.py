# Auxiliary functions for generating and manipulating multi-QPU targets in Qiskit.
# Multi-QPU Heron r1 target generation and manipulation utilities.

from dataclasses import dataclass
import networkx as nx
import numpy as np
import random

import pytket_dqc
from pytket_dqc.utils.gateset import start_proc
from pytket_dqc.circuits.distribution import is_robust_start_proc
from pytket import Circuit, OpType, Qubit
from pytket.circuit import CustomGateDef

from qiskit_aer import AerSimulator
from qiskit.circuit import (
    Measure, Delay, Parameter,
    Reset, QuantumCircuit, IfElseOp,
    ClassicalRegister, QuantumRegister, Gate
)
from qiskit.circuit.library import (
    XGate, SXGate, RZGate,
    CZGate, IGate, CXGate, Barrier
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Target, InstructionProperties, Layout, PassManager
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes import (
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout,
    SabreSwap,
    SabreLayout,
    Unroll3qOrMore,
    RemoveBarriers,
    RemoveFinalMeasurements,
    BarrierBeforeFinalMeasurements,
)
from qiskit.providers import BackendV2, Options
from qiskit.providers.fake_provider import GenericBackendV2

from pytket_dqc import NISQNetwork
from pytket_dqc.utils import DQCPass
from pytket_dqc.distributors import (
    CoverEmbedding,
    CoverEmbeddingSteinerDetached,
    PartitioningHeterogeneousEmbedding
)
from pytket_dqc.refiners import (
    NeighbouringDTypeMerge,
    IntertwinedDTypeMerge,
    SequenceRefiner,
    RepeatRefiner,
)

from pytket.extensions.qiskit.qiskit_convert import qiskit_to_tk
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements, Unroll3qOrMore

############################################################################################
###### Flattening Nested Lists #############################################################
############################################################################################

def flatten_pairs(nested_list):
    """Flatten a nested list of coordinate pairs into a single list."""
    return [pair for group in nested_list for pair in group]

############################################################################################
###### Heron R1 Coordinates Generation #####################################################
############################################################################################

def generate_heron_r1_coordinates(n_qpus: int = 1, m_offset: int = 5, n_offset: int = 15):
    """
    Generate coordinates for a 3-Heron QPU layout with L-couplers.

    Args:
        n_qpus (int): Number of QPUs.
        m_offset (int): Offset for the row coordinates.
        n_offset (int): Offset for the column coordinates.

    Returns:
        List of coordinates for each QPU in the format [[row, col], ...].
    """
    if n_qpus < 1:
        raise ValueError("Number of QPUs must be at least 1.")

    mqpu_coordinates = []

    row_patterns = [
        (0, range(15)),
        (1, range(0, 15, 4)),
        (2, range(15)),
        (3, range(2, 15, 4)),
        (4, range(15)),
        (5, range(0, 15, 4)),
        (6, range(15)),
        (7, range(2, 15, 4)),
        (8, range(15)),
        (9, range(0, 15, 4)),
        (10, range(15)),
        (11, range(2, 15, 4)),
        (12, range(15)),
        (13, range(0, 15, 4)),
    ]

    for qpu_index in range(n_qpus):
        qpu_coords = [
            [ [row + qpu_index * m_offset, col + qpu_index * n_offset] for col in cols ]
            for row, cols in row_patterns
        ]
        mqpu_coordinates.append(flatten_pairs(qpu_coords))

    return flatten_pairs(mqpu_coordinates)

############################################################################################
###### Multi-QPU Coordinates Generation ####################################################
############################################################################################

def generate_mqpu_coordinates_from_monolithic_coordinates(n_qpus: int = 1, monolithic_coordinates: list = None, m_offset: int = 5, n_offset: int = 15):
    """
    Generate multi-QPU coordinates from monolithic coordinates.

    Args:
        n_qpus (int): Number of QPUs.
        monolithic_coordinates (list): List of monolithic coordinates.
        m_offset (int): Offset for the row coordinates.
        n_offset (int): Offset for the column coordinates.

    Returns:
        List of multi-QPU coordinates.
    """
    if n_qpus < 1:
        raise ValueError("Number of QPUs must be at least 1.")

    mqpu_coordinates = []

    for qpu_index in range(n_qpus):
        for coord in monolithic_coordinates:
            shifted_coord = [coord[0] + qpu_index * m_offset, coord[1] + qpu_index * n_offset]
            mqpu_coordinates.append(shifted_coord)

    return mqpu_coordinates

############################################################################################
###### Coupling Map Shifting ###############################################################
############################################################################################

def shift_coupling_map(coupling_map, offset: int = 0):
    """
    Shifts the coupling map by the given offset.

    Parameters
    ----------
    coupling_map : list
        The coupling map to shift.
    offset : int
        The offset to shift the coupling map by.

    Returns
    -------
    list
        The shifted coupling map.
    """
    return [[i + offset, j + offset] for (i, j) in coupling_map]

############################################################################################
###### Coupling Map Generation #############################################################
############################################################################################

def generate_mqpu_coupling_map_from_monolithic_backend(
    n_qpus:                int = 1,
    monolithic_backend:    BackendV2 = None,
    coupling_map:          list = None,
    inter_qpu_connections: list = None,
    base_qubits:           list = None,
):
    """
    Generate a combined coupling map for multiple QPUs, based on a monolithic backend.
    Either a monolithic backend or a (monolithic) coupling map must be provided.

    Args:
        n_qpus (int): Number of QPUs to replicate.
        monolithic_backend (BackendV2): Backend representing a single QPU.
        coupling_map (list, optional): Custom coupling map for a single QPU. If None, uses the backend's coupling map.
        inter_qpu_connections (list, optional): List of connections between QPUs.
        base_qubits (list, optional): Base qubit indices for inter-QPU connections. If None, defaults to None.

    Returns:
        CouplingMap: Combined coupling map including intra- and inter-QPU connections.
    """

    # Monolithic backend flag. Useful to avoid `generate_inter_qpu_links` function "Warning" messages. Ad-hoc solution.
    MONOLITHIC_BACKEND_FLAG = False

    if n_qpus < 1:
        raise ValueError("[generate_mqpu_coupling_map_from_monolithic_backend] Number of QPUs must be at least 1.")
    
    if not monolithic_backend: # If no backend is provided, use the provided coupling map.
        if coupling_map is None:
            raise ValueError("[generate_mqpu_coupling_map_from_monolithic_backend] Either a monolithic backend or a coupling map must be provided.")
        # Retrieve coupling map and number of qubits from the provided coupling map
        monolithic_coupling_map = coupling_map
        monolithic_num_qubits = max(max(pair) for pair in coupling_map) + 1

    else: # If a backend is provided, use its coupling map. If both are provided, use the user-provided coupling map.
        MONOLITHIC_BACKEND_FLAG = True
        if coupling_map is not None:
            print("[generate_mqpu_coupling_map_from_monolithic_backend] Warning: Both a monolithic backend and a coupling map were provided. Using the provided coupling map.")
            monolithic_coupling_map = coupling_map
            monolithic_num_qubits = max(max(pair) for pair in coupling_map) + 1
        else: # Only a backend is provided.
            if not isinstance(monolithic_backend, BackendV2):
                raise TypeError("[generate_mqpu_coupling_map_from_monolithic_backend] Monolithic backend must be an instance of BackendV2.")
            # Retrieve coupling map and number of qubits from the backend
            monolithic_coupling_map = monolithic_backend.coupling_map.get_edges()
            monolithic_num_qubits = monolithic_backend.num_qubits

    if base_qubits is None: # If base qubits is None, inter_qpu_connections must be provided.
        base_qubits = []
        if inter_qpu_connections is None:
            raise ValueError("[generate_mqpu_coupling_map_from_monolithic_backend] Either base_qubits or inter_qpu_connections must be provided.")
    else: # If both base_qubits and inter_qpu_connections are provided, use the provided inter_qpu_connections.
        if inter_qpu_connections is not None:
            print("[generate_mqpu_coupling_map_from_monolithic_backend] Warning: Both base_qubits and inter_qpu_connections were provided. Using the provided inter_qpu_connections.")
            inter_qpu_connections = inter_qpu_connections
        else: # Only base_qubits is provided.
            if MONOLITHIC_BACKEND_FLAG: # Avoid "Warning" message in `generate_inter_qpu_links` function. Use only one of coupling_map or monolithic_backend.
                inter_qpu_connections = generate_inter_qpu_links(n_qpus, base_qubits=base_qubits, monolithic_backend=monolithic_backend)[1]
            else:
                inter_qpu_connections = generate_inter_qpu_links(n_qpus, base_qubits=base_qubits, coupling_map=monolithic_coupling_map)[1]

    # Intra-QPU coupling maps with shifted indices
    coupling_map = [
        pair
        for qpu_index in range(n_qpus)
        for pair in shift_coupling_map(monolithic_coupling_map, offset=qpu_index * monolithic_num_qubits)
    ]

    # Add inter-QPU connections if provided
    if inter_qpu_connections:
        coupling_map.extend(inter_qpu_connections)

    return CouplingMap(coupling_map)

############################################################################################
###### Inter-QPU Links Generation ##########################################################
############################################################################################

def generate_inter_qpu_links(
    n_qpus:             int,
    base_qubits:        list,
    monolithic_backend: BackendV2 = None,
    coupling_map:       list = None,
    neighbor_offsets:   list = [1],
):
    """
    Generate inter-QPU links based on base qubit indices and neighbor distances.
    (By default, connects each base qubit to the corresponding base qubit in the next QPU.)
    (Linear inter-QPU connectivity is assumed by default.)

    Args:
        n_qpus (int): Number of QPUs.
        base_qubits (list): Base qubit indices (e.g., L coupler qubits).
        monolithic_backend (BackendV2): Backend representing a single QPU.
        coupling_map (list, optional): Custom coupling map for a single QPU. If None, uses the backend's coupling map.
        neighbor_offsets (list): List of neighbor distances to connect (e.g., [1, 2]).

    Returns:
        tuple: (shifted_qubits, inter_qpu_connections)
            - shifted_qubits (list): All shifted qubit indices across QPUs.
            - inter_qpu_connections (list): Bidirectional inter-QPU connections.
    """
    
    if n_qpus < 1:
        raise ValueError("[generate_inter_qpu_links] Number of QPUs must be at least 1.")

    if not monolithic_backend: # If no backend is provided, use the provided coupling map.
        if coupling_map is None:
            raise ValueError("[generate_inter_qpu_links] Either a monolithic backend or a coupling map must be provided.")
        num_qubits = max(max(pair) for pair in coupling_map) + 1

    else: # If a backend is provided, use its coupling map. If both are provided, use the user-provided coupling map.
        if coupling_map is not None:
            print("[generate_inter_qpu_links] Warning: Both a monolithic backend and a coupling map were provided. Using the provided coupling map.")
            num_qubits = max(max(pair) for pair in coupling_map) + 1
        else: # Only a backend is provided.
            if not isinstance(monolithic_backend, BackendV2):
                raise TypeError("[generate_inter_qpu_links] Monolithic backend must be an instance of BackendV2.")
            num_qubits = monolithic_backend.num_qubits

    # Shift base qubits across QPUs
    shifted_qubits = [
        q + offset * num_qubits
        for offset in range(n_qpus)
        for q in base_qubits
    ]

    inter_qpu_connections = []

    for offset in neighbor_offsets:
        step = offset * len(base_qubits)
        for i in range(len(shifted_qubits) - step):
            src = shifted_qubits[i]
            tgt = shifted_qubits[i + step]
            inter_qpu_connections.append((src, tgt))
            inter_qpu_connections.append((tgt, src))  # Bidirectional

    return shifted_qubits, inter_qpu_connections

###############################################################################################
###### CZ Error and Duration Retrieval ########################################################
###############################################################################################

def retrieve_median_cz_error(backend: BackendV2):
    return sorted([v.error for v in backend.target['cz'].values()])[len(backend.target['cz'].values()) // 2]

def retrieve_median_cz_duration(backend: BackendV2):
    return sorted([v.duration for v in backend.target['cz'].values()])[len(backend.target['cz'].values()) // 2]

def retrieve_avg_cz_error(backend: BackendV2):
    return sum(v.error for v in backend.target['cz'].values()) / len(backend.target['cz'].values())

def retrieve_avg_cz_duration(backend: BackendV2):
    return sum(v.duration for v in backend.target['cz'].values()) / len(backend.target['cz'].values())

###############################################################################################
###### CX Error and Duration Retrieval ########################################################
###############################################################################################

def retrieve_median_cx_error(backend: BackendV2):
    return sorted([v.error for v in backend.target['cx'].values()])[len(backend.target['cx'].values()) // 2]

def retrieve_median_cx_duration(backend: BackendV2):
    return sorted([v.duration for v in backend.target['cx'].values()])[len(backend.target['cx'].values()) // 2]

def retrieve_avg_cx_error(backend: BackendV2):
    return sum(v.error for v in backend.target['cx'].values()) / len(backend.target['cx'].values())

def retrieve_avg_cx_duration(backend: BackendV2):
    return sum(v.duration for v in backend.target['cx'].values()) / len(backend.target['cx'].values())

#############################################################################################
###### Multi-QPU Target Generation ##########################################################
#############################################################################################

def generate_multi_qpu_target(
    backend, # Monolithic backend representing a single QPU.
    n_qpus: int,
    inter_qpu_links: list,
    inter_qpu_error_rate: float,
    inter_qpu_duration: float,
):
    """
    Generate a multi-QPU target based on the provided backend and parameters.
    """

    # Auxiliary mapping of strictly physical/standard operations
    op_mapping = {
        'cx': CXGate(), # For Falcon (ibm_paris)
        'cz': CZGate(), # For Heron (ibm_torino)
        'x': XGate(),
        'sx': SXGate(),
        'rz': RZGate(Parameter('λ')),
        'reset': Reset(),
        'measure': Measure(),
        'delay': Delay(Parameter('t')),
        'id': IGate()
    }

    # Retrieve operation names from the backend, aggressively filtering out 
    # unsupported dynamic ops (like 'switch_case', 'if_else', 'for_loop')
    op_names = [op for op in backend.operation_names if op in op_mapping]
    op_instructions = [op_mapping[op] for op in op_names]

    # Number of qubits in the backend
    num_qubits = backend.num_qubits

    # Create the target object
    target = Target()

    # Add instructions to the target
    for op_name, op_instruction in zip(op_names, op_instructions):
        op_properties = {}
        base_properties = backend.target[op_name]

        if op_name in ['x', 'sx', 'delay', 'rz', 'reset', 'measure', 'id']:
            # Single-qubit gates
            for qpu_idx in range(n_qpus):
                shifted = {(k[0] + qpu_idx * num_qubits,): v for k, v in base_properties.items()}
                op_properties.update(shifted)
        else:
            # Multi-qubit gates (e.g., CZ, CX)
            for qpu_idx in range(n_qpus):
                shifted = {
                    (k[0] + qpu_idx * num_qubits, k[1] + qpu_idx * num_qubits): v
                    for k, v in base_properties.items()
                }
                op_properties.update(shifted)

            # Add the inter-QPU links
            inter_props = {
                (qa, qb): InstructionProperties(duration=inter_qpu_duration, error=inter_qpu_error_rate)
                for qa, qb in inter_qpu_links
            }
            op_properties.update(inter_props)

        target.add_instruction(op_instruction, op_properties)

    return target

#############################################################################################
###### Custom Backend Definition ############################################################
#############################################################################################

# Very primitive custom backend definition, just to test transpilation onto the custom target. (For the `plot_circuit_layout` function.)
class custom_mqpu_backend(BackendV2):
    """Custom-defined mQPU Backend."""

    def __init__(self, name: str, n_qpus: int, target: Target, inter_qpu_links=None, inter_qpu_error_rate=None, inter_qpu_duration=None):
        super().__init__(name=name)
        self._n_qpus = n_qpus
        self._target = target
        self._inter_qpu_links = inter_qpu_links
        self._inter_qpu_error_rate = inter_qpu_error_rate
        self._inter_qpu_duration = inter_qpu_duration
        # This `graph` should be a CouplingMap or similar structure, if needed. Tells you which qubits can perform two-qubit gates with each other.
        # And which is used as control and which is used as target. (If that's relevant for the gate.)
        # Removed: Can simply access `CouplingMap` with `custom_mqpu_backend.coupling_map`.

    # We don't define a setter for these properties, so they are read-only.
    @property
    def n_qpus(self):
        return self._n_qpus
    
    @property
    def target(self):
        return self._target # This is a read-only property, since there's no setter defined.
    
    @property
    def inter_qpu_links(self):
        return self._inter_qpu_links
    
    # Retrieve inter-QPU error rate.
    @property
    def inter_qpu_error_rate(self):
        return self._inter_qpu_error_rate

    # Retrieve inter-QPU duration.
    @property
    def inter_qpu_duration(self):
        return self._inter_qpu_duration

    @property
    def max_circuits(self):
        return None
 
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
    
    def run(self, circuit, **kwargs): # This'd be used to run the circuit on the backend. But then we'd need an actual physical device (or simulator).
        raise NotImplementedError(    # Since we're only testing transpilation, we don't need to implement this method.
            "This backend does not contain a run method"
        )
    
    # # Overriding the `plot_gate_map` method to visualize the gate map of the custom backend. Could be interesting to implement.

    # # Example usage:
    # ```
    # mqpu_coordinates = generate_mqpu_coordinates(n_qpus=n_qpus, monolithic_coordinates=FALCON_R4_QUBIT_COORDINATES, m_offset=5, n_offset=0.5)
    # inter_qpu_edges = generate_inter_qpu_links(n_qpus=n_qpus, base_qubits=base_qubits, monolithic_backend=monolithic_backend)[1]
    #
    # line_colors = ["#adaaab" for edge in mqpu_backend_paris.coupling_map.get_edges()]
    # for i, edge in enumerate(mqpu_backend_paris.coupling_map.get_edges()):
    #     if edge in inter_qpu_edges:
    #         line_colors[i] = "#000000"
    # ```

    # # Define line colors for inter-QPU connections (black) and intra-QPU connections (gray).
    # plot_gate_map(mqpu_backend_paris, qubit_coordinates=mqpu_coordinates, plot_directed=True, line_color=line_colors)
    # def plot_gate_map(self):
    #     return plot_gate_map(self.target)

###########################################################################################
###### Multi-QPU Backend Generation #######################################################
###########################################################################################

# Function that directly generates a multi-QPU backend. A monolithic backend must be provided, which represents a single QPU.
# Developed with 'ibm_torino' in mind, as we retrieve the median CZ error and duration.
# If another 2-qubit gate is used, the error and duration retrieval functions should be adapted accordingly. (Not yet supported.)
def generate_multi_qpu_backend_from_monolithic_backend_with_links(
    n_qpus: int = 3,
    monolithic_backend: BackendV2 = None, # Could add a `coupling_map` parameter, where the user would have the option to provide this instead of a `backend`.
    inter_qpu_links: list = None,
    backend_name: str = "mqpu_backend"
):
    """
    Generate a multi-QPU backend based on the provided parameters.

    Args:
        n_qpus (int): Number of QPUs.
        monolithic_backend (BackendV2): Backend representing a single QPU.
        inter_qpu_links (list): List of inter-QPU links (qubit pairs).
        backend_name (str): Name for the generated multi-QPU backend.

    Returns:
        custom_mqpu_backend: The generated multi-QPU backend.
    """
    if not monolithic_backend:
        raise ValueError("Monolithic backend must be provided.")
    if not isinstance(monolithic_backend, BackendV2):
        raise TypeError("Monolithic backend must be an instance of BackendV2.")
    if n_qpus < 1:
        raise ValueError("Number of QPUs must be at least 1.")
    if inter_qpu_links is None:
        raise ValueError("Inter-QPU links must be provided.")

    # # To be stored in the backend object, under the `graph` property. # This is unnecessary, as the `Target` object generates the `CouplingMap` internally (`target._build_coupling_graph`).
    # coupling_map = generate_mqpu_coupling_map_from_monolithic_backend(
    #     n_qpus=n_qpus,
    #     monolithic_backend=monolithic_backend,
    #     inter_qpu_connections=inter_qpu_links
    # ).get_edges()

    # 'ibm_torino' (Heron r1): 2-qubit gate - CZ gate. 'ibm_paris' (Falcon r4): 2-qubit gate - CX gate.
    if('cz' in monolithic_backend.operation_names):
        inter_qpu_error_rate = 10 * retrieve_median_cz_error(monolithic_backend)
        inter_qpu_duration = 2 * retrieve_median_cz_duration(monolithic_backend)
    elif('cx' in monolithic_backend.operation_names):
        inter_qpu_error_rate = 10 * retrieve_median_cx_error(monolithic_backend)
        inter_qpu_duration = 2 * retrieve_median_cx_duration(monolithic_backend)
    else:
        raise ValueError("Monolithic backend must support either 'cz' or 'cx' operations.")

    target = generate_multi_qpu_target(
        backend=monolithic_backend,
        n_qpus=n_qpus,
        inter_qpu_links=inter_qpu_links,
        inter_qpu_error_rate=inter_qpu_error_rate,
        inter_qpu_duration=inter_qpu_duration
    )

    return custom_mqpu_backend(name=backend_name, n_qpus=n_qpus, target=target, inter_qpu_links=inter_qpu_links, inter_qpu_error_rate=inter_qpu_error_rate, inter_qpu_duration=inter_qpu_duration)

# Function that directly generates a multi-QPU backend.
# A monolithic backend must be provided, which represents a single QPU, along with the base qubits for inter-QPU connections.
# The inter-QPU connections will be generated automatically based on the base qubits.
def generate_multi_qpu_backend_from_monolithic_backend_with_base_qubits(
    n_qpus: int = 3,
    monolithic_backend: BackendV2 = None, # Could add a `coupling_map` parameter, where the user would have the option to provide this instead of a `backend`.
    base_qubits: list = None,
    backend_name: str = "mqpu_backend"
):
    
    ###############################################################################################################
    ### This function could, in principle, be merged with the previous one, but it's kept separate for clarity. ###
    ###############################################################################################################

    """
    Generate a multi-QPU backend based on the provided parameters.

    Args:
        n_qpus (int): Number of QPUs.
        monolithic_backend (BackendV2): Backend representing a single QPU.
        base_qubits (list): Base qubit indices for inter-QPU connections.
        backend_name (str): Name for the generated multi-QPU backend.

    Returns:
        custom_mqpu_backend: The generated multi-QPU backend.
    """
    if not monolithic_backend:
        raise ValueError("Monolithic backend must be provided.")
    if not isinstance(monolithic_backend, BackendV2):
        raise TypeError("Monolithic backend must be an instance of BackendV2.")
    if n_qpus < 1:
        raise ValueError("Number of QPUs must be at least 1.")
    if base_qubits is None:
        raise ValueError("Base qubits must be provided.")

    # # To be stored in the backend object, under the `graph` property. # This is unnecessary, as the `Target` object generates the `CouplingMap` internally (`target._build_coupling_graph`).
    # coupling_map = generate_mqpu_coupling_map_from_monolithic_backend(
    #     n_qpus=n_qpus,
    #     monolithic_backend=monolithic_backend,
    #     base_qubits=base_qubits
    # ).get_edges()

    # Generate inter-QPU links based on base qubits.
    inter_qpu_links = generate_inter_qpu_links(n_qpus=n_qpus, base_qubits=base_qubits, monolithic_backend=monolithic_backend)[1]

    # 'ibm_torino' (Heron r1): 2-qubit gate - CZ gate. 'ibm_paris' (Falcon r4): 2-qubit gate - CX gate.
    if('cz' in monolithic_backend.operation_names):
        inter_qpu_error_rate = 10 * retrieve_median_cz_error(monolithic_backend)
        inter_qpu_duration = 2 * retrieve_median_cz_duration(monolithic_backend)
    elif('cx' in monolithic_backend.operation_names):
        inter_qpu_error_rate = 10 * retrieve_median_cx_error(monolithic_backend)
        inter_qpu_duration = 2 * retrieve_median_cx_duration(monolithic_backend)
    else:
        raise ValueError("Monolithic backend must support either 'cz' or 'cx' operations.")

    # Define multi-QPU target.
    target = generate_multi_qpu_target(
        backend=monolithic_backend,
        n_qpus=n_qpus,
        inter_qpu_links=inter_qpu_links,
        inter_qpu_error_rate=inter_qpu_error_rate,
        inter_qpu_duration=inter_qpu_duration,
    )

    return custom_mqpu_backend(name=backend_name, n_qpus=n_qpus, target=target, inter_qpu_links=inter_qpu_links, inter_qpu_error_rate=inter_qpu_error_rate, inter_qpu_duration=inter_qpu_duration)

# Generate a multi-QPU backend from a monolithic backend and an inter-QPU coupling map.
def generate_mqpu_backend_from_monolithic_backend_and_inter_qpu_coupling_map_with_base_qubits(
    monolithic_backend: BackendV2,
    inter_qpu_coupling_map: list[tuple[int, int]],
    base_qubits: list[int],
    backend_name: str = "mqpu_backend",
):
    """
    Generate a multi-QPU backend from a monolithic backend and an inter-QPU coupling map.

    Each `base_qubit` in QPU j is connected to the same `base_qubit` in QPU k 
    if (j, k) exists in `inter_qpu_coupling_map`. For example, qubit 0 in QPU 0 
    connects to qubit 0 in QPU 1 if (0, 1) is in the map.

    Args:
        monolithic_backend (BackendV2): The base monolithic backend.
        inter_qpu_coupling_map (list[tuple[int, int]]): Pairs of connected QPUs.
        base_qubits (list[int]): Indices of base qubits used for QPU-to-QPU connections.
        backend_name (str): Name of the resulting backend. Defaults to "mqpu_backend".

    Returns:
        BackendV2: A multi-QPU backend with the specified connectivity.
        list[tuple[int, int]]: List of inter-QPU connections.
    """
    n_qpus = max(max(pair) for pair in inter_qpu_coupling_map) + 1
    n_qubits_m = monolithic_backend.num_qubits
    inter_qpu_connections = []

    for src_qpu, tgt_qpu in inter_qpu_coupling_map:
        for base_qubit in base_qubits:
            src_qubit = base_qubit + src_qpu * n_qubits_m
            tgt_qubit = base_qubit + tgt_qpu * n_qubits_m
            inter_qpu_connections.extend([
                (src_qubit, tgt_qubit),
                (tgt_qubit, src_qubit),  # Bidirectional
            ])

    return generate_multi_qpu_backend_from_monolithic_backend_with_links(
        n_qpus=n_qpus,
        monolithic_backend=monolithic_backend,
        inter_qpu_links=inter_qpu_connections,
        backend_name=backend_name,
    ), inter_qpu_connections

#############################################################################################
###### Rectangular Backend Generation #######################################################
#############################################################################################

def rectangular_backend(a: int, b: int, basis_gates: list, seed: int) -> GenericBackendV2:
    """Generate a rectangular backend with a x b qubits in a grid layout.
    Args:
        a (int): Number of rows.
        b (int): Number of columns.
        basis_gates (list): List of basis gates for the backend.
        seed (int): Seed for noise model generation.
    Returns:
        GenericBackendV2: A rectangular backend with a x b qubits in a grid layout.
    """
    couplinglist = []
    for i in range(a):
        for j in range(b):
            if j < b-1:
                couplinglist.append([i*b+j, i*b+(j+1)])
                couplinglist.append([i*b+(j+1), i*b+j])
            if i < a-1:
                couplinglist.append([i*b+j, (i+1)*b+j])
                couplinglist.append([(i+1)*b+j, i*b+j])
    return GenericBackendV2(num_qubits=a*b, coupling_map=couplinglist, basis_gates=basis_gates, noise_info=True, seed=seed)

#############################################################################################
###### Rectangular Backend Coordinates Generation ###########################################
#############################################################################################

def rectangular_backend_coordinates(a: int, b: int) -> list:
    """Generate coordinates for a rectangular backend with a x b qubits in a grid layout.
    Args:
        a (int): Number of rows.
        b (int): Number of columns.
    Returns:
        list: A list of coordinates for a rectangular backend with a x b qubits in a grid layout.
    """
    qubit_coordinates = []
    for i in range(a):
        for j in range(b):
            qubit_coordinates.append((j,i))
    return qubit_coordinates

#############################################################################################
###### Post-processing Functions ############################################################
#############################################################################################

# Function to count the number of inter-QPU links in a transpiled circuit.
def get_num_inter_qpu_links(circuit, inter_qpu_links):
    """
    Counts the number of inter-QPU links in the transpiled circuit.
    Considers both 'cz' and 'cx' instructions, as different QPUs may use different 2-qubit gates.
    (Only one of them will be present in the circuit, depending on the QPU type.)
    (For Heron r1, it's 'cz'. For old Falcon r4, it's 'cx'.)

    Parameters
    ----------
    circuit : QuantumCircuit
        The transpiled quantum circuit.
    inter_qpu_links : list
        The list of inter-QPU links.

    Returns
    -------
    int
        The number of inter-QPU links in the circuit.
    """
    count = 0

    # Retrieve CZ and CX instructions from the circuit
    cz_instructions = circuit.get_instructions('cz') # Heron r1 uses 'cz'.
    cx_instructions = circuit.get_instructions('cx') # Old Falcon r4 uses 'cx'.

    # Count inter-QPU links based on CZ instructions.
    if cz_instructions:
        for instruction in cz_instructions:
            link = tuple([circuit.find_bit(qarg)[0] for qarg in instruction.qubits])
            if link in inter_qpu_links:
                count += 1
        return count # Return early if CZ instructions were found. Only one of CZ or CX will be present.
    
    # Some QPUs use 'cx' instead of 'cz'.
    if cx_instructions:
        for instruction in cx_instructions:
            link = tuple([circuit.find_bit(qarg)[0] for qarg in instruction.qubits])
            if link in inter_qpu_links:
                count += 1
        return count # Return the count based on CX instructions. Only one of CZ or CX will be present.

# Function to compute the total accumulated multiplicative fidelity of a circuit.
# This is the product of (1 - error) for each operation in the circuit, disregarding measurement operations.
# This is a very primitive way to compute the fidelity, but it gives an idea of the accumulated error in the circuit.
# It assumes that the errors are independent and multiplicative.
def get_multiplicative_fidelity(circuit, backend): # Should implement the option to consider only 2-qubit gates.
    """
    Computes the total accumulated error in a circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to analyze.
    backend : BackendV2
        The backend to use for the analysis.

    Returns
    -------
    float
        The total accumulated multiplicative fidelity of the circuit.
    """
    multiplicative_fidelity = 1.0
    for instruction in circuit.data:
        op_name = instruction.operation.name
        if op_name not in backend.operation_names:
            continue
        elif op_name == 'measure': # Disregard measurement operations for fidelity calculation.
            continue
        qubits_involved = [circuit.find_bit(q).index for q in instruction.qubits] # Apparently, instruction.qubits[0]._index worked. No need to use `circuit.find_bit()`.
        error = backend.target[op_name][tuple(qubits_involved)].error # if tuple(qubits_involved) in backend.target[op_name] else 0
        if error == 1.0: # These are cases in which the operation is defective.
            continue # Skip operations with error 1.0 (or handle differently)
        multiplicative_fidelity *= (1 - error)
    return multiplicative_fidelity

#############################################################################################
###### Miscellaneous Functions ##############################################################
#############################################################################################

# Function to compute a mixed distance matrix based on topology, error rates, and gate durations.
def compute_mixed_distance_matrix(backend, alpha_1, alpha_2, alpha_3):
    """
    Compute a mixed distance matrix based on topology (D), error rates (E),
    and gate durations (T) for a given quantum backend. This is an implementation of
    "A Hardware-Aware Heuristic for the Qubit Mapping Problem in the NISQ Era",
    Niu S. et al. (https://arxiv.org/abs/2010.03397).

    The mixed distance matrix is computed as a weighted sum of the three matrices:
        S = (alpha_1 * D + alpha_2 * E + alpha_3 * T) / (alpha_1 + alpha_2 + alpha_3)
    where alpha_1, alpha_2, and alpha_3 are user-defined weights.

    Parameters
    ----------
    backend : BackendV2
        The quantum backend to analyze.
    alpha_1 : float
        Weight for the topology-based distance matrix (D).
    alpha_2 : float
        Weight for the error-based distance matrix (E).
    alpha_3 : float
        Weight for the duration-based distance matrix (T).

    Returns
    -------
    S : np.ndarray
        The mixed distance matrix.
    """

    two_qubit_op = "cx" if "cx" in backend.operation_names else "cz"
    edges = backend.coupling_map.get_edges()
    nodes = list(backend.coupling_map.physical_qubits)  # fixed node ordering
    target = backend.target[two_qubit_op]

    # Precompute weights
    D_edges, E_edges, T_edges = [], [], []
    for source, sink in edges:
        # Topology (uniform weights)
        D_edges.append((source, sink, 1.0))

        # Error-weighted
        fwd = 1 - target[(source, sink)].error
        rev = 1 - target[(sink, source)].error
        swap_fidelity = fwd * rev * max(fwd, rev)
        E_edges.append((source, sink, 1 - swap_fidelity))

        # Duration-weighted
        d_fwd = target[(source, sink)].duration
        d_rev = target[(sink, source)].duration
        swap_duration = d_fwd + d_rev + min(d_fwd, d_rev)
        T_edges.append((source, sink, swap_duration))

    def floyd_warshall_from_edges(edges):
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        M = nx.floyd_warshall_numpy(G, nodelist=nodes)
        return M / np.linalg.norm(M, ord="fro")

    # Compute normalized matrices
    D = floyd_warshall_from_edges(D_edges)
    E = floyd_warshall_from_edges(E_edges)
    T = floyd_warshall_from_edges(T_edges)

    # Weighted mixture
    alpha_sum = alpha_1 + alpha_2 + alpha_3
    return (alpha_1 * D + alpha_2 * E + alpha_3 * T) / alpha_sum

def generate_pytket_dqc_init_layout(qc: QuantumCircuit,
                                    mqpu_backend: custom_mqpu_backend,
                                    inter_qpu_coupling_map: list[tuple[int, int]],
                                    method: str = 'CoverEmbedding+Refiners',
                                    verbose: bool = False,
                                    seed: int = 0) -> Layout:
    # Qiskit -> Passes -> Pytket -> DQCPass
    ur = Unroll3qOrMore()          # Unroll arbitrary gate definitions, e.g., 'qft' block
    rb = RemoveBarriers()          # Remove barriers, so Pytket doesn't complain
    rm = RemoveFinalMeasurements() # Remove final measurements, so Pytket doesn't complain
    qc = rm(rb(ur(qc))) 

    try: # If an `Exception` is raised here, try again after `decompose`ing the circuit.
        circ = qiskit_to_tk(qc) # Convert to Pytket circuit to apply `DQCPass`
    except Exception:
        if verbose: print('[generate_pytket_dqc_init_layout] Warning: qiskit_to_tk failed, trying again after decomposing the circuit.')
        circ = qiskit_to_tk(qc.decompose()) # Convert to Pytket circuit to apply `DQCPass`

    DQCPass().apply(circ)

    # Transforming `qubit_qpu_map` into a dictionary, for later use
    qubit_qpu_map = {}
    for qpu_id in range(mqpu_backend.n_qpus):
        qubit_qpu_map[qpu_id] = [q for q in range(mqpu_backend.num_qubits) if q // (mqpu_backend.num_qubits // mqpu_backend.n_qpus) == qpu_id]
    if verbose: print(f'qubit_qpu_map={qubit_qpu_map}')

    server_coupling = inter_qpu_coupling_map
    server_qubits = qubit_qpu_map
    network = NISQNetwork(
        server_coupling=server_coupling, server_qubits=server_qubits
    )

    # Match `method`:
    # 'CoverEmbedding';
    # 'CoverEmbedding+Refiners';
    # 'CoverEmbeddingSteinerDetached';
    # 'PartitioningHeterogeneousEmbedding'.

    if verbose: print(f'[generate_pytket_dqc_init_layout] Using method: {method}')
    match method:
        case 'CoverEmbedding':
            distribution = CoverEmbedding().distribute(circ, network, seed=seed) # Initial distribution
        case 'CoverEmbedding+Refiners':
            if verbose: print('[generate_pytket_dqc_init_layout] Using refiners with CoverEmbedding.')
            distribution = CoverEmbedding().distribute(circ, network, seed=seed) # Initial distribution
            if verbose: print('[generate_pytket_dqc_init_layout] Initial distribution complete.')
            
            refiner_list = [
                NeighbouringDTypeMerge(),
                IntertwinedDTypeMerge(),
            ]
            refiner = RepeatRefiner(SequenceRefiner(refiner_list))
            if verbose: print('[generate_pytket_dqc_init_layout] Starting refinement process.')
            refiner.refine(distribution) # Refine the distribution
            if verbose: print('[generate_pytket_dqc_init_layout] Refinement process complete.')
        case 'CoverEmbeddingSteinerDetached':
            distribution = CoverEmbeddingSteinerDetached().distribute(circ, network, seed=seed)
        case 'PartitioningHeterogeneousEmbedding':
            distribution = PartitioningHeterogeneousEmbedding().distribute(circ, network, seed=seed)
        case _:
            raise ValueError(f'Unknown method: {method}')

    new_layout = pytket_dqc_init_layout_to_qiskit_initial_layout(qc, distribution, server_qubits, verbose=verbose, seed=seed)

    return new_layout

# Pytket doesn't perform intra-QPU mapping yet, so we can't use it for that.
# It merely assigns virtual qubits to QPUs. This isn't enough, we also need
# to map virtual qubits to physical qubits within each QPU. The simplest way
# is to randomly assign virtual qubits to physical qubits within each QPU.
def pytket_dqc_init_layout_to_qiskit_initial_layout(qc: QuantumCircuit,
                                                    distribution: pytket_dqc.circuits.distribution.Distribution,
                                                    server_qubits: dict[int, list[int]],
                                                    verbose: bool = False,
                                                    seed: int = 0) -> Layout:
    initial_layout = distribution.placement.to_dict() # retrieve `pytket_dqc`'s results
    qiskit_init_layout = {}                           # virtual_qubit -> physical_qubit

    random.seed(seed) # set random seed
    for v_qubit, qpu_no in initial_layout.items():
        if(v_qubit > qc.num_qubits-1): # `circ.n_qubits` -> `qc.num_qubits`: Should be equivalent.
            break
        p_qubit = random.choice(server_qubits[qpu_no]) # pick a random non-allocated physical qubit from the QPU
        while p_qubit in qiskit_init_layout.values():
            p_qubit = random.choice(server_qubits[qpu_no])
        qiskit_init_layout[v_qubit] = p_qubit

    # Assigning the new layout to Qiskit Layout object
    new_qr = qc.qregs[0]
    new_layout_dict = {v: new_qr[k] for k, v in qiskit_init_layout.items() if v is not None}
    new_layout = Layout()
    new_layout.from_dict(new_layout_dict)
    if verbose: print(f'pytket_dqc_init_layout={new_layout}') # use this for the `initial_layout` in our method, instead of the "trivial layout"

    return new_layout

############################################################################################
###### Pytket-dqc + SABRE Integration ######################################################
############################################################################################

def rectangular_backend_with_links(a: int, b: int, basis_gates: list, seed: int) -> GenericBackendV2:
    couplinglist = []
    num_grid_qubits = a * b
    
    # Standard Grid Connections
    for i in range(a):
        for j in range(b):
            curr = i * b + j
            # Connect Right (Increasing x / j)
            if j < b - 1:
                right = i * b + (j + 1)
                couplinglist.extend([[curr, right], [right, curr]])
            # Connect Up (Increasing y / i)
            if i < a - 1:
                up = (i + 1) * b + j
                couplinglist.extend([[curr, up], [up, curr]])

    # Add 'a' Link Qubits at x = b
    for i in range(a):
        # The grid qubit at the end of each row (column b-1)
        grid_qubit = i * b + (b - 1)
        # The link qubit (index starts after the grid)
        link_qubit = num_grid_qubits + i
        
        couplinglist.extend([[grid_qubit, link_qubit], [link_qubit, grid_qubit]])

    return GenericBackendV2(
        num_qubits=num_grid_qubits + a, 
        basis_gates=basis_gates,
        coupling_map=couplinglist, 
        seed=seed
    )

def rectangular_backend_with_links_coordinates(a: int, b: int) -> list:
    """Generate coordinates: (0,0) is bottom-left, (1,0) is right, (0,1) is up."""
    qubit_coordinates = []
    
    # 1. Grid Qubits
    # i loop handles rows (y-axis), j loop handles columns (x-axis)
    for i in range(a): # Rows
        for j in range(b): # Columns
            qubit_coordinates.append([i, j]) # (i, j) corresponds to (y, x)
            
    # 2. Link Qubits 
    # Placed to the right of the grid at x = b
    for i in range(a):
        qubit_coordinates.append([i, b])
        
    return qubit_coordinates

def generate_hand_drawn_layout_smooth(a, b):
    template = rectangular_backend_with_links_coordinates(a, b)
    # Center the template so it rotates around its own midpoint
    center = np.mean(template, axis=0)
    centered = template - center

    def rotate_and_place(coords, angle_deg, tx, ty):
        rad = np.radians(angle_deg)
        rot_mat = np.array([[np.cos(rad), -np.sin(rad)], 
                            [np.sin(rad),  np.cos(rad)]])
        # Keep your 2.5 scaling
        transformed = (coords @ rot_mat.T) * 2.5 + [tx, ty]
        return transformed

    # --- Vertex Placement preserved from your code ---
    # QPU 0 (Bottom Right): Rotate 90 CW (-90 deg)
    q0 = rotate_and_place(centered, -90, 15, 20)
    
    # QPU 1 (Top Right): Rotate 90 CCW (+90 deg)
    q1 = rotate_and_place(centered, 90, 35, 20)
    
    # QPU 2 (Left Center): Rotate +0 CCW (+0 deg)
    q2 = rotate_and_place(centered, +0, 25, 10)

    combined = np.vstack([q0, q1, q2])

    # Normalize: Keep as floats to prevent misalignment when nodes are large
    min_x, min_y = np.min(combined, axis=0)
    final_coords = []
    
    for pt in combined:
        # We subtract the min to keep things positive, but NO INT ROUNDING
        fx = pt[0] - min_x
        fy = pt[1] - min_y
        final_coords.append([fx, fy])
        
    return final_coords

def build_cz_fraction_circuit(n: int, d: int, p: float, seed: int = None) -> QuantumCircuit:
    """
    Implements Algorithm 2: Building an instance of CZ Fraction.
    
    Args:
        n (int): Number of qubits (Width).
        d (int): Number of layers (Depth).
        p (float): Fraction probability in [0, 1].
        seed (int): Optional seed for reproducibility.
        
    Returns:
        QuantumCircuit: The generated CZ Fraction circuit.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    
    for t in range(d):
        # List to track qubits that did NOT receive an H gate in this layer
        qubits_without_h = []
        
        # 1. Apply H with probability 1 - p
        for i in range(n):
            if rng.random() < (1 - p):
                qc.h(i)
            else:
                qubits_without_h.append(i)
        
        # 2. Randomly pair qubits to which no H was acted
        rng.shuffle(qubits_without_h)
        
        # 3. Apply CZ to each pair
        # If there's an odd number of qubits without H, the last one is left alone
        for i in range(0, len(qubits_without_h) // 2 * 2, 2):
            q_a = qubits_without_h[i]
            q_b = qubits_without_h[i+1]
            qc.cz(q_a, q_b)
            
        # qc.barrier() # Optional: keeps layers visually distinct
            
    return qc

def check_violations(final_circuit, server_link_capacities, verbose=False) -> bool:
    """
    Check for violations of inter-QPU link capacities in the final distributed circuit.
    Prints any violations found during the simulation of the circuit execution.
    Args:
        final_circuit (pytket.Circuit): The final distributed circuit to check.
        server_link_capacities (dict): A dictionary mapping inter-QPU link tuples to their capacities.
    Returns:
        bool: False if no violations are found, True otherwise.
    """
    # Dictionary to track occupation of inter-QPU links (for visualization/debugging purposes).
    occupation = {
        (0, 1): 0,
        (0, 2): 0,
        (1, 2): 0
    }

    def _get_link(cmd):
        """
        Helper function to identify which inter-QPU link a command corresponds to.
        Works with "Starting" and "Ending" process commands only. These are custom gates.
        """
        q1, q2 = cmd.args
        s1, s2 = int(str(q1).split('_')[1][0]), int(str(q2).split('_')[1][0])
        t = sorted((s1, s2))
        return tuple(t)
    
    # Track violations
    v_flag = False

    # Print operations
    for cmd in final_circuit.get_commands(): # Compare with `final_circuit_without_barriers`: removal of barriers can result in violation of link capacities.
        if "starting_process" in str(cmd.op):
            link = _get_link(cmd)
            occupation[link] += 1
            if occupation[link] > server_link_capacities.get(link, 99):
                if verbose:
                    print(f"!!! VIOLATION DETECTED !!!")
                    print(f"Command: {cmd}")
                    print(f"Link: {link}, Occupation: {occupation[link]}, Capacity: {server_link_capacities.get(link, 'Unknown')}")
                v_flag = True
        elif "ending_process" in str(cmd.op):
            link = _get_link(cmd)
            occupation[link] -= 1
    # So, "detached gates" seems to be messing up the occupation count.
    # For now, I'll try to distribute without "detached gates".

    return v_flag

def optimize_circuit_barriers(
    circuit: Circuit, 
    capacities: dict[tuple[int, int], int], 
    barrier_index_to_check: int = 0,
    removed_indices: list[int] = None,
    current_original_mapping: list[int] = None
    ) -> tuple[Circuit, list[int]]:
    """
    Recursively removes barriers one by one. 
    Returns the optimized circuit AND a list of the original barrier indices that were removed.
    """
    
    # Initialization on the very first run
    if removed_indices is None:
        removed_indices = []
    if current_original_mapping is None:
        # Count total barriers in the starting circuit to create an index map [0, 1, 2, 3...]
        total_barriers = sum(1 for cmd in circuit.get_commands() if cmd.op.type == OpType.Barrier)
        current_original_mapping = list(range(total_barriers))

    # BASE CASE: We've checked all remaining barriers
    if barrier_index_to_check >= len(current_original_mapping):
        return circuit, removed_indices

    # 1. Build a "Trial Circuit" that removes ONLY the barrier at 'barrier_index_to_check'
    trial_circ = Circuit()
    for q in circuit.qubits: 
        trial_circ.add_qubit(q)
    for b in circuit.bits: # Good practice to keep bits too!
        trial_circ.add_bit(b)

    current_barrier_idx = 0

    for cmd in circuit.get_commands():
        if cmd.op.type == OpType.Barrier:
            # Check if this is the barrier we want to test removing
            if current_barrier_idx == barrier_index_to_check:
                pass # SKIP adding this barrier to the trial circuit
            else:
                trial_circ.add_barrier(cmd.qubits)
            current_barrier_idx += 1
        else:
            # Keep all gates
            if cmd.op.type == OpType.CZ: 
                trial_circ.add_gate(OpType.CU1, 1.0, cmd.qubits)
            elif is_robust_start_proc(cmd): 
                trial_circ.add_custom_gate(start_proc(), [], cmd.qubits)
            else:
                trial_circ.add_gate(cmd.op, cmd.qubits)

    # 2. Check if the removal caused a violation
    violation_flag = False
    try:
        violation_flag = check_violations(trial_circ, capacities, verbose=False)
    except Exception:
        violation_flag = True

    # 3. Recursive Decision
    if not violation_flag:
        # SUCCESS: The barrier was not needed.
        # Figure out what the ORIGINAL index of this barrier was
        original_idx = current_original_mapping[barrier_index_to_check]
        removed_indices.append(original_idx)
        
        # Remove this barrier from our mapping tracker
        new_mapping = current_original_mapping.copy()
        new_mapping.pop(barrier_index_to_check)
        
        # Recurse. Keep 'barrier_index_to_check' the same because the list shifted left!
        return optimize_circuit_barriers(
            trial_circ, capacities, barrier_index_to_check, removed_indices, new_mapping
        )
    
    else:
        # FAILURE: The barrier was needed to prevent capacity overflow.
        # Recurse. Move to the next index, keeping the mapping exactly as it is.
        return optimize_circuit_barriers(
            circuit, capacities, barrier_index_to_check + 1, removed_indices, current_original_mapping
        )

# --- Helper Wrapper to make it easy to call ---
def optimize_barriers_wrapper(dist, circuit):
    """
    Wrapper to handle the dependencies (capacities, check function) 
    if you are running this outside the class.
    """
    # Extract capacities from your network object
    raw_capacities = getattr(dist.network, "server_link_capacities", {})
    capacities = {}
    for (u, v), cap in raw_capacities.items():
        capacities[tuple(sorted((int(u), int(v))))] = cap

    return optimize_circuit_barriers(circuit, capacities)

def create_subcircuit(n_computation_qubits, n_link_qubits, n_virtual_qubits, server_id):
    subcircuit = Circuit(n_computation_qubits + n_link_qubits + n_virtual_qubits)
    # Create the qubit mapping: `n_computation_qubits` computation qubits followed by `n_link_qubits` link qubits and `n_virtual_qubits` virtual sink qubits
    qubit_map = {
        Qubit(i): Qubit(f"server_{server_id}", i) for i in range(n_computation_qubits)
    }
    qubit_map.update({
        Qubit(i + n_computation_qubits): Qubit(f"server_{server_id}_link_register", i) for i in range(n_link_qubits)
    })
    qubit_map.update({
        Qubit(i + n_computation_qubits + n_link_qubits): Qubit(f"server_{server_id}_virtual_sink", i) for i in range(n_virtual_qubits)
    })
    subcircuit.rename_units(qubit_map)
    return subcircuit

# Not required for the current implementation.
def squash_placeholders_per_pair(subcircuit, group_name="routing_placeholder"):
    new_circ = Circuit()
    # 1. Replicate the register structure
    for q in subcircuit.qubits: new_circ.add_qubit(q)
    for b in subcircuit.bits: new_circ.add_bit(b)
    
    # 2. Track active placeholders per qubit pair
    # We use a set of frozensets so that (q0, q1) is the same as (q1, q0)
    active_placeholder_pairs = set()

    for cmd in subcircuit.get_commands():
        if cmd.opgroup == group_name:
            # Identify the specific pair of qubits
            pair = frozenset(cmd.qubits)
            
            if pair not in active_placeholder_pairs:
                # First time seeing it for this pair, add it
                new_circ.add_gate(cmd.op, cmd.args, opgroup=cmd.opgroup)
                active_placeholder_pairs.add(pair)
            else:
                # Already active on these qubits, skip it (squash)
                pass
        elif cmd.op.type == OpType.Barrier:
            # Barriers don't break the continuity of placeholders, so we keep them as is.
            new_circ.add_barrier(cmd.qubits)
        else:
            # 3. For any other gate, reset the flag for the qubits involved
            # If a Hadamard or CZ happens on a qubit, it "breaks" the 
            # continuity of the placeholder sequence for that qubit.
            for q in cmd.qubits:
                # Remove any pair from the set if it contains this qubit
                active_placeholder_pairs = {p for p in active_placeholder_pairs if q not in p}
            
            # Add the gate (handling the NoneType opgroup issue)
            if cmd.opgroup:
                new_circ.add_gate(cmd.op, cmd.args, opgroup=cmd.opgroup)
            else:
                new_circ.add_gate(cmd.op, cmd.args) # No `opgroup`, since `NoneType` `opgroup`
                                                    # causes issues with the backend's C++ code
                                                    # `std::bad_cast` error.
                
    return new_circ

# Auxiliary function to extract server ID from qubit name, e.g., "server_0_link_register[2]" -> 0
def get_server_id(qubit):
    try:
        return int(qubit.reg_name.split("_")[1])
    except:
        return None

def parse_q(q):
    try:
        parts = q.reg_name.split("_")
        server = int(parts[1])
        if "link" in parts: return "link", server, q.index[0]
        elif "virtual" in parts: return "virtual", server, q.index[0]
        else: return "comp", server, q.index[0]
    except: return "unknown", -1, -1

def sanitize_qiskit_labels(qc):
    # 1. Create sanitized registers
    new_qregs = [QuantumRegister(qr.size, qr.name.replace("_", "-")) for qr in qc.qregs]
    new_cregs = [ClassicalRegister(cr.size, cr.name.replace("_", "-")) for cr in qc.cregs]
    
    # 2. Initialize circuit with these specific registers
    safe_qc = QuantumCircuit()
    for qr in new_qregs: safe_qc.add_register(qr)
    for cr in new_cregs: safe_qc.add_register(cr)
    
    # 3. Create a direct mapping between old qubit objects and new ones
    # Qiskit maps bits based on their position in the circuit's 'qubits' list
    qubit_map = {old_q: new_q for old_q, new_q in zip(qc.qubits, safe_qc.qubits)}
    clbit_map = {old_c: new_c for old_c, new_c in zip(qc.clbits, safe_qc.clbits)}

    # 4. Re-apply the instructions
    for instr in qc.data:
        # Map the qubits/clbits found in the instruction to our new registers
        new_qubits = [qubit_map[q] for q in instr.qubits]
        new_clbits = [clbit_map[c] for c in instr.clbits]
        safe_qc.append(instr.operation, new_qubits, new_clbits)
        
    return safe_qc

# Simulation-based identity test for two circuits, with placeholder replacement.
def replace_placeholders(qc):
    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instr in qc.data:
        if instr.operation.name == "routing_placeholder":
            # Replace with CZ for testing purposes (this is a simplification)
            new_qc.cz(instr.qubits[0], instr.qubits[1])
        else:
            new_qc.append(instr.operation, instr.qubits, instr.clbits)
    return new_qc

def fast_identity_test(qc1, qc2, sim=AerSimulator(method='matrix_product_state')):
    # Replace `routing_placeholder` gates with CZ gates for the purpose of this test, since the placeholder gates are just stand-ins and don't have a direct representation in Qiskit.
    qc1 = replace_placeholders(qc1)
    qc2 = replace_placeholders(qc2)

    # Ensure they have the same qubits
    test_circ = qc1.compose(qc2.inverse())
    test_circ.measure_all()
    
    # 100 shots is plenty for an identity check
    result = sim.run(test_circ, shots=1024).result()
    counts = result.get_counts()
    
    expected = '0' * qc1.num_qubits
    return list(counts.keys()) == [expected]

# def identity_test(qc1, qc2):
    # # Replace `routing_placeholder` gates with CZ gates for the purpose of this test, since the placeholder gates are just stand-ins and don't have a direct representation in Qiskit.
    # qc1 = replace_placeholders(qc1)
    # qc2 = replace_placeholders(qc2)

    # test_circ = qc1.compose(qc2.inverse())
    # test_circ.measure_all()
    
    # sampler = AerSimulator()
    # result = sampler.run(test_circ, shots=1024).result()
    # counts = result.get_counts()
    
    # # If the circuits are equal, the only key in counts should be '0' * num_qubits
    # expected_key = '0' * qc1.num_qubits
    # return list(counts.keys()) == [expected_key]

# This function creates a hardware layout mapping for a quantum circuit, based on the registers present in the circuit. It maps:
# - 'virtual-sink' register -> Indices 0-15 (Computation grid)
# - 'link-register' register -> Indices 16-19 (Right-edge links)
# - 'virtual' or 'server' register -> Indices 20-21 (Virtual sinks)
def create_hardware_layout(qc: QuantumCircuit, seed: int = 42, verbose: bool = False,
                           comp_phys_qubits=None, link_phys_qubits=None, sink_phys_qubits=None) -> Layout:
    """
    Creates a fixed initial layout mapping logical registers to physical indices.
    Handles non-contiguous physical indices (e.g., 0-15 and 18-19) safely.
    """
    # 0. Set defaults for backward compatibility (Standard 16-4-8)
    if comp_phys_qubits is None: comp_phys_qubits = list(range(16))
    if link_phys_qubits is None: link_phys_qubits = list(range(16, 20))
    if sink_phys_qubits is None: sink_phys_qubits = list(range(20, 28))

    layout_dict = {}
    
    def get_reg(name_part):
        try:
            return next(reg for reg in qc.qregs if name_part in reg.name)
        except StopIteration:
            raise ValueError(f"Register containing '{name_part}' not found in circuit.")

    comp_reg = get_reg('server')
    link_reg = get_reg('link-register')
    sink_reg = get_reg('virtual-sink')

    # 1. Map Computational Qubits
    # We use the list of logical qubits in the register
    comp_logical_qubits = list(comp_reg)
    
    if seed is not None:
        # Shuffling allows the layout to start from different randomized initial states
        random.seed(seed)
        random.shuffle(comp_logical_qubits)

    # Map each logical qubit to the specific physical index provided in the list
    for i, logical_q in enumerate(comp_logical_qubits):
        if i < len(comp_phys_qubits):
            layout_dict[logical_q] = comp_phys_qubits[i]
        else:
            if verbose: print(f"Warning: Not enough physical comp qubits for {logical_q}")

    # 2. Map Link Qubits (directly to the provided indices, skipping any "gaps")
    for i, logical_q in enumerate(link_reg):
        if i < len(link_phys_qubits):
            layout_dict[logical_q] = link_phys_qubits[i]
        
    # 3. Map Virtual Sinks
    for i, logical_q in enumerate(sink_reg):
        if i < len(sink_phys_qubits):
            layout_dict[logical_q] = sink_phys_qubits[i]
            
    if verbose:
        print(f"Layout created for {len(layout_dict)} total qubits.")
        # Debugging non-contiguous gaps:
        phys_indices = sorted(layout_dict.values())
        print(f"Physical indices used: {phys_indices}")
        
    return Layout(layout_dict)

# 1. Define the opaque gate (as we did before)
class RoutingPlaceholder(Gate):
    """An opaque 2-qubit gate that the transpiler cannot simplify."""
    def __init__(self):
        super().__init__("routing_placeholder", 2, [])

# 2. Define the custom pass to inject it
class MakePlaceholdersOpaque(TransformationPass):
    """
    Finds existing 'routing_placeholder' gates with math definitions 
    and replaces them with our opaque version.
    """
    def run(self, dag):
        for node in dag.op_nodes():
            # Check if the operation has the name we are looking for
            if node.op.name == "routing_placeholder":
                # Substitute the math-heavy node with our opaque black-box gate
                dag.substitute_node(node, RoutingPlaceholder(), inplace=True)
        return dag

def create_partial_hardware_layout(qc: QuantumCircuit) -> Layout:
    layout_dict = {}
    
    link_reg = next(reg for reg in qc.qregs if 'link-register' in reg.name)
    sink_reg = next(reg for reg in qc.qregs if 'virtual-sink' in reg.name)

    # Lock Link Qubits to 16-19
    for i in range(4):
        layout_dict[link_reg[i]] = 16 + i
        
    # Lock Virtual Sinks to 20-27
    for i in range(8):
        layout_dict[sink_reg[i]] = 20 + i
        
    return Layout(layout_dict)

@dataclass
class TranspilerConfig:
    seed: int = 42
    trials: int = 5
    heuristic: str = 'lookahead'
    optimization_level: int = 0
    pm_name: str = "(1e{0,2,4,6})_lookahead_sabre"

def create_custom_pm(backend,
                     monolithic_virtual_map,
                     initial_layout,
                     config,
                     distance_matrix):
    
    try:
        base_gates = backend.configuration().basis_gates
    except AttributeError:
        base_gates = ['id', 'rz', 'sx', 'x', 'cz', 'measure']

    custom_mapping = {"routing_placeholder": RoutingPlaceholder()}
    
    translation_target = Target.from_configuration(
        basis_gates=base_gates + ["routing_placeholder"],
        num_qubits=monolithic_virtual_map.size(),
        coupling_map=monolithic_virtual_map, 
        custom_name_mapping=custom_mapping
    )

    pm = generate_preset_pass_manager(
        optimization_level=0, 
        initial_layout=initial_layout, 
        target=translation_target,
        seed_transpiler=config.seed
    )
    pm.init.append(MakePlaceholdersOpaque())

    real_qubit_limit = backend.num_qubits 
    virtual_edges = [
        (src, dst) for src, dst in monolithic_virtual_map.get_edges() 
        if src >= real_qubit_limit or dst >= real_qubit_limit
    ]
    qubit_qpu_map = [0 for _ in range(monolithic_virtual_map.size())]

    # --- Revert to SabreSwap ---
    routing_pass = SabreSwap(
        coupling_map=monolithic_virtual_map,
        heuristic=config.heuristic if hasattr(config, 'heuristic') else 'lookahead',
        seed=config.seed,
        trials=5, # 5 routing trials per layout!
        penalized_swaps=virtual_edges,
        qubit_qpu_map=qubit_qpu_map,
        inter_qpu_coupling_map=[],
        alpha=100000.0,
        beta=0.0
    )
    routing_pass._routing_target.set_distance_matrix(distance_matrix)

    # Inject SabreSwap back into the routing phase
    pm.routing.replace(
        index=1, 
        passes=(BarrierBeforeFinalMeasurements(), routing_pass)
    )
    
    return pm

def remove_routing_placeholders(qc):
    """
    Creates a new QuantumCircuit identical to the input, 
    but perfectly stripped of any 'routing_placeholder' gates.
    """
    # 1. Create a blank clone (keeps the exact layout, qubits, and clbits)
    clean_qc = qc.copy_empty_like()
    
    removed_count = 0
    
    # 2. Iterate through the original instructions
    for instr in qc.data:
        # Check the name of the operation
        if instr.operation.name == "routing_placeholder":
            removed_count += 1
        else:
            # 3. Append valid instructions to the new circuit
            clean_qc.append(instr)
            
    # Optional: You can print this out if you want to verify 
    # that the number of removed placeholders matches your expectations
    # print(f"  -> Stripped {removed_count} routing placeholders.")
    
    return clean_qc

def build_distributed_subcircuits(
    final_optimized_circuit,
    network,
    available_links,
    busy_links,
    link_to_virtual_mapping,
    link_to_virtual_return,
    # virtual_to_comp_mapping,
    n_servers,
    n_links,
    offsets=None # [[comp offset 0, link offset 0, virtual offset 0], [comp offset 1, link offset 1, virtual offset 1], [comp offset 2, link offset 2, virtual offset 2]]
):

    # *Preamble*: define a dummy gate that lo`oks like a CU1 but has a unique name
    dummy_circ = Circuit(2)
    dummy_circ.add_gate(OpType.CU1, 1.0, [0, 1])
    route_gate_def = CustomGateDef.define("routing_placeholder", dummy_circ, [])

    # All operations in the distributed circuit
    all_cmds = final_optimized_circuit.get_commands()

    ## --- 1. Initialization ---
    n_comp    = len(network.server_qubits[0]) if offsets == None else len(network.server_qubits[1]) # Line topology: `len(network.server_qubits[1])` (Center Core)
    n_virtual = 2 * n_links

    # 1. Initialization
    if offsets == None:
        subcircuits = {
            i: create_subcircuit(n_comp, n_links, n_virtual, server_id=i) 
            for i in range(n_servers)
        }
    else:
        subcircuits = {
            i: create_subcircuit(n_comp + offsets[0][i], n_links + offsets[1][i], n_virtual + offsets[2][i], server_id=i) 
            for i in range(n_servers)
        }
    
    # Initialize the Master Distributed Circuit
    distributed_circ = Circuit()
    for q in final_optimized_circuit.qubits:
        distributed_circ.add_qubit(q)
    for b in final_optimized_circuit.bits:
        distributed_circ.add_bit(b)

    # State tracking for logical-to-physical mapping
    equivalence_dict = {}

    # 2. Main Loop: Process each command and build subcircuits with routing placeholders
    for cmd in all_cmds:
        if "starting_process" in str(cmd.op).lower():
            qubit_0, qubit_1 = cmd.qubits
            type_0, server_0, _ = parse_q(qubit_0)
            type_1, server_1, _ = parse_q(qubit_1)
            link_count = sum([type_0 == "link", type_1 == "link"])
            link = tuple(sorted((server_0, server_1)))

            reverse = True if server_0 > server_1 else False

            if available_links[link]: # Link: (0, 1)
                chosen_link = available_links[link].pop(0) # Get the first available physical link: (0, 3)
                busy_links[link].append(chosen_link) # Mark it as busy: (0, 3)
                if not reverse: # E.g.: Link going from cores 0 to 2
                    chosen_link_qubit_sender = Qubit(f"server_{server_0}_link_register", chosen_link[0]) # Get the corresponding link qubit: "server_0_link_register[0]
                    chosen_link_qubit_receiver = Qubit(f"server_{server_1}_link_register", chosen_link[1]) # Get the corresponding link qubit: "server_1_link_register[3]
                else: # E.g.: Link going from cores 2 to 0
                    chosen_link_qubit_sender = Qubit(f"server_{server_0}_link_register", chosen_link[1]) # Get the corresponding link qubit: "server_0_link_register[0]
                    chosen_link_qubit_receiver = Qubit(f"server_{server_1}_link_register", chosen_link[0]) # Get the corresponding link qubit: "server_1_link_register[3]
                equivalence_dict[str(qubit_1)] = chosen_link_qubit_receiver # Map the command to the chosen physical link

                # --- MASTER CIRCUIT UPDATE ---
                # Dynamically register the physical link qubits if they aren't in the circuit yet!
                if chosen_link_qubit_receiver not in distributed_circ.qubits:
                    distributed_circ.add_qubit(chosen_link_qubit_receiver)
                # if chosen_link_qubit_sender not in distributed_circ.qubits:
                #     distributed_circ.add_qubit(chosen_link_qubit_sender)

                mapped_q0 = equivalence_dict.get(str(qubit_0), qubit_0)
                
                # Safety check for computational qubits too
                if mapped_q0 not in distributed_circ.qubits:
                    distributed_circ.add_qubit(mapped_q0)

                # Now it is safe to add the gate
                # print(f"STARTING_PROCESS: {mapped_q0} and {chosen_link_qubit_receiver} for link between Server {server_0} and Server {server_1} using physical link qubits {chosen_link_qubit_sender} and {chosen_link_qubit_receiver}.") # Debugging print statement
                distributed_circ.add_gate(cmd.op, [mapped_q0, chosen_link_qubit_receiver])

            else:
                raise Exception(f"No available physical links for logical link between Server {server_0} and Server {server_1}")

            if(link_count == 1): # Comp. -> Link (e.g., "server_0[5]" -> "server_1_link_register[0]") (0 <-> 1);
                                 # Want to have "server_0_link_register[0]": "server_0_link_register[3]",
                                 # and block (0, 3).
                # Add placeholder routing gate: Comp. to Virtual
                subcircuit = subcircuits[server_0]
                from_q = qubit_0 # `if str(qubit_0) not in equivalence_dict else equivalence_dict[str(qubit_0)]` # Link - Must be mapped to the chosen physical link qubit
                to_q = link_to_virtual_mapping[str(chosen_link_qubit_sender)]
                subcircuit.add_custom_gate(route_gate_def, [], [from_q, to_q], opgroup="routing_placeholder")
            else: # `link_count == 2`: Need 2 routing placeholder gates - back and forth.
                subcircuit = subcircuits[server_0]
                
                # 1st gate: same as above; Link to Virtual
                from_q_0 = qubit_0 if str(qubit_0) not in equivalence_dict else equivalence_dict[str(qubit_0)] # Link - Must be mapped to the chosen physical link qubit
                to_q_0 = link_to_virtual_mapping[str(chosen_link_qubit_sender)] # Virtual
                subcircuit.add_custom_gate(route_gate_def, [], [from_q_0, to_q_0], opgroup="routing_placeholder") # Link to Virtual -> Move link qubit to the vicinity of another link qubit

                # 2nd gate: "reversed".
                # Either use ALL virtual qubits, or insert SWAPs in both places for consistency (both are viable strategies);
                # For now, I shall use ALL virtual qubits.
                # `from_q_1` is the computational qubit on the sender side, connected to the chosen link qubit's virtual sink qubit.
                from_q_1 = from_q_0 # Comp. # Previously (Wrong): `from_q_1 = virtual_to_comp_mapping[str(to_q_0)]`.
                to_q_1 = link_to_virtual_return[str(from_q_0)] # Virtual
                
                # --- DAG Synchronization ---
                # Force SABRE to completely finish the first routing operation before it is even allowed to look at the second one.
                subcircuit.add_barrier([from_q_0, to_q_0, to_q_1]) # from_q_1 is same as from_q_0, so we don't need to add it again.

                subcircuit.add_custom_gate(route_gate_def, [], [from_q_1, to_q_1], opgroup="routing_placeholder") # Comp. to Virtual -> Move computational qubit to the vicinity of the link qubit
                                                                                                                  # (*return path* from the first movement of the link qubit, which we want to *be* in its original place before the ending process);
                                                                                                                  # We need to guarantee that the reverse path is taken, which I reckon is (with the correct custom distance matrix).

        elif "ending_process" in str(cmd.op).lower():
            qubit_0, qubit_1 = cmd.qubits
            _, server_0, _ = parse_q(qubit_0)
            _, server_1, _ = parse_q(qubit_1)
            link = tuple(sorted((server_0, server_1)))

            # --- MASTER CIRCUIT UPDATE ---
            # Add the ending process using the current physical mapped states
            mapped_q0 = equivalence_dict.get(str(qubit_0), qubit_0)
            mapped_q1 = equivalence_dict.get(str(qubit_1), qubit_1)
            # print(f"Adding ENDING_PROCESS gate between {mapped_q0} and {mapped_q1} for logical link between Server {server_0} and Server {server_1}.") # Debugging print statement
            distributed_circ.add_gate(cmd.op, [mapped_q0, mapped_q1])

            if busy_links[link]: # Link, e.g., (0, 1)
                corresponding_link_qubit_index = equivalence_dict.get(str(qubit_0)).index[0] # Get the corresponding link qubit index, e.g., "server_1_link_register[3]" -> 3.
                for busy_link in busy_links[link]:
                    if corresponding_link_qubit_index in busy_link: # Select the correct physical link to free.
                        released_link = busy_link
                        break
                busy_links[link].remove(released_link) # Remove the released physical link from busy list, e.g., (0, 3)
                available_links[link].append(released_link) # Mark it as available again, e.g., (0, 3)
                # Also, we should (probably) remove the equivalence mapping for this qubit (`qubit_0`), since the logical link is now released, and this qubit is no longer connected to the remote server's qubit.
                del equivalence_dict[str(qubit_0)]
            else:
                raise Exception(f"No busy physical links to release for logical link between Server {server_0} and Server {server_1}")
            
        elif cmd.op.type == OpType.Barrier:
            for subcircuit in subcircuits.values():
                # Adding barrier on all qubits in the subcircuit, since barriers are for synchronization and don't correspond to actual gates.
                # I want to make sure they are present in all subcircuits, since they are needed to prevent capacity violations.
                subcircuit.add_barrier(subcircuit.qubits) 

            # --- MASTER CIRCUIT UPDATE ---
            # Prevent DAG-slide on newly allocated physical link wires by applying 
            # the barrier globally across all currently active qubits in the master circuit.
            if len(distributed_circ.qubits) > 0:
                distributed_circ.add_barrier(distributed_circ.qubits)

        elif cmd.op.type != OpType.CustomGate: # Local operations
            # This is required to "tag" "detached gates"
            n_qubits = len(cmd.qubits)
            if(n_qubits == 2):
                qubit_0, qubit_1 = cmd.qubits
                type_0, server_0, _ = parse_q(qubit_0)
                type_1, server_1, _ = parse_q(qubit_1)
                link_count = sum([type_0 == "link", type_1 == "link"])
                subcircuit = subcircuits[server_0] # Both qubits are on the same server, so it doesn't matter if I use server_0 or server_1 here.
            else: # This is required for single-qubit gates
                qubit_0 = cmd.qubits[0]
                type_0, server_0, _ = parse_q(qubit_0)
                link_count = sum([type_0 == "link"]) # Should never be "2" for single-qubit gates
                assert link_count != 2, "Single-qubit gate cannot be a 'detached gate' with 2 link qubits."
                subcircuit = subcircuits[server_0] 

            if link_count == 2: # "detached gate": Exceptional case; needs special handling since it acts on two link qubits 
                # Logic for handling "detached gates" that act on two link qubits without an active logical link (i.e., not within a starting/ending process).
                # print(f"Warning: Found a \"detached gate\" between qubits {cmd.qubits[0]} and {cmd.qubits[1]}") # Debugging print statement
                
                # 1st gate: same as above; Link to Virtual
                from_q_0 = qubit_0 if str(qubit_0) not in equivalence_dict else equivalence_dict[str(qubit_0)] # Link - Must be mapped to the chosen physical link qubit
                destination_qubit = qubit_1 if str(qubit_1) not in equivalence_dict else equivalence_dict[str(qubit_1)]
                to_q_0 = link_to_virtual_mapping[str(destination_qubit)] # <--- Only change from the "starting_process" case 
                # print(f"Adding first routing placeholder gate to move link qubit {from_q_0} to the vicinity of link qubit {destination_qubit} for executing the detached gate {cmd.op} between them in Server {server_0}'s subcircuit.") # Debugging print statement
                # print(f"{from_q_0} -> {to_q_0} (Link to Virtual)") # Debugging print statement
                
                # 2nd gate: "reversed".
                from_q_1 = from_q_0 # Comp.
                to_q_1 = link_to_virtual_return[str(from_q_0)] # Virtual

                # # Adding barrier to ensure the first routing placeholder is executed before the second one, to prevent SABRE from trying to optimize them together and breaking the intended sequence.
                # subcircuit.add_barrier(subcircuit.qubits) # from_q_1 is same as from_q_0, so we don't need to add it again.
                
                # First routing placeholder to move the link qubit to the vicinity of the other link qubit, so that the local gate can be executed between them in the "virtual" space.
                subcircuit.add_custom_gate(route_gate_def, [], [from_q_0, to_q_0], opgroup="routing_placeholder") # Link to Virtual -> Move link qubit to the vicinity of another link qubit

                # --- DAG Synchronization ---
                # Force SABRE to completely finish the first routing operation before it is even allowed to look at the second one.
                subcircuit.add_barrier([from_q_0, to_q_0, to_q_1, destination_qubit]) # from_q_1 is same as from_q_0, so we don't need to add it again.

                # Local gate is now executable in the subcircuit (After 1st "routing_placeholder" gate)
                # print(f"Adding local gate {cmd.op} on qubits {from_q_0} and {destination_qubit} in Server {server_0}'s subcircuit, as a \"detached gate\" acting on two link qubits without an active logical link.") # Debugging print statement
                subcircuit.add_gate(cmd.op, [from_q_0, destination_qubit]) # Local gate between the two link qubits, now in the "virtual" space near each other.

                # Add barrier to ensure the local gate is executed before the reverse routing happens
                subcircuit.add_barrier([from_q_0, to_q_0, to_q_1, destination_qubit])

                # Reverse path back to the original position of the link qubit, to ensure that the "detached gate" is properly executed without affecting the state of the link qubits for future operations.
                # print(f"Adding reverse routing placeholder gate to move link qubit {from_q_1} back to its original position in Server {server_0}'s subcircuit.") # Debugging print statement
                # print(f"{from_q_1} -> {to_q_1} (Comp. to Virtual)") # Debugging print statement
                subcircuit.add_custom_gate(route_gate_def, [], [from_q_1, to_q_1], opgroup="routing_placeholder") # Comp. to Virtual -> Move computational qubit to the vicinity of the link qubit
                
                # # Fool-proof barrier to ensure the reverse routing is completed before any subsequent gates are executed, to prevent SABRE from trying to optimize them together and breaking the intended sequence.
                # subcircuit.add_barrier(subcircuit.qubits) # from_q_1 is same as from_q_0, so we don't need to add it again.

                # --- MASTER CIRCUIT UPDATE ---
                # print(f"Adding local gate {cmd.op} on qubits {mapped_qubits} in Server {server_0}'s subcircuit.") # Debugging print statement
                distributed_circ.add_gate(cmd.op, [from_q_0, destination_qubit])           

            else: # "non-detached gate": Default case
                # Need to guarantee that we use the right qubits with the equivalence_dict mapping here, otherwise we might end up with gates on the wrong qubits in the subcircuits.
                mapped_qubits = []
                for q in cmd.qubits:
                    if str(q) in equivalence_dict:
                        mapped_qubits.append(equivalence_dict[str(q)])
                    else:
                        mapped_qubits.append(q)
                subcircuit.add_gate(cmd.op, mapped_qubits)

                # --- MASTER CIRCUIT UPDATE ---
                # print(f"Adding local gate {cmd.op} on qubits {mapped_qubits} in Server {server_0}'s subcircuit.") # Debugging print statement
                distributed_circ.add_gate(cmd.op, mapped_qubits)
        else:
            # Any other type of command (e.g., measurements) can be handled here if needed.
            # For now, do nothing. (We shouldn't be entering this branch anyways.)
            print(f"Warning: Unhandled command type {cmd.op.type} for command {cmd}")
            
    return subcircuits, distributed_circ

def verify_fixed_qubits_layout(transpiled_qc, original_qc, verbose=True, 
                               link_phys_qubits=None, sink_phys_qubits=None):
    """
    Verifies that the link and sink qubits are locked to their 
    designated physical hardware indices, accounting for different core topologies.
    """
    # 0. Default to standard indices if not provided
    if link_phys_qubits is None: link_phys_qubits = list(range(16, 20))
    if sink_phys_qubits is None: sink_phys_qubits = list(range(20, 28))
    
    # Create sets for O(1) lookup
    link_phys_set = set(link_phys_qubits)
    sink_phys_set = set(sink_phys_qubits)

    initial_layout = transpiled_qc.layout.initial_layout.get_virtual_bits()
    final_layout = transpiled_qc.layout.final_layout
        
    errors = []
    
    for v_qubit, p_idx in initial_layout.items():
        # 1. Ask the ORIGINAL circuit which registers this qubit belongs to
        try:
            bit_locations = original_qc.find_bit(v_qubit)
            reg_names = [reg.name for reg, idx in bit_locations.registers]
        except ValueError:
            continue
            
        # 2. Safely extract the final physical index (to ensure they didn't move during SWAP)
        final_p_idx = p_idx
        if final_layout is not None:
            final_mapping = final_layout.get_virtual_bits()
            if v_qubit in final_mapping:
                final_p_idx = final_mapping[v_qubit]
            else:
                # After ApplyLayout, map through the physical qubit reference
                phys_qubit = transpiled_qc.qubits[p_idx]
                final_p_idx = final_mapping.get(phys_qubit, p_idx)
                
        # 3. Check Link Qubits
        if any('link-register' in name for name in reg_names):
            if p_idx not in link_phys_set:
                errors.append(f"❌ Initial Layout Error: {v_qubit} mapped to {p_idx} (Expected one of {link_phys_qubits})")
            if final_p_idx != p_idx:
                errors.append(f"❌ Routing Error: Link Qubit {v_qubit} was moved during SabreSwap to {final_p_idx}")
                # Print Layout for debugging
                if verbose:
                    print("Current Layout Mapping (Virtual -> Physical):")
                    for v_q, p_i in initial_layout.items():
                        print(f"  {v_q} -> {p_i}")
                    print("Final Layout Mapping (Virtual -> Physical):")
                    if final_layout is not None:
                        for v_q, p_i in final_layout.get_virtual_bits().items():
                            print(f"  {v_q} -> {p_i}")
                    else:
                        print("  No final layout available.")
                
        # 4. Check Virtual Sinks
        elif any('virtual-sink' in name for name in reg_names):
            if p_idx not in sink_phys_set:
                errors.append(f"❌ Initial Layout Error: {v_qubit} mapped to {p_idx} (Expected one of {sink_phys_qubits})")
            if final_p_idx != p_idx:
                errors.append(f"❌ Routing Error: Sink Qubit {v_qubit} was moved during SabreSwap to {final_p_idx}")
                # Print Layout for debugging
                if verbose:
                    print("Current Layout Mapping (Virtual -> Physical):")
                    for v_q, p_i in initial_layout.items():
                        print(f"  {v_q} -> {p_i}")
                    print("Final Layout Mapping (Virtual -> Physical):")
                    if final_layout is not None:
                        for v_q, p_i in final_layout.get_virtual_bits().items():
                            print(f"  {v_q} -> {p_i}")
                    else:
                        print("  No final layout available.")

    if errors:
        if verbose:
            for error in errors:
                print(error)
        return False
        
    if verbose:
        print("✅ Verification Passed: All fixed-position qubits remained locked.")
        
    return True

def get_optimized_locked_layout(qc, monolithic_virtual_map, seed=42, 
                                comp_phys_qubits=None, link_phys_qubits=None, sink_phys_qubits=None):
    """
    Runs SabreLayout ONLY on the defined computational core,
    then forcefully locks the link and sink qubits to their designated physical indices.
    
    Defaults to the standard 16-core, 4-link, 8-sink architecture if no physical lists are provided.
    """
    # 0. Set defaults to maintain backward compatibility
    if comp_phys_qubits is None: comp_phys_qubits = list(range(16))
    if link_phys_qubits is None: link_phys_qubits = list(range(16, 20))
    if sink_phys_qubits is None: sink_phys_qubits = list(range(20, 28))

    # 1. Identify registers
    def get_reg(name_part):
        return next(reg for reg in qc.qregs if name_part in reg.name)
        
    comp_reg = get_reg('server')
    link_reg = get_reg('link-register')
    sink_reg = get_reg('virtual-sink')

    # 2. Extract ONLY the gates acting on the logical computational qubits
    core_qc = QuantumCircuit(comp_reg)
    comp_qubit_set = set(comp_reg)
    
    for instruction in qc.data:
        if all(q in comp_qubit_set for q in instruction.qubits):
            core_qc.append(instruction)

    # 3. Create a contiguous Coupling Map for SABRE
    comp_phys_set = set(comp_phys_qubits)
    core_edges = [
        edge for edge in monolithic_virtual_map.get_edges() 
        if edge[0] in comp_phys_set and edge[1] in comp_phys_set
    ]
    
    # Map true physical indices to contiguous SABRE indices (0 to N-1) to avoid routing errors
    true_to_sabre = {phys: i for i, phys in enumerate(comp_phys_qubits)}
    sabre_to_true = {i: phys for i, phys in enumerate(comp_phys_qubits)}
    
    contiguous_edges = [(true_to_sabre[u], true_to_sabre[v]) for u, v in core_edges]
    core_cmap = CouplingMap(contiguous_edges)

    # 4. Run SABRE's fw/bw passes strictly on the contiguous core
    sabre_pm = PassManager(SabreLayout(coupling_map=core_cmap, seed=seed))
    mapped_core_qc = sabre_pm.run(core_qc)
    core_layout = mapped_core_qc.layout.initial_layout

    # 5. Stitch the optimized core and the locked peripherals together
    layout_dict = {}
    
    # Map SABRE's contiguous indices back to the TRUE physical indices
    for v_q, sabre_p_idx in core_layout.get_virtual_bits().items():
        layout_dict[v_q] = sabre_to_true[sabre_p_idx]
        
    # Force lock Link Qubits to the provided physical link indices
    for i, q in enumerate(link_reg):
        layout_dict[q] = link_phys_qubits[i]
        
    # Force lock Virtual Sinks to the provided physical sink indices
    for i, q in enumerate(sink_reg):
        layout_dict[q] = sink_phys_qubits[i]
        
    return Layout(layout_dict)