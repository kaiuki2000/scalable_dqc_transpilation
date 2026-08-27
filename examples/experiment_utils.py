"""Results bookkeeping and subcircuit helpers shared by the example scripts.

Merged from the original `experiment_utils.py` and `circuit_utils.py` in
kaiuki2000/mqpu-ustutt-ibm at commit 48dbc57f2ff97b55898f217e004de322e1a3ca3e; see `examples/README.md`.
"""

import json
import os
import numpy as np

def compute_statistics(data_list):
    """
    Computes the median and the bounds for the middle 75% of the data.
    Returns: (median, lower_bound, upper_bound)
    """
    median = np.median(data_list)
    lower_bound = np.percentile(data_list, 12.5)
    upper_bound = np.percentile(data_list, 87.5)
    return median, lower_bound, upper_bound

def save_experiment_database(db, filepath=None):
    """Saves the database to disk."""
    if filepath is None:
        raise ValueError("Filepath must be provided to save the experiment database.")
    with open(filepath, 'w') as f:
        json.dump(db, f, indent=4)
    """Saves the database to disk."""
    if filepath is None:
        raise ValueError("Filepath must be provided to save the experiment database.")
    with open(filepath, 'w') as f:
        json.dump(db, f, indent=4)

def load_experiment_database(filepath=None):
    """Loads existing results. Does NOT hardcode methods anymore."""
    if filepath is None:
        raise ValueError("Filepath must be provided to load the experiment database.")
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            db = json.load(f)
            
        # Auto-upgrade: Add cz_frac_list
        if "cz_frac_list" not in db:
            db["cz_frac_list"] = []
            
        # Auto-upgrade: Ensure raw_* arrays exist
        for m in db.setdefault("methods_data", []):
            target_len = len(m.get("ebits", []))
            for key in ["raw_ebits", "raw_swaps", "raw_time"]:
                m.setdefault(key, [])
                while len(m[key]) < target_len:
                    m[key].append([])
        return db
        
    # Return empty structure if no file exists
    return {
        "circuit_list": [],
        "seed_history": [],
        "cz_frac_list": [],
        "methods_data": []
    }

def get_or_create_method_index(db, label, color):
    """Finds a method by label, or appends a new one to the database."""
    for i, m in enumerate(db["methods_data"]):
        if m["label"] == label:
            return i
            
    # Not found? Create it!
    db["methods_data"].append({
        "label": label, "color": color,
        "ebits": [], "swaps": [], "time": [],
        "raw_ebits": [], "raw_swaps": [], "raw_time": []
    })
    return len(db["methods_data"]) - 1

def prepare_sorted_data_for_plotting(db):
    """Sorts data by fraction and dynamically orders an arbitrary number of methods."""
    if not db["cz_frac_list"]:
        return [], []
        
    sorted_indices = np.argsort(db["cz_frac_list"])
    sorted_fracs = [db["cz_frac_list"][i] for i in sorted_indices]
    
    sorted_methods = []
    for method in db["methods_data"]:
        sorted_method = method.copy()
        sorted_method["ebits"] = [method["ebits"][i] for i in sorted_indices]
        sorted_method["swaps"] = [method["swaps"][i] for i in sorted_indices]
        sorted_method["time"] = [method["time"][i] for i in sorted_indices]
        sorted_methods.append(sorted_method)
        
    # NEW DYNAMIC SORTING: 
    # 1. Pytket runs, 2. Custom Lookahead, 3. (1,10) SABRE, 4. Default
    def sort_key(m):
        label = m["label"]
        if "Pytket-DQC" in label: return 0
        if "Custom Lookahead" in label: return 1
        if "(1,10)" in label: return 2
        if "Default" in label: return 3
        return 4
        
    sorted_methods.sort(key=sort_key)
    return sorted_fracs, sorted_methods

# ---------------------------------------------------------------------------
# From the original circuit_utils.py
# ---------------------------------------------------------------------------

from pytket import Circuit, OpType

def sync_subcircuit_barriers(subcircuits, indices_to_remove):
    """
    Takes the dictionary of subcircuits and removes barriers at the specified indices.
    indices_to_remove: A list or set of the Nth barrier indices that were removed 
                       from the master distributed circuit.
    """
    optimized_subcircuits = {}
    
    for server_id, subcirc in subcircuits.items():
        # Create a blank slate with the exact same qubits/bits
        new_circ = Circuit()
        for q in subcirc.qubits:
            new_circ.add_qubit(q)
        for b in subcirc.bits:
            new_circ.add_bit(b)
            
        barrier_counter = 0
        
        for cmd in subcirc.get_commands():
            if cmd.op.type == OpType.Barrier:
                # Only add the barrier if its index wasn't flagged for removal
                if barrier_counter not in indices_to_remove:
                    new_circ.add_barrier(cmd.qubits)
                barrier_counter += 1
            else:
                # Add all normal gates exactly as they are
                new_circ.add_gate(cmd.op, cmd.qubits)
                
        optimized_subcircuits[server_id] = new_circ
        
    return optimized_subcircuits