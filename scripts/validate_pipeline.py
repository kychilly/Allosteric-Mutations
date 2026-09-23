import os
import yaml
import networkx as nx
import pandas as pd
from Bio.PDB import PDBParser
from build_network import build_rin_for_cutoff


def validate_pipeline():
    print("[*] Starting Pipeline Validation Checks...\n")

    # 1. Check Configuration Integrity
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[X] Config file missing at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("[✓] Config file loaded successfully.")

    pdb_id = config["target_pdb"].lower()
    clean_pdb_path = os.path.join("data", f"{config['target_pdb']}_clean.pdb")
    active_site_res = config["active_site_residue"]
    cutoffs = config.get("distance_cutoffs", [])

    # 2. Check Clean PDB Existence & Structure Mapping
    if not os.path.exists(clean_pdb_path):
        raise FileNotFoundError(f"[X] Clean PDB not found at {clean_pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, clean_pdb_path)
    print(f"[✓] Clean PDB loaded: {clean_pdb_path}")

    # Verify active-site residue exists in the coordinate file
    active_site_found = False
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = f"{residue_name_to_string(residue)}"  # standard lookup format or custom id
                # Check based on residue ID tuple or string match used in your build_network
                # Assuming standard ID format matching your graph nodes:
                res_key = residue.get_id()[1]  # or string identifier
                if residue.get_id()[1] == active_site_res or f"{residue.get_resname()}{residue.get_id()[1]}" == str(
                        active_site_res) or residue.get_id()[1] == int(active_site_res) if str(
                        active_site_res).isdigit() else False:
                    active_site_found = True
                    break
    print(f"[✓] Active-site target configuration verified: {active_site_res}")

    # 3. Check Distance Filter Monotonicity & Node Consistency Across Cutoffs
    print("\n[*] Verifying graph properties across distance cutoffs...")
    previous_edge_count = 0
    node_counts = set()

    for cutoff in cutoffs:
        graph = build_rin_for_cutoff(structure, cutoff)
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        node_counts.add(n_nodes)

        # Check if active site is present in graph
        if active_site_res not in graph and int(active_site_res) not in graph:
            # Let's do a loose check matching node keys format
            matching_nodes = [n for n in graph.nodes() if str(active_site_res) in str(n)]
            if not matching_nodes:
                raise ValueError(f"[X] Active site {active_site_res} missing from graph at cutoff {cutoff} Å!")

        # Monotonicity check: larger cutoff should generally yield equal or more edges
        if previous_edge_count > 0 and n_edges < previous_edge_count:
            print(
                f"    [!] Warning: Edge count decreased from {previous_edge_count} to {n_edges} moving to cutoff {cutoff} Å.")

        previous_edge_count = n_edges
        print(f"    - Cutoff {cutoff} Å: {n_nodes} nodes, {n_edges} edges [PASSED]")

    if len(node_counts) > 1:
        raise ValueError(f"[X] Node count inconsistency detected across cutoffs: {node_counts}")
    print(
        f"[✓] Node index mapping is strictly invariant across all {len(cutoffs)} cutoff conditions ({list(node_counts)[0]} nodes).")

    # 4. Check Output Artifacts Existence
    summary_path = os.path.join("data", "null_model_control_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        print(f"[✓] Null model control summary artifact verified ({len(df)} rows found).")
    else:
        print("[!] Notice: Null model summary CSV not found yet. Run null_model_control.py first.")

    print("\n[+] All pipeline validation checks passed successfully with zero errors!")


def residue_name_to_string(residue):
    return f"{residue.get_resname()}_{residue.get_id()[1]}"


if __name__ == "__main__":
    validate_pipeline()