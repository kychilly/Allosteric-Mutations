import os
import yaml
import networkx as nx
import pandas as pd
from Bio.PDB import PDBParser

# This is the control for the experiment
def is_heavy_atom(atom):
    """Safely check if an atom is a heavy atom (non-hydrogen)."""
    element = getattr(atom, 'element', None)
    if element:
        if str(element).strip().upper() == 'H':
            return False

    name = atom.get_name().strip().upper()
    if name.startswith('H') and (len(name) == 1 or name[1].isdigit()):
        return False

    return True


def build_rin_for_cutoff(structure, cutoff):
    """
    Build a Residue Interaction Network (RIN) using a specific distance cutoff.
    Nodes = residues, Edges = heavy-atom distance < cutoff between any two atoms of different residues.
    """
    model = structure[0]
    chain = model['A']

    graph = nx.Graph()

    # Collect valid standard amino acid residues and their heavy atom coordinates
    residues = []
    for res in chain:
        if res.id[0] == ' ' and res.get_resname().strip() in [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"
        ]:
            res_id = res.id[1]
            residues.append((res_id, res))
            graph.add_node(res_id, resname=res.get_resname().strip())

    # Calculate pairwise distances to create edges
    n = len(residues)
    for i in range(n):
        res_id_i, res_i = residues[i]
        atoms_i = [a for a in res_i if is_heavy_atom(a)]

        for j in range(i + 1, n):
            res_id_j, res_j = residues[j]
            atoms_j = [a for a in res_j if is_heavy_atom(a)]

            # Check minimum distance between any heavy atom pair
            min_dist = float('inf')
            for ai in atoms_i:
                for aj in atoms_j:
                    diff = ai.get_coord() - aj.get_coord()
                    dist = (diff @ diff) ** 0.5
                    if dist < min_dist:
                        min_dist = dist

            if min_dist <= cutoff:
                # Add edge with distance as weight
                graph.add_edge(res_id_i, res_id_j, weight=round(float(min_dist), 2))

    return graph


def run_network_analysis():
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pdb_id = config["target_pdb"].lower()
    clean_pdb_path = os.path.join("data", f"{config['target_pdb']}_clean.pdb")

    if not os.path.exists(clean_pdb_path):
        raise FileNotFoundError(f"Clean PDB file not found at {clean_pdb_path}. Run preprocessing first!")

    print(f"[*] Loading clean structure from {clean_pdb_path}...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, clean_pdb_path)

    cutoffs = config.get("distance_cutoffs", [config.get("distance_cutoff", 5.0)])
    active_site_res = config["active_site_residue"]

    results_summary = []

    for cutoff in cutoffs:
        print(f"[*] Building RIN with distance cutoff: {cutoff} Å...")
        graph = build_rin_for_cutoff(structure, cutoff)

        print(f"    - Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        if active_site_res not in graph:
            print(f"[!] Warning: Active site residue {active_site_res} not found in graph nodes.")
            continue

        print(f"    - Computing baseline centralities (including Current-Flow Betweenness)...")
        degree_cent = nx.degree_centrality(graph)
        betweenness_cent = nx.betweenness_centrality(graph, weight='weight')

        if nx.is_connected(graph):
            cf_betweenness_cent = nx.current_flow_betweenness_centrality(graph, weight='weight')
        else:
            print(f"    - Graph is fragmented. Computing Current-Flow Betweenness on largest connected component...")
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc).copy()
            sub_cf_cent = nx.current_flow_betweenness_centrality(subgraph, weight='weight')
            cf_betweenness_cent = {node: sub_cf_cent.get(node, 0.0) for node in graph.nodes()}

        active_site_degree = degree_cent.get(active_site_res, 0)
        active_site_betweenness = betweenness_cent.get(active_site_res, 0)
        active_site_cf_betweenness = cf_betweenness_cent.get(active_site_res, 0)

        results_summary.append({
            "cutoff_angstroms": cutoff,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "active_site_residue": active_site_res,
            "active_site_degree_centrality": round(active_site_degree, 4),
            "active_site_betweenness_centrality": round(active_site_betweenness, 4),
            "active_site_current_flow_betweenness": round(active_site_cf_betweenness, 4)
        })

    summary_df = pd.DataFrame(results_summary)
    output_csv = os.path.join("data", "network_metrics_summary.csv")
    summary_df.to_csv(output_csv, index=False)
    print(f"[+] Network sensitivity analysis complete! Summary saved to {output_csv}")


if __name__ == "__main__":
    run_network_analysis()