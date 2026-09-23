import os
import yaml
import networkx as nx
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from build_network import build_rin_for_cutoff, is_heavy_atom


def apply_mutation_perturbation(graph, mutated_residue, perturbation_type="truncation"):
    """
    Simulate a mutation at a specific residue node:
    - 'truncation': Removes edges connected to the mutated residue (mimics Ala/Gly knockout of side-chain).
    - 'steric_strain': Scales edge weights up/down to mimic volumetric disruption.
    """
    perturbed_graph = graph.copy()

    if mutated_residue not in perturbed_graph:
        return perturbed_graph

    if perturbation_type == "truncation":
        perturbed_graph.remove_edges_from(list(perturbed_graph.edges(mutated_residue)))
    elif perturbation_type == "steric_strain":
        for u, v in list(perturbed_graph.edges(mutated_residue)):
            current_weight = perturbed_graph[u][v].get('weight', 1.0)
            perturbed_graph[u][v]['weight'] = round(current_weight * 1.35, 2)

    return perturbed_graph


def run_perturbation_analysis():
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pdb_id = config["target_pdb"].lower()
    clean_pdb_path = os.path.join("data", f"{config['target_pdb']}_clean.pdb")

    if not os.path.exists(clean_pdb_path):
        raise FileNotFoundError(f"Clean PDB file not found at {clean_pdb_path}. Run preprocessing first!")

    # Ensure output directory for individual runs exists
    runs_dir = os.path.join("data", "perturbation_runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"[*] Loading structure for multi-file perturbation analysis from {clean_pdb_path}...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, clean_pdb_path)

    cutoffs = config.get("distance_cutoffs", [5.0])
    active_site_res = config["active_site_residue"]

    # Loop through each distance cutoff and save to its own independent file
    for cutoff in cutoffs:
        print(f"\n[*] --- Processing Distance Cutoff: {cutoff} Å ---")
        base_graph = build_rin_for_cutoff(structure, cutoff)

        print(f"    - Base RIN Nodes: {base_graph.number_of_nodes()}, Edges: {base_graph.number_of_edges()}")

        if active_site_res not in base_graph:
            print(f"    [!] Warning: Active site {active_site_res} not found in graph at cutoff {cutoff}. Skipping.")
            continue

        # Compute baseline active site centrality
        if nx.is_connected(base_graph):
            base_cf = nx.current_flow_betweenness_centrality(base_graph, weight='weight')
        else:
            largest_cc = max(nx.connected_components(base_graph), key=len)
            sub_cf = nx.current_flow_betweenness_centrality(base_graph.subgraph(largest_cc), weight='weight')
            base_cf = {node: sub_cf.get(node, 0.0) for node in base_graph.nodes()}

        baseline_active_centrality = base_cf.get(active_site_res, 0.0)
        print(f"    - Baseline Active Site Centrality: {baseline_active_centrality:.5f}")

        test_nodes = [node for node in base_graph.nodes() if node != active_site_res]
        print(f"    - Running edge-truncation simulations across {len(test_nodes)} nodes...")

        cutoff_results = []
        for node in test_nodes:
            pert_graph = apply_mutation_perturbation(base_graph, node, perturbation_type="truncation")

            if nx.is_connected(pert_graph):
                pert_cf = nx.current_flow_betweenness_centrality(pert_graph, weight='weight')
            else:
                comps = list(nx.connected_components(pert_graph))
                if comps:
                    largest_cc = max(comps, key=len)
                    sub_cf = nx.current_flow_betweenness_centrality(pert_graph.subgraph(largest_cc), weight='weight')
                    pert_cf = {n: sub_cf.get(n, 0.0) for n in pert_graph.nodes()}
                else:
                    pert_cf = {n: 0.0 for n in pert_graph.nodes()}

            new_active_centrality = pert_cf.get(active_site_res, 0.0)
            delta_centrality = new_active_centrality - baseline_active_centrality

            cutoff_results.append({
                "cutoff_angstroms": cutoff,
                "mutated_residue": node,
                "baseline_active_centrality": round(baseline_active_centrality, 5),
                "perturbed_active_centrality": round(new_active_centrality, 5),
                "delta_centrality": round(delta_centrality, 5)
            })

        # Save individual file for this specific cutoff
        df_cutoff = pd.DataFrame(cutoff_results)
        file_name = f"perturbation_cutoff_{cutoff}.csv"
        out_path = os.path.join(runs_dir, file_name)
        df_cutoff.to_csv(out_path, index=False)
        print(f"    [+] Saved results for {cutoff} Å to: {out_path}")

    print(f"\n[+] All perturbation runs successfully saved to '{runs_dir}/'!")


if __name__ == "__main__":
    run_perturbation_analysis()