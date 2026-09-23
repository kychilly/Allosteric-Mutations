# scripts/null_model_control.py
import os
import yaml
import networkx as nx
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from build_network import build_rin_for_cutoff


def run_null_model_analysis():
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pdb_id = config["target_pdb"].lower()
    clean_pdb_path = os.path.join("data", f"{config['target_pdb']}_clean.pdb")

    if not os.path.exists(clean_pdb_path):
        raise FileNotFoundError(f"Clean PDB file not found at {clean_pdb_path}. Run preprocessing first!")

    print(f"[*] Loading structure for multi-cutoff null-model control from {clean_pdb_path}...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, clean_pdb_path)

    cutoffs = config.get("distance_cutoffs", [5.0])
    active_site_res = config["active_site_residue"]
    base_seed = config.get("random_seed", 1)
    num_random_iter = 100

    all_control_results = []

    for cutoff in cutoffs:
        print(f"\n[*] --- Processing Configuration Model Control for Cutoff: {cutoff} Å ---")
        true_graph = build_rin_for_cutoff(structure, cutoff)

        print(f"    - True RIN Nodes: {true_graph.number_of_nodes()}, Edges: {true_graph.number_of_edges()}")

        if active_site_res not in true_graph:
            print(f"    [!] Warning: Active site {active_site_res} not found in graph at cutoff {cutoff}. Skipping.")
            continue

        # Compute true biological active site centrality
        if nx.is_connected(true_graph):
            true_cf = nx.current_flow_betweenness_centrality(true_graph, weight='weight')
        else:
            largest_cc = max(nx.connected_components(true_graph), key=len)
            sub_cf = nx.current_flow_betweenness_centrality(true_graph.subgraph(largest_cc), weight='weight')
            true_cf = {node: sub_cf.get(node, 0.0) for node in true_graph.nodes()}

        true_active_centrality = true_cf.get(active_site_res, 0.0)
        print(f"    - True Biological Active-Site Centrality: {true_active_centrality:.5f}")

        # Extract the degree sequence from the true graph to preserve hub structure
        degree_sequence = [d for n, d in true_graph.degree()]

        print(f"    - Generating {num_random_iter} degree-preserving configuration models...")
        randomized_centralities = []
        success_count = 0

        for i in range(num_random_iter):
            try:
                # Generate configuration model (returns a MultiGraph with potential self-loops)
                rand_multi = nx.configuration_model(degree_sequence, seed=base_seed + i + int(cutoff * 100))

                # Convert to a clean simple graph (removes multi-edges) and drop self-loops
                rand_g = nx.Graph(rand_multi)
                rand_g.remove_edges_from(nx.selfloop_edges(rand_g))

                # Assign random weights to remaining edges
                for u, v in rand_g.edges():
                    rand_g[u][v]['weight'] = np.random.uniform(1.0, 5.0)

                if nx.is_connected(rand_g) and rand_g.number_of_nodes() > 0:
                    rand_cf = nx.current_flow_betweenness_centrality(rand_g, weight='weight')
                else:
                    if rand_g.number_of_nodes() == 0:
                        continue
                    lcc = max(nx.connected_components(rand_g), key=len)
                    sub_rand_cf = nx.current_flow_betweenness_centrality(rand_g.subgraph(lcc), weight='weight')
                    rand_cf = {n: sub_rand_cf.get(n, 0.0) for n in rand_g.nodes()}

                rand_active_val = rand_cf.get(active_site_res, 0.0)
                randomized_centralities.append(rand_active_val)
                success_count += 1
            except Exception as e:
                continue

        if randomized_centralities:
            mean_rand = np.mean(randomized_centralities)
            std_rand = np.std(randomized_centralities)
            z_score = (true_active_centrality - mean_rand) / (std_rand if std_rand > 0 else 1e-6)
        else:
            mean_rand, std_rand, z_score = 0.0, 0.0, 0.0

        print(f"    - Successful Iterations: {success_count}/{num_random_iter}")
        print(f"    - Null Model Mean: {mean_rand:.5f} (± {std_rand:.5f}) | Z-Score: {z_score:.2f}")

        all_control_results.append({
            "cutoff_angstroms": cutoff,
            "true_active_centrality": round(true_active_centrality, 5),
            "null_model_mean": round(float(mean_rand), 5),
            "null_model_std": round(float(std_rand), 5),
            "z_score": round(float(z_score), 4),
            "successful_iterations": success_count
        })

    df_control = pd.DataFrame(all_control_results)
    out_path = os.path.join("data", "null_model_control_summary.csv")
    df_control.to_csv(out_path, index=False)
    print(f"\n[+] Configuration model control analysis complete! Summary saved to {out_path}")


if __name__ == "__main__":
    run_null_model_analysis()