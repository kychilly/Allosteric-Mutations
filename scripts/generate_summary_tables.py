import os
import pandas as pd
from sklearn.linear_model import LogisticRegression


def generate_tables():
    print("[*] Generating Publication Summary Tables (Markdown & LaTeX)...\n")
    eval_dir = os.path.join("data", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Load Data
    mutation_path = os.path.join("data", "mutation_perturbation_results.csv")
    null_path = os.path.join("data", "null_model_control_summary.csv")

    if not os.path.exists(mutation_path):
        raise FileNotFoundError(f"[X] Missing {mutation_path}")

    df_mut = pd.read_csv(mutation_path)

    # Train a quick logistic regression to extract feature importance coefficients
    feature_cols = [
        "delta_shortest_path",
        "global_efficiency_drop",
        "delta_current_flow",
        "delta_bridging_betweenness"
    ]
    X = df_mut[feature_cols].fillna(0.0)
    y = df_mut["label"].values

    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    # --- TABLE 1: Representative Top Mutations Summary ---
    print("--- Table 1: Representative Mutation Perturbation Impacts ---")
    top_muts = df_mut.head(10).copy()
    top_muts["Target"] = "TEM-1 (1ZG4)"
    top_muts["Classification Score"] = clf.predict_proba(top_muts[feature_cols])[:, 1].round(4)

    t1_display = pd.DataFrame({
        "Target": top_muts["Target"],
        "Mutation ID": top_muts["mutation_id"],
        "True Label": top_muts["label"],
        "Path Delta": top_muts["delta_shortest_path"].round(3),
        "Efficiency Drop": top_muts["global_efficiency_drop"].round(3),
        "Pred. Score": top_muts["Classification Score"]
    })

    print(t1_display.to_markdown(index=False))
    print("\n[LaTeX Code]:")
    print(t1_display.to_latex(index=False))
    print("\n" + "=" * 50 + "\n")

    # --- TABLE 2: Feature Importance Coefficients (Recommended Addition) ---
    print("--- Table 2: Network Disruption Feature Coefficients (Logistic Regression) ---")
    df_coef = pd.DataFrame({
        "Network Disruption Feature": [
            "Δ Shortest Path Length",
            "Global Efficiency Drop",
            "Δ Current-Flow Centrality",
            "Δ Bridging Betweenness"
        ],
        "Coefficient (Beta)": clf.coef_[0].round(4),
        "Absolute Weight": abs(clf.coef_[0]).round(4)
    }).sort_values(by="Absolute Weight", ascending=False)

    print(df_coef.to_markdown(index=False))
    print("\n[LaTeX Code]:")
    print(df_coef.to_latex(index=False))
    print("\n" + "=" * 50 + "\n")

    # --- TABLE 3: Multi-Cutoff Robustness Summary (Recommended Addition) ---
    if os.path.exists(null_path):
        print("--- Table 3: Multi-Cutoff Null Model Validation Summary ---")
        df_null = pd.read_csv(null_path)
        print(df_null.to_markdown(index=False))
        print("\n[LaTeX Code]:")
        print(df_null.to_latex(index=False))

    # Save tables to disk
    t1_display.to_csv(os.path.join(eval_dir, "table1_top_mutations.csv"), index=False)
    df_coef.to_csv(os.path.join(eval_dir, "table2_feature_coefficients.csv"), index=False)
    print(f"\n[+] All summary tables successfully compiled and saved to {eval_dir}/")


if __name__ == "__main__":
    generate_tables()