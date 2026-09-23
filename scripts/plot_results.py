import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


def generate_plots():
    print("[*] Starting Results Plotting Pipeline...\n")

    # 1. Setup paths & style
    eval_dir = os.path.join("data", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Use clean academic plot aesthetics
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = ["#4C72B0", "#C44E52"]  # Custom palette: Neutral (Blue), Resistance (Red)

    data_path = os.path.join("data", "mutation_perturbation_results.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[X] Dataset not found at {data_path}. Run perturbation and evaluation steps first.")

    df = pd.read_csv(data_path)
    print(f"[✓] Loaded dataset with {len(df)} samples for plotting.")

    # Map numeric labels to descriptive names for cleaner plot legends
    df["Mutation Type"] = df["label"].map({0: "Neutral (0)", 1: "Resistance (1)"})

    # 2. Plot Grouped Violin/Box Plots for Network Disruption Features
    feature_cols = [
        "delta_shortest_path",
        "global_efficiency_drop",
        "delta_current_flow",
        "delta_bridging_betweenness"
    ]

    # Clean display names for axes
    feature_labels = {
        "delta_shortest_path": "Δ Shortest Path Length",
        "global_efficiency_drop": "Global Efficiency Drop",
        "delta_current_flow": "Δ Current-Flow Centrality",
        "delta_bridging_betweenness": "Δ Bridging Betweenness"
    }

    print("[*] Generating grouped violin/box perturbation comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        # Create a hybrid violin + box plot to show distributions and summary stats clearly
        sns.violinplot(
            data=df,
            x="Mutation Type",
            y=feature,
            hue="Mutation Type",
            palette=palette,
            inner="box",
            ax=ax,
            cut=0,
            legend=False
        )
        ax.set_title(feature_labels.get(feature, feature), fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Disruption Magnitude", fontsize=10)

    plt.suptitle("Structural Network Disruption Profiles: Neutral vs. Resistance Mutations", fontsize=14,
                 fontweight='bold', y=0.98)
    plt.tight_layout()

    dist_plot_path = os.path.join(eval_dir, "network_disruption_violin_plots.png")
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved distribution plots to {dist_plot_path}")

    # 3. Generate ROC-AUC Curve Plot
    print("[*] Training model and generating ROC-AUC curve...")
    X = df[feature_cols].fillna(0.0)
    y = df["label"].values

    # Train-test split for ROC visualization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)
    y_scores = classifier.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"Logistic Regression (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='bold')
    plt.title("ROC-AUC Curve for Allosteric Mutation Classification", fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    roc_plot_path = os.path.join(eval_dir, "roc_auc_curve.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved ROC-AUC plot to {roc_plot_path}")

    print(f"\n[+] Plotting complete! All visual assets successfully compiled into {eval_dir}/")


if __name__ == "__main__":
    generate_plots()