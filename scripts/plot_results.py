import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


def truncate_auc(val, decimals=3):
    """Truncate a float to the specified number of decimal places without rounding."""
    multiplier = 10 ** decimals
    return int(val * multiplier) / multiplier


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

    # 3. Generate ROC-AUC Curve Plot using 5-Fold Cross-Validation
    print("[*] Performing 5-fold cross-validation and generating ROC-AUC curves...")
    X = df[feature_cols].fillna(0.0)
    y = df["label"].values

    # Logistic Regression Model (5-Fold CV Probabilities)
    classifier_lr = LogisticRegression(random_state=42)
    y_scores_lr = cross_val_predict(classifier_lr, X, y, cv=5, method="predict_proba")[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y, y_scores_lr)
    roc_auc_lr = truncate_auc(auc(fpr_lr, tpr_lr), 3)

    # Random Forest Classifier Model (5-Fold CV Probabilities)
    classifier_rf = RandomForestClassifier(random_state=42)
    y_scores_rf = cross_val_predict(classifier_rf, X, y, cv=5, method="predict_proba")[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y, y_scores_rf)
    roc_auc_rf = truncate_auc(auc(fpr_rf, tpr_rf), 3)

    # Plotting both curves with truncated values
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, color="#2ca02c", lw=2.5, label=f"Logistic Regression (AUC = {roc_auc_lr:.3f})")
    plt.plot(fpr_rf, tpr_rf, color="#ff7f0e", lw=2.0, linestyle="-.",
             label=f"Random Forest Classifier (AUC = {roc_auc_rf:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")

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