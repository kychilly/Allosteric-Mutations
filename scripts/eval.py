import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer


def run_evaluation_pipeline():
    print("[*] Starting Quantitative Evaluation & Predictive Modeling Pipeline...\n")

    # 1. Load Configuration
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[X] Config file missing at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Locate Mutation Perturbation & Feature Data
    # Expecting a CSV containing mutation records with network disruption features and a binary label (1 = Resistance, 0 = Neutral)
    data_path = os.path.join("data", "mutation_perturbation_results.csv")

    if not os.path.exists(data_path):
        print(f"[!] Notice: {data_path} not found. Generating synthetic template/mock feature set for demonstration.")
        generate_mock_mutation_dataset(data_path)

    df = pd.read_csv(data_path)
    print(f"[✓] Loaded mutation dataset with {len(df)} entries from {data_path}")

    # Required feature columns expected in the dataset
    feature_cols = [
        "delta_shortest_path",
        "global_efficiency_drop",
        "delta_current_flow",
        "delta_bridging_betweenness"
    ]

    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"[X] Expected feature column '{col}' missing from dataset CSV.")

    if "label" not in df.columns:
        raise KeyError("[X] Expected target column 'label' (1 for resistance, 0 for neutral) missing from dataset.")

    # 3. Statistical Rigor: Mann-Whitney U Tests
    print("\n--- 3. Statistical Validation (Mann-Whitney U Tests) ---")
    resistance_group = df[df["label"] == 1]
    neutral_group = df[df["label"] == 0]

    stat_results = []
    for col in feature_cols:
        res_vals = resistance_group[col].dropna()
        neut_vals = neutral_group[col].dropna()

        # Two-sided Mann-Whitney U test
        u_stat, p_val = mannwhitneyu(res_vals, neut_vals, alternative='two-sided')

        sig_label = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "NS"))
        print(f"    - Feature: {col:<30} | U-Stat: {u_stat:<8.2f} | p-value: {p_val:.4e} [{sig_label}]")

        stat_results.append({
            "feature": col,
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05
        })

    # 4. Predictive Modeling: Classifier & Cross-Validation
    print("\n--- 4. Predictive Modeling & Machine Learning Evaluation ---")
    X = df[feature_cols].fillna(0.0)
    y = df["label"].values

    # Initialize models
    models = {
        "Logistic Regression (L2)": LogisticRegression(penalty='l2', C=1.0, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_results = []

    for name, model in models.items():
        # Perform 5-fold cross-validation using ROC-AUC
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)

        print(f"    - {name:<30} | 5-Fold CV ROC-AUC: {mean_auc:.4f} (± {std_auc:.4f})")

        eval_results.append({
            "model": name,
            "mean_roc_auc": round(float(mean_auc), 4),
            "std_roc_auc": round(float(std_auc), 4)
        })

        # 5. Save Evaluation Summary Artifacts to a dedicated subfolder
        eval_dir = os.path.join("data", "eval")
        os.makedirs(eval_dir, exist_ok=True)

        stats_df = pd.DataFrame(stat_results)
        stats_df.to_csv(os.path.join(eval_dir, "statistical_tests_summary.csv"), index=False)

        ml_df = pd.DataFrame(eval_results)
        ml_df.to_csv(os.path.join(eval_dir, "ml_classification_summary.csv"), index=False)

        print(f"\n[+] Evaluation pipeline complete! Summaries saved to {eval_dir}/")

def generate_mock_mutation_dataset(filepath):
    """Generates a synthetic feature dataset matching structural perturbation outputs if missing."""
    np.random.seed(42)
    n_samples = 150

    # Simulate data with distinct distributions for resistance (label=1) vs neutral (label=0)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    data = {
        "mutation_id": [f"MUT_{i}" for i in range(n_samples)],
        "label": labels,
        "delta_shortest_path": np.where(labels == 1, np.random.normal(2.5, 0.8, n_samples),
                                        np.random.normal(0.4, 0.3, n_samples)),
        "global_efficiency_drop": np.where(labels == 1, np.random.normal(0.18, 0.05, n_samples),
                                           np.random.normal(0.02, 0.01, n_samples)),
        "delta_current_flow": np.where(labels == 1, np.random.normal(0.035, 0.01, n_samples),
                                       np.random.normal(0.005, 0.002, n_samples)),
        "delta_bridging_betweenness": np.where(labels == 1, np.random.normal(0.015, 0.004, n_samples),
                                               np.random.normal(0.001, 0.0005, n_samples))
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"    -> Generated mock mutation dataset at {filepath}")


if __name__ == "__main__":
    run_evaluation_pipeline()