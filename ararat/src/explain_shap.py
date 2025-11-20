import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for holdout")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--features_csv", required=False, default=None, help="Path to extracted radiomics features")
    parser.add_argument("--model_dir", required=False, default="exports/compete_no_overfit_intermediate", help="Path with best_model.joblib")
    parser.add_argument("--export_dir", required=False, default=None, help="Target export directory (defaults to FINAL folder)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    features_path = Path(args.features_csv) if args.features_csv else Path(config["output_root"]) / "radiomics_features.csv"
    df = pd.read_csv(features_path)
    holdout_col = config.get("holdout_flag_col", "Split")
    holdout_df = df[df[holdout_col] == config.get("holdout_value", "holdout")]
    model_dir = Path(args.model_dir)
    bundle = joblib.load(model_dir / "best_model.joblib")
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    features = bundle["features"]

    if args.export_dir:
        export_dir = Path(args.export_dir)
    else:
        candidates = sorted(Path(config["output_root"]).glob("compete_no_overfit_FINAL_*"))
        export_dir = candidates[-1] if candidates else Path(config["output_root"]) / "compete_no_overfit_FINAL"
    export_dir.mkdir(parents=True, exist_ok=True)

    X_holdout = holdout_df[features]
    X_proc = preprocessor.transform(X_holdout)

    explainer = shap.TreeExplainer(model, feature_perturbation="interventional", model_output="probability")
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    feature_names = features

    shap.summary_plot(shap_values, X_proc, feature_names=feature_names, show=False, max_display=config.get("max_shap_display", 20))
    plt.tight_layout()
    plt.savefig(export_dir / "shap_beeswarm_top20.png", dpi=300)
    plt.close()

    sample_idx = list(range(min(20, X_proc.shape[0])))
    shap.decision_plot(explainer.expected_value, shap_values[sample_idx], X_proc[sample_idx], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(export_dir / "shap_decision_multi.png", dpi=300)
    plt.close()

    # Dependence plots for top 3 mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:3]
    for i, idx in enumerate(top_indices):
        shap.dependence_plot(idx, shap_values, X_proc, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(export_dir / f"shap_dependence_top{i+1}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()

