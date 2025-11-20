import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from train_validate import compute_thresholds

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibrated model on holdout set")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--features_csv", required=False, default=None, help="Path to extracted radiomics features")
    parser.add_argument("--model_dir", required=False, default="exports/compete_no_overfit_intermediate", help="Path with best_model.joblib")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_holdout(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    holdout_col = config.get("holdout_flag_col", "Split")
    holdout_value = config.get("holdout_value", "holdout")
    if holdout_col in df.columns:
        return df[df[holdout_col] == holdout_value].copy()
    return pd.DataFrame(columns=df.columns)


def patient_bootstrap_metric(df: pd.DataFrame, patient_col: str, target_col: str, prob_col: str, func, n_iter: int, seed: int) -> Tuple[float, Tuple[float, float], List[float]]:
    rng = np.random.default_rng(seed)
    patients = df[patient_col].unique()
    boot_vals = []
    for _ in range(n_iter):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = df[patient_col].isin(sampled)
        boot_vals.append(func(df.loc[idx, target_col], df.loc[idx, prob_col]))
    point = func(df[target_col], df[prob_col])
    ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
    return point, (ci_low, ci_high), boot_vals


def threshold_metrics(y_true: np.ndarray, probas: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probas >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1 = f1_score(y_true, preds)
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}


def plot_roc_pr(y_true: np.ndarray, probas: np.ndarray, export_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probas)
    auc = roc_auc_score(y_true, probas)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(export_dir / "roc_holdout.png", dpi=300)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, probas)
    ap = average_precision_score(y_true, probas)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(export_dir / "pr_holdout.png", dpi=300)
    plt.close()


def plot_calibration(y_true: np.ndarray, probas: np.ndarray, export_dir: Path) -> float:
    prob_true, prob_pred = calibration_curve(y_true, probas, n_bins=10, strategy="uniform")
    ece = np.mean(np.abs(prob_true - prob_pred))
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.tight_layout()
    plt.savefig(export_dir / "calibration_plot_BestModel.png", dpi=300)
    plt.close()
    return ece


def plot_threshold_curves(y_true: np.ndarray, probas: np.ndarray, export_dir: Path) -> None:
    prec, rec, thresh = precision_recall_curve(y_true, probas)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    plt.figure()
    plt.plot(thresh, prec[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(export_dir / "precision_vs_threshold.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(thresh, rec[:-1])
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(export_dir / "recall_vs_threshold.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(thresh, f1)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(export_dir / "f1_vs_threshold.png", dpi=300)
    plt.close()


def plot_score_distributions(df: pd.DataFrame, prob_col: str, target_col: str, export_dir: Path) -> None:
    plt.figure()
    sns.histplot(data=df, x=prob_col, hue=target_col, bins=20, element="step", stat="density", common_norm=False)
    plt.tight_layout()
    plt.savefig(export_dir / "score_hist_BestModel.png", dpi=300)
    plt.close()

    plt.figure()
    sns.violinplot(data=df, x=target_col, y=prob_col)
    plt.tight_layout()
    plt.savefig(export_dir / "prob_by_class_violin.png", dpi=300)
    plt.close()


def plot_confusion_matrices(y_true: np.ndarray, probas: np.ndarray, thresholds: Dict[str, float], export_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(thresholds), figsize=(12, 4))
    for ax, (label, thr) in zip(axes, thresholds.items()):
        preds = (probas >= thr).astype(int)
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Threshold {label}={thr:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(export_dir / "confusion_matrices_thresholds.png", dpi=300)
    plt.close()


def plot_reliability_diagram(y_true: np.ndarray, probas: np.ndarray, export_dir: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, probas, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="s")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Predicted probability bin")
    plt.ylabel("Observed fraction")
    plt.tight_layout()
    plt.savefig(export_dir / "reliability_diagram_binned.png", dpi=300)
    plt.close()


def plot_bootstrap_distributions(values: List[float], metric_name: str, export_dir: Path) -> None:
    plt.figure()
    sns.histplot(values, kde=True)
    plt.xlabel(metric_name)
    plt.tight_layout()
    plt.savefig(export_dir / f"{metric_name.lower()}_bootstrap_distribution.png", dpi=300)
    plt.close()


def plot_population_table(df: pd.DataFrame, target_col: str, export_dir: Path) -> None:
    counts = df[target_col].value_counts().sort_index()
    plt.figure()
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(export_dir / "population_table_plot.png", dpi=300)
    plt.close()


def plot_model_comparison(cv_path: Path, export_dir: Path) -> None:
    if not cv_path.exists():
        return
    df = pd.read_csv(cv_path)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="model", y="mean_f1", hue="k")
    plt.tight_layout()
    plt.savefig(export_dir / "model_comparison_bar.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    features_path = Path(args.features_csv) if args.features_csv else Path(config["output_root"]) / "radiomics_features.csv"
    df = pd.read_csv(features_path)
    holdout_df = filter_holdout(df, config)
    if holdout_df.empty:
        raise ValueError("No holdout data found. Ensure Split column marks holdout.")

    model_dir = Path(args.model_dir)
    bundle = joblib.load(model_dir / "best_model.joblib")
    calibrator = joblib.load(model_dir / "calibrator.joblib")
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    features = bundle["features"]

    export_dir = Path(config["output_root"]) / f"compete_no_overfit_FINAL_{datetime.now().strftime('%Y%m%d_%H%M')}"
    export_dir.mkdir(parents=True, exist_ok=True)

    X_holdout = holdout_df[features]
    y_holdout = holdout_df[config["stratify_label"]]
    X_holdout_proc = preprocessor.transform(X_holdout)
    raw_probs = model.predict_proba(X_holdout_proc)[:, 1]
    calibrated_probs = calibrator.predict_proba(raw_probs)[:, 1]

    thresholds = compute_thresholds(y_holdout.to_numpy(), calibrated_probs)

    metrics_records = []
    for label, thr in thresholds.items():
        metrics_records.append(
            {
                "threshold": label,
                **threshold_metrics(y_holdout.to_numpy(), calibrated_probs, thr),
            }
        )
    overall = {
        "auc": roc_auc_score(y_holdout, calibrated_probs),
        "ap": average_precision_score(y_holdout, calibrated_probs),
        "brier": brier_score_loss(y_holdout, calibrated_probs),
    }
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(export_dir / "metrics_summary_holdout.csv", index=False)
    (export_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2))

    oof_path = model_dir / "oof_predictions.csv"
    if oof_path.exists():
        Path(export_dir / "oof_predictions.csv").write_text(oof_path.read_text())
    plot_model_comparison(model_dir / "model_comparison.csv", export_dir)

    pd.DataFrame({"PatientID": holdout_df[config["patient_id_col"]], "LesionID": holdout_df[config["lesion_id_col"]], "Target": y_holdout, "raw_prob": raw_probs}).to_csv(export_dir / "holdout_predictions_raw.csv", index=False)
    pd.DataFrame({"PatientID": holdout_df[config["patient_id_col"]], "LesionID": holdout_df[config["lesion_id_col"]], "Target": y_holdout, "calibrated_prob": calibrated_probs}).to_csv(export_dir / "holdout_predictions_calibrated.csv", index=False)

    plot_roc_pr(y_holdout.to_numpy(), calibrated_probs, export_dir)
    ece = plot_calibration(y_holdout.to_numpy(), calibrated_probs, export_dir)
    plot_reliability_diagram(y_holdout.to_numpy(), calibrated_probs, export_dir)
    plot_threshold_curves(y_holdout.to_numpy(), calibrated_probs, export_dir)
    plot_score_distributions(holdout_df.assign(prob=calibrated_probs), "prob", config["stratify_label"], export_dir)
    plot_confusion_matrices(y_holdout.to_numpy(), calibrated_probs, thresholds, export_dir)
    plot_population_table(holdout_df, config["stratify_label"], export_dir)

    auc_point, auc_ci, auc_boot = patient_bootstrap_metric(
        holdout_df.assign(prob=calibrated_probs),
        config["patient_id_col"],
        config["stratify_label"],
        "prob",
        roc_auc_score,
        config.get("bootstrap_iterations", 1000),
        config.get("bootstrap_seed", 123),
    )
    ap_point, ap_ci, ap_boot = patient_bootstrap_metric(
        holdout_df.assign(prob=calibrated_probs),
        config["patient_id_col"],
        config["stratify_label"],
        "prob",
        average_precision_score,
        config.get("bootstrap_iterations", 1000),
        config.get("bootstrap_seed", 123),
    )

    plot_bootstrap_distributions(auc_boot, "AUC", export_dir)
    plot_bootstrap_distributions(ap_boot, "AP", export_dir)

    with open(export_dir / "report.md", "w") as f:
        f.write("# Holdout Evaluation\n")
        f.write(f"Best model: {bundle['config'].get('best_model', 'from CV selection')}\n\n")
        f.write(f"AUC: {auc_point:.3f} (95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f})\n\n")
        f.write(f"AP: {ap_point:.3f} (95% CI {ap_ci[0]:.3f}-{ap_ci[1]:.3f})\n\n")
        f.write(f"Thresholds used: {json.dumps(thresholds)}\n")
        f.write(f"ECE (calibrated): {ece:.3f}\n")
        f.write(f"Brier score: {overall['brier']:.3f}\n")


if __name__ == "__main__":
    main()

