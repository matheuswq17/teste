import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from classes.preprocessing import TabularPreprocessor

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate classical models for GGG classification")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--features_csv", required=False, default=None, help="Path to extracted radiomics features")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_holdout(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    holdout_col = config.get("holdout_flag_col", "Split")
    holdout_value = config.get("holdout_value", "holdout")
    if holdout_col in df.columns:
        holdout_df = df[df[holdout_col] == holdout_value].copy()
        train_df = df[df[holdout_col] != holdout_value].copy()
    else:
        holdout_df = pd.DataFrame(columns=df.columns)
        train_df = df.copy()
    return train_df, holdout_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("original_")]


def rank_features_mutual_info(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    scores = mutual_info_classif(X, y, random_state=SEED)
    ranking = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    return ranking.head(k).index.tolist()


def rank_features_auc(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    aucs = {}
    for col in X.columns:
        try:
            aucs[col] = roc_auc_score(y, X[col])
        except ValueError:
            aucs[col] = 0.5
    ranking = pd.Series(aucs).sort_values(ascending=False)
    return ranking.head(k).index.tolist()


def remove_correlated(features: pd.DataFrame, threshold: float) -> List[str]:
    corr = features.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    keep = [c for c in features.columns if c not in to_drop]
    return keep


def make_models(config: Dict) -> Dict[str, object]:
    models = {
        "log_reg": LogisticRegression(
            C=float(config["log_reg"].get("C", 1.0)),
            penalty=config["log_reg"].get("penalty", "l2"),
            class_weight=config["log_reg"].get("class_weight", "balanced"),
            solver=config["log_reg"].get("solver", "liblinear"),
            max_iter=500,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=config["random_forest"].get("n_estimators", 500),
            max_depth=config["random_forest"].get("max_depth", 8),
            min_samples_leaf=config["random_forest"].get("min_samples_leaf", 2),
            class_weight=config["random_forest"].get("class_weight", "balanced_subsample"),
            random_state=SEED,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(**config["lightgbm"], random_state=SEED),
    }
    return models


def evaluate_predictions(y_true: np.ndarray, probas: np.ndarray) -> Dict[str, float]:
    preds = (probas >= 0.5).astype(int)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probas)
    ap = average_precision_score(y_true, probas)
    return {"f1": f1, "auc": auc, "ap": ap}


def cross_validate_models(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[config["stratify_label"]]
    groups = df[config["patient_id_col"]]

    models = make_models(config)
    selectors = ["mutual_info", "auc"]
    ks = config.get("feature_selection_ks", [40, 60, 80])
    results: List[Dict] = []
    oof_predictions: Dict[str, List] = {}

    for model_name, model in models.items():
        for selector in selectors:
            for k in ks:
                cv = StratifiedGroupKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_seed"])
                fold_probs = np.zeros_like(y, dtype=float)
                fold_ids = np.zeros_like(y, dtype=int)
                per_fold_scores = []
                for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    if selector == "mutual_info":
                        cols = rank_features_mutual_info(X_train, y_train, k)
                    else:
                        cols = rank_features_auc(X_train, y_train, k)
                    cols = remove_correlated(X_train[cols], config.get("correlation_threshold", 0.9))

                    preprocessor = TabularPreprocessor()
                    X_train_proc = preprocessor.fit_transform(X_train[cols])
                    X_val_proc = preprocessor.transform(X_val[cols])

                    clf = model
                    if model_name == "lightgbm":
                        clf = LGBMClassifier(**config["lightgbm"], random_state=SEED)
                        clf.fit(
                            X_train_proc,
                            y_train,
                            eval_set=[(X_val_proc, y_val)],
                            eval_metric=config["lightgbm"].get("eval_metric", "binary_logloss"),
                            verbose=False,
                        )
                    else:
                        clf.fit(X_train_proc, y_train)

                    val_proba = clf.predict_proba(X_val_proc)[:, 1]
                    fold_probs[val_idx] = val_proba
                    fold_ids[val_idx] = fold
                    per_fold_scores.append(evaluate_predictions(y_val, val_proba))

                mean_f1 = np.mean([m["f1"] for m in per_fold_scores])
                mean_auc = np.mean([m["auc"] for m in per_fold_scores])
                mean_ap = np.mean([m["ap"] for m in per_fold_scores])
                results.append(
                    {
                        "model": model_name,
                        "selector": selector,
                        "k": k,
                        "mean_f1": mean_f1,
                        "mean_auc": mean_auc,
                        "mean_ap": mean_ap,
                    }
                )
                key = f"{model_name}_{selector}_{k}"
                oof_predictions[key] = [fold_probs.tolist(), fold_ids.tolist()]

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values(by=["mean_f1", "mean_auc"], ascending=False).iloc[0]
    selection = {
        "model": best_row["model"],
        "selector": best_row["selector"],
        "k": int(best_row["k"]),
    }
    best_key = f"{selection['model']}_{selection['selector']}_{selection['k']}"
    return results_df, {"selection": selection, "oof": oof_predictions[best_key]}


def fit_preprocess_model(df: pd.DataFrame, config: Dict, selection: Dict) -> Tuple[object, TabularPreprocessor, List[str]]:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[config["stratify_label"]]

    if selection["selector"] == "mutual_info":
        cols = rank_features_mutual_info(X, y, selection["k"])
    else:
        cols = rank_features_auc(X, y, selection["k"])
    cols = remove_correlated(X[cols], config.get("correlation_threshold", 0.9))

    preprocessor = TabularPreprocessor()
    X_proc = preprocessor.fit_transform(X[cols])

    models = make_models(config)
    model = models[selection["model"]]
    if selection["model"] == "lightgbm":
        model = LGBMClassifier(**config["lightgbm"], random_state=SEED)
    model.fit(X_proc, y)
    return model, preprocessor, cols


class ProbabilityCalibrator:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, raw: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict(raw.reshape(-1, 1))
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        return np.vstack([1 - calibrated, calibrated]).T


def calibrate_probabilities(y_true: np.ndarray, probas: np.ndarray, method: str) -> ProbabilityCalibrator:
    if method == "platt":
        model = LogisticRegression()
        model.fit(probas.reshape(-1, 1), y_true)
        return ProbabilityCalibrator(model)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probas, y_true)

    class _IsoWrapper:
        def __init__(self, iso_model):
            self.iso_model = iso_model

        def predict(self, raw: np.ndarray) -> np.ndarray:
            return self.iso_model.predict(raw.reshape(-1))

    return ProbabilityCalibrator(_IsoWrapper(iso))


def compute_thresholds(y_true: np.ndarray, probas: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = thresholds[youden_idx]

    target_spec = 0.6
    spec = 1 - fpr
    spec_idx = np.argmin(np.abs(spec - target_spec))
    spec_threshold = thresholds[spec_idx]

    return {
        "default": 0.5,
        "youden": float(youden_threshold),
        "spec60": float(spec_threshold),
    }


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    features_path = Path(args.features_csv) if args.features_csv else Path(config["output_root"]) / "radiomics_features.csv"
    df = pd.read_csv(features_path)

    train_df, holdout_df = filter_holdout(df, config)
    cv_results, best_bundle = cross_validate_models(train_df, config)

    export_dir = Path(config["output_root"]) / "compete_no_overfit_intermediate"
    export_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(export_dir / "model_comparison.csv", index=False)

    best_selection = best_bundle["selection"]
    model, preprocessor, cols = fit_preprocess_model(train_df, config, best_selection)

    feature_map = pd.DataFrame({"feature_name": cols})
    feature_map.to_csv(export_dir / "feature_name_mapping.csv", index=False)

    # Recompute OOF predictions for selected model
    cv = StratifiedGroupKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_seed"])
    X = train_df[cols]
    y = train_df[config["stratify_label"]]
    groups = train_df[config["patient_id_col"]]
    oof_probs = np.zeros_like(y, dtype=float)
    oof_folds = np.zeros_like(y, dtype=int)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        pre = TabularPreprocessor()
        X_train_proc = pre.fit_transform(X_train)
        X_val_proc = pre.transform(X_val)
        clf = make_models(config)[best_selection["model"]]
        if best_selection["model"] == "lightgbm":
            clf = LGBMClassifier(**config["lightgbm"], random_state=SEED)
        clf.fit(X_train_proc, y_train)
        oof_probs[val_idx] = clf.predict_proba(X_val_proc)[:, 1]
        oof_folds[val_idx] = fold

    calibrator = calibrate_probabilities(y.to_numpy(), oof_probs, config.get("calibration", "isotonic"))
    thresholds = compute_thresholds(y.to_numpy(), calibrator.predict_proba(oof_probs.reshape(-1, 1))[:, 1])

    oof_df = pd.DataFrame(
        {
            "PatientID": train_df[config["patient_id_col"]],
            "LesionID": train_df[config["lesion_id_col"]],
            "Target": y,
            "oof_prob": oof_probs,
            "fold": oof_folds,
        }
    )
    oof_df.to_csv(export_dir / "oof_predictions.csv", index=False)

    summary = {
        "best_model": best_selection["model"],
        "selector": best_selection["selector"],
        "k": best_selection["k"],
        "thresholds": thresholds,
        "calibration": config.get("calibration", "isotonic"),
    }
    (export_dir / "cv_validation_summary.json").write_text(json.dumps(summary, indent=2))

    joblib.dump({"model": model, "preprocessor": preprocessor, "features": cols, "config": config}, export_dir / "best_model.joblib")
    joblib.dump(calibrator, export_dir / "calibrator.joblib")

    # Placeholder for holdout; only saving data here
    if not holdout_df.empty:
        holdout_df.to_csv(export_dir / "holdout_ready.csv", index=False)


if __name__ == "__main__":
    main()

