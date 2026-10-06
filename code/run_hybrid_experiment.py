"""Run the locked expert-only, ML-only, and hybrid DevCARE-SE experiment.

The final 30% repository-wise temporal test block is preserved.  The earlier
70% is split again by time into development train/validation blocks so the
hybrid mixing coefficient is never selected on the final test observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics import classification_metrics
from pipeline_utils import FEATURES, queue_metric


SEED = 42
WEIGHTS = np.asarray([0.32, 0.24, 0.18, 0.26])
CODE_WEIGHTS = WEIGHTS[:3] / WEIGHTS[:3].sum()


# Fixed to the original cluster ordering so finite Monte Carlo intervals remain
# exactly reproducible after replacing private repository identifiers with R01--R14.
CLUSTER_BOOTSTRAP_ORDER = [
    "R02", "R03", "R04", "R05", "R06", "R07", "R08",
    "R01", "R14", "R09", "R10", "R13", "R11", "R12",
]


def cluster_bootstrap_order(repositories: np.ndarray) -> np.ndarray:
    observed = set(np.unique(repositories))
    if observed == set(CLUSTER_BOOTSTRAP_ORDER):
        return np.asarray(CLUSTER_BOOTSTRAP_ORDER, dtype=object)
    return np.unique(repositories)


def temporal_three_way_split(df: pd.DataFrame) -> pd.Series:
    split = pd.Series(index=df.index, dtype="string")
    for _, group in df.groupby("repository_id"):
        ordered = group.sort_values(["merged_at", "pr_id"], kind="stable")
        n = len(ordered)
        test_start = max(2, min(n - 1, math.floor(0.70 * n)))
        development = ordered.iloc[:test_start]
        validation_start = max(1, min(len(development) - 1, math.floor(0.75 * len(development))))
        split.loc[development.index[:validation_start]] = "train"
        split.loc[development.index[validation_start:]] = "validation"
        split.loc[ordered.index[test_start:]] = "test"
    return split


def thresholds(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([float(np.percentile(frame[column], 90)) for column in FEATURES[:3]])


def expert_scores(frame: pd.DataFrame, caps: np.ndarray, context: bool = True) -> np.ndarray:
    normalized = []
    for column, cap in zip(FEATURES[:3], caps):
        values = frame[column].to_numpy(float)
        normalized.append(np.minimum(1.0, values / cap) if cap > 0 else (values > 0).astype(float))
    matrix = np.column_stack(normalized)
    if context:
        return matrix @ WEIGHTS[:3] + WEIGHTS[3] * frame[FEATURES[3]].to_numpy(float)
    return matrix @ CODE_WEIGHTS


def estimators() -> dict[str, tuple[object, list[str]]]:
    code = FEATURES[:3]
    context = FEATURES
    return {
        "Logistic Code": (Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000, random_state=SEED)),
            ]
        ), code),
        "Logistic Context": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2", C=1.0, solver="lbfgs", max_iter=3000, random_state=SEED
                    ),
                ),
            ]
        ),
        "Random Forest Code": (RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ), code),
        "Random Forest Context": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        ),
        "LightGBM Code": (LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        ), code),
        "LightGBM Context": LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def normalized_single_scores(frame: pd.DataFrame, caps: np.ndarray) -> dict[str, np.ndarray]:
    scores = {}
    for index, name in enumerate(("C1 Churn", "C2 Complexity", "C3 Static")):
        values = frame[FEATURES[index]].to_numpy(float)
        cap = caps[index]
        scores[name] = np.minimum(1.0, values / cap) if cap > 0 else (values > 0).astype(float)
    scores["C4 Unfamiliarity"] = frame[FEATURES[3]].to_numpy(float)
    return scores


def compact_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    result = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }
    for fraction in (0.05, 0.10, 0.20):
        queue = queue_metric(y, score, fraction)
        tag = int(fraction * 100)
        result[f"k_at_{tag}"] = queue["k"]
        result[f"positives_at_{tag}"] = queue["positives_at_k"]
        result[f"precision_at_{tag}"] = queue["precision_at_k"]
        result[f"recall_at_{tag}"] = queue["recall_at_k"]
        result[f"lift_at_{tag}"] = queue["lift_at_k"]
    return result


def paired_bootstrap(
    y: np.ndarray, reference: np.ndarray, candidate: np.ndarray, samples: int = 5000
) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(samples):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        differences.append(
            average_precision_score(y[indices], candidate[indices])
            - average_precision_score(y[indices], reference[indices])
        )
    values = np.asarray(differences)
    return {
        "observed_delta_pr_auc": float(
            average_precision_score(y, candidate) - average_precision_score(y, reference)
        ),
        "ci_2_5": float(np.percentile(values, 2.5)),
        "ci_97_5": float(np.percentile(values, 97.5)),
        "positive_rate": float((values > 0).mean()),
        "valid_samples": int(values.size),
    }


def repository_cluster_bootstrap(
    repositories: np.ndarray,
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    samples: int = 5000,
) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    unique = cluster_bootstrap_order(repositories)
    differences = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(repositories == repository) for repository in sampled])
        if np.unique(y[indices]).size < 2:
            continue
        differences.append(
            average_precision_score(y[indices], candidate[indices])
            - average_precision_score(y[indices], reference[indices])
        )
    values = np.asarray(differences)
    return {
        "observed_delta_pr_auc": float(
            average_precision_score(y, candidate) - average_precision_score(y, reference)
        ),
        "ci_2_5": float(np.percentile(values, 2.5)),
        "ci_97_5": float(np.percentile(values, 97.5)),
        "positive_rate": float((values > 0).mean()),
        "valid_samples": int(values.size),
        "clusters": int(unique.size),
    }


def repository_cluster_point_bootstrap(
    repositories: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    samples: int = 5000,
) -> dict[str, float | int]:
    """Bootstrap a pooled PR-AUC while preserving repository-level dependence."""
    rng = np.random.default_rng(SEED)
    unique = cluster_bootstrap_order(repositories)
    estimates = []
    for _ in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(repositories == repository) for repository in sampled])
        if np.unique(y[indices]).size < 2:
            continue
        estimates.append(average_precision_score(y[indices], score[indices]))
    values = np.asarray(estimates)
    return {
        "observed_pr_auc": float(average_precision_score(y, score)),
        "ci_2_5": float(np.percentile(values, 2.5)),
        "ci_97_5": float(np.percentile(values, 97.5)),
        "valid_samples": int(values.size),
        "clusters": int(unique.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="results/training/devcare_outcome_dataset.csv")
    parser.add_argument("--output-dir", default="ml/hybrid")
    parser.add_argument("--provenance", default="Frozen labeled dataset artifact.")
    parser.add_argument(
        "--max-analysis-error-ratio",
        type=float,
        default=None,
        help="Optional post-hoc completeness sensitivity filter; 0 keeps only rows with no analyzer-error files.",
    )
    args = parser.parse_args()
    source = Path(args.dataset)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    source_rows_before_filter = len(df)
    if args.max_analysis_error_ratio is not None:
        if "analysis_error_ratio" not in df.columns:
            raise ValueError("analysis_error_ratio is required for the completeness sensitivity filter")
        df = df[df.analysis_error_ratio <= args.max_analysis_error_ratio].copy()
        if df.empty:
            raise ValueError("completeness sensitivity filter removed all rows")
    df["merged_at"] = pd.to_datetime(df["merged_at"], utc=True, errors="raise", format="mixed")
    df["split"] = temporal_three_way_split(df)
    train = df[df.split == "train"].copy()
    validation = df[df.split == "validation"].copy()
    test = df[df.split == "test"].copy()
    development = df[df.split != "test"].copy()
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if part.outcome_proxy.nunique() != 2:
            raise RuntimeError(f"{name} does not contain both outcome classes")

    train_caps = thresholds(train)
    validation_expert = expert_scores(validation, train_caps)
    y_train = train.outcome_proxy.astype(int).to_numpy()
    y_validation = validation.outcome_proxy.astype(int).to_numpy()
    y_test = test.outcome_proxy.astype(int).to_numpy()

    alpha_grid = np.round(np.linspace(0, 1, 21), 2)
    selection_rows = []
    selected_alpha: dict[str, float] = {}
    context_estimators = {name: value for name, value in estimators().items() if name.endswith("Context")}
    for model_name, estimator in context_estimators.items():
        if isinstance(estimator, tuple):
            estimator, feature_columns = estimator
        else:
            feature_columns = FEATURES
        estimator.fit(train[feature_columns], y_train)
        validation_ml = estimator.predict_proba(validation[feature_columns])[:, 1]
        model_rows = []
        for alpha in alpha_grid:
            hybrid = alpha * validation_expert + (1 - alpha) * validation_ml
            row = {
                "model": model_name,
                "alpha_expert": float(alpha),
                "validation_pr_auc": float(average_precision_score(y_validation, hybrid)),
            }
            selection_rows.append(row)
            model_rows.append(row)
        best = max(model_rows, key=lambda row: (row["validation_pr_auc"], -row["alpha_expert"]))
        selected_alpha[model_name] = best["alpha_expert"]

    development_caps = thresholds(development)
    test_expert = expert_scores(test, development_caps)
    test_code_rule = expert_scores(test, development_caps, context=False)
    predictions = test[["repository_id", "repository_name", "pr_id", "merged_at", "outcome_proxy"]].copy()
    predictions["Expert Code-only"] = test_code_rule
    predictions["Expert DevCARE-SE"] = test_expert
    single_scores = normalized_single_scores(test, development_caps)
    for single_name, single_score in single_scores.items():
        predictions[single_name] = single_score
    score_map: dict[str, np.ndarray] = {
        "Expert Code-only": test_code_rule,
        "Expert DevCARE-SE": test_expert,
        **single_scores,
    }

    for model_name, estimator_value in estimators().items():
        if isinstance(estimator_value, tuple):
            estimator, feature_columns = estimator_value
        else:
            estimator, feature_columns = estimator_value, FEATURES
        estimator.fit(development[feature_columns], development.outcome_proxy.astype(int))
        ml_score = estimator.predict_proba(test[feature_columns])[:, 1]
        score_map[model_name] = ml_score
        predictions[model_name] = ml_score
        if model_name.endswith("Context"):
            alpha = selected_alpha[model_name]
            hybrid_name = f"Hybrid Expert + {model_name.replace(' Context', '')}"
            hybrid_score = alpha * test_expert + (1 - alpha) * ml_score
            score_map[hybrid_name] = hybrid_score
            predictions[hybrid_name] = hybrid_score

    result_rows = [{"method": name, **compact_metrics(y_test, score)} for name, score in score_map.items()]
    result_frame = pd.DataFrame(result_rows).sort_values("pr_auc", ascending=False)

    rf_name = "Random Forest Context"
    hybrid_rf_name = "Hybrid Expert + Random Forest"
    hybrid_logistic_name = "Hybrid Expert + Logistic"
    uncertainty = {
        "expert_context_vs_code": paired_bootstrap(
            y_test, score_map["Expert Code-only"], score_map["Expert DevCARE-SE"]
        ),
        "logistic_context_vs_code": paired_bootstrap(
            y_test, score_map["Logistic Code"], score_map["Logistic Context"]
        ),
        "lightgbm_context_vs_code": paired_bootstrap(
            y_test, score_map["LightGBM Code"], score_map["LightGBM Context"]
        ),
        "hybrid_rf_vs_expert": paired_bootstrap(
            y_test, score_map["Expert DevCARE-SE"], score_map[hybrid_rf_name]
        ),
        "hybrid_rf_vs_rf": paired_bootstrap(y_test, score_map[rf_name], score_map[hybrid_rf_name]),
        "rf_context_vs_expert": paired_bootstrap(
            y_test, score_map["Expert DevCARE-SE"], score_map[rf_name]
        ),
        "rf_context_vs_code": paired_bootstrap(
            y_test, score_map["Random Forest Code"], score_map[rf_name]
        ),
        "rf_context_vs_expert_clustered": repository_cluster_bootstrap(
            test.repository_id.to_numpy(), y_test, score_map["Expert DevCARE-SE"], score_map[rf_name]
        ),
        "rf_context_vs_code_clustered": repository_cluster_bootstrap(
            test.repository_id.to_numpy(), y_test, score_map["Random Forest Code"], score_map[rf_name]
        ),
        "hybrid_logistic_vs_c2": paired_bootstrap(
            y_test, score_map["C2 Complexity"], score_map[hybrid_logistic_name]
        ),
        "hybrid_logistic_vs_expert": paired_bootstrap(
            y_test, score_map["Expert DevCARE-SE"], score_map[hybrid_logistic_name]
        ),
        "hybrid_logistic_vs_expert_clustered": repository_cluster_bootstrap(
            test.repository_id.to_numpy(),
            y_test,
            score_map["Expert DevCARE-SE"],
            score_map[hybrid_logistic_name],
        ),
    }
    selected_point_intervals = {
        name: repository_cluster_point_bootstrap(
            test.repository_id.to_numpy(), y_test, score_map[name]
        )
        for name in (
            "C2 Complexity",
            "Expert DevCARE-SE",
            "Logistic Context",
            hybrid_logistic_name,
        )
    }

    per_repository = []
    for repository, group in predictions.groupby("repository_name"):
        indices = group.index.to_numpy()
        positions = test.index.get_indexer(indices)
        labels = y_test[positions]
        for name, score in score_map.items():
            repository_score = score[positions]
            row = {"repository": repository, "method": name, "n": len(labels), "positives": int(labels.sum())}
            if np.unique(labels).size == 2:
                row.update(
                    roc_auc=float(roc_auc_score(labels, repository_score)),
                    pr_auc=float(average_precision_score(labels, repository_score)),
                )
            else:
                row.update(roc_auc=np.nan, pr_auc=np.nan)
            per_repository.append(row)

    split_summary = (
        df.groupby("split")
        .agg(n=("pr_id", "size"), positives=("outcome_proxy", "sum"))
        .reset_index()
    )
    split_summary["prevalence"] = split_summary.positives / split_summary.n

    pd.DataFrame(selection_rows).to_csv(output / "alpha_validation_grid.csv", index=False)
    result_frame.to_csv(output / "hybrid_model_comparison.csv", index=False)
    predictions.to_csv(output / "hybrid_test_predictions.csv", index=False)
    per_repository_frame = pd.DataFrame(per_repository)
    per_repository_frame.to_csv(output / "per_repository_metrics.csv", index=False)
    macro_by_repository = (
        per_repository_frame.dropna(subset=["roc_auc", "pr_auc"])
        .groupby("method", as_index=False)
        .agg(
            macro_roc_auc=("roc_auc", "mean"),
            macro_pr_auc=("pr_auc", "mean"),
            valid_repositories=("repository", "nunique"),
        )
        .sort_values("macro_pr_auc", ascending=False)
    )
    macro_by_repository.to_csv(output / "macro_by_repository_metrics.csv", index=False)
    split_summary.to_csv(output / "split_summary.csv", index=False)
    protocol = {
        "source_artifact": str(source),
        "source_rows": len(df),
        "source_rows_before_filter": source_rows_before_filter,
        "max_analysis_error_ratio": args.max_analysis_error_ratio,
        "split_policy": "repository-wise temporal; first 70% development, last 30% test; last 25% of development used for alpha validation",
        "expert_weights": WEIGHTS.tolist(),
        "normalization": "P90 caps estimated without the evaluated block",
        "selected_alpha_expert": selected_alpha,
        "bootstrap": uncertainty,
        "selected_pr_auc_repository_cluster_intervals": selected_point_intervals,
        "data_provenance": args.provenance,
    }
    (output / "hybrid_protocol_and_uncertainty.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(result_frame.to_string(index=False))
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
