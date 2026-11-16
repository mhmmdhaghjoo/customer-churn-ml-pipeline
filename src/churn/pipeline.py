"""The end-to-end run: load, validate, snapshot, test, segment, fit, evaluate, report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from churn import plots
from churn.config import Settings
from churn.evaluate import bootstrap_auc_difference, decile_table, evaluate
from churn.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_snapshot,
    feature_matrix,
    orders_table,
)
from churn.model import candidate_models
from churn.segment import segment_customers
from churn.stats import categorical_tests, numeric_tests
from churn.validate import load_data

logger = logging.getLogger(__name__)


def run(settings: Settings) -> dict[str, object]:
    """Execute the whole pipeline and write every artefact to ``settings.report_dir``."""
    report_dir = settings.report_dir
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    customers, products, order_lines = load_data(settings.data_dir)
    orders = orders_table(order_lines, products)

    def snapshot(cutoff: str) -> pd.DataFrame:
        return build_snapshot(orders, customers, cutoff, settings.horizon_days, settings.active_days)

    train = pd.concat([snapshot(c) for c in settings.train_cutoffs], ignore_index=True)
    test = snapshot(settings.test_cutoff)
    X_train, y_train = feature_matrix(train)
    X_test, y_test = feature_matrix(test)

    # 1. Do churned and retained customers differ? Tested on the latest
    # training snapshot only: pooling snapshots would count the same customer
    # several times and make the p-values look better than they are.
    latest_train = train[train["cutoff"] == train["cutoff"].max()]
    numeric_tests(latest_train, NUMERIC_FEATURES).to_csv(report_dir / "tests_numeric.csv", index=False)
    categorical_tests(latest_train, CATEGORICAL_FEATURES).to_csv(
        report_dir / "tests_categorical.csv", index=False
    )

    # 2. Segments, described on the test snapshot (the most recent customers).
    _, profile, silhouettes = segment_customers(test, settings.segment_range, settings.seed)
    profile.to_csv(report_dir / "segments.csv")
    silhouettes.to_csv(report_dir / "segments_silhouette.csv", index=False)

    # 3. Fit every candidate on the earlier snapshots, score the later one.
    metrics: dict[str, dict[str, float]] = {}
    scores: dict[str, pd.Series] = {}
    fitted = {}
    for name, pipeline in candidate_models(settings.seed).items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        scores[name] = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=X_test.index)
        metrics[name] = evaluate(y_test, scores[name].to_numpy())
        logger.info(
            "%-20s ROC AUC %.3f  PR AUC %.3f", name, metrics[name]["roc_auc"], metrics[name]["pr_auc"]
        )

    full_models = [name for name in metrics if name != "recency_only"]
    best = max(full_models, key=lambda name: metrics[name]["roc_auc"])
    gain, low, high = bootstrap_auc_difference(
        y_test.to_numpy(), scores[best].to_numpy(), scores["recency_only"].to_numpy(), seed=settings.seed
    )

    # 4. Explain and persist the best model.
    importance = permutation_importance(
        fitted[best], X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=settings.seed
    )
    importance_table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    deciles = decile_table(y_test, scores[best].to_numpy())

    importance_table.to_csv(report_dir / "feature_importance.csv", index=False)
    deciles.to_csv(report_dir / "deciles.csv")
    pd.DataFrame(metrics).T.to_csv(report_dir / "model_comparison.csv", index_label="model")
    joblib.dump(fitted[best], report_dir / "best_model.joblib")

    aucs = {name: values["roc_auc"] for name, values in metrics.items()}
    plots.roc_curves(y_test, scores, aucs, report_dir / "figures" / "roc_curves.png")
    plots.decile_chart(deciles, float(y_test.mean()), report_dir / "figures" / "deciles.png")
    plots.importance_chart(importance_table, report_dir / "figures" / "feature_importance.png")

    summary: dict[str, object] = {
        "train_cutoffs": list(settings.train_cutoffs),
        "test_cutoff": settings.test_cutoff,
        "horizon_days": settings.horizon_days,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_churn_rate": float(train[TARGET].mean()),
        "test_churn_rate": float(test[TARGET].mean()),
        "best_model": best,
        "auc_gain_over_recency_baseline": {"estimate": gain, "ci95_low": low, "ci95_high": high},
        "metrics": metrics,
    }
    _write_json(report_dir / "metrics.json", summary)
    logger.info(
        "best model: %s (ROC AUC +%.3f over the recency baseline, 95%% CI %.3f to %.3f)",
        best,
        gain,
        low,
        high,
    )
    return summary


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
