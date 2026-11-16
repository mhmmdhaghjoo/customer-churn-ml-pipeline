"""Metrics for a churn model whose scores are used to rank customers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def lift_at(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.1) -> float:
    """Churn rate among the top ``fraction`` highest-scored customers divided
    by the overall churn rate. 1.0 means no better than picking at random."""
    y_true = np.asarray(y_true)
    n_top = max(1, int(round(len(y_true) * fraction)))
    top = np.argsort(-np.asarray(scores), kind="stable")[:n_top]
    return float(y_true[top].mean() / y_true.mean())


def evaluate(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "lift_top_10pct": lift_at(y, scores, 0.10),
        "lift_top_20pct": lift_at(y, scores, 0.20),
    }


def bootstrap_auc_difference(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """ROC AUC of model A minus model B on the same customers, with a 95 %
    bootstrap interval. If the interval excludes 0, the gap is not noise."""
    y_true, scores_a, scores_b = map(np.asarray, (y_true, scores_a, scores_b))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    while len(differences) < n_boot:
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue  # a resample needs both classes
        differences.append(
            roc_auc_score(y_true[idx], scores_a[idx]) - roc_auc_score(y_true[idx], scores_b[idx])
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    point = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    return float(point), float(low), float(high)


def decile_table(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Observed vs. predicted churn per score decile (1 = highest risk)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": scores})
    frame["decile"] = 10 - pd.qcut(frame["score"].rank(method="first"), 10, labels=False)
    table = frame.groupby("decile").agg(
        customers=("y", "size"), predicted_churn=("score", "mean"), observed_churn=("y", "mean")
    )
    table["cumulative_share_of_churners"] = frame.groupby("decile")["y"].sum().cumsum() / frame["y"].sum()
    return table
