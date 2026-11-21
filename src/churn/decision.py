"""Turn churn scores into a contact decision: thresholds, campaign value and calibration.

The model ranks customers by their risk of not ordering again. It does not
estimate what a retention offer would change. Everything here that involves
money therefore depends on assumptions the user supplies (cost per contact,
value of a retained customer, share of contacted churners an offer retains),
and the functions keep those assumptions explicit instead of hiding them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.config import Settings
from churn.features import TARGET, build_snapshot, feature_matrix, orders_table
from churn.segment import segment_customers
from churn.validate import load_data


@dataclass(frozen=True)
class CampaignAssumptions:
    """Inputs the data cannot provide. They must be agreed with the business."""

    contact_cost: float  # EUR per contacted customer (voucher, call, email production)
    value_per_retained: float  # EUR margin from a customer who would have churned and stays
    save_rate: float  # share of contacted churners the offer retains (0 to 1)

    def __post_init__(self) -> None:
        if self.contact_cost < 0 or self.value_per_retained < 0:
            raise ValueError("costs and values must not be negative")
        if not 0 <= self.save_rate <= 1:
            raise ValueError("save_rate must be between 0 and 1")


def _arrays(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y_true and scores differ in length: {y.shape} vs {p.shape}")
    if len(y) == 0:
        raise ValueError("no customers to score")
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("scores must be probabilities between 0 and 1")
    return y, p


def threshold_table(
    y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, thresholds: list[float] | np.ndarray
) -> pd.DataFrame:
    """For each threshold: how many customers are flagged, and how many of them really churn.

    A customer is flagged when their score is at or above the threshold.
    Precision is undefined when nobody is flagged and is reported as NaN.
    """
    y, p = _arrays(y_true, scores)
    total_churners = int(y.sum())
    rows = []
    for t in thresholds:
        flagged = p >= t
        n_flagged = int(flagged.sum())
        tp = int((flagged & (y == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "flagged": n_flagged,
                "flagged_share": n_flagged / len(y),
                "churners_flagged": tp,
                "precision": tp / n_flagged if n_flagged else np.nan,
                "recall": tp / total_churners if total_churners else np.nan,
            }
        )
    return pd.DataFrame(rows)


def campaign_value(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    threshold: float,
    assumptions: CampaignAssumptions,
) -> dict[str, float]:
    """Value of contacting everyone at or above ``threshold``, computed two ways.

    ``backtest_*`` uses the churn that actually happened in the scored period.
    ``expected_*`` uses the model's own probabilities, which is all a real
    campaign has before the outcome is known. If the two differ a lot, the
    model is miscalibrated for that group and its probabilities should not be
    read as literal chances.
    """
    y, p = _arrays(y_true, scores)
    flagged = p >= threshold
    n_flagged = int(flagged.sum())
    cost = n_flagged * assumptions.contact_cost
    backtest_churners = float(y[flagged].sum())
    expected_churners = float(p[flagged].sum())
    unit = assumptions.save_rate * assumptions.value_per_retained
    return {
        "flagged": float(n_flagged),
        "cost": cost,
        "backtest_churners_reached": backtest_churners,
        "backtest_net_value": backtest_churners * unit - cost,
        "expected_churners_reached": expected_churners,
        "expected_net_value": expected_churners * unit - cost,
    }


def break_even_threshold(assumptions: CampaignAssumptions) -> float:
    """Lowest calibrated churn probability at which a contact pays for itself.

    Contacting a customer with churn probability q is worth
    q * save_rate * value_per_retained - contact_cost, which is positive when
    q > contact_cost / (save_rate * value_per_retained). Only meaningful if
    the scores are calibrated.
    """
    unit = assumptions.save_rate * assumptions.value_per_retained
    if unit == 0:
        return float("inf")
    return assumptions.contact_cost / unit


def calibration_table(
    y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Mean predicted vs. observed churn in equal-sized score bins, with a Wilson 95 % interval."""
    y, p = _arrays(y_true, scores)
    if n_bins < 1 or n_bins > len(y):
        raise ValueError("n_bins must be between 1 and the number of customers")
    order = np.argsort(p, kind="stable")
    rows = []
    for idx in np.array_split(order, n_bins):
        n = len(idx)
        observed = y[idx].mean()
        low, high = _wilson(observed, n)
        rows.append(
            {
                "customers": n,
                "mean_predicted": float(p[idx].mean()),
                "observed": float(observed),
                "observed_low": low,
                "observed_high": high,
            }
        )
    return pd.DataFrame(rows)


def _wilson(share: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    centre = (share + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(share * (1 - share) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def score_test_snapshot(settings: Settings, model_path: Path | None = None) -> pd.DataFrame:
    """Rebuild the out-of-time test snapshot and score it with the saved model.

    Returns one row per customer with the features, the observed label,
    the churn score and the K-Means cluster used in the README.
    """
    model_path = model_path or settings.report_dir / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found: run `make data run` first")
    customers, products, order_lines = load_data(settings.data_dir)
    orders = orders_table(order_lines, products)
    snapshot = build_snapshot(
        orders, customers, settings.test_cutoff, settings.horizon_days, settings.active_days
    )
    X, _ = feature_matrix(snapshot)
    model = joblib.load(model_path)
    clusters, _, _ = segment_customers(snapshot, settings.segment_range, settings.seed)
    return snapshot.assign(score=model.predict_proba(X)[:, 1], cluster=clusters.to_numpy()).rename(
        columns={TARGET: "churned"}
    )
