"""Build one row per customer as of a cutoff date.

Everything in a snapshot is computed from orders placed strictly before the
cutoff; the label looks only at the window after it. Keeping those two apart
is the whole point of this module, and `tests/test_features.py` checks it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "recency_days",
    "tenure_days",
    "n_orders",
    "revenue_total",
    "avg_order_value",
    "avg_items_per_order",
    "n_categories",
    "discount_share",
    "mean_gap_days",
    "recency_to_gap_ratio",
    "orders_last_90d",
    "orders_prev_90d",
    "age",
]
CATEGORICAL_FEATURES = ["country", "acquisition_channel"]
TARGET = "churned"


def orders_table(order_lines: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Collapse order lines to one row per order."""
    lines = order_lines.merge(products[["product_id", "category"]], on="product_id", how="left")
    lines["revenue"] = lines["quantity"] * lines["unit_price"] * (1 - lines["discount_pct"])
    lines["discounted"] = lines["discount_pct"] > 0
    return lines.groupby(["order_id", "customer_id"], as_index=False).agg(
        order_ts=("order_ts", "min"),
        revenue=("revenue", "sum"),
        items=("quantity", "sum"),
        any_discount=("discounted", "max"),
        categories=("category", lambda s: frozenset(s)),
    )


def build_snapshot(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int = 90,
    active_days: int = 365,
) -> pd.DataFrame:
    """Features and churn label for every customer active before ``cutoff``."""
    cutoff = pd.Timestamp(cutoff)
    label_end = cutoff + pd.Timedelta(days=horizon_days)
    if label_end > orders["order_ts"].max() + pd.Timedelta(days=1):
        raise ValueError(f"label window ends {label_end.date()}, after the last order in the data")

    history = orders[orders["order_ts"] < cutoff].sort_values(["customer_id", "order_ts"])
    future = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < label_end)]

    days_before_cutoff = (cutoff - history["order_ts"]).dt.days
    history = history.assign(
        gap_days=history.groupby("customer_id")["order_ts"].diff().dt.total_seconds() / 86400,
        in_last_90d=days_before_cutoff < 90,
        in_prev_90d=days_before_cutoff.between(90, 179),
    )
    grouped = history.groupby("customer_id")
    snapshot = grouped.agg(
        last_order=("order_ts", "max"),
        first_order=("order_ts", "min"),
        n_orders=("order_id", "count"),
        revenue_total=("revenue", "sum"),
        avg_items_per_order=("items", "mean"),
        discount_share=("any_discount", "mean"),
        mean_gap_days=("gap_days", "mean"),
        orders_last_90d=("in_last_90d", "sum"),
        orders_prev_90d=("in_prev_90d", "sum"),
    )
    snapshot["n_categories"] = grouped["categories"].agg(lambda sets: len(frozenset().union(*sets)))
    snapshot["recency_days"] = (cutoff - snapshot["last_order"]).dt.days
    snapshot["tenure_days"] = (cutoff - snapshot["first_order"]).dt.days
    snapshot["avg_order_value"] = snapshot["revenue_total"] / snapshot["n_orders"]
    # How overdue is the customer relative to their own rhythm? One-time
    # buyers have no rhythm yet, so the value stays missing for them.
    snapshot["recency_to_gap_ratio"] = snapshot["recency_days"] / snapshot["mean_gap_days"].clip(lower=1)

    snapshot = snapshot[snapshot["recency_days"] <= active_days]
    snapshot = snapshot.join(
        customers.set_index("customer_id")[["country", "acquisition_channel", "birth_year"]]
    )
    snapshot["age"] = cutoff.year - snapshot["birth_year"]
    snapshot[TARGET] = (~snapshot.index.isin(future["customer_id"])).astype(int)
    snapshot["cutoff"] = cutoff

    snapshot = snapshot.drop(columns=["last_order", "first_order", "birth_year"]).reset_index()
    logger.info(
        "snapshot %s: %d customers, churn rate %.1f%%",
        cutoff.date(),
        len(snapshot),
        100 * snapshot[TARGET].mean(),
    )
    return snapshot


def feature_matrix(snapshot: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a snapshot into model inputs and target."""
    X = snapshot[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
    return X, snapshot[TARGET]
