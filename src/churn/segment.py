"""Unsupervised customer segmentation on recency, frequency and monetary value."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

RFM_COLUMNS = ["recency_days", "n_orders", "revenue_total"]


def segment_customers(
    snapshot: pd.DataFrame, k_values: tuple[int, ...] = (3, 4, 5, 6), seed: int = 42
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Cluster with K-Means and choose k by silhouette score.

    Returns the cluster label per customer, a profile table (one row per
    cluster) and the silhouette score for every k that was tried.
    """
    # Counts and money are right-skewed; log1p keeps a few big spenders from
    # dominating the Euclidean distances K-Means relies on.
    scaled = StandardScaler().fit_transform(np.log1p(snapshot[RFM_COLUMNS].to_numpy(dtype=float)))

    scores = []
    models = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled)
        models[k] = model
        scores.append(
            {"k": k, "silhouette": silhouette_score(scaled, model.labels_), "inertia": model.inertia_}
        )
    score_table = pd.DataFrame(scores)
    best_k = int(score_table.loc[score_table["silhouette"].idxmax(), "k"])
    logger.info("K-Means: best k = %d (silhouette %.3f)", best_k, score_table["silhouette"].max())

    labels = pd.Series(models[best_k].labels_, index=snapshot.index, name="cluster")
    profile = (
        snapshot.assign(cluster=labels)
        .groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            median_recency_days=("recency_days", "median"),
            median_orders=("n_orders", "median"),
            median_revenue=("revenue_total", "median"),
            revenue_share=("revenue_total", "sum"),
            churn_rate=("churned", "mean"),
        )
    )
    profile["revenue_share"] /= profile["revenue_share"].sum()
    return labels, profile.sort_values("median_revenue", ascending=False), score_table
