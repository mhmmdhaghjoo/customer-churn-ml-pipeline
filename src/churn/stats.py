"""Hypothesis tests that check, before any modelling, whether churned and
retained customers differ at all."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from churn.features import TARGET


def numeric_tests(snapshot: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mann-Whitney U per numeric feature, with Holm-adjusted p-values.

    The features are heavily skewed, so a rank test is used instead of a
    t-test. The effect size is the rank-biserial correlation
    (-1 to 1; positive means higher values among churned customers).
    """
    churned = snapshot[snapshot[TARGET] == 1]
    retained = snapshot[snapshot[TARGET] == 0]
    rows = []
    for column in columns:
        a, b = churned[column].dropna(), retained[column].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "feature": column,
                "median_churned": a.median(),
                "median_retained": b.median(),
                "rank_biserial": 2 * u / (len(a) * len(b)) - 1,
                "p_value": p,
            }
        )
    result = pd.DataFrame(rows).sort_values("p_value", ignore_index=True)
    # Holm step-down correction for testing many features at once.
    m = len(result)
    adjusted = (result["p_value"] * (m - np.arange(m))).cummax().clip(upper=1.0)
    result["p_holm"] = adjusted
    result["significant_5pct"] = adjusted < 0.05
    return result


def categorical_tests(snapshot: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Chi-square test of independence per categorical feature, with Cramér's V."""
    rows = []
    for column in columns:
        table = pd.crosstab(snapshot[column].fillna("missing"), snapshot[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        rows.append(
            {
                "feature": column,
                "chi2": chi2,
                "dof": dof,
                "cramers_v": float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))),
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)
