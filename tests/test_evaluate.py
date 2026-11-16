from __future__ import annotations

import numpy as np
import pytest

from churn.evaluate import bootstrap_auc_difference, decile_table, evaluate, lift_at


def test_lift_of_a_perfect_ranking() -> None:
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1, 0, 100)
    assert lift_at(y, scores, 0.10) == pytest.approx(10.0)  # top 10 % are all churners; base rate is 10 %
    assert lift_at(y, scores, 0.20) == pytest.approx(5.0)


def test_lift_of_a_constant_score_is_about_one() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 10_000)
    assert lift_at(y, rng.random(10_000), 0.10) == pytest.approx(1.0, abs=0.1)


def test_evaluate_on_perfect_scores() -> None:
    y = np.array([0, 0, 1, 1])
    result = evaluate(y, np.array([0.0, 0.0, 1.0, 1.0]))
    assert result["roc_auc"] == 1.0
    assert result["brier"] == 0.0


def test_bootstrap_interval_contains_zero_for_identical_models() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    scores = rng.random(500)
    point, low, high = bootstrap_auc_difference(y, scores, scores, n_boot=200)
    assert point == low == high == 0.0


def test_decile_table_covers_all_churners() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    table = decile_table(y, rng.random(1000))
    assert table["customers"].sum() == 1000
    assert table["cumulative_share_of_churners"].iloc[-1] == pytest.approx(1.0)
