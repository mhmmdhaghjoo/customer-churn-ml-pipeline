from __future__ import annotations

import numpy as np
import pytest

from churn.decision import (
    CampaignAssumptions,
    break_even_threshold,
    calibration_table,
    campaign_value,
    threshold_table,
)

# Six customers, worked out by hand.
Y = np.array([1, 1, 0, 1, 0, 0])
P = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.1])


def test_threshold_table_counts_by_hand() -> None:
    table = threshold_table(Y, P, [0.75, 0.5, 0.0, 0.95]).set_index("threshold")
    # >= 0.75: customers 1 and 2, both churn
    assert table.loc[0.75, "flagged"] == 2
    assert table.loc[0.75, "precision"] == 1.0
    assert table.loc[0.75, "recall"] == pytest.approx(2 / 3)
    # >= 0.5: customers 1, 2, 3; two of three churn
    assert table.loc[0.5, "precision"] == pytest.approx(2 / 3)
    # everyone flagged: precision is the base rate, recall is 1
    assert table.loc[0.0, "flagged_share"] == 1.0
    assert table.loc[0.0, "recall"] == 1.0
    # nobody flagged: precision undefined, not zero
    assert table.loc[0.95, "flagged"] == 0
    assert np.isnan(table.loc[0.95, "precision"])


def test_campaign_value_backtest_and_expected() -> None:
    a = CampaignAssumptions(contact_cost=5.0, value_per_retained=100.0, save_rate=0.2)
    out = campaign_value(Y, P, threshold=0.5, assumptions=a)
    assert out["flagged"] == 3
    assert out["cost"] == 15.0
    # backtest: 2 real churners * 0.2 * 100 - 15 = 25
    assert out["backtest_net_value"] == pytest.approx(25.0)
    # expected: (0.9 + 0.8 + 0.7) * 0.2 * 100 - 15 = 33
    assert out["expected_net_value"] == pytest.approx(33.0)


def test_break_even_threshold() -> None:
    a = CampaignAssumptions(contact_cost=5.0, value_per_retained=100.0, save_rate=0.2)
    assert break_even_threshold(a) == pytest.approx(0.25)
    assert break_even_threshold(CampaignAssumptions(1.0, 100.0, 0.0)) == float("inf")


def test_calibration_table_bins_and_interval() -> None:
    table = calibration_table(Y, P, n_bins=2)
    assert table["customers"].tolist() == [3, 3]
    # lowest three scores 0.1, 0.2, 0.4 -> one churner
    assert table.loc[0, "mean_predicted"] == pytest.approx(0.7 / 3)
    assert table.loc[0, "observed"] == pytest.approx(1 / 3)
    assert (table["observed_low"] <= table["observed"]).all()
    assert (table["observed"] <= table["observed_high"]).all()


def test_perfectly_calibrated_scores_match_observed_rates() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.size) < p).astype(int)
    table = calibration_table(y, p, n_bins=10)
    assert np.allclose(table["mean_predicted"], table["observed"], atol=0.01)


@pytest.mark.parametrize(
    ("y", "p"),
    [
        (np.array([1, 0]), np.array([0.5])),  # length mismatch
        (np.array([1, 0]), np.array([0.5, 1.2])),  # not a probability
        (np.array([1, 0]), np.array([0.5, np.nan])),  # missing score
        (np.array([], dtype=int), np.array([])),  # nothing to score
    ],
)
def test_invalid_inputs_are_rejected(y: np.ndarray, p: np.ndarray) -> None:
    with pytest.raises(ValueError):
        threshold_table(y, p, [0.5])


def test_assumptions_are_validated() -> None:
    with pytest.raises(ValueError):
        CampaignAssumptions(contact_cost=5.0, value_per_retained=100.0, save_rate=1.5)
    with pytest.raises(ValueError):
        CampaignAssumptions(contact_cost=-1.0, value_per_retained=100.0, save_rate=0.2)
