from __future__ import annotations

import pandas as pd
import pytest

from churn.features import NUMERIC_FEATURES, build_snapshot, feature_matrix, orders_table

CUTOFF = "2024-07-01"


def snapshot(tiny_shop, **kwargs) -> pd.DataFrame:
    customers, products, order_lines = tiny_shop
    orders = orders_table(order_lines, products)
    return build_snapshot(orders, customers, CUTOFF, **kwargs).set_index("customer_id")


def test_orders_table_has_one_row_per_order_with_net_revenue(tiny_shop) -> None:
    _, products, order_lines = tiny_shop
    orders = orders_table(order_lines, products).set_index("order_id")
    assert len(orders) == 6
    assert orders.loc["O1", "revenue"] == pytest.approx(2 * 10 + 1 * 20 * 0.5)
    assert orders.loc["O1", "items"] == 3
    assert orders.loc["O1", "categories"] == frozenset({"Books", "Toys"})


def test_features_match_hand_calculation(tiny_shop) -> None:
    row = snapshot(tiny_shop).loc[1]
    assert row["n_orders"] == 2  # O3 is after the cutoff
    assert row["revenue_total"] == pytest.approx(30 + 10)
    assert row["avg_order_value"] == pytest.approx(20)
    assert row["recency_days"] == 53  # 9 May -> 1 July
    assert row["tenure_days"] == 173  # 10 January -> 1 July
    assert row["mean_gap_days"] == pytest.approx(120)
    assert row["recency_to_gap_ratio"] == pytest.approx(53 / 120)
    assert row["n_categories"] == 2
    assert row["discount_share"] == pytest.approx(0.5)
    assert row["orders_last_90d"] == 1
    assert row["orders_prev_90d"] == 1  # 10 January is 173 days before the cutoff
    assert row["age"] == 34


def test_label_is_defined_by_the_window_after_the_cutoff(tiny_shop) -> None:
    result = snapshot(tiny_shop)
    assert result.loc[1, "churned"] == 0  # ordered on 15 July
    assert result.loc[2, "churned"] == 1


def test_inactive_customers_are_not_scored(tiny_shop) -> None:
    assert 3 not in snapshot(tiny_shop).index
    assert 3 in snapshot(tiny_shop, active_days=1000).index


def test_one_time_buyer_has_missing_gap_not_zero(tiny_shop) -> None:
    row = snapshot(tiny_shop).loc[2]
    assert pd.isna(row["mean_gap_days"])
    assert pd.isna(row["recency_to_gap_ratio"])


def test_no_leakage_future_orders_do_not_change_features(tiny_shop) -> None:
    customers, products, order_lines = tiny_shop
    before = snapshot(tiny_shop)

    extra = order_lines.iloc[[0]].assign(order_id="O99", order_ts=pd.Timestamp("2024-07-02"), quantity=50)
    after = snapshot((customers, products, pd.concat([order_lines, extra], ignore_index=True)))

    pd.testing.assert_frame_equal(before[NUMERIC_FEATURES], after[NUMERIC_FEATURES])


def test_label_window_must_fit_inside_the_data(tiny_shop) -> None:
    customers, products, order_lines = tiny_shop
    orders = orders_table(order_lines, products)
    with pytest.raises(ValueError, match="label window"):
        build_snapshot(orders, customers, "2024-12-01")


def test_feature_matrix_is_numeric_and_finite(tiny_shop) -> None:
    X, y = feature_matrix(snapshot(tiny_shop).reset_index())
    assert X[NUMERIC_FEATURES].dtypes.eq(float).all()
    assert not X[NUMERIC_FEATURES].isin([float("inf"), float("-inf")]).any().any()
    assert set(y) <= {0, 1}
