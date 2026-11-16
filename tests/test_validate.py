from __future__ import annotations

from pathlib import Path

import pytest

from churn.validate import DataValidationError, load_data, validate


def test_valid_data_passes(tiny_shop) -> None:
    validate(*tiny_shop)


def test_missing_column_is_reported(tiny_shop) -> None:
    customers, products, order_lines = tiny_shop
    with pytest.raises(DataValidationError, match="missing columns.*discount_pct"):
        validate(customers, products, order_lines.drop(columns="discount_pct"))


def test_all_problems_are_reported_together(tiny_shop) -> None:
    customers, products, order_lines = tiny_shop
    broken = order_lines.copy()
    broken.loc[0, "quantity"] = -1
    broken.loc[1, "customer_id"] = 999
    with pytest.raises(DataValidationError) as error:
        validate(customers, products, broken)
    assert "non-positive quantity" in str(error.value)
    assert "customer_id not found" in str(error.value)


def test_missing_files_give_an_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="churn simulate"):
        load_data(tmp_path)
