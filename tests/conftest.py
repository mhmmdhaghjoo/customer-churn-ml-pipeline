from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.simulate import simulate


@pytest.fixture
def tiny_shop() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three customers with hand-written histories, so expected values can be worked out by hand."""
    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "signup_date": pd.to_datetime(["2024-01-10", "2024-03-01", "2023-01-01"]),
            "country": ["DE", "AT", None],
            "acquisition_channel": ["email", "social", "organic"],
            "birth_year": [1990, 1985, 2000],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["A", "B"],
            "category": ["Books", "Toys"],
            "unit_price": [10.0, 20.0],
            "unit_cost": [6.0, 12.0],
        }
    )
    order_lines = pd.DataFrame(
        [
            # customer 1: two orders before the cutoff, one after it
            ("O1", "2024-01-10", 1, "A", 2, 10.0, 0.0),
            ("O1", "2024-01-10", 1, "B", 1, 20.0, 0.5),
            ("O2", "2024-05-09", 1, "A", 1, 10.0, 0.0),
            ("O3", "2024-07-15", 1, "B", 1, 20.0, 0.0),
            # customer 2: one order before the cutoff, nothing after
            ("O4", "2024-06-01", 2, "B", 3, 20.0, 0.0),
            # customer 3: last order too long ago to count as active
            ("O5", "2023-01-01", 3, "A", 1, 10.0, 0.0),
            # far-future order so label windows fit inside the data
            ("O6", "2024-12-31", 3, "A", 1, 10.0, 0.0),
        ],
        columns=[
            "order_id",
            "order_ts",
            "customer_id",
            "product_id",
            "quantity",
            "unit_price",
            "discount_pct",
        ],
    )
    order_lines["order_ts"] = pd.to_datetime(order_lines["order_ts"])
    return customers, products, order_lines


@pytest.fixture(scope="session")
def simulated_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sim")
    simulate(out, seed=3, n_customers=1200)
    return out
