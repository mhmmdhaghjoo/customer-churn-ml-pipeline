"""Load the raw files and refuse to continue if they are not what the pipeline expects."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: dict[str, set[str]] = {
    "customers": {"customer_id", "signup_date", "country", "acquisition_channel", "birth_year"},
    "products": {"product_id", "category", "unit_price", "unit_cost"},
    "order_lines": {
        "order_id",
        "order_ts",
        "customer_id",
        "product_id",
        "quantity",
        "unit_price",
        "discount_pct",
    },
}


class DataValidationError(ValueError):
    """Raised when the input data breaks an assumption the pipeline relies on."""


def validate(customers: pd.DataFrame, products: pd.DataFrame, order_lines: pd.DataFrame) -> None:
    """Check structure and integrity. Collects every problem before raising."""
    problems: list[str] = []
    frames = {"customers": customers, "products": products, "order_lines": order_lines}
    for name, frame in frames.items():
        missing = REQUIRED_COLUMNS[name] - set(frame.columns)
        if missing:
            problems.append(f"{name}: missing columns {sorted(missing)}")
    if problems:  # the remaining checks need these columns
        raise DataValidationError("; ".join(problems))

    if customers["customer_id"].duplicated().any():
        problems.append("customers: customer_id is not unique")
    if products["product_id"].duplicated().any():
        problems.append("products: product_id is not unique")
    if order_lines[["order_id", "customer_id", "product_id", "order_ts"]].isna().any().any():
        problems.append("order_lines: null keys or timestamps")
    if (order_lines["quantity"] <= 0).any():
        problems.append("order_lines: non-positive quantity")
    if not order_lines["discount_pct"].between(0, 1, inclusive="left").all():
        problems.append("order_lines: discount_pct outside [0, 1)")
    if not set(order_lines["customer_id"]) <= set(customers["customer_id"]):
        problems.append("order_lines: customer_id not found in customers")
    if not set(order_lines["product_id"]) <= set(products["product_id"]):
        problems.append("order_lines: product_id not found in products")

    if problems:
        raise DataValidationError("; ".join(problems))


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three CSV files, parse dates and validate them."""
    try:
        customers = pd.read_csv(data_dir / "customers.csv", parse_dates=["signup_date"])
        products = pd.read_csv(data_dir / "products.csv")
        order_lines = pd.read_csv(data_dir / "order_lines.csv", parse_dates=["order_ts"])
    except FileNotFoundError as error:
        raise DataValidationError(
            f"{error.filename} not found. Run `churn simulate` first or point --data-dir at your files."
        ) from error
    validate(customers, products, order_lines)
    logger.info(
        "loaded %d customers, %d products, %d order lines",
        len(customers),
        len(products),
        len(order_lines),
    )
    return customers, products, order_lines
