"""All tunable settings in one place, so a run is fully described by one object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/raw")
    report_dir: Path = Path("reports")
    seed: int = 42

    # A customer has churned if they place no order in the `horizon_days`
    # after the cutoff. Only customers who ordered in the `active_days`
    # before the cutoff are scored; the others are already gone.
    horizon_days: int = 90
    active_days: int = 365

    # Out-of-time evaluation: fit on one snapshot, test on a later one.
    train_cutoffs: tuple[str, ...] = ("2024-10-01", "2025-01-01", "2025-04-01")
    test_cutoff: str = "2025-10-01"

    # k = 2 only separates one-time from repeat buyers, which needs no
    # clustering, so the search starts at three segments.
    segment_range: tuple[int, ...] = field(default=(3, 4, 5, 6))

    def __post_init__(self) -> None:
        horizon = pd.Timedelta(days=self.horizon_days)
        last_train_label_day = max(pd.Timestamp(c) for c in self.train_cutoffs) + horizon
        if last_train_label_day > pd.Timestamp(self.test_cutoff):
            raise ValueError(
                "training labels would overlap the test snapshot: "
                f"{last_train_label_day.date()} > {self.test_cutoff}"
            )
