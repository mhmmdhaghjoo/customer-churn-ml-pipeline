"""End-to-end smoke test on a small simulation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from churn.cli import main
from churn.config import Settings


def test_full_run_writes_all_artefacts(simulated_dir: Path, tmp_path: Path) -> None:
    exit_code = main(["run", "--data-dir", str(simulated_dir), "--report-dir", str(tmp_path)])
    assert exit_code == 0

    summary = json.loads((tmp_path / "metrics.json").read_text())
    assert set(summary["metrics"]) == {
        "recency_only",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    # Loose sanity bound: every model must clearly beat a coin flip.
    assert all(m["roc_auc"] > 0.65 for m in summary["metrics"].values())
    for name in ("model_comparison.csv", "segments.csv", "deciles.csv", "best_model.joblib"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "figures" / "roc_curves.png").stat().st_size > 0


def test_cli_returns_error_code_for_missing_data(tmp_path: Path) -> None:
    assert main(["run", "--data-dir", str(tmp_path / "nowhere"), "--report-dir", str(tmp_path)]) == 2


def test_settings_reject_overlapping_train_and_test_windows() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Settings(train_cutoffs=("2025-09-01",), test_cutoff="2025-10-01")
