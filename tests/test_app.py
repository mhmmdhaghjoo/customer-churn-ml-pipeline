"""Smoke test: the Streamlit app renders without exceptions on the pipeline's outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "reports" / "best_model.joblib").exists() or not (ROOT / "data" / "raw").exists(),
    reason="needs `make data run` first",
)
def test_app_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Churn decision simulator"
    at.sidebar.slider[1].set_value(0.5).run()  # move the threshold
    assert not at.exception
