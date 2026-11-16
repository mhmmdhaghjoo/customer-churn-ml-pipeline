"""Figures for the report. Kept apart from the analysis so the pipeline can
run headless and the charts can be restyled without touching any logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

STYLE: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.titlelocation": "left",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
}


def roc_curves(y_true: pd.Series, scores: dict[str, pd.Series], aucs: dict[str, float], path: Path) -> None:
    colors = [MUTED, BLUE, ORANGE, AQUA]
    with plt.rc_context(STYLE):  # type: ignore[arg-type]
        fig, ax = plt.subplots(figsize=(5.6, 5))
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
        for (name, score), color in zip(scores.items(), colors, strict=False):
            fpr, tpr, _ = roc_curve(y_true, score)
            ax.plot(
                fpr, tpr, color=color, linewidth=2, label=f"{name.replace('_', ' ')} (AUC {aucs[name]:.3f})"
            )
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves on the out-of-time test snapshot")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def decile_chart(deciles: pd.DataFrame, base_rate: float, path: Path) -> None:
    with plt.rc_context(STYLE):  # type: ignore[arg-type]
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.38
        x = deciles.index.to_numpy()
        ax.bar(x - width / 2, 100 * deciles["predicted_churn"], width, color="#9ec5f4", label="Predicted")
        ax.bar(x + width / 2, 100 * deciles["observed_churn"], width, color=BLUE, label="Observed")
        ax.axhline(100 * base_rate, color=MUTED, linewidth=1, linestyle="--")
        ax.text(
            10.45, 100 * base_rate + 1, f"overall {100 * base_rate:.0f}%", ha="right", color=MUTED, fontsize=9
        )
        ax.set_xticks(x)
        ax.set_xlabel("Risk decile (1 = highest predicted risk)")
        ax.set_title("Churn rate per risk decile (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def importance_chart(importance: pd.DataFrame, path: Path, top: int = 10) -> None:
    data = importance.head(top).iloc[::-1]
    with plt.rc_context(STYLE):  # type: ignore[arg-type]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            data["feature"], data["importance_mean"], xerr=data["importance_std"], color=BLUE, ecolor=MUTED
        )
        ax.tick_params(axis="y", colors=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Drop in ROC AUC when the feature is shuffled")
        ax.set_title("Permutation importance on the test snapshot")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
