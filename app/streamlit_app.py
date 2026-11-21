"""Churn decision simulator: who to contact, at what threshold, and how far to trust the scores.

Run from the repository root after `make data run`:

    streamlit run app/streamlit_app.py

The data is simulated. The money figures depend entirely on the assumptions
entered in the sidebar; the model predicts churn risk, not the effect of an offer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from churn.config import Settings
from churn.decision import (
    CampaignAssumptions,
    break_even_threshold,
    calibration_table,
    campaign_value,
    score_test_snapshot,
    threshold_table,
)

st.set_page_config(page_title="Churn decision simulator", layout="wide")


@st.cache_data(show_spinner="Scoring the October 2025 snapshot ...")
def load_scores() -> pd.DataFrame:
    return score_test_snapshot(Settings())


try:
    scores = load_scores()
except FileNotFoundError as error:
    st.error(f"{error}. From the repository root run `make data run`, then reload this page.")
    st.stop()

y = scores["churned"].to_numpy()
p = scores["score"].to_numpy()

st.title("Churn decision simulator")
st.caption(
    "Simulated online shop. Logistic regression trained on snapshots from October 2024 to April 2025, "
    f"scored on the October 2025 snapshot ({len(scores):,} active customers, "
    f"{y.mean():.1%} did not order in the next 90 days). Figures describe the simulation, not a real company."
)

with st.sidebar:
    st.header("Business assumptions")
    st.write("The data cannot supply these. Agree them with marketing before reading any euro figure.")
    contact_cost = st.number_input("Cost per contacted customer (EUR)", 0.0, 100.0, 5.0, 0.5)
    value = st.number_input("Margin from one retained churner (EUR)", 0.0, 5000.0, 60.0, 5.0)
    save_rate = st.slider("Share of contacted churners the offer retains", 0.0, 1.0, 0.10, 0.01)
    st.divider()
    threshold = st.slider("Contact everyone with a score at or above", 0.0, 1.0, 0.80, 0.01)

assumptions = CampaignAssumptions(contact_cost, value, save_rate)
result = campaign_value(y, p, threshold, assumptions)
row = threshold_table(y, p, [threshold]).iloc[0]

tab_decision, tab_segments, tab_calibration, tab_audit = st.tabs(
    ["Threshold and value", "Segments", "Calibration", "How to check this"]
)

with tab_decision:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Customers contacted ({row.flagged_share:.0%} of active)", f"{int(row.flagged):,}")
    c2.metric("Of whom churn (precision)", "n/a" if np.isnan(row.precision) else f"{row.precision:.0%}")
    c3.metric("Churners reached (recall)", f"{row.recall:.0%}")
    c4.metric("Campaign cost", f"EUR {result['cost']:,.0f}")

    c5, c6 = st.columns(2)
    c5.metric(
        "Net value, expected from scores",
        f"EUR {result['expected_net_value']:,.0f}",
        help="What a campaign planner would estimate before the outcome is known.",
    )
    c6.metric(
        "Net value, backtest on what happened",
        f"EUR {result['backtest_net_value']:,.0f}",
        help="Uses the churn observed after October 2025. Only possible in hindsight.",
    )
    gap = result["expected_net_value"] - result["backtest_net_value"]
    if abs(gap) > 0.1 * max(abs(result["backtest_net_value"]), 1.0):
        st.warning(
            f"The scores promise EUR {gap:,.0f} more than actually happened. The model over-predicts churn "
            "for lower-risk customers (see Calibration), so expected values below the top deciles "
            "are too high."
        )

    be = break_even_threshold(assumptions)
    st.write(
        f"**Break-even probability:** a contact pays for itself when the true churn probability is above "
        f"{'never' if np.isinf(be) else f'{be:.0%}'} "
        "(cost / (save rate x margin)). This holds only for calibrated probabilities."
    )

    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    curve = pd.DataFrame([{"threshold": t, **campaign_value(y, p, t, assumptions)} for t in grid]).set_index(
        "threshold"
    )
    st.subheader("Net value by threshold")
    st.line_chart(
        curve[["expected_net_value", "backtest_net_value"]].rename(
            columns={"expected_net_value": "Expected from scores", "backtest_net_value": "Backtest"}
        ),
        x_label="Threshold",
        y_label="Net value (EUR)",
    )
    st.dataframe(
        threshold_table(y, p, grid).style.format(
            {"flagged_share": "{:.0%}", "precision": "{:.0%}", "recall": "{:.0%}", "threshold": "{:.2f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

with tab_segments:
    st.write(
        "K-Means clusters on log-scaled recency, frequency and monetary value of the same snapshot. "
        "Cluster numbers are arbitrary; read them by their medians."
    )
    flagged = scores["score"] >= threshold
    segments = (
        scores.assign(flagged=flagged)
        .groupby("cluster")
        .agg(
            customers=("customer_id", "size"),
            median_orders=("n_orders", "median"),
            median_recency_days=("recency_days", "median"),
            median_revenue_eur=("revenue_total", "median"),
            mean_score=("score", "mean"),
            observed_churn=("churned", "mean"),
            contacted=("flagged", "sum"),
        )
        .sort_values("median_revenue_eur", ascending=False)
    )
    st.dataframe(
        segments.style.format(
            {
                "mean_score": "{:.0%}",
                "observed_churn": "{:.0%}",
                "median_revenue_eur": "{:,.0f}",
                "median_recency_days": "{:.0f}",
            }
        ),
        use_container_width=True,
    )
    st.write(
        "A high score is not a reason to contact on its own: one-time buyers churn most, but the offer "
        "may not change their behaviour and their margin is small. Whether an offer works needs an "
        "experiment with a hold-out group, not this model."
    )

with tab_calibration:
    bins = st.slider("Bins", 5, 20, 10)
    cal = calibration_table(y, p, bins)
    st.line_chart(
        cal.set_index("mean_predicted")[["observed"]].assign(perfect=cal.set_index("mean_predicted").index),
        x_label="Mean predicted churn",
        y_label="Observed churn",
    )
    st.dataframe(
        cal.style.format("{:.3f}", subset=cal.columns[1:]), hide_index=True, use_container_width=True
    )
    st.write(
        "Points below the diagonal mean the model predicts more churn than happened. The test window is "
        "the Christmas quarter, when more customers return than in the training windows. A production "
        "version would recalibrate on recent data or add seasonal features before using probabilities "
        "as literal chances."
    )

with tab_audit:
    st.markdown(
        """
**What this app does and does not tell you**

- It ranks customers by risk of not ordering in 90 days. It does **not** estimate the effect of an offer;
  the save rate is your assumption.
- The data is simulated, so the figures show the method, not a real shop's economics.

**How I would check an AI-generated or colleague's version of this analysis**

1. Recompute one number by hand: precision at the chosen threshold equals churners among flagged
   customers divided by flagged customers (`tests/test_decision.py` does this on six customers).
2. Check that features use only data before the cutoff (`tests/test_features.py` adds future orders
   and asserts no feature changes).
3. Compare against the recency-only baseline before trusting a more complex model.
4. Compare expected value with the backtest; a large gap means the probabilities are not calibrated.
5. Ask who owns the decision: the model produces a list, a person decides who is contacted and why.
"""
    )
