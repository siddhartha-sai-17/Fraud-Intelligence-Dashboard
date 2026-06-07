"""
Fraud Intelligence Dashboard (Task 7 + Bonus)
- CSV upload & batch prediction
- Fraud probability & high-risk transactions
- Attention visualization
- Real-time fraud detection simulation
"""

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* =========================
   GLOBAL THEME
========================= */
.main {
    background-color: #0b1220;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* =========================
   HEADER CARD
========================= */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0b1220 100%);
    padding: 2.2rem;
    border-radius: 18px;
    margin-bottom: 2rem;
    border: 1px solid #334155;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.main-header h1 {
    color: #38bdf8;
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
}

.main-header p {
    color: #94a3b8;
    margin-top: 0.5rem;
}

/* =========================
   METRICS CARDS
========================= */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid #1f2937;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

[data-testid="metric-container"] label {
    color: #94a3b8 !important;
}

[data-testid="metric-container"] div {
    color: #38bdf8 !important;
    font-weight: 600;
}

/* =========================
   DATAFRAMES
========================= */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #1f2937;
}

/* =========================
   BUTTONS
========================= */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #38bdf8);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(56,189,248,0.3);
}

/* =========================
   SIDEBAR
========================= */
section[data-testid="stSidebar"] {
    background-color: #0a0f1c;
    border-right: 1px solid #1f2937;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

/* =========================
   TABS
========================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #111827;
    border-radius: 10px;
    padding: 10px 14px;
    color: #94a3b8;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #38bdf8);
    color: white !important;
}

/* =========================
   INFO / WARNING BOXES
========================= */
.stAlert {
    border-radius: 12px;
}

/* =========================
   TABLES
========================= */
table {
    border-radius: 12px !important;
    overflow: hidden;
}

/* =========================
   FILE UPLOADER
========================= */
[data-testid="stFileUploader"] {
    background: #0f172a;
    border: 1px dashed #334155;
    border-radius: 12px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    from src.inference import load_artifacts as _load
    return _load()


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Deep Learning Fraud Intelligence Dashboard</h1>
        <p>Sequential transaction fraud detection with LSTM + Attention + Positional Encoding</p>
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(level: str) -> str:
    css = {"High Risk": "risk-high", "Medium Risk": "risk-medium", "Low Risk": "risk-low"}
    return f'<span class="{css.get(level, "")}">{level}</span>'


try:
    artifacts = load_artifacts()
    models = artifacts["models"]
    extractor = artifacts["extractor"]
    scaler = artifacts["scaler"]
    models_loaded = len(models) > 0
except Exception as e:
    models_loaded = False
    load_error = str(e)


render_header()

if not models_loaded:
    st.warning(
        "Models not trained yet. Run training first:\n\n"
        "```\npython -m src.train\n```\n\n"
        "You can still explore EDA in the notebook."
    )
    if "load_error" in dir():
        st.error(f"Load error: {load_error}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
selected_model = st.sidebar.selectbox(
    "Model",
    list(models.keys()),
    format_func=lambda x: {
        "dense": "Model A — Dense Network",
        "lstm": "Model B — LSTM",
        "lstm_attention": "Model C — LSTM + Attention",
        "lstm_pe_attention": "Model D — LSTM + PE + Attention",
    }.get(x, x),
)
threshold = st.sidebar.slider("Fraud Threshold", 0.1, 0.9, 0.5, 0.05)
high_risk_threshold = st.sidebar.slider("High Risk Threshold", 0.5, 0.95, 0.70, 0.05)

model = models[selected_model]

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_upload, tab_attention, tab_realtime, tab_info = st.tabs([
    "📤 CSV Upload",
    "🔍 Attention Investigation",
    "⚡ Real-Time Simulation",
    "📚 Business Context",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CSV Upload
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Upload Transaction CSV")
    st.caption("Requires columns: Time, V1–V28, Amount. Sequences of 4 prior txns → predict next txn fraud.")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded:
        from src.inference import csv_to_sequences, predict_fraud, get_risk_level

        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head(10), use_container_width=True, width='stretch')

            X, meta = csv_to_sequences(df, scaler)
            probs, preds = predict_fraud(model, X, threshold)
            risk_levels = [get_risk_level(p) for p in probs]

            results = meta.copy()
            results["Fraud Probability"] = probs.round(4)
            results["Prediction"] = np.where(preds == 1, "Fraud", "Non-Fraud")
            results["Risk Level"] = risk_levels

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Sequences", len(results))
            with c2:
                st.metric("Predicted Fraud", int(preds.sum()))
            with c3:
                st.metric("High Risk", int((probs >= high_risk_threshold).sum()))
            with c4:
                st.metric("Avg Fraud Prob", f"{probs.mean():.3f}")

            st.subheader("All Predictions")
            st.dataframe(results, use_container_width=True, width='stretch')

            high_risk = results[results["Fraud Probability"] >= high_risk_threshold]
            st.subheader("🚨 High Risk Transactions")
            if len(high_risk) > 0:
                st.dataframe(high_risk.sort_values("Fraud Probability", ascending=False),
                             use_container_width=True, width='stretch')
            else:
                st.success("No high-risk transactions detected.")

            csv_out = results.to_csv(index=False)
            st.download_button("Download Results", csv_out, "fraud_predictions.csv", "text/csv")

            st.session_state["X_sequences"] = X
            st.session_state["results"] = results

        except Exception as exc:
            st.error(f"Error processing file: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Attention Investigation (Task 6)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_attention:
    st.subheader("Which Transaction Influenced the Fraud Prediction Most?")

    if extractor is None:
        st.info("Attention extractor not available. Re-run training.")
    elif "X_sequences" not in st.session_state:
        st.info("📤 Upload a CSV in the **CSV Upload** tab first, then come back here.")
    else:
        from src.inference import analyze_attention
        from src.attention_utils import visualize_attention_bar, visualize_attention_heatmap

        # Warn if selected model doesn't use attention
        if selected_model not in ("lstm_attention", "lstm_pe_attention"):
            st.warning(
                "⚠️ The attention extractor is based on the **LSTM + PE + Attention** model. "
                "Switch to **Model C** or **Model D** in the sidebar for best results. "
                "Visualization below uses the dedicated attention extractor regardless."
            )

        X = st.session_state["X_sequences"]
        results = st.session_state["results"]

        seq_idx = st.selectbox(
            "Select sequence to investigate",
            range(len(X)),
            format_func=lambda i: (
                f"Seq {i} — Prob: {results.iloc[i]['Fraud Probability']:.3f} "
                f"({results.iloc[i]['Prediction']})"
            ),
        )

        try:
            analysis = analyze_attention(extractor, X, seq_idx)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Fraud Probability", f"{analysis['fraud_probability']:.4f}")
            with c2:
                st.metric("Most Influential Transaction",
                          f"{analysis['top_transaction']} ({analysis['top_score']:.3f})")

            st.subheader("Transaction Influence Ranking")
            rank_df = pd.DataFrame(analysis["ranked"])
            st.dataframe(rank_df, width='stretch')

            col1, col2 = st.columns(2)
            with col1:
                fig_bar = visualize_attention_bar(
                    analysis["importance"],
                    analysis["txn_labels"],
                    title=f"Attention — Sequence {seq_idx}",
                )
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            with col2:
                fig_heat = visualize_attention_heatmap(
                    analysis["attention_matrix"],
                    analysis["txn_labels"],
                    title="Attention Matrix",
                )
                st.pyplot(fig_heat)
                plt.close(fig_heat)

            top = analysis["ranked"][0]
            st.success(
                f"**{top['transaction']}** had the highest attention weight "
                f"({top['importance']:.3f}), meaning it most influenced the fraud prediction "
                f"for this sequence."
            )

        except Exception as exc:
            st.error(f"❌ Attention analysis failed: {exc}")
            with st.expander("Full traceback"):
                import traceback
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Real-Time Simulation (Bonus)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_realtime:
    st.subheader("⚡ Real-Time Fraud Detection Simulation")
    st.caption("Simulates incoming transactions one-by-one. After 4 transactions, predicts fraud on the next.")

    from src.inference import build_single_sequence_from_history, predict_fraud, get_risk_level

    if "sim_history" not in st.session_state:
        st.session_state.sim_history = []
    if "sim_log" not in st.session_state:
        st.session_state.sim_log = []

    sim_cols = st.columns(3)
    with sim_cols[0]:
        sim_amount = st.number_input("Amount", min_value=0.0, value=100.0, key="sim_amt")
    with sim_cols[1]:
        sim_time = st.number_input("Time (seconds)", min_value=0.0, value=0.0, key="sim_time")
    with sim_cols[2]:
        auto_sim = st.toggle("Auto-stream (1 txn/sec)", value=False)

    if st.button("➕ Add Transaction", type="primary") or auto_sim:
        txn = {"Time": sim_time, "Amount": sim_amount}
        for i in range(1, 29):
            txn[f"V{i}"] = float(np.random.randn())
        st.session_state.sim_history.append(txn)

        if len(st.session_state.sim_history) >= 4:
            try:
                X_single = build_single_sequence_from_history(
                    st.session_state.sim_history, scaler
                )
                prob, pred = predict_fraud(model, X_single, threshold)
                risk = get_risk_level(prob[0])

                entry = {
                    "txn_num": len(st.session_state.sim_history),
                    "amount": sim_amount,
                    "fraud_prob": round(float(prob[0]), 4),
                    "prediction": "Fraud" if pred[0] == 1 else "Non-Fraud",
                    "risk": risk,
                }
                st.session_state.sim_log.append(entry)

                if pred[0] == 1:
                    st.error(f"🚨 FRAUD DETECTED — Probability: {prob[0]:.4f} | {risk}")
                else:
                    st.success(f"✅ Legitimate — Probability: {prob[0]:.4f} | {risk}")
            except Exception as exc:
                st.warning(str(exc))
        else:
            st.info(f"Collecting history... ({len(st.session_state.sim_history)}/4 transactions)")

        if auto_sim:
            time.sleep(1)
            st.rerun()

    if st.button("🔄 Reset Simulation"):
        st.session_state.sim_history = []
        st.session_state.sim_log = []
        st.rerun()

    if st.session_state.sim_log:
        st.subheader("Detection Log")
        st.dataframe(pd.DataFrame(st.session_state.sim_log), use_container_width=True, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Business Context (Task 1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.subheader("Business Understanding")

    st.markdown("""
    ### Why is fraud detection difficult?

    1. **Extreme class imbalance** — Fraudulent transactions are typically < 0.2% of all
       transactions. A model predicting "no fraud" always would appear highly accurate.
    2. **Evolving fraud patterns** — Fraudsters continuously adapt tactics, making
       historical patterns less reliable over time.
    3. **Sequential context matters** — A single transaction may look normal, but its
       position in a spending sequence (e.g., rapid small purchases before a large one)
       can signal fraud.
    4. **High cost of errors** — False negatives (missed fraud) cost money; false positives
       (blocking legitimate customers) damage trust and revenue.
    5. **Anonymized features** — PCA-transformed features (V1–V28) lose direct
       interpretability, making root-cause analysis harder.

    ### Why is accuracy alone misleading?

    | Scenario | Accuracy | Problem |
    |----------|----------|---------|
    | 99.8% legit data | 99.8% accuracy by predicting all legit | **0% fraud caught** |
    | Block 1% randomly | ~99% accuracy | Massive false positive rate |

    **Better metrics for imbalanced fraud detection:**
    - **Precision** — Of flagged transactions, how many are actually fraud?
    - **Recall** — Of all fraud, how much did we catch?
    - **F1 Score** — Harmonic mean of precision and recall
    - **ROC-AUC / PR-AUC** — Threshold-independent ranking quality

    ### Why does transaction order matter?

    Fraud is often a **sequence pattern**, not a single event:
    - A small "test" transaction followed by a large purchase (card testing)
    - Multiple rapid transactions across locations (account takeover)
    - Unusual spending after a period of inactivity

    **Positional Encoding** tells the attention mechanism *where* each transaction
    sits in the sequence, so the model can learn order-dependent patterns like
    "Txn4 right after three small Txns" vs. the same Txn4 in isolation.
    """)

    comparison_path = ROOT / "outputs" / "model_comparison.json"
    if comparison_path.exists():
        import json
        with open(comparison_path) as f:
            comparison = json.load(f)
        st.subheader("Model Comparison Results")
        rows = []
        for name, m in comparison.items():
            rows.append({
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "ROC-AUC": m["roc_auc"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, width='stretch')
