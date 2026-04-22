from __future__ import annotations

# Streamlit frontend for SMS Spam Shield.
# Features:
# - Real-time single-message classification with threshold control
# - Batch CSV scoring and export
# - Model dashboard with ROC, confusion matrix, and token insights
# - Session log for analyzed messages

from datetime import datetime
import io
import json
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from pandas.errors import ParserError

from text_preprocessing import normalize_text as shared_normalize_text

MODEL_PATH = Path("artifacts/spam_model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")


# Backward compatibility: older artifacts may reference __main__.normalize_text.
def normalize_text(text: str) -> str:
    return shared_normalize_text(text)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;500;700;800&family=Outfit:wght@400;600;700&display=swap');

        :root {
            --bg-a: #f5fffd;
            --bg-b: #edf7ff;
            --bg-c: #fff8ef;
            --ink: #0f172a;
            --muted: #334155;
            --line: #cbd5e1;
            --surface: #ffffff;
            --surface-soft: #f8fafc;
            --accent: #0f766e;
            --accent-2: #0891b2;
            --danger: #be123c;
            --sidebar-bg: #0b1220;
            --sidebar-bg-2: #111827;
            --sidebar-text: #f1f5f9;
            --sidebar-muted: #cbd5e1;
            --sidebar-line: rgba(148, 163, 184, 0.28);
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Keep header visible for sidebar controls */
        header {
            background: rgba(11, 18, 32, 0.95) !important;
            border-bottom: 1px solid var(--sidebar-line);
        }

        [data-testid="stHeader"] button,
        [data-testid="stHeader"] svg {
            color: var(--sidebar-text) !important;
            fill: var(--sidebar-text) !important;
        }

        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.6rem;
            max-width: 1250px;
        }

        .stApp {
            font-family: 'Sora', sans-serif;
            background:
                radial-gradient(circle at 8% 16%, rgba(8, 145, 178, 0.15), transparent 38%),
                radial-gradient(circle at 88% 8%, rgba(245, 158, 11, 0.12), transparent 32%),
                radial-gradient(circle at 80% 84%, rgba(190, 18, 60, 0.08), transparent 34%),
                linear-gradient(135deg, var(--bg-a) 0%, var(--bg-b) 46%, var(--bg-c) 100%);
            color: var(--ink);
        }

        /* Main content text colors */
        [data-testid="stAppViewContainer"] .main h1,
        [data-testid="stAppViewContainer"] .main h2,
        [data-testid="stAppViewContainer"] .main h3,
        [data-testid="stAppViewContainer"] .main h4,
        [data-testid="stAppViewContainer"] .main h5,
        [data-testid="stAppViewContainer"] .main p,
        [data-testid="stAppViewContainer"] .main li,
        [data-testid="stAppViewContainer"] .main label,
        [data-testid="stAppViewContainer"] .main span {
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] .main .stCaption,
        [data-testid="stAppViewContainer"] .main .stMarkdown p,
        [data-testid="stAppViewContainer"] .main .stMarkdown li {
            color: var(--muted) !important;
        }

        .main .stAlert p,
        .main .stAlert div {
            color: var(--ink) !important;
        }

        /* Decorative ambient */
        .ambient {
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            overflow: hidden;
        }

        .orb {
            position: absolute;
            border-radius: 999px;
            filter: blur(2px);
            opacity: 0.25;
        }

        .orb-a {
            width: 280px;
            height: 280px;
            background: radial-gradient(circle, rgba(20,184,166,0.45), rgba(20,184,166,0));
            top: 10%;
            left: -50px;
            animation: driftA 13s ease-in-out infinite;
        }

        .orb-b {
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(245,158,11,0.35), rgba(245,158,11,0));
            top: 52%;
            right: -70px;
            animation: driftB 16s ease-in-out infinite;
        }

        .orb-c {
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(190,18,60,0.26), rgba(190,18,60,0));
            bottom: 6%;
            left: 24%;
            animation: driftC 15s ease-in-out infinite;
        }

        @keyframes driftA {
            0% { transform: translate(0px, 0px) scale(1); }
            50% { transform: translate(40px, 35px) scale(1.08); }
            100% { transform: translate(0px, 0px) scale(1); }
        }

        @keyframes driftB {
            0% { transform: translate(0px, 0px) scale(1); }
            50% { transform: translate(-35px, -30px) scale(1.05); }
            100% { transform: translate(0px, 0px) scale(1); }
        }

        @keyframes driftC {
            0% { transform: translate(0px, 0px) scale(1); }
            50% { transform: translate(25px, -25px) scale(1.12); }
            100% { transform: translate(0px, 0px) scale(1); }
        }

        .hero {
            padding: 1.2rem 1.4rem 1.35rem;
            border-radius: 22px;
            background: linear-gradient(150deg, rgba(255,255,255,0.96), rgba(255,255,255,0.84));
            border: 1px solid rgba(148, 163, 184, 0.4);
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.10);
            backdrop-filter: blur(10px);
            margin-bottom: 0.7rem;
        }

        .hero h1 {
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.02em;
            margin-bottom: 0.4rem;
        }

        .hero p {
            margin-top: 0;
            color: var(--muted);
        }

        .metric-card {
            border-radius: 14px;
            padding: 0.8rem;
            background: var(--glass);
            border: 1px solid var(--line);
            text-align: center;
            margin-bottom: 0.5rem;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        .metric-card strong {
            color: #334155;
        }

        .quick-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.9rem 0 0.2rem;
        }

        .quick-card {
            border: 1px solid rgba(148, 163, 184, 0.4);
            background: rgba(255,255,255,0.92);
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
            box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
            animation: cardLift 4s ease-in-out infinite;
        }

        .quick-card b {
            color: var(--ink);
            font-size: 1.1rem;
        }

        .quick-card p {
            margin: 0;
            color: var(--muted) !important;
            font-size: 0.86rem;
        }

        @keyframes cardLift {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-2px); }
            100% { transform: translateY(0px); }
        }

        .result-spam {
            color: var(--coral);
            font-weight: 700;
            font-size: 1.3rem;
        }

        .result-ham {
            color: var(--teal);
            font-weight: 700;
            font-size: 1.3rem;
        }

        .result-card {
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.4);
            background: rgba(255,255,255,0.94);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.8rem;
        }

        .meter {
            width: 100%;
            height: 18px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.12);
            overflow: hidden;
            border: 1px solid rgba(15, 23, 42, 0.1);
            margin: 0.35rem 0 0.8rem;
        }

        .meter > span {
            display: block;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #0f766e 0%, #14b8a6 45%, #f59e0b 75%, #be123c 100%);
            animation: pulseFill 1.8s ease-in-out infinite;
        }

        @keyframes pulseFill {
            0% { filter: saturate(1); }
            50% { filter: saturate(1.25); }
            100% { filter: saturate(1); }
        }

        .chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
            font-size: 0.78rem;
            border: 1px solid rgba(15, 23, 42, 0.22);
            background: rgba(255,255,255,0.95);
        }

        .signal-spam {
            border-color: rgba(190, 18, 60, 0.35);
            color: #9f1239;
            background: rgba(251, 207, 232, 0.44);
        }

        .signal-ham {
            border-color: rgba(15, 118, 110, 0.35);
            color: #0f766e;
            background: rgba(153, 246, 228, 0.34);
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 42, 0.22);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            font-weight: 600;
            background: linear-gradient(120deg, #0f172a 0%, #1f2937 100%) !important;
            color: #f8fafc !important;
            min-height: 44px;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 16px rgba(15, 23, 42, 0.12);
            border-color: rgba(20, 184, 166, 0.9);
        }

        .stButton > button p {
            color: #f8fafc !important;
        }

        .main [data-testid="stWidgetLabel"] p {
            color: var(--ink) !important;
            font-weight: 600;
        }

        .main .stTextArea textarea,
        .main .stTextInput input,
        .main .stNumberInput input {
            background: var(--surface) !important;
            color: var(--ink) !important;
            border: 1px solid rgba(148, 163, 184, 0.55) !important;
            border-radius: 12px !important;
        }

        .main .stTextArea textarea::placeholder,
        .main .stTextInput input::placeholder {
            color: #64748b !important;
        }

        .main .stTextArea textarea {
            caret-color: var(--ink) !important;
        }

        .main .stTextArea textarea:focus,
        .main .stTextInput input:focus,
        .main .stNumberInput input:focus {
            border-color: rgba(20, 184, 166, 0.9) !important;
            box-shadow: 0 0 0 1px rgba(20, 184, 166, 0.5) !important;
        }

        .main div[data-baseweb="select"] > div {
            background: var(--surface) !important;
            color: var(--ink) !important;
            border: 1px solid rgba(148, 163, 184, 0.55) !important;
            border-radius: 12px !important;
        }

        .main [data-baseweb="slider"] [role="slider"] {
            border: 2px solid #0f766e !important;
            background: #14b8a6 !important;
        }

        /* Tabs */
        [data-baseweb="tab-list"] {
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 0.7rem;
            position: sticky;
            top: 0.15rem;
            z-index: 10;
            backdrop-filter: blur(6px);
            background: linear-gradient(90deg, rgba(255,255,255,0.86), rgba(255,255,255,0.7));
            border-radius: 12px;
            padding: 0.3rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
        }

        [data-baseweb="tab"] {
            border-radius: 10px !important;
            background: rgba(255,255,255,0.98) !important;
            border: 1px solid rgba(148, 163, 184, 0.4) !important;
            color: var(--ink) !important;
            min-height: 42px;
            padding: 0.3rem 0.9rem !important;
            font-weight: 600 !important;
        }

        [data-baseweb="tab"] * {
            color: var(--ink) !important;
            opacity: 1 !important;
        }

        [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(236, 253, 245, 1) !important;
            border: 1px solid rgba(15, 118, 110, 0.65) !important;
            box-shadow: 0 6px 14px rgba(15, 118, 110, 0.12);
        }

        [data-baseweb="tab"][aria-selected="true"] * {
            color: #0f766e !important;
            font-weight: 700 !important;
        }

        .guide-box {
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.45);
            background: rgba(255,255,255,0.92);
            padding: 0.9rem;
            margin-bottom: 0.6rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--sidebar-bg-2) 100%) !important;
            border-right: 1px solid var(--sidebar-line);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li {
            color: var(--sidebar-text) !important;
        }

        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown li,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: var(--sidebar-muted) !important;
        }

        [data-testid="stSidebar"] .metric-card {
            background: rgba(30, 41, 59, 0.92) !important;
            border: 1px solid rgba(148, 163, 184, 0.36) !important;
            box-shadow: 0 8px 16px rgba(2, 6, 23, 0.35);
        }

        [data-testid="stSidebar"] .metric-card strong,
        [data-testid="stSidebar"] .metric-card,
        [data-testid="stSidebar"] .metric-card p,
        [data-testid="stSidebar"] .metric-card span,
        [data-testid="stSidebar"] .metric-card div {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
            color: var(--sidebar-text) !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: #1e293b !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] .stButton > button p {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] [data-baseweb="slider"] div {
            color: var(--sidebar-text) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            border-color: #22d3ee !important;
            background: #14b8a6 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stTickBar"] div {
            background: rgba(203, 213, 225, 0.6) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="switch"] > div {
            background: rgba(100, 116, 139, 0.45) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="switch"] input:checked + div,
        [data-testid="stSidebar"] [data-baseweb="switch"] [aria-checked="true"] {
            background: linear-gradient(90deg, #14b8a6, #22d3ee) !important;
        }

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stTextArea textarea {
            background: rgba(30, 41, 59, 0.9) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.38) !important;
        }

        [data-testid="stSidebar"] .stTextInput input::placeholder,
        [data-testid="stSidebar"] .stTextArea textarea::placeholder {
            color: #cbd5e1 !important;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(30, 41, 59, 0.9) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.38) !important;
        }

        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            color: var(--sidebar-muted) !important;
        }

        [data-testid="stSidebar"]::-webkit-scrollbar,
        [data-testid="stVerticalBlock"]::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        [data-testid="stSidebar"]::-webkit-scrollbar-thumb,
        [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.5);
            border-radius: 999px;
        }

        [data-testid="stSidebar"]::-webkit-scrollbar-track,
        [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
            background: transparent;
        }

        @media (max-width: 1100px) {
            .quick-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 2rem;
            }

            [data-baseweb="tab-list"] {
                top: 0.2rem;
            }

            [data-testid="stAppViewContainer"] .main .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_background_orbs() -> None:
    st.markdown(
        """
        <div class="ambient">
            <div class="orb orb-a"></div>
            <div class="orb orb-b"></div>
            <div class="orb orb-c"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_payload(path: str, mtime: float) -> dict[str, Any]:
    del mtime
    try:
        return joblib.load(path)
    except AttributeError as exc:
        raise RuntimeError(
            "Incompatible model artifact found. Please run: .\\.venv\\Scripts\\python.exe train_model.py"
        ) from exc


def predict_text(model, text: str, threshold: float) -> tuple[str, float, float]:
    proba = float(model.predict_proba([text])[0][1])
    label = "Spam" if proba >= threshold else "Ham"
    confidence = proba if label == "Spam" else 1 - proba
    return label, confidence, proba


def render_probability_meter(probability: float, threshold: float) -> None:
    st.markdown(
        f"""
        <div style="font-size:0.9rem; margin-top:0.25rem;">
            Spam Probability: <strong>{probability:.2%}</strong> | Threshold: <strong>{threshold:.2%}</strong>
        </div>
        <div class="meter"><span style="width:{probability * 100:.2f}%"></span></div>
        """,
        unsafe_allow_html=True,
    )


def get_signal_matches(text: str, metrics: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    normalized = normalize_text(text)
    top_keywords = metrics.get("top_keywords", {})

    matches: list[dict[str, Any]] = []
    for cls in ("spam", "ham"):
        for item in top_keywords.get(cls, []):
            token = str(item.get("token", "")).strip()
            if token and token in normalized:
                matches.append(
                    {
                        "class": cls,
                        "token": token,
                        "weight": float(item.get("weight", 0.0)),
                    }
                )

    matches.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return matches[:limit]


def add_to_history(text: str, label: str, confidence: float, probability: float, threshold: float) -> None:
    if "analysis_log" not in st.session_state:
        st.session_state.analysis_log = []

    st.session_state.analysis_log.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": text,
            "label": label,
            "confidence": round(confidence, 4),
            "spam_probability": round(probability, 4),
            "threshold": round(threshold, 4),
        },
    )
    st.session_state.analysis_log = st.session_state.analysis_log[:250]


def render_sidebar(metrics: dict[str, Any]) -> tuple[float, bool]:
    with st.sidebar:
        st.header("Control Center")
        threshold = st.slider("Spam Threshold", min_value=0.20, max_value=0.80, value=0.50, step=0.01)
        realtime = st.toggle("Real-time analysis", value=True, help="When on, predictions refresh as message changes.")

        st.markdown("### Model Snapshot")
        for name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if name in metrics:
                st.markdown(
                    f'<div class="metric-card"><strong>{name.upper()}</strong><br>{metrics[name]:.4f}</div>',
                    unsafe_allow_html=True,
                )
        if metrics.get("model_version"):
            st.caption(f"Version: {metrics['model_version']}")
        if metrics.get("trained_at"):
            st.caption(f"Trained: {metrics['trained_at']}")
        if "dataset_rows" in metrics:
            st.caption(f"Rows: {metrics['dataset_rows']} | Test: {metrics.get('test_rows', 'n/a')}")

    return threshold, realtime


def read_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Read user-uploaded CSV with common encoding fallbacks.

    Returns:
        (dataframe, encoding_used)
    """
    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("Uploaded file is empty.")

    attempts = [
        ("utf-8-sig", {"engine": "c"}),
        ("utf-8", {"engine": "c"}),
        ("cp1252", {"engine": "c"}),
        ("latin-1", {"engine": "c"}),
        ("utf-8-sig", {"engine": "python", "sep": None}),
        ("utf-8", {"engine": "python", "sep": None}),
        ("cp1252", {"engine": "python", "sep": None}),
        ("latin-1", {"engine": "python", "sep": None}),
    ]

    last_error: Exception | None = None
    for encoding, kwargs in attempts:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding, **kwargs)
            return df, encoding
        except (UnicodeDecodeError, ParserError, ValueError) as exc:
            last_error = exc
            continue

    raise ValueError(
        "Could not parse CSV with supported encodings (utf-8, utf-8-sig, cp1252, latin-1)."
    ) from last_error


def load_metrics_fallback(payload_metrics: Any) -> dict[str, Any]:
    """Prefer payload metrics, fallback to artifacts/metrics.json when needed."""
    if isinstance(payload_metrics, dict) and payload_metrics:
        return payload_metrics

    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {}


def validate_prediction_schema(df: pd.DataFrame) -> None:
    required_columns = {"spam_probability", "prediction"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Prediction output missing required columns: {sorted(missing)}")

    probabilities = pd.to_numeric(df["spam_probability"], errors="coerce")
    if probabilities.isna().any():
        raise ValueError("spam_probability contains non-numeric values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("spam_probability must stay in [0, 1].")

    allowed_labels = {"Spam", "Ham"}
    predicted_labels = set(df["prediction"].dropna().astype(str))
    invalid_labels = sorted(predicted_labels - allowed_labels)
    if invalid_labels:
        raise ValueError(f"prediction contains invalid labels: {invalid_labels}")


def show_roc(metrics: dict) -> None:
    roc = metrics.get("roc_curve")
    if not roc:
        st.info("ROC curve data not available. Retrain model to populate this section.")
        return

    fpr_values = roc.get("fpr", [])
    tpr_values = roc.get("tpr", [])
    if not isinstance(fpr_values, list) or not isinstance(tpr_values, list) or len(fpr_values) != len(tpr_values):
        st.warning("ROC data is malformed. Retrain model to refresh metrics payload.")
        return

    roc_df = pd.DataFrame({"FPR": fpr_values, "TPR": tpr_values})
    if roc_df.empty:
        st.info("ROC curve data not available.")
        return

    diag_df = pd.DataFrame({"FPR": [0.0, 1.0], "TPR": [0.0, 1.0]})

    chart = (
        alt.Chart(roc_df)
        .mark_line(strokeWidth=3, color="#0f766e")
        .encode(
            x=alt.X("FPR:Q", title="False Positive Rate"),
            y=alt.Y("TPR:Q", title="True Positive Rate"),
            tooltip=["FPR", "TPR"],
        )
        .properties(height=320)
    )

    baseline = (
        alt.Chart(diag_df)
        .mark_line(strokeDash=[6, 4], color="#64748b")
        .encode(x="FPR:Q", y="TPR:Q")
        .properties(height=320)
    )
    st.altair_chart(chart + baseline, use_container_width=True)


def show_confusion_matrix(metrics: dict) -> None:
    cm = metrics.get("confusion_matrix")
    if not cm or len(cm) != 2 or any(len(row) != 2 for row in cm):
        st.info("Confusion matrix not available. Retrain model to populate this section.")
        return

    cm_df = pd.DataFrame(
        [
            {"Actual": "Ham", "Predicted": "Ham", "Count": int(cm[0][0])},
            {"Actual": "Ham", "Predicted": "Spam", "Count": int(cm[0][1])},
            {"Actual": "Spam", "Predicted": "Ham", "Count": int(cm[1][0])},
            {"Actual": "Spam", "Predicted": "Spam", "Count": int(cm[1][1])},
        ]
    )

    chart = (
        alt.Chart(cm_df)
        .mark_rect(cornerRadius=8)
        .encode(
            x=alt.X("Predicted:N"),
            y=alt.Y("Actual:N"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="teals")),
            tooltip=["Actual", "Predicted", "Count"],
        )
        .properties(height=240)
    )
    text = chart.mark_text(baseline="middle", fontWeight="bold", color="#0f172a").encode(text="Count:Q")
    st.altair_chart(chart + text, use_container_width=True)


def show_top_keywords(metrics: dict) -> None:
    top = metrics.get("top_keywords", {})
    spam_words = top.get("spam", [])
    ham_words = top.get("ham", [])
    if not spam_words and not ham_words:
        st.info("Top keyword analysis unavailable.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Spam Signals")
        if spam_words:
            spam_df = pd.DataFrame(spam_words)
            chart = alt.Chart(spam_df).mark_bar(color="#be123c").encode(
                x=alt.X("weight:Q", title="Weight"),
                y=alt.Y("token:N", sort="-x", title="Token"),
                tooltip=["token", "weight"],
            )
            st.altair_chart(chart.properties(height=330), use_container_width=True)

    with col2:
        st.subheader("Top Ham Signals")
        if ham_words:
            ham_df = pd.DataFrame(ham_words)
            ham_df["ham_weight"] = ham_df["weight"].abs()
            chart = alt.Chart(ham_df).mark_bar(color="#0f766e").encode(
                x=alt.X("ham_weight:Q", title="Absolute Weight"),
                y=alt.Y("token:N", sort="-x", title="Token"),
                tooltip=["token", "weight"],
            )
            st.altair_chart(chart.properties(height=330), use_container_width=True)


def show_threshold_sweep(metrics: dict[str, Any]) -> None:
    sweep = metrics.get("threshold_sweep", [])
    if not isinstance(sweep, list) or not sweep:
        st.info("Threshold sweep data is unavailable. Retrain to populate this analysis.")
        return

    sweep_df = pd.DataFrame(sweep)
    required_cols = {"threshold", "precision", "recall", "f1_score", "spam_alert_rate"}
    if not required_cols.issubset(set(sweep_df.columns)):
        st.warning("Threshold sweep payload is malformed. Retrain model to refresh metrics.")
        return

    line_df = sweep_df.melt(
        id_vars=["threshold"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="metric",
        value_name="value",
    )
    line_chart = (
        alt.Chart(line_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("threshold:Q", title="Threshold"),
            y=alt.Y("value:Q", title="Score"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["threshold:Q", "metric:N", "value:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(line_chart, use_container_width=True)

    st.caption("Spam alert rate helps explain business impact (how many messages are flagged).")
    sweep_display = sweep_df.copy()
    for col in ["precision", "recall", "f1_score", "spam_alert_rate"]:
        sweep_display[col] = sweep_display[col].map(lambda v: f"{float(v):.2%}")
    st.dataframe(sweep_display, use_container_width=True, hide_index=True)


def render_live_tab(model, metrics: dict[str, Any], threshold: float, realtime: bool) -> None:
    st.subheader("Live Message Lab")
    st.caption("Tune threshold and inspect why the model flags a message.")

    if "live_message" not in st.session_state:
        st.session_state.live_message = ""

    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button("Load Spam-like Example", use_container_width=True):
            st.session_state.live_message = "Urgent! You have won a free prize. Click now to claim."
            st.rerun()
    with sample_col2:
        if st.button("Load Ham-like Example", use_container_width=True):
            st.session_state.live_message = "Hey, can we meet tomorrow at 6 PM near the station?"
            st.rerun()

    user_text = st.text_area(
        "Type SMS text",
        key="live_message",
        height=170,
        placeholder="Paste or type a message to classify...",
    )

    analyze_clicked = st.button("Analyze Message", type="primary", use_container_width=True)
    previous_threshold = st.session_state.get("live_last_threshold")
    threshold_changed = previous_threshold is not None and abs(float(previous_threshold) - float(threshold)) > 1e-12
    should_analyze = bool(user_text.strip()) and (realtime or analyze_clicked or threshold_changed)

    if not should_analyze:
        st.session_state.live_last_threshold = threshold
        st.info("Enter text and click Analyze Message, or keep real-time analysis enabled.")
        return

    label, confidence, probability = predict_text(model, user_text, threshold)
    st.session_state.live_last_threshold = threshold
    baseline_label = "Spam" if probability >= 0.5 else "Ham"
    css_class = "result-spam" if label == "Spam" else "result-ham"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="{css_class}">{label}</div>
            <div style="margin-top:0.2rem; font-size:0.95rem;">Confidence: <strong>{confidence:.2%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_probability_meter(probability, threshold)

    change_note = (
        f"Decision changed by threshold ({threshold:.2f})" if label != baseline_label else "Same as 0.50 baseline"
    )
    change_class = "signal-spam" if label != baseline_label else "signal-ham"
    st.markdown(
        f'<span class="chip {change_class}">{change_note}</span>',
        unsafe_allow_html=True,
    )

    if st.button("Add This Result To Session Log", use_container_width=True):
        add_to_history(user_text, label, confidence, probability, threshold)
        st.success("Added to session log.")

    matches = get_signal_matches(user_text, metrics)
    if matches:
        st.markdown("#### Matched Signal Tokens")
        for match in matches:
            cls = "signal-spam" if match["class"] == "spam" else "signal-ham"
            st.markdown(
                f'<span class="chip {cls}">{match["class"].upper()} | {match["token"]} ({match["weight"]:+.2f})</span>',
                unsafe_allow_html=True,
            )

    with st.expander("Normalized Text Used By Model"):
        st.code(normalize_text(user_text), language="text")


def render_batch_tab(model, threshold: float) -> None:
    st.subheader("Batch CSV Studio")
    st.caption("Upload a dataset and score all rows in one shot.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to start batch scoring.")
        return

    try:
        df, used_encoding = read_uploaded_csv(uploaded)
    except Exception as exc:
        st.error("Could not read CSV file. Try saving it as UTF-8 or Latin-1 and upload again.")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    st.caption(f"Loaded file with encoding: {used_encoding}")

    if df.empty:
        st.warning("Uploaded CSV is empty.")
        return

    candidate_cols = [col for col in df.columns if df[col].dtype == "object"]
    if not candidate_cols:
        candidate_cols = [df.columns[0]]
    text_col = st.selectbox("Text column", options=candidate_cols, index=0)

    source_id = f"{uploaded.name}:{uploaded.size}:{len(df)}:{text_col}"
    run_requested = st.button("Run Batch Scoring", type="primary", use_container_width=True)

    if (
        run_requested
        or st.session_state.get("batch_source_id") != source_id
        or "batch_probs" not in st.session_state
    ):
        texts = df[text_col].fillna("").astype(str)
        probs = model.predict_proba(texts)[:, 1]
        st.session_state.batch_source_id = source_id
        st.session_state.batch_base_df = df.copy()
        st.session_state.batch_text_col = text_col
        st.session_state.batch_probs = probs

    if "batch_probs" not in st.session_state or "batch_base_df" not in st.session_state:
        return

    out = st.session_state.batch_base_df.copy()
    out["spam_probability"] = st.session_state.batch_probs
    out["prediction"] = out["spam_probability"].apply(lambda p: "Spam" if p >= threshold else "Ham")
    try:
        validate_prediction_schema(out)
    except ValueError as exc:
        st.error("Prediction schema validation failed. Please retrain the model and try again.")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    baseline_spam = int((out["spam_probability"] >= 0.5).sum())
    threshold_spam = int((out["prediction"] == "Spam").sum())
    delta = threshold_spam - baseline_spam
    delta_sign = "+" if delta >= 0 else ""
    previous_threshold = st.session_state.get("batch_last_threshold")
    flipped_count = 0
    if previous_threshold is not None and abs(float(previous_threshold) - float(threshold)) > 1e-12:
        previous_labels = pd.Series(st.session_state.batch_probs).apply(
            lambda p: "Spam" if p >= float(previous_threshold) else "Ham"
        )
        flipped_count = int((previous_labels.values != out["prediction"].values).sum())

    st.session_state.batch_last_threshold = threshold

    st.markdown(
        f'<span class="chip signal-ham">Threshold {threshold:.2f} active | Spam count delta vs 0.50: {delta_sign}{delta}</span>',
        unsafe_allow_html=True,
    )
    if previous_threshold is not None:
        st.markdown(
            f'<span class="chip signal-spam">Rows flipped vs previous threshold ({previous_threshold:.2f} → {threshold:.2f}): {flipped_count}</span>',
            unsafe_allow_html=True,
        )
    total_rows = int(len(out))
    spam_rows = int((out["prediction"] == "Spam").sum())
    ham_rows = total_rows - spam_rows
    avg_prob = float(out["spam_probability"].mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rows", f"{total_rows}")
    k2.metric("Spam Rows", f"{spam_rows}")
    k3.metric("Ham Rows", f"{ham_rows}")
    k4.metric("Avg Spam Prob", f"{avg_prob:.2%}")

    hist = (
        alt.Chart(out)
        .mark_bar(color="#0f766e")
        .encode(
            x=alt.X("spam_probability:Q", bin=alt.Bin(maxbins=25), title="Spam Probability"),
            y=alt.Y("count()", title="Row Count"),
            tooltip=[alt.Tooltip("count()", title="Rows")],
        )
        .properties(height=260)
    )
    st.altair_chart(hist, use_container_width=True)

    st.dataframe(out.head(40), use_container_width=True)

    top_risk = out.sort_values("spam_probability", ascending=False).head(15)
    st.markdown("#### Highest-Risk Messages")
    st.dataframe(top_risk[[text_col, "spam_probability", "prediction"]], use_container_width=True)

    st.download_button(
        label="Download Predictions CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="spam_predictions.csv",
        mime="text/csv",
    )


def render_dashboard_tab(metrics: dict[str, Any]) -> None:
    st.subheader("Model Dashboard")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### ROC Curve")
        show_roc(metrics)
    with col2:
        st.markdown("#### Confusion Matrix")
        show_confusion_matrix(metrics)

    st.markdown("#### Most Influential Tokens")
    show_top_keywords(metrics)
    st.markdown("#### Threshold Sweep (Business Tradeoff)")
    show_threshold_sweep(metrics)


def render_session_log_tab() -> None:
    st.subheader("Session Log")
    history = st.session_state.get("analysis_log", [])
    if not history:
        st.info("No saved predictions yet. Use the Live tab and click 'Add This Result To Session Log'.")
        return

    hist_df = pd.DataFrame(history)
    st.dataframe(hist_df, use_container_width=True)

    btn1, btn2 = st.columns(2)
    with btn1:
        st.download_button(
            label="Download Session Log CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="session_prediction_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with btn2:
        if st.button("Clear Session Log", use_container_width=True):
            st.session_state.analysis_log = []
            st.rerun()


def render_user_guide_tab() -> None:
    st.subheader("Product Guide")
    st.markdown('<div class="guide-box"><strong>Step 1.</strong> Open <em>Live Message Lab</em> and type/paste text.</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><strong>Step 2.</strong> Set threshold in sidebar and classify instantly.</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><strong>Step 3.</strong> Use <em>Batch CSV Studio</em> to score entire files.</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><strong>Step 4.</strong> Explore quality metrics in <em>Model Dashboard</em>.</div>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><strong>Step 5.</strong> Save and export decisions from <em>Session Log</em>.</div>', unsafe_allow_html=True)

    st.markdown("#### Pro Tips")
    st.markdown("- Use threshold > 0.50 when you want fewer false spam alerts.")
    st.markdown("- Use threshold < 0.50 when you want to catch maximum spam.")
    st.markdown("- Retrain after dataset updates using `& .\\.venv\\Scripts\\python.exe train_model.py`.")


def render_quick_stats_strip(metrics: dict[str, Any]) -> None:
    accuracy = float(metrics.get("accuracy", 0.0))
    f1_score = float(metrics.get("f1_score", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))

    st.markdown(
        f"""
        <div class="quick-grid">
            <div class="quick-card"><p>Accuracy</p><b>{accuracy:.2%}</b></div>
            <div class="quick-card"><p>F1 Score</p><b>{f1_score:.2%}</b></div>
            <div class="quick-card"><p>ROC AUC</p><b>{roc_auc:.2%}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="SMS Spam Shield", page_icon="📩", layout="wide")
    inject_styles()
    inject_background_orbs()

    if not MODEL_PATH.exists():
        st.error("Model artifact missing. Train first with: & .\\.venv\\Scripts\\python.exe train_model.py")
        st.stop()

    model_mtime = MODEL_PATH.stat().st_mtime
    try:
        payload = load_model_payload(str(MODEL_PATH), model_mtime)
    except Exception as exc:
        st.error("Unable to load model artifact. Please retrain and restart the app.")
        st.exception(exc)
        st.stop()

    st.markdown(
        '<div class="hero"><h1>SMS Spam Shield</h1><p>Dynamic spam detection interface with live scoring, interactive analytics, and batch intelligence.</p></div>',
        unsafe_allow_html=True,
    )

    render_quick_stats_strip(payload.get("training_metrics", {}))

    model = payload["model"]
    metrics = load_metrics_fallback(payload.get("training_metrics", {}))

    threshold, realtime = render_sidebar(metrics)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Live Message Lab",
            "Batch CSV Studio",
            "Model Dashboard",
            "Session Log",
            "User Guide",
        ]
    )

    with tab1:
        render_live_tab(model, metrics, threshold, realtime)

    with tab2:
        render_batch_tab(model, threshold)

    with tab3:
        render_dashboard_tab(metrics)

    with tab4:
        render_session_log_tab()

    with tab5:
        render_user_guide_tab()


if __name__ == "__main__":
    main()
