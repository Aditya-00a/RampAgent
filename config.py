import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """Read from env vars first, fall back to Streamlit secrets (for Cloud deploy)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


# Cerebras (primary LLM — free, fast)
CEREBRAS_API_KEY = _get_secret("CEREBRAS_API_KEY")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_MODEL = "llama3.1-8b"

RISK_THRESHOLDS = {
    "critical": 0.65,
    "high": 0.45,
    "medium": 0.25,
    "low": 0.0,
}

ANOMALY_WEIGHTS = {
    "isolation_forest": 0.30,
    "zscore": 0.20,
    "duplicate": 0.25,
    "temporal": 0.10,
    "velocity": 0.10,
    "first_time_vendor": 0.05,
}

CATEGORIES = [
    "Meals & Dining",
    "Travel - Flights",
    "Travel - Hotels",
    "Travel - Ground Transport",
    "Software & SaaS",
    "Office Supplies",
    "Entertainment",
    "Professional Services",
    "Telecommunications",
    "Training & Education",
]
