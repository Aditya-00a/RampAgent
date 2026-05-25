import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

# Cerebras (primary LLM — free, fast)
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
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
