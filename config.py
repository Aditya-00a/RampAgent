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

# ── Enterprise Risk Framework ────────────────────────────────────────────────
# Maps every detection signal to (a) the automated control that produced it and
# (b) the enterprise risk it evidences. This is what turns raw anomaly flags into
# a risk taxonomy, a control library, and a risk register that a GRC / internal
# audit team can act on.
#
# impact: 1-5 inherent-impact rating used for the risk register's impact axis.

RISK_SIGNALS = {
    "isolation_forest": {
        "control_id": "CTRL-02",
        "control_name": "Multivariate Anomaly Detection",
        "control_objective": "Identify transactions that deviate across amount, timing and category simultaneously.",
        "risk_id": "R-01",
        "risk_name": "Anomalous / Potentially Fraudulent Transaction",
        "risk_category": "Financial Fraud",
        "impact": 4,
    },
    "zscore_outlier": {
        "control_id": "CTRL-01",
        "control_name": "Statistical Spend-Outlier Detection",
        "control_objective": "Flag transactions exceeding normal spend for their category (>2.5σ).",
        "risk_id": "R-02",
        "risk_name": "Overspend / Budget Leakage",
        "risk_category": "Financial Loss",
        "impact": 3,
    },
    "potential_duplicate": {
        "control_id": "CTRL-03",
        "control_name": "Duplicate Payment Detection",
        "control_objective": "Detect repeated charges to the same vendor for a similar amount within 48h.",
        "risk_id": "R-03",
        "risk_name": "Duplicate / Erroneous Payment",
        "risk_category": "Financial Loss",
        "impact": 4,
    },
    "velocity_spike": {
        "control_id": "CTRL-05",
        "control_name": "Spend-Velocity Monitoring",
        "control_objective": "Detect employees whose weekly spend spikes beyond their rolling baseline.",
        "risk_id": "R-04",
        "risk_name": "Spend-Velocity / Budget Overrun",
        "risk_category": "Financial Loss",
        "impact": 3,
    },
    "first_time_vendor": {
        "control_id": "CTRL-06",
        "control_name": "New-Vendor Screening",
        "control_objective": "Surface first-time / low-history vendors for third-party due diligence.",
        "risk_id": "R-05",
        "risk_name": "Unvetted Third-Party Vendor",
        "risk_category": "Third-Party Risk",
        "impact": 3,
    },
    "late_night": {
        "control_id": "CTRL-04",
        "control_name": "Off-Hours Transaction Monitoring",
        "control_objective": "Flag activity in the 11PM-5AM window for review.",
        "risk_id": "R-06",
        "risk_name": "Off-Hours / Timing Anomaly",
        "risk_category": "Operational Process",
        "impact": 2,
    },
    "weekend_transaction": {
        "control_id": "CTRL-04",
        "control_name": "Off-Hours Transaction Monitoring",
        "control_objective": "Flag weekend activity for review.",
        "risk_id": "R-06",
        "risk_name": "Off-Hours / Timing Anomaly",
        "risk_category": "Operational Process",
        "impact": 2,
    },
    "missing_receipt": {
        "control_id": "CTRL-07",
        "control_name": "Receipt & Documentation Compliance",
        "control_objective": "Ensure supporting documentation exists for transactions over threshold.",
        "risk_id": "R-07",
        "risk_name": "Documentation / Audit-Trail Gap",
        "risk_category": "Compliance",
        "impact": 2,
    },
    "policy_violation": {
        "control_id": "CTRL-08",
        "control_name": "Expense-Policy Limit Enforcement",
        "control_objective": "Test every transaction against codified policy spend limits.",
        "risk_id": "R-08",
        "risk_name": "Expense-Policy Breach",
        "risk_category": "Compliance",
        "impact": 3,
    },
    "preapproval": {
        "control_id": "CTRL-09",
        "control_name": "Pre-Approval / Authorization Enforcement",
        "control_objective": "Verify spend over approval thresholds carries the required authorization.",
        "risk_id": "R-09",
        "risk_name": "Authorization / Approval Control Gap",
        "risk_category": "Governance",
        "impact": 4,
    },
}

# Risk appetite: thresholds (in %) above which a Key Risk Indicator turns
# amber, then red. Lower is better for all of these indicators.
RISK_APPETITE = {
    "control_failure_rate": {"name": "Control Failure Rate", "amber": 10.0, "red": 20.0},
    "policy_violation_rate": {"name": "Policy Violation Rate", "amber": 8.0, "red": 15.0},
    "duplicate_payment_rate": {"name": "Duplicate Payment Rate", "amber": 1.0, "red": 3.0},
    "documentation_gap_rate": {"name": "Documentation Gap Rate", "amber": 5.0, "red": 10.0},
    "off_hours_rate": {"name": "Off-Hours Activity Rate", "amber": 5.0, "red": 12.0},
    "critical_exposure_rate": {"name": "Critical Exposure (% of spend)", "amber": 3.0, "red": 8.0},
}

# Loss-recovery assumption used to estimate recoverable value from flagged spend.
RECOVERY_RATE = 0.15
