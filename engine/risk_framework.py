"""Enterprise risk framework.

Turns the per-transaction risk output (anomaly flags, policy violations and a
combined risk score) into the artifacts a GRC / internal-audit / risk-consulting
team actually works with:

  * Exposure & expected loss   — how much money is at risk, probability-weighted
  * Key Risk Indicators (KRIs) — measured against a defined risk appetite (RAG)
  * Risk Register              — inherent risks ranked by likelihood × impact
  * Continuous Controls Monitoring (CCM) — each automated control's exception
                                 rate and effectiveness rating

Everything here is pure (DataFrame in, plain dicts/lists out) so it can drive
the Streamlit app, the static dashboard precompute, and any export.
"""

from __future__ import annotations

import pandas as pd

from config import RISK_SIGNALS, RISK_APPETITE, RECOVERY_RATE


# ── Signal extraction ────────────────────────────────────────────────────────

def _signals_for_row(row) -> set[str]:
    """Return the set of canonical signal keys a transaction triggered."""
    signals: set[str] = set()

    flags = row.get("anomaly_flags", [])
    if isinstance(flags, list):
        for f in flags:
            fs = str(f).lower()
            if "isolation_forest" in fs:
                signals.add("isolation_forest")
            if "zscore" in fs:
                signals.add("zscore_outlier")
            if "duplicate" in fs:
                signals.add("potential_duplicate")
            if "velocity" in fs:
                signals.add("velocity_spike")
            if "first_time_vendor" in fs:
                signals.add("first_time_vendor")
            if "late_night" in fs:
                signals.add("late_night")
            if "weekend" in fs:
                signals.add("weekend_transaction")
            if "missing_receipt" in fs:
                signals.add("missing_receipt")

    violations = row.get("policy_violations", [])
    if isinstance(violations, list):
        for v in violations:
            rule_id = v.get("rule_id", "") if isinstance(v, dict) else ""
            text = (v.get("explanation", "") if isinstance(v, dict) else str(v)).lower()
            if "approv" in str(rule_id).lower() or "pre-approv" in text or "pre approval" in text:
                signals.add("preapproval")
            else:
                signals.add("policy_violation")

    return signals


# ── Exposure / value-at-risk ─────────────────────────────────────────────────

def compute_exposure(risk_df: pd.DataFrame) -> dict:
    total_spend = float(risk_df["amount"].sum())
    flagged = risk_df[risk_df["risk_level"] != "LOW"]
    critical = risk_df[risk_df["risk_level"] == "CRITICAL"]

    flagged_exposure = float(flagged["amount"].sum())
    # Probability-weighted (expected-loss style) exposure across flagged spend.
    expected_loss = float((flagged["amount"] * flagged["combined_score"]).sum())
    critical_exposure = float(critical["amount"].sum())

    return {
        "total_spend": total_spend,
        "monitored_spend": total_spend,            # 100% of spend is screened
        "coverage_pct": 100.0,
        "flagged_exposure": flagged_exposure,
        "exposure_rate": (flagged_exposure / total_spend * 100) if total_spend else 0.0,
        "expected_loss": expected_loss,
        "critical_exposure": critical_exposure,
        "recoverable_value": flagged_exposure * RECOVERY_RATE,
    }


# ── Key Risk Indicators ──────────────────────────────────────────────────────

def _rag(value: float, appetite_key: str) -> str:
    """Red / Amber / Green status for a 'lower is better' indicator."""
    band = RISK_APPETITE[appetite_key]
    if value >= band["red"]:
        return "RED"
    if value >= band["amber"]:
        return "AMBER"
    return "GREEN"


def compute_kris(risk_df: pd.DataFrame) -> list[dict]:
    n = len(risk_df)
    if n == 0:
        return []

    pct = lambda count: count / n * 100

    flagged = int((risk_df["risk_level"] != "LOW").sum())
    policy = int(risk_df["has_policy_violation"].sum())

    signal_counts = {k: 0 for k in RISK_SIGNALS}
    off_hours = 0
    for _, row in risk_df.iterrows():
        sigs = _signals_for_row(row)
        for s in sigs:
            signal_counts[s] = signal_counts.get(s, 0) + 1
        if "late_night" in sigs or "weekend_transaction" in sigs:
            off_hours += 1

    total_spend = float(risk_df["amount"].sum())
    critical_exposure = float(risk_df[risk_df["risk_level"] == "CRITICAL"]["amount"].sum())
    critical_exposure_rate = (critical_exposure / total_spend * 100) if total_spend else 0.0

    rows = [
        ("control_failure_rate", pct(flagged), f"{flagged} of {n} transactions"),
        ("policy_violation_rate", pct(policy), f"{policy} policy breaches"),
        ("duplicate_payment_rate", pct(signal_counts.get("potential_duplicate", 0)),
         f"{signal_counts.get('potential_duplicate', 0)} suspected duplicates"),
        ("documentation_gap_rate", pct(signal_counts.get("missing_receipt", 0)),
         f"{signal_counts.get('missing_receipt', 0)} missing receipts"),
        ("off_hours_rate", pct(off_hours), f"{off_hours} off-hours transactions"),
        ("critical_exposure_rate", critical_exposure_rate,
         f"${critical_exposure:,.0f} in critical risk"),
    ]

    kris = []
    for key, value, detail in rows:
        band = RISK_APPETITE[key]
        kris.append({
            "id": key,
            "name": band["name"],
            "value": round(value, 1),
            "unit": "%",
            "amber": band["amber"],
            "red": band["red"],
            "status": _rag(value, key),
            "detail": detail,
        })
    return kris


# ── Continuous Controls Monitoring (CCM) ─────────────────────────────────────

def _effectiveness(exception_rate: float) -> str:
    if exception_rate >= 15.0:
        return "Ineffective"
    if exception_rate >= 5.0:
        return "Needs Improvement"
    return "Effective"


def compute_controls(risk_df: pd.DataFrame) -> list[dict]:
    n = len(risk_df)
    # Aggregate exceptions and exposure per control.
    controls: dict[str, dict] = {}
    for sig, meta in RISK_SIGNALS.items():
        cid = meta["control_id"]
        controls.setdefault(cid, {
            "control_id": cid,
            "control_name": meta["control_name"],
            "objective": meta["control_objective"],
            "exceptions": 0,
            "exposure": 0.0,
        })

    for _, row in risk_df.iterrows():
        sigs = _signals_for_row(row)
        seen_controls = set()
        for s in sigs:
            cid = RISK_SIGNALS[s]["control_id"]
            if cid in seen_controls:
                continue  # count a transaction once per control
            seen_controls.add(cid)
            controls[cid]["exceptions"] += 1
            controls[cid]["exposure"] += float(row["amount"])

    out = []
    for cid in sorted(controls):
        c = controls[cid]
        rate = (c["exceptions"] / n * 100) if n else 0.0
        out.append({
            **c,
            "tested": n,
            "exception_rate": round(rate, 1),
            "effectiveness": _effectiveness(rate),
        })
    # Highest exception rate first — where attention is needed.
    out.sort(key=lambda c: c["exception_rate"], reverse=True)
    return out


# ── Risk Register ────────────────────────────────────────────────────────────

_LIKELIHOOD_BANDS = [
    (0.5, "Rare", 1), (2.0, "Unlikely", 2), (5.0, "Possible", 3),
    (12.0, "Likely", 4), (float("inf"), "Almost Certain", 5),
]


def _likelihood(rate_pct: float) -> tuple[str, int]:
    """Map event-frequency (% of all transactions) to a likelihood band + level (1-5)."""
    for threshold, label, level in _LIKELIHOOD_BANDS:
        if rate_pct < threshold:
            return label, level
    return "Almost Certain", 5


def _impact_label(impact: int) -> str:
    return {1: "Insignificant", 2: "Minor", 3: "Moderate",
            4: "Major", 5: "Severe"}.get(impact, "Moderate")


def _rating(likelihood_level: int, impact_level: int) -> str:
    """Inherent-risk rating from a standard 5x5 likelihood x impact matrix (score 1-25)."""
    score = likelihood_level * impact_level
    if score >= 15:
        return "CRITICAL"
    if score >= 9:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def compute_risk_register(risk_df: pd.DataFrame) -> list[dict]:
    """Aggregate transaction-level events into inherent risks."""
    n = len(risk_df)
    register: dict[str, dict] = {}

    for _, row in risk_df.iterrows():
        score = float(row["combined_score"])
        amount = float(row["amount"])
        for sig in _signals_for_row(row):
            meta = RISK_SIGNALS[sig]
            rid = meta["risk_id"]
            entry = register.setdefault(rid, {
                "risk_id": rid,
                "risk_name": meta["risk_name"],
                "risk_category": meta["risk_category"],
                "control_id": meta["control_id"],
                "control_name": meta["control_name"],
                "impact": meta["impact"],
                "events": 0,
                "exposure": 0.0,
                "_score_sum": 0.0,
            })
            entry["events"] += 1
            entry["exposure"] += amount
            entry["_score_sum"] += score

    out = []
    for entry in register.values():
        events = entry["events"]
        likelihood_pct = (events / n * 100) if n else 0.0
        likelihood_label, likelihood_level = _likelihood(likelihood_pct)
        impact_level = entry["impact"]
        avg_score = entry["_score_sum"] / events if events else 0.0
        out.append({
            "risk_id": entry["risk_id"],
            "risk_name": entry["risk_name"],
            "risk_category": entry["risk_category"],
            "control_id": entry["control_id"],
            "control_name": entry["control_name"],
            "events": events,
            "exposure": entry["exposure"],
            "likelihood": likelihood_label,
            "likelihood_pct": round(likelihood_pct, 1),
            "impact": impact_level,
            "impact_label": _impact_label(impact_level),
            "inherent_score": likelihood_level * impact_level,
            "avg_risk_score": round(avg_score, 3),
            "rating": _rating(likelihood_level, impact_level),
        })

    # Order by inherent severity: rating first, then exposure.
    rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    out.sort(key=lambda r: (rank.get(r["rating"], 4), -r["exposure"]))
    return out


def build_risk_framework(risk_df: pd.DataFrame) -> dict:
    """One call → all enterprise-risk artifacts."""
    return {
        "exposure": compute_exposure(risk_df),
        "kris": compute_kris(risk_df),
        "controls": compute_controls(risk_df),
        "risk_register": compute_risk_register(risk_df),
        "trends": compute_trends(risk_df),
    }


# ── Trends / period-over-period ──────────────────────────────────────────────

# For each metric, whether an *increase* is bad ("risk") or just informational
# ("neutral"). Drives the red/green direction colouring in the UI.
_TREND_SENTIMENT = {
    "total_spend": "neutral",
    "flagged_exposure": "risk",
    "expected_loss": "risk",
    "critical_count": "risk",
    "control_exceptions": "risk",
}


def _period_aggregate(d: pd.DataFrame) -> dict:
    flagged = d[d["risk_level"] != "LOW"]
    return {
        "total_spend": float(d["amount"].sum()),
        "flagged_exposure": float(flagged["amount"].sum()),
        "expected_loss": float((flagged["amount"] * flagged["combined_score"]).sum()),
        "critical_count": int((d["risk_level"] == "CRITICAL").sum()),
        "control_exceptions": int((d["risk_level"] != "LOW").sum()),
        "transactions": int(len(d)),
    }


def compute_trends(risk_df: pd.DataFrame) -> dict:
    """Split the data into a current vs. prior period (by date midpoint) and
    compute deltas, plus a weekly time series for trend charts."""
    if "date" not in risk_df.columns or risk_df.empty:
        return {"available": False, "metrics": {}, "weekly": []}

    df = risk_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return {"available": False, "metrics": {}, "weekly": []}

    dmin, dmax = df["date"].min(), df["date"].max()
    mid = dmin + (dmax - dmin) / 2
    current = df[df["date"] >= mid]
    prior = df[df["date"] < mid]

    cur_agg, prior_agg = _period_aggregate(current), _period_aggregate(prior)

    metrics = {}
    for key, sentiment in _TREND_SENTIMENT.items():
        cv, pv = cur_agg[key], prior_agg[key]
        delta_pct = ((cv - pv) / pv * 100) if pv else (100.0 if cv else 0.0)
        direction = "up" if cv > pv else ("down" if cv < pv else "flat")
        good = None if sentiment == "neutral" else (direction == "down")
        metrics[key] = {
            "current": cv, "prior": pv,
            "delta_pct": round(delta_pct, 1),
            "direction": direction, "good": good,
        }

    # Weekly time series (week-starting dates).
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    weekly = []
    for wk, g in df.groupby("week"):
        fl = g[g["risk_level"] != "LOW"]
        weekly.append({
            "week": str(wk.date()),
            "spend": float(g["amount"].sum()),
            "exposure": float(fl["amount"].sum()),
            "expected_loss": float((fl["amount"] * fl["combined_score"]).sum()),
            "exceptions": int((g["risk_level"] != "LOW").sum()),
        })

    return {
        "available": True,
        "period_days": int((dmax - mid).days),
        "current": cur_agg,
        "prior": prior_agg,
        "metrics": metrics,
        "weekly": weekly,
    }
