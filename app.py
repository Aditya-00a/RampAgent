import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config import RISK_THRESHOLDS, CEREBRAS_API_KEY
from utils.data_loader import load_transactions, load_policy, preprocess_for_anomaly
from engine.anomaly_detector import AnomalyDetector
from engine.policy_checker import PolicyChecker
from engine.risk_scorer import RiskScorer
from engine.risk_framework import build_risk_framework

# -- Page config ---------------------------------------------------------------

st.set_page_config(
    page_title="RampAgent — Risk & Controls Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Ramp Brand Colors ---------------------------------------------------------
# Primary: #1C1B17 (near-black), #010518 (dark navy)
# Accent:  #E4FF54 (chartreuse/lime — Ramp's signature highlight)
# Grays:   #2B2E35, #2D3748, #4A5568, #A0AEC0, #E2E8F0
# Alert:   #E53E3E (crimson)

RAMP_CHARTREUSE = "#E4FF54"
RAMP_DARK = "#010518"
RAMP_CARD = "#111318"
RAMP_BORDER = "#1e2028"
RAMP_GRAY = "#4A5568"
RAMP_LIGHT_GRAY = "#A0AEC0"

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────── */
    .main .block-container { padding-top: 1rem; max-width: 1200px; }
    h1, h2, h3 { letter-spacing: -0.3px; }

    /* ── Header banner ──────────────────────────────── */
    .ramp-header {
        background: #010518;
        border: 1px solid #1e2028;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    .ramp-header::before {
        content: '';
        position: absolute;
        top: -40%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(228,255,84,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .ramp-header h1 {
        margin: 0 0 0.25rem 0;
        color: #fff;
        font-size: 1.6rem !important;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .ramp-header .ramp-logo {
        color: #E4FF54;
        font-weight: 800;
    }
    .ramp-header .sub {
        color: #A0AEC0;
        font-size: 0.85rem;
        margin: 0;
    }
    .ramp-header .badge {
        display: inline-block;
        background: rgba(228,255,84,0.12);
        border: 1px solid rgba(228,255,84,0.3);
        color: #E4FF54;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 12px;
        vertical-align: middle;
        letter-spacing: 1px;
    }

    /* ── Metric cards ───────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #111318;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid #1e2028;
    }
    div[data-testid="stMetric"] label {
        color: #A0AEC0 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.7rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #E4FF54 !important;
    }

    /* ── Expander cards ─────────────────────────────── */
    div[data-testid="stExpander"] {
        border: 1px solid #1e2028;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(228,255,84,0.35);
    }

    /* ── Risk score bar ─────────────────────────────── */
    .risk-bar {
        width: 100%;
        height: 5px;
        background: #1e2028;
        border-radius: 3px;
        overflow: hidden;
        margin-top: 4px;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* ── Severity badges ────────────────────────────── */
    .sev-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        margin-right: 6px;
        vertical-align: middle;
    }
    .sev-critical { background: rgba(229,62,62,0.15); color: #fc8181; border: 1px solid rgba(229,62,62,0.3); }
    .sev-high { background: rgba(237,137,54,0.15); color: #fbd38d; border: 1px solid rgba(237,137,54,0.3); }
    .sev-medium { background: rgba(228,255,84,0.12); color: #E4FF54; border: 1px solid rgba(228,255,84,0.25); }
    .sev-low { background: rgba(72,187,120,0.15); color: #9ae6b4; border: 1px solid rgba(72,187,120,0.3); }

    /* ── AI narrative box ───────────────────────────── */
    .ai-narrative {
        background: rgba(228,255,84,0.04);
        border: 1px solid rgba(228,255,84,0.15);
        border-left: 3px solid #E4FF54;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
        color: #E2E8F0;
    }
    .ai-narrative .ai-label {
        color: #E4FF54;
        font-weight: 700;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    /* ── Executive insight cards ────────────────────── */
    .insight-card {
        background: #111318;
        border: 1px solid #1e2028;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .insight-card .insight-title {
        color: #A0AEC0;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .insight-card .insight-value {
        color: #E2E8F0;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .insight-card strong {
        color: #fff;
    }

    /* ── KRI / RAG cards ────────────────────────────── */
    .kri-card {
        background: #111318;
        border: 1px solid #1e2028;
        border-left-width: 4px;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        height: 100%;
    }
    .kri-card.rag-red    { border-left-color: #E53E3E; }
    .kri-card.rag-amber  { border-left-color: #ED8936; }
    .kri-card.rag-green  { border-left-color: #48BB78; }
    .kri-name {
        color: #A0AEC0;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kri-value { font-size: 1.5rem; font-weight: 800; }
    .kri-card.rag-red    .kri-value { color: #fc8181; }
    .kri-card.rag-amber  .kri-value { color: #fbd38d; }
    .kri-card.rag-green  .kri-value { color: #9ae6b4; }
    .kri-detail { color: #4A5568; font-size: 0.7rem; margin-top: 2px; }
    .kri-appetite { color: #4A5568; font-size: 0.65rem; margin-top: 4px; }
    .rag-pill {
        display: inline-block; font-size: 0.6rem; font-weight: 700;
        padding: 1px 7px; border-radius: 10px; letter-spacing: 0.5px;
        vertical-align: middle; margin-left: 6px;
    }
    .rag-pill.rag-red   { background: rgba(229,62,62,0.15); color: #fc8181; }
    .rag-pill.rag-amber { background: rgba(237,137,54,0.15); color: #fbd38d; }
    .rag-pill.rag-green { background: rgba(72,187,120,0.15); color: #9ae6b4; }

    /* ── Tabs ───────────────────────────────────────── */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    /* ── Footer ─────────────────────────────────────── */
    .ramp-footer {
        text-align: center;
        color: #4A5568;
        font-size: 0.75rem;
        padding: 1.5rem 0;
        border-top: 1px solid #1e2028;
        margin-top: 1rem;
    }
    .ramp-footer a { color: #E4FF54; text-decoration: none; }
    .ramp-footer strong { color: #A0AEC0; }

    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #0a0c10;
        border-right: 1px solid #1e2028;
    }
</style>
""", unsafe_allow_html=True)

# -- Header --------------------------------------------------------------------

col_title, col_upload = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div class="ramp-header">
        <h1><span class="ramp-logo">RampAgent</span> Risk Intelligence <span class="badge">CONTINUOUS CONTROLS MONITORING</span></h1>
        <p class="sub">AI-driven anomaly detection, automated control testing &amp; enterprise risk scoring across 100% of corporate spend</p>
    </div>
    """, unsafe_allow_html=True)

# -- Data loading ---------------------------------------------------------------

with col_upload:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["date"])
    df["receipt_attached"] = df["receipt_attached"].astype(bool)
    df["pre_approved"] = df["pre_approved"].astype(bool)
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
else:
    df = load_transactions()

df = preprocess_for_anomaly(df)

# -- Load policy ---------------------------------------------------------------

policy_text = load_policy()

# -- Run engines ---------------------------------------------------------------

@st.cache_data(show_spinner="Detecting anomalies across transactions...")
def run_anomaly_detection(_df):
    detector = AnomalyDetector(_df)
    return detector.run_all()

@st.cache_data(show_spinner="Extracting policy rules & checking compliance...")
def run_policy_check(_df, _policy_text):
    if not CEREBRAS_API_KEY or CEREBRAS_API_KEY == "":
        results = []
        for _, row in _df.iterrows():
            violations = []
            has_v = False

            if row["category"] == "Meals & Dining" and row["amount"] > 75:
                has_v = True
                violations.append({
                    "rule_id": "meal_limit",
                    "severity": "medium",
                    "explanation": f"${row['amount']:.2f} exceeds $75/person client meal limit",
                })
            if row["category"] == "Travel - Flights" and row["amount"] > 500 and not row["pre_approved"]:
                has_v = True
                violations.append({
                    "rule_id": "flight_preapproval",
                    "severity": "high",
                    "explanation": f"Flight ${row['amount']:.2f} exceeds $500 — requires pre-approval",
                })
            if row["category"] == "Travel - Hotels" and row["amount"] > 250:
                has_v = True
                violations.append({
                    "rule_id": "hotel_nightly",
                    "severity": "medium",
                    "explanation": f"${row['amount']:.2f} exceeds $250/night hotel limit",
                })
            if row["category"] == "Software & SaaS" and row["amount"] > 200 and not row["pre_approved"]:
                has_v = True
                violations.append({
                    "rule_id": "software_it_approval",
                    "severity": "high",
                    "explanation": f"Software purchase ${row['amount']:.2f} exceeds $200 — requires IT sign-off",
                })
            if row["category"] == "Entertainment" and row["amount"] > 100:
                has_v = True
                violations.append({
                    "rule_id": "entertainment_limit",
                    "severity": "medium",
                    "explanation": f"${row['amount']:.2f} exceeds $100/person entertainment limit",
                })
            if not row["receipt_attached"] and row["amount"] > 25:
                has_v = True
                violations.append({
                    "rule_id": "receipt_required",
                    "severity": "low",
                    "explanation": f"Missing receipt for ${row['amount']:.2f} transaction (required over $25)",
                })
            if row.get("hour", 12) >= 23 or row.get("hour", 12) <= 4:
                has_v = True
                violations.append({
                    "rule_id": "late_night",
                    "severity": "medium",
                    "explanation": f"Transaction at {row.get('hour', 0):02d}:00 — auto-flagged (11PM-5AM policy)",
                })

            score = max(0.0, 1.0 - 0.25 * len(violations)) if violations else 1.0
            results.append({
                "has_violation": has_v,
                "violations": violations,
                "compliance_score": score,
            })
        return pd.DataFrame(results, index=_df.index)

    checker = PolicyChecker(_policy_text)
    checker.extract_rules()
    return checker.check_all(_df, use_llm_for_ambiguous=True)


# -- Run pipeline ---------------------------------------------------------------

anomaly_results = run_anomaly_detection(df)
policy_results = run_policy_check(df, policy_text)

scorer = RiskScorer()
risk_df = scorer.compute_risk(df, anomaly_results, policy_results)

# Generate LLM narratives
if CEREBRAS_API_KEY and CEREBRAS_API_KEY != "":
    try:
        risk_df = scorer.generate_narratives(risk_df)
    except Exception:
        pass

# Fill fallback narratives for any still empty
for idx, row in risk_df.iterrows():
    if row["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL") and (not row.get("narrative") or str(row.get("narrative", "")).strip() == ""):
        parts = []
        flags = row.get("anomaly_flags", [])
        violations = row.get("policy_violations", [])

        # Build a specific narrative from available signals
        amount = row["amount"]
        vendor = row["vendor"]
        category = row["category"]
        employee = row["employee_name"]

        if isinstance(flags, list) and flags:
            flag_details = []
            for f in flags[:3]:
                fs = str(f)
                if "zscore" in fs:
                    flag_details.append(f"statistical outlier for {category}")
                elif "velocity" in fs:
                    flag_details.append(f"spending velocity spike detected")
                elif "first_time" in fs:
                    flag_details.append(f"{vendor} is a first-time vendor for this employee")
                elif "duplicate" in fs:
                    flag_details.append("potential duplicate transaction")
                elif "isolation_forest" in fs:
                    flag_details.append("multivariate anomaly detected")
                elif "late_night" in fs or "weekend" in fs:
                    flag_details.append("unusual timing pattern")
                elif "missing_receipt" in fs:
                    flag_details.append("missing receipt documentation")
                else:
                    flag_details.append(fs)
            parts.append(f"${amount:,.2f} charge at {vendor} was flagged: {', '.join(flag_details)}.")

        if isinstance(violations, list) and violations:
            v_strs = []
            for v in violations[:2]:
                if isinstance(v, dict):
                    v_strs.append(v.get("explanation", str(v)))
                else:
                    v_strs.append(str(v))
            parts.append(f"Policy: {'; '.join(v_strs)}.")

        if not parts:
            parts.append(f"${amount:,.2f} transaction at {vendor} by {employee} warrants review.")

        parts.append("Recommend manual review by finance team.")
        risk_df.at[idx, "narrative"] = " ".join(parts)

# -- Enterprise risk framework --------------------------------------------------

framework = build_risk_framework(risk_df)
exposure = framework["exposure"]
kris = framework["kris"]
controls = framework["controls"]
risk_register = framework["risk_register"]

# -- Helper functions -----------------------------------------------------------

def risk_color(level):
    return {"CRITICAL": "#E53E3E", "HIGH": "#ED8936", "MEDIUM": "#E4FF54", "LOW": "#48BB78"}.get(level, "#4A5568")

def severity_badge(sev):
    sev_lower = sev.lower()
    return f'<span class="sev-badge sev-{sev_lower}">{sev.upper()}</span>'

def risk_score_bar(score, level):
    color = risk_color(level)
    pct = min(score * 100, 100)
    return f"""<div style="display:flex;align-items:center;gap:8px;margin-top:2px;">
        <span style="color:{color};font-weight:700;font-size:0.85rem;">{score:.0%}</span>
        <div class="risk-bar" style="flex:1;"><div class="risk-bar-fill" style="width:{pct}%;background:{color};"></div></div>
    </div>"""

# -- Metrics row ----------------------------------------------------------------

total_spend = df["amount"].sum()
flagged = risk_df[risk_df["risk_level"] != "LOW"]
flagged_count = len(flagged)
flagged_pct = flagged_count / len(df) * 100
policy_violations = risk_df[risk_df["has_policy_violation"] == True]
critical_count = len(risk_df[risk_df["risk_level"] == "CRITICAL"])
flagged_amount = flagged["amount"].sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Spend Under Monitoring", f"${total_spend:,.0f}", f"{len(df)} txns · 100% coverage")
m2.metric("Exposure at Risk", f"${exposure['flagged_exposure']:,.0f}", f"{exposure['exposure_rate']:.1f}% of spend", delta_color="inverse")
m3.metric("Expected Loss", f"${exposure['expected_loss']:,.0f}", "probability-weighted", delta_color="off")
m4.metric("Critical Risks", f"{critical_count}", delta=f"{critical_count}" if critical_count > 0 else None, delta_color="inverse")
m5.metric("Control Exceptions", f"{flagged_count}", f"{flagged_pct:.1f}% failure rate", delta_color="inverse")

# -- Key Risk Indicators (KRIs) -------------------------------------------------

st.markdown("")  # spacer
st.markdown("##### Key Risk Indicators &nbsp;·&nbsp; <span style='color:#4A5568;font-size:0.8rem;font-weight:400;'>measured against defined risk appetite</span>", unsafe_allow_html=True)

kri_cols = st.columns(3)
for i, k in enumerate(kris):
    rag = k["status"].lower()
    with kri_cols[i % 3]:
        st.markdown(f"""<div class="kri-card rag-{rag}">
            <div class="kri-name">{k['name']}<span class="rag-pill rag-{rag}">{k['status']}</span></div>
            <div class="kri-value">{k['value']:.1f}{k['unit']}</div>
            <div class="kri-detail">{k['detail']}</div>
            <div class="kri-appetite">Appetite: amber ≥ {k['amber']:.0f}{k['unit']} · red ≥ {k['red']:.0f}{k['unit']}</div>
        </div>""", unsafe_allow_html=True)

# -- Executive Insights ---------------------------------------------------------

st.markdown("")  # spacer

# Find the riskiest employee and category
emp_flags = risk_df.groupby("employee_name").agg(
    flags=("risk_level", lambda x: (x != "LOW").sum()),
    spend=("amount", "sum"),
    critical=("risk_level", lambda x: (x == "CRITICAL").sum()),
).sort_values("flags", ascending=False)

cat_flags = risk_df.groupby("category").agg(
    flags=("risk_level", lambda x: (x != "LOW").sum()),
    spend=("amount", "sum"),
).sort_values("flags", ascending=False)

top_emp = emp_flags.index[0] if len(emp_flags) > 0 else "N/A"
top_emp_flags = int(emp_flags.iloc[0]["flags"]) if len(emp_flags) > 0 else 0
top_emp_critical = int(emp_flags.iloc[0]["critical"]) if len(emp_flags) > 0 else 0
top_cat = cat_flags.index[0] if len(cat_flags) > 0 else "N/A"
top_cat_flags = int(cat_flags.iloc[0]["flags"]) if len(cat_flags) > 0 else 0

# Detection methods used
methods_used = set()
for _, r in risk_df.iterrows():
    fl = r.get("anomaly_flags", [])
    if isinstance(fl, list):
        for f in fl:
            fs = str(f).split(" ")[0].split("(")[0].strip()
            methods_used.add(fs)

ic1, ic2, ic3 = st.columns(3)
with ic1:
    st.markdown(f"""<div class="insight-card">
        <div class="insight-title">Highest-Risk Employee</div>
        <div class="insight-value"><strong>{top_emp}</strong> — {top_emp_flags} flagged transactions, {top_emp_critical} critical</div>
    </div>""", unsafe_allow_html=True)
with ic2:
    st.markdown(f"""<div class="insight-card">
        <div class="insight-title">Highest-Risk Category</div>
        <div class="insight-value"><strong>{top_cat}</strong> — {top_cat_flags} flagged across ${cat_flags.iloc[0]['spend']:,.0f} spend</div>
    </div>""", unsafe_allow_html=True)
with ic3:
    methods_str = ", ".join(sorted(methods_used)[:6]) if methods_used else "N/A"
    st.markdown(f"""<div class="insight-card">
        <div class="insight-title">Detection Methods Active</div>
        <div class="insight-value">{len(methods_used)} methods: {methods_str}</div>
    </div>""", unsafe_allow_html=True)

# -- Charts row -----------------------------------------------------------------

st.divider()
chart1, chart2 = st.columns(2)

color_map = {"CRITICAL": "#E53E3E", "HIGH": "#ED8936", "MEDIUM": "#E4FF54", "LOW": "#48BB78"}

with chart1:
    st.subheader("Risk Distribution")
    risk_counts = risk_df["risk_level"].value_counts()
    fig_risk = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map=color_map,
        hole=0.45,
    )
    fig_risk.update_traces(textposition="inside", textinfo="value+label")
    fig_risk.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with chart2:
    st.subheader("Flagged Transactions Timeline")
    flagged_plot = risk_df[risk_df["risk_level"] != "LOW"].copy()
    if not flagged_plot.empty:
        flagged_plot["date"] = pd.to_datetime(flagged_plot["date"])
        fig_timeline = px.scatter(
            flagged_plot,
            x="date",
            y="amount",
            color="risk_level",
            color_discrete_map=color_map,
            size="combined_score",
            hover_data=["employee_name", "vendor", "category"],
            size_max=18,
        )
        fig_timeline.update_layout(
            xaxis_title="",
            yaxis_title="Amount ($)",
            showlegend=True,
            legend_title="Risk Level",
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("No flagged transactions to display.")

# -- Flagged transactions detail -------------------------------------------------

st.divider()
st.subheader("Control Exceptions / Risk Events")
st.caption("Transactions where one or more automated controls raised an exception, ranked by risk score.")

level_filter = st.multiselect(
    "Filter by risk level",
    ["CRITICAL", "HIGH", "MEDIUM"],
    default=["CRITICAL", "HIGH", "MEDIUM"],
)

flagged_display = risk_df[risk_df["risk_level"].isin(level_filter)].sort_values(
    "combined_score", ascending=False
).head(25)

st.caption(f"Showing top {len(flagged_display)} of {len(risk_df[risk_df['risk_level'].isin(level_filter)])} flagged transactions")

risk_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}

# Ramp-style icons for expander headers
risk_dots = {
    "CRITICAL": '<span style="color:#E53E3E;font-size:0.9rem;">&#x25CF;</span>',
    "HIGH": '<span style="color:#ED8936;font-size:0.9rem;">&#x25CF;</span>',
    "MEDIUM": '<span style="color:#E4FF54;font-size:0.9rem;">&#x25CF;</span>',
    "LOW": '<span style="color:#48BB78;font-size:0.9rem;">&#x25CF;</span>',
}

for _, row in flagged_display.iterrows():
    icon = risk_icons.get(row["risk_level"], "⚪")
    with st.expander(
        f"{icon} **{row['risk_level']}** — {row['vendor']} · ${row['amount']:,.2f} · {row['employee_name']}"
    ):
        # Top row: key facts
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Employee:** {row['employee_name']} ({row['department']})")
        c2.markdown(f"**Category:** {row['category']}")
        c3.markdown(f"**Date:** {str(row['date'])[:19]}")

        c4, c5, c6 = st.columns(3)
        c4.markdown(f"**Amount:** ${row['amount']:,.2f}")
        c5.markdown(f"**Receipt:** {'✅ Yes' if row['receipt_attached'] else '❌ **Missing**'}")
        c6.markdown(f"**Pre-approved:** {'✅ Yes' if row['pre_approved'] else '➖ No'}")

        if row.get("description") and str(row["description"]) != "nan":
            st.markdown(f"**Description:** {row['description']}")

        # Risk score bar
        st.markdown(risk_score_bar(row["combined_score"], row["risk_level"]), unsafe_allow_html=True)

        # Anomaly flags
        flags = row.get("anomaly_flags", [])
        if isinstance(flags, list) and flags:
            st.markdown("**Anomaly Flags:**")
            for f in flags:
                st.markdown(f"- `{f}`")

        # Policy violations with severity badges
        violations = row.get("policy_violations", [])
        if isinstance(violations, list) and violations:
            st.markdown("**Policy Violations:**")
            for v in violations:
                if isinstance(v, dict):
                    sev = v.get("severity", "medium")
                    explanation = v.get("explanation", str(v))
                    st.markdown(
                        f"{severity_badge(sev)} {explanation}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"- {v}")

        # AI narrative
        narrative = row.get("narrative", "")
        if narrative and str(narrative).strip():
            st.markdown(
                f"""<div class="ai-narrative">
                    <div class="ai-label">🤖 AI Risk Analysis</div>
                    {narrative}
                </div>""",
                unsafe_allow_html=True,
            )

        st.caption(f"Risk Score: {row['combined_score']:.3f} | Transaction ID: {row['transaction_id']}")

# -- Analytics section -----------------------------------------------------------

st.divider()
st.subheader("Risk & Controls Analytics")

tab_register, tab_ccm, tab_cat, tab_emp, tab_vendor, tab_policy = st.tabs(
    ["🗂 Risk Register", "🛡 Controls (CCM)", "📊 By Category", "👤 By Employee", "🏢 By Vendor", "📋 Policy Violations"]
)

with tab_register:
    st.caption("Inherent risks aggregated from transaction-level control exceptions, ranked by severity. "
               "Each risk maps to the automated control that detects it.")
    if risk_register:
        reg_df = pd.DataFrame([{
            "Risk ID": r["risk_id"],
            "Risk": r["risk_name"],
            "Category": r["risk_category"],
            "Likelihood": r["likelihood"],
            "Impact": r["impact_label"],
            "Rating": r["rating"],
            "Events": r["events"],
            "Exposure": r["exposure"],
            "Avg Score": r["avg_risk_score"],
            "Mapped Control": f"{r['control_id']} · {r['control_name']}",
        } for r in risk_register])

        def _rate_style(val):
            colors = {"CRITICAL": "#E53E3E", "HIGH": "#ED8936", "MEDIUM": "#E4FF54", "LOW": "#48BB78"}
            return f"color: {colors.get(val, '#A0AEC0')}; font-weight: 700;"

        styled = (reg_df.style
                  .format({"Exposure": "${:,.0f}", "Avg Score": "{:.3f}"})
                  .map(_rate_style, subset=["Rating"]))
        st.dataframe(styled, use_container_width=True, hide_index=True)

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Distinct Risks", len(risk_register))
        rc2.metric("Total Risk Exposure", f"${sum(r['exposure'] for r in risk_register):,.0f}")
        crit_high = sum(1 for r in risk_register if r["rating"] in ("CRITICAL", "HIGH"))
        rc3.metric("Critical / High Risks", crit_high)
    else:
        st.info("No risks recorded.")

with tab_ccm:
    st.caption("Continuous Controls Monitoring — every automated control runs against 100% of transactions. "
               "Exception rate drives the effectiveness rating.")
    if controls:
        ccm_df = pd.DataFrame([{
            "Control": f"{c['control_id']} · {c['control_name']}",
            "Objective": c["objective"],
            "Tested": c["tested"],
            "Exceptions": c["exceptions"],
            "Exception Rate": c["exception_rate"],
            "Exposure": c["exposure"],
            "Effectiveness": c["effectiveness"],
        } for c in controls])

        def _eff_style(val):
            colors = {"Effective": "#48BB78", "Needs Improvement": "#ED8936", "Ineffective": "#E53E3E"}
            return f"color: {colors.get(val, '#A0AEC0')}; font-weight: 700;"

        styled = (ccm_df.style
                  .format({"Exception Rate": "{:.1f}%", "Exposure": "${:,.0f}"})
                  .map(_eff_style, subset=["Effectiveness"]))
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.bar_chart(ccm_df.set_index("Control")["Exception Rate"], height=280)
    else:
        st.info("No controls evaluated.")

with tab_cat:
    cat_risk = risk_df.groupby("category").agg(
        total_spend=("amount", "sum"),
        flagged_count=("risk_level", lambda x: (x != "LOW").sum()),
        avg_risk=("combined_score", "mean"),
    ).sort_values("flagged_count", ascending=False)

    st.bar_chart(cat_risk["flagged_count"], height=300)
    st.dataframe(cat_risk.style.format({
        "total_spend": "${:,.0f}", "avg_risk": "{:.3f}",
    }), use_container_width=True)

with tab_emp:
    emp_risk = risk_df.groupby("employee_name").agg(
        total_spend=("amount", "sum"),
        transaction_count=("transaction_id", "count"),
        flagged_count=("risk_level", lambda x: (x != "LOW").sum()),
        critical_count=("risk_level", lambda x: (x == "CRITICAL").sum()),
        avg_risk=("combined_score", "mean"),
    ).sort_values("flagged_count", ascending=False)

    st.bar_chart(emp_risk.head(10)["flagged_count"], height=300)
    st.dataframe(emp_risk.head(10).style.format({
        "total_spend": "${:,.0f}", "avg_risk": "{:.3f}",
    }), use_container_width=True)

with tab_vendor:
    vendor_risk = risk_df.groupby("vendor").agg(
        total_spend=("amount", "sum"),
        transaction_count=("transaction_id", "count"),
        flagged_count=("risk_level", lambda x: (x != "LOW").sum()),
        avg_risk=("combined_score", "mean"),
    ).sort_values("flagged_count", ascending=False)

    st.dataframe(
        vendor_risk.head(15).style.format({
            "total_spend": "${:,.2f}",
            "avg_risk": "{:.3f}",
        }),
        use_container_width=True,
    )

with tab_policy:
    all_violations = []
    for _, row in risk_df.iterrows():
        violations = row.get("policy_violations", [])
        if isinstance(violations, list):
            for v in violations:
                if isinstance(v, dict):
                    all_violations.append({
                        "rule_id": v.get("rule_id", "unknown"),
                        "severity": v.get("severity", "medium"),
                        "explanation": v.get("explanation", str(v)),
                        "employee": row["employee_name"],
                        "amount": row["amount"],
                    })

    if all_violations:
        viol_df = pd.DataFrame(all_violations)

        # Severity distribution
        sev_counts = viol_df["severity"].value_counts()
        st.bar_chart(sev_counts, height=200)

        # Rule ID distribution
        rule_counts = viol_df["rule_id"].value_counts()
        st.bar_chart(rule_counts, height=300)
        st.dataframe(
            viol_df.style.format({"amount": "${:,.2f}"}),
            use_container_width=True,
        )
    else:
        st.info("No policy violations detected.")

# -- AI Chat sidebar -------------------------------------------------------------

with st.sidebar:
    st.markdown("### Ask the Risk Analyst")
    st.caption("Powered by Llama 3.1 via Cerebras")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_q := st.chat_input("e.g. Where is our largest risk exposure?"):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Build context summary for LLM
        summary_stats = {
            "total_transactions": len(risk_df),
            "total_spend": float(risk_df["amount"].sum()),
            "flagged_count": int((risk_df["risk_level"] != "LOW").sum()),
            "critical_count": int((risk_df["risk_level"] == "CRITICAL").sum()),
            "high_count": int((risk_df["risk_level"] == "HIGH").sum()),
            "medium_count": int((risk_df["risk_level"] == "MEDIUM").sum()),
        }

        top_flagged = risk_df[risk_df["risk_level"] != "LOW"].nlargest(10, "combined_score")[
            ["transaction_id", "employee_name", "vendor", "category", "amount",
             "risk_level", "anomaly_flags", "policy_violations"]
        ].to_dict("records")

        emp_summary = risk_df.groupby("employee_name").agg(
            total_spend=("amount", "sum"),
            flagged=("risk_level", lambda x: (x != "LOW").sum()),
        ).sort_values("flagged", ascending=False).head(10).to_dict("index")

        cat_summary = risk_df.groupby("category").agg(
            total_spend=("amount", "sum"),
            flagged=("risk_level", lambda x: (x != "LOW").sum()),
        ).sort_values("flagged", ascending=False).to_dict("index")

        context = f"""You are an AI risk analyst for an enterprise risk & controls monitoring platform covering corporate spend. Answer the user's question using ONLY the data provided. Speak the language of a risk / internal audit team: exposure, control exceptions, risk rating, likelihood and impact. Be specific with dollar amounts, names, and percentages. Keep answers concise (3-5 sentences max). Sound like an automated risk-intelligence system — data-driven, precise, actionable.

SUMMARY:
{json.dumps(summary_stats, indent=2)}

TOP 10 FLAGGED TRANSACTIONS:
{json.dumps(top_flagged, indent=2, default=str)}

EMPLOYEE SUMMARY (top 10 by flags):
{json.dumps(emp_summary, indent=2, default=str)}

CATEGORY SUMMARY:
{json.dumps(cat_summary, indent=2, default=str)}

USER QUESTION: {user_q}"""

        if CEREBRAS_API_KEY:
            try:
                from openai import OpenAI as _OAI
                from config import CEREBRAS_BASE_URL, CEREBRAS_MODEL
                _client = _OAI(base_url=CEREBRAS_BASE_URL, api_key=CEREBRAS_API_KEY)
                _resp = _client.chat.completions.create(
                    model=CEREBRAS_MODEL,
                    messages=[{"role": "user", "content": context}],
                    temperature=0.3,
                    max_tokens=512,
                )
                answer = _resp.choices[0].message.content.strip()
            except Exception as e:
                answer = f"Error querying AI: {e}"
        else:
            answer = "Configure CEREBRAS_API_KEY in .env to enable AI chat."

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# -- Footer --------------------------------------------------------------------

st.markdown("""
<div class="ramp-footer">
    <strong>RampAgent</strong> — Continuous Controls Monitoring &amp; Risk Intelligence &nbsp;&middot;&nbsp;
    Ensemble ML anomaly detection &nbsp;&middot;&nbsp;
    LLM-based control testing (Llama 3.1 via Cerebras) &nbsp;&middot;&nbsp;
    9 automated controls · risk register · KRIs &nbsp;&middot;&nbsp;
    Built by <strong>Aditya Sakhale</strong>
</div>
""", unsafe_allow_html=True)
