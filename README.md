# RampAgent — Continuous Risk & Controls Monitoring

**An AI-driven continuous controls monitoring (CCM) and financial-risk intelligence platform for corporate spend.** RampAgent screens 100% of transactions through a library of automated controls, quantifies financial exposure, and rolls everything up into the artifacts a risk, internal-audit, or GRC team actually works with: a **risk register**, **Key Risk Indicators (KRIs)** measured against a defined risk appetite, and **control-effectiveness ratings**.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Why this matters

Traditional expense review is **sample-based, after-the-fact, and manual** — auditors test a small sample of transactions weeks or months after the spend. RampAgent demonstrates the shift to **continuous controls monitoring**: every transaction is tested against every control, in real time, with risks quantified in dollars and prioritised by inherent severity.

| Traditional spend audit | RampAgent (continuous monitoring) |
|---|---|
| Sample-based (e.g. 5–10%) | 100% transaction coverage |
| Periodic / retrospective | Continuous |
| Findings as a list | Risk register + KRIs + control ratings |
| Qualitative judgement | Quantified exposure & expected loss |
| Manual narrative write-up | LLM-generated risk narratives |

---

## The risk framework

RampAgent layers an enterprise-risk model on top of a transaction-scoring engine. Every detection signal maps to **both** the automated control that produced it **and** the enterprise risk it evidences.

### 1. Automated control library (9 controls)

Nine controls run against 100% of transactions. Each reports an **exception rate** and an **effectiveness rating** (Effective / Needs Improvement / Ineffective).

| Control | Detects | Risk it mitigates |
|---|---|---|
| CTRL-01 Statistical Spend-Outlier Detection | Transactions >2.5σ above category norm | Overspend / budget leakage |
| CTRL-02 Multivariate Anomaly Detection | Outliers across amount, time & category (Isolation Forest) | Anomalous / fraudulent transactions |
| CTRL-03 Duplicate Payment Detection | Same vendor + amount within 48h | Duplicate / erroneous payment |
| CTRL-04 Off-Hours Transaction Monitoring | Late-night (11PM–5AM) & weekend activity | Timing / process anomalies |
| CTRL-05 Spend-Velocity Monitoring | Weekly spend spikes vs. rolling baseline | Budget overrun |
| CTRL-06 New-Vendor Screening | First-time / low-history vendors | Unvetted third-party risk |
| CTRL-07 Receipt & Documentation Compliance | Missing receipts over threshold | Audit-trail gaps |
| CTRL-08 Expense-Policy Limit Enforcement | Breaches of codified policy limits (LLM) | Compliance breach |
| CTRL-09 Pre-Approval / Authorization Enforcement | Spend over approval thresholds without sign-off | Authorization control gap |

### 2. Risk register

Transaction-level exceptions are aggregated into **inherent risks**, each rated on a standard **5×5 likelihood × impact matrix** (LOW → CRITICAL), with the financial exposure and mapped control attached. This is the board-level view of *where* the risk concentrates.

### 3. Key Risk Indicators (KRIs)

Six KRIs are measured against a defined **risk appetite** and flagged Red / Amber / Green:

- Control Failure Rate · Policy Violation Rate · Duplicate Payment Rate
- Documentation Gap Rate · Off-Hours Activity Rate · Critical Exposure (% of spend)

### 4. Exposure & expected loss

- **Exposure at risk** — total spend touched by a control exception
- **Expected loss** — probability-weighted exposure (Σ amount × risk score)
- **Critical exposure** — spend sitting in CRITICAL-rated risk
- **Recoverable value** — estimated recovery at a configurable rate

### 5. AI risk narratives

For every flagged event, an LLM generates a concise, audit-ready narrative explaining what was detected, the exposure, and the recommended action.

---

## How a transaction is scored

1. **Ensemble anomaly detection** — six statistical / ML methods score each transaction:

   | Method | Weight | What it catches |
   |--------|--------|----------------|
   | Isolation Forest | 30% | Multivariate outliers |
   | Z-Score (by category) | 20% | Spend >2.5σ from category mean |
   | Duplicate Detection | 25% | Same vendor + amount within 48h |
   | Temporal Patterns | 10% | Weekend / late-night activity |
   | Velocity Checks | 10% | Spend >2σ above weekly baseline |
   | First-Time Vendor | 5% | Unseen vendors |

2. **LLM control testing** — parses the policy document, extracts structured rules, and tests every transaction for limit, approval, and documentation breaches.

3. **Combined risk score** — anomaly score (55%) + policy non-compliance (45%) → unified risk level (LOW / MEDIUM / HIGH / CRITICAL), which feeds exposure, KRIs, and the register.

---

## Tech stack

- **UI:** Streamlit dashboard (dark, risk-themed) + a zero-dependency static HTML dashboard
- **ML:** scikit-learn (Isolation Forest), NumPy / Pandas (statistical methods)
- **Risk engine:** `engine/risk_framework.py` — exposure, KRIs, register, CCM (pure Python, no LLM required)
- **LLM:** Llama 3.1-8B via Cerebras (control testing + narratives + analyst chat)
- **Charts:** Plotly

## Quick start

```bash
git clone https://github.com/Aditya-00a/RampAgent.git
cd RampAgent
pip install -r requirements.txt

# Optional — enables LLM control testing, narratives & the AI risk analyst
echo "CEREBRAS_API_KEY=your_key_here" > .env

streamlit run app.py
```

Free Cerebras API key: [cloud.cerebras.ai](https://cloud.cerebras.ai). The risk framework (exposure, KRIs, register, controls) runs **with or without** an API key — the key only adds LLM-based policy testing and narratives.

## Project structure

```
RampAgent/
├── app.py                        # Streamlit risk dashboard
├── config.py                     # Risk thresholds, control library, risk appetite
├── data/
│   ├── sample_transactions.csv   # 495 synthetic transactions with planted risk events
│   └── sample_policy.md          # Corporate expense policy (control source)
├── engine/
│   ├── anomaly_detector.py       # Ensemble ML anomaly detection
│   ├── policy_checker.py         # LLM-based control testing
│   ├── risk_scorer.py            # Combined risk scoring + narratives
│   └── risk_framework.py         # Exposure, KRIs, risk register, CCM   ← risk layer
├── scripts/precompute.py         # Builds the static dashboard JSON
├── public/                       # Static HTML dashboard + precomputed results
└── api/chat.py                   # Serverless AI risk-analyst endpoint
```

## What the dashboard shows

- **Risk metrics** — spend under monitoring, exposure at risk, expected loss, critical risks, control exceptions
- **KRI panel** — six indicators with RAG status vs. risk appetite
- **Risk register** — inherent risks ranked by likelihood × impact, with exposure and mapped controls
- **Controls (CCM)** — per-control exception rate and effectiveness rating
- **Control exceptions** — drill-down cards with anomaly flags, policy breaches, and AI risk narratives
- **AI risk analyst** — ask natural-language questions about exposure, controls, and the register

## Sample data

495 synthetic transactions across 20 employees and 10 categories, with ~100 planted risk events: duplicate payments, over-limit spend, weekend/late-night activity, velocity spikes, missing documentation, and unauthorized travel.

## Deploy

- **Streamlit Cloud** — point [share.streamlit.io](https://share.streamlit.io) at `app.py`; add `CEREBRAS_API_KEY` under *Advanced settings → Secrets*.
- **Vercel (static)** — the `public/` dashboard reads a precomputed `results.json` (regenerate with `python scripts/precompute.py`); `api/chat.py` powers the AI analyst.

---

Built by **Aditya Sakhale** | [LinkedIn](https://linkedin.com/in/adityasakhale)
