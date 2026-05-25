# Ramp Expense Intelligence Agent

AI-powered corporate expense review system that combines **ensemble anomaly detection** with **LLM-based policy enforcement** to flag risky transactions, surface compliance violations, and generate actionable risk narratives.

Built as a technical demo inspired by [Ramp's](https://ramp.com) AI-first approach to corporate spend management.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## How It Works

**1. Ensemble Anomaly Detection** — Six detection methods scored and weighted:

| Method | Weight | What it catches |
|--------|--------|----------------|
| Isolation Forest | 30% | Multivariate outliers across amount, time, category |
| Z-Score | 20% | Transactions >3 std devs from category mean |
| Duplicate Detection | 25% | Same vendor + similar amount within 48 hours |
| Temporal Patterns | 10% | Weekend charges, late-night spending (11pm-5am) |
| Velocity Checks | 10% | Spending >2x employee's rolling average |
| First-Time Vendor | 5% | Vendors never seen before for that employee |

**2. LLM Policy Enforcement** — Parses a company expense policy document, extracts structured rules via LLM, then checks every transaction against them. Catches limit overages, missing receipts, pre-approval violations, and more.

**3. Risk Scoring + Narratives** — Combines anomaly scores (55%) and policy compliance (45%) into a unified risk level (LOW / MEDIUM / HIGH / CRITICAL). For flagged transactions, generates specific narratives explaining what was detected and what action to take.

## Tech Stack

- **UI:** Streamlit with custom Ramp-branded dark theme
- **ML:** scikit-learn (Isolation Forest), NumPy/Pandas (statistical methods)
- **LLM:** Llama 3.1-8B via Cerebras (free, fast inference)
- **Charts:** Plotly (interactive risk distribution, timelines, category breakdowns)

## Quick Start

```bash
# Clone
git clone https://github.com/Aditya-00a/RampAgent.git
cd RampAgent

# Install dependencies
pip install -r requirements.txt

# Add your Cerebras API key
echo "CEREBRAS_API_KEY=your_key_here" > .env

# Run
streamlit run app.py
```

Get a free Cerebras API key at [cloud.cerebras.ai](https://cloud.cerebras.ai).

## Project Structure

```
RampAgent/
├── app.py                        # Streamlit dashboard
├── config.py                     # Risk thresholds, anomaly weights, model config
├── requirements.txt
├── data/
│   ├── sample_transactions.csv   # 495 synthetic transactions with planted anomalies
│   └── sample_policy.md          # Corporate expense policy document
├── engine/
│   ├── anomaly_detector.py       # Ensemble ML anomaly detection
│   ├── policy_checker.py         # LLM-based policy extraction + checking
│   └── risk_scorer.py            # Combined risk scoring + narrative generation
└── utils/
    └── data_loader.py            # CSV parsing, validation, preprocessing
```

## What the Dashboard Shows

- **Risk metrics** — total spend, flag rate, policy violations, estimated savings
- **Risk distribution** — pie chart of LOW / MEDIUM / HIGH / CRITICAL breakdown
- **Flagged transactions** — expandable cards with severity badges, risk score bars, and AI-generated narratives
- **Executive insights** — highest-risk employee, highest-risk category, active detection methods
- **Analytics tabs** — spending by category, department, and vendor breakdowns
- **AI Chat** — ask natural-language questions about the expense data

## Deploy to Streamlit Cloud

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repo
4. Set **Main file path** to `app.py`
5. Under **Advanced settings → Secrets**, add:
   ```toml
   CEREBRAS_API_KEY = "your_key_here"
   ```
6. Click **Deploy**

Free Cerebras API key: [cloud.cerebras.ai](https://cloud.cerebras.ai)

## Sample Data

The included dataset has 495 synthetic transactions across 20 employees and 10 categories, with ~100 planted anomalies including duplicates, over-limit meals, weekend entertainment, velocity spikes, missing receipts, and unapproved travel.

---

Built by **Aditya Sakhale** | [LinkedIn](https://linkedin.com/in/adityasakhale)
