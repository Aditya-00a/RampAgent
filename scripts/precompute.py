"""Pre-compute analysis results into a static JSON for the Vercel frontend."""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import load_transactions, load_policy, preprocess_for_anomaly
from engine.anomaly_detector import AnomalyDetector
from engine.policy_checker import PolicyChecker, fallback_policy_check
from engine.risk_scorer import RiskScorer
from engine.risk_framework import build_risk_framework
from config import CEREBRAS_API_KEY

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "data", "results.json")


def main():
    print("Loading data...")
    tx_path = os.path.join(DATA_DIR, "sample_transactions.csv")
    pol_path = os.path.join(DATA_DIR, "sample_policy.md")
    df = load_transactions(tx_path)
    policy_text = load_policy(pol_path)
    df_feat = preprocess_for_anomaly(df)

    print(f"Loaded {len(df)} transactions")

    # --- Anomaly Detection ---
    print("Running anomaly detection...")
    detector = AnomalyDetector(df_feat)
    anomaly_results = detector.run_all()

    # --- Policy Checking ---
    if CEREBRAS_API_KEY:
        print("Extracting policy rules (LLM)...")
        try:
            checker = PolicyChecker(policy_text)
            rules = checker.extract_rules()
            print(f"Extracted {len(rules)} policy rules")
            policy_results = checker.check_all(df_feat)
        except Exception as e:
            print(f"LLM policy check failed ({e}); using rule-based fallback")
            policy_results = fallback_policy_check(df_feat)
    else:
        print("No CEREBRAS_API_KEY — using rule-based policy fallback")
        policy_results = fallback_policy_check(df_feat)

    # --- Risk Scoring ---
    print("Computing risk scores...")
    scorer = RiskScorer()
    risk_df = scorer.compute_risk(df_feat, anomaly_results, policy_results)

    print("Generating narratives...")
    risk_df = scorer.generate_narratives(risk_df)

    # --- Enterprise risk framework ---
    print("Building risk framework (exposure, KRIs, register, controls)...")
    framework = build_risk_framework(risk_df)

    # --- Build output JSON ---
    print("Building output JSON...")

    total_spend = float(risk_df["amount"].sum())
    flagged = risk_df[risk_df["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])]
    flagged_count = len(flagged)
    flag_rate = flagged_count / len(risk_df) * 100 if len(risk_df) > 0 else 0
    violation_count = int(risk_df["has_policy_violation"].sum())
    savings = float(flagged["amount"].sum() * 0.15)

    risk_dist = risk_df["risk_level"].value_counts().to_dict()

    # Category spend
    cat_spend = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    category_data = {"labels": cat_spend.index.tolist(), "values": [float(v) for v in cat_spend.values]}

    # Department spend
    dept_spend = df.groupby("department")["amount"].sum().sort_values(ascending=False)
    department_data = {"labels": dept_spend.index.tolist(), "values": [float(v) for v in dept_spend.values]}

    # Flagged transactions
    flagged_list = []
    for _, row in flagged.sort_values("combined_score", ascending=False).iterrows():
        violations = row.get("policy_violations", [])
        if isinstance(violations, list):
            v_strs = []
            for v in violations:
                if isinstance(v, dict):
                    v_strs.append(v.get("explanation", str(v)))
                else:
                    v_strs.append(str(v))
        else:
            v_strs = []

        flags = row.get("anomaly_flags", [])
        if not isinstance(flags, list):
            flags = []

        flagged_list.append({
            "transaction_id": str(row["transaction_id"]),
            "employee_name": str(row["employee_name"]),
            "department": str(row["department"]),
            "vendor": str(row["vendor"]),
            "category": str(row["category"]),
            "amount": float(row["amount"]),
            "description": str(row["description"]),
            "date": str(row["date"]),
            "receipt_attached": bool(row["receipt_attached"]),
            "pre_approved": bool(row["pre_approved"]),
            "risk_level": str(row["risk_level"]),
            "combined_score": float(row["combined_score"]),
            "anomaly_flags": [str(f) for f in flags],
            "policy_violations": v_strs,
            "narrative": str(row.get("narrative", "")),
        })

    # Executive insights
    emp_flags = risk_df[risk_df["risk_level"].isin(["HIGH", "CRITICAL"])].groupby("employee_name").size()
    top_emp = emp_flags.idxmax() if not emp_flags.empty else "N/A"
    top_emp_count = int(emp_flags.max()) if not emp_flags.empty else 0

    cat_flags = risk_df[risk_df["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])].groupby("category")
    if not cat_flags.ngroups == 0:
        cat_flag_counts = cat_flags.size()
        top_cat = cat_flag_counts.idxmax()
        top_cat_count = int(cat_flag_counts.max())
        top_cat_spend = float(cat_flags["amount"].sum().loc[top_cat])
    else:
        top_cat, top_cat_count, top_cat_spend = "N/A", 0, 0.0

    # All transactions for table
    all_tx = []
    for _, row in risk_df.iterrows():
        all_tx.append({
            "transaction_id": str(row["transaction_id"]),
            "employee_name": str(row["employee_name"]),
            "vendor": str(row["vendor"]),
            "category": str(row["category"]),
            "amount": float(row["amount"]),
            "date": str(row["date"]),
            "risk_level": str(row["risk_level"]),
            "combined_score": float(row["combined_score"]),
        })

    output = {
        "summary": {
            "total_transactions": len(risk_df),
            "total_spend": total_spend,
            "flagged_count": flagged_count,
            "flag_rate": flag_rate,
            "violation_count": violation_count,
            "estimated_savings": savings,
            "flagged_exposure": framework["exposure"]["flagged_exposure"],
            "exposure_rate": framework["exposure"]["exposure_rate"],
            "expected_loss": framework["exposure"]["expected_loss"],
            "critical_exposure": framework["exposure"]["critical_exposure"],
        },
        "exposure": framework["exposure"],
        "kris": framework["kris"],
        "controls": framework["controls"],
        "risk_register": framework["risk_register"],
        "risk_distribution": risk_dist,
        "category_spend": category_data,
        "department_spend": department_data,
        "flagged_transactions": flagged_list,
        "executive_insights": {
            "highest_risk_employee": top_emp,
            "highest_risk_employee_flags": top_emp_count,
            "highest_risk_category": top_cat,
            "highest_risk_category_flags": top_cat_count,
            "highest_risk_category_spend": top_cat_spend,
            "detection_methods": 6,
        },
        "all_transactions": all_tx,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Written {os.path.getsize(OUT_PATH)} bytes to {OUT_PATH}")
    print(f"  {len(risk_df)} transactions, {flagged_count} flagged, {violation_count} violations")


if __name__ == "__main__":
    main()
