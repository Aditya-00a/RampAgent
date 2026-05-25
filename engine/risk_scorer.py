"""Combined risk scoring and LLM narrative generation — powered by Cerebras."""

import json
import pandas as pd
from openai import OpenAI
from config import CEREBRAS_API_KEY, CEREBRAS_BASE_URL, CEREBRAS_MODEL, RISK_THRESHOLDS


BATCH_NARRATIVE_PROMPT = """You are a corporate expense risk analyst at a company like Ramp. Generate brief, specific risk narratives for these flagged transactions. Write like a Ramp AI expense alert — direct, data-driven, actionable.

FLAGGED TRANSACTIONS:
{transactions_json}

For each transaction, write a 2-3 sentence narrative explaining why it was flagged and what action the finance team should take. Reference specific dollar amounts, policy limits, and detected patterns. Sound like an automated AI system, not a human.

Return a JSON array where each item is:
{{"transaction_id": "TXN-XXXX", "narrative": "Your 2-3 sentence narrative here."}}

Return ONLY valid JSON, no markdown."""


class RiskScorer:
    def __init__(self):
        if CEREBRAS_API_KEY:
            self.client = OpenAI(base_url=CEREBRAS_BASE_URL, api_key=CEREBRAS_API_KEY)
        else:
            self.client = None

    def compute_risk(
        self,
        df: pd.DataFrame,
        anomaly_results: pd.DataFrame,
        policy_results: pd.DataFrame,
    ) -> pd.DataFrame:
        risk_df = df[["transaction_id", "employee_name", "department", "vendor",
                       "category", "amount", "description", "date",
                       "receipt_attached", "pre_approved"]].copy()

        risk_df["anomaly_score"] = anomaly_results["anomaly_score"].values
        risk_df["anomaly_flags"] = anomaly_results["anomaly_flags"].values

        if "has_violation" in policy_results.columns:
            risk_df["has_policy_violation"] = policy_results["has_violation"].values
            risk_df["policy_violations"] = policy_results["violations"].values
            risk_df["compliance_score"] = policy_results["compliance_score"].values
        else:
            risk_df["has_policy_violation"] = False
            risk_df["policy_violations"] = [[] for _ in range(len(df))]
            risk_df["compliance_score"] = 1.0

        risk_df["combined_score"] = self._compute_combined_score(risk_df)
        risk_df["risk_level"] = risk_df["combined_score"].apply(self._score_to_level)
        risk_df["narrative"] = ""

        return risk_df

    def generate_narratives(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        if not self.client:
            return self._fill_fallback_narratives(risk_df)

        flagged = risk_df[risk_df["risk_level"].isin(["MEDIUM", "HIGH", "CRITICAL"])].copy()
        if flagged.empty:
            return risk_df

        batch_items = []
        for _, row in flagged.head(25).iterrows():
            batch_items.append({
                "transaction_id": row["transaction_id"],
                "employee_name": row["employee_name"],
                "department": row["department"],
                "vendor": row["vendor"],
                "category": row["category"],
                "amount": float(row["amount"]),
                "description": row["description"],
                "date": str(row["date"]),
                "receipt_attached": bool(row["receipt_attached"]),
                "pre_approved": bool(row["pre_approved"]),
                "anomaly_flags": row["anomaly_flags"] if isinstance(row["anomaly_flags"], list) else [],
                "policy_violations": (
                    [v.get("explanation", str(v)) if isinstance(v, dict) else str(v)
                     for v in row["policy_violations"]]
                    if isinstance(row["policy_violations"], list) else []
                ),
                "risk_level": row["risk_level"],
            })

        try:
            for i in range(0, len(batch_items), 8):
                batch = batch_items[i : i + 8]
                prompt = BATCH_NARRATIVE_PROMPT.format(
                    transactions_json=json.dumps(batch, indent=2, default=str)
                )
                response = self.client.chat.completions.create(
                    model=CEREBRAS_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=4096,
                )
                text = response.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                narratives = json.loads(text)
                for n in narratives:
                    mask = risk_df["transaction_id"] == n["transaction_id"]
                    risk_df.loc[mask, "narrative"] = n.get("narrative", "")
        except Exception:
            pass

        return self._fill_fallback_narratives(risk_df)

    def _fill_fallback_narratives(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        for idx, row in risk_df.iterrows():
            if row["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL") and not row.get("narrative"):
                risk_df.at[idx, "narrative"] = self._fallback_narrative(row)
        return risk_df

    def _compute_combined_score(self, risk_df: pd.DataFrame) -> pd.Series:
        anomaly_weight = 0.55
        policy_weight = 0.45

        policy_score = 1.0 - risk_df["compliance_score"]
        combined = (
            risk_df["anomaly_score"] * anomaly_weight
            + policy_score * policy_weight
        )

        has_violation = risk_df["has_policy_violation"].astype(float)
        combined = combined + has_violation * 0.15

        has_flags = risk_df["anomaly_flags"].apply(
            lambda x: 1.0 if any("duplicate" in str(f) for f in x) else 0.0
        )
        combined = combined + has_flags * 0.1

        return combined.clip(0, 1)

    def _score_to_level(self, score: float) -> str:
        for level in ["critical", "high", "medium", "low"]:
            if score >= RISK_THRESHOLDS[level]:
                return level.upper()
        return "LOW"

    def _fallback_narrative(self, row) -> str:
        parts = []
        flags = row.get("anomaly_flags", [])
        violations = row.get("policy_violations", [])

        if isinstance(flags, list) and flags:
            flag_strs = [str(f) for f in flags[:3]]
            parts.append(f"Anomaly detection flagged: {', '.join(flag_strs)}.")

        if isinstance(violations, list) and violations:
            v_strs = [v.get("explanation", str(v)) if isinstance(v, dict) else str(v) for v in violations[:2]]
            parts.append(f"Policy violations: {'; '.join(v_strs)}.")

        parts.append("Recommend manual review by finance team.")
        return " ".join(parts)
