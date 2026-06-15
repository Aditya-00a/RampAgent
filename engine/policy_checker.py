"""LLM-based expense policy enforcement engine — powered by Cerebras."""

import json
import pandas as pd
from openai import OpenAI
from config import CEREBRAS_API_KEY, CEREBRAS_BASE_URL, CEREBRAS_MODEL


RULE_EXTRACTION_PROMPT = """You are a corporate expense policy analyst. Extract all quantitative rules from this expense policy document into a structured JSON array.

Each rule should have:
- "id": short unique identifier (e.g., "meal_solo_limit")
- "category": the expense category it applies to
- "rule_text": the original rule in plain English
- "limit_amount": the dollar limit (null if not a dollar limit)
- "condition": what triggers this rule (e.g., "per_person", "total", "nightly", "requires_preapproval")
- "severity": how serious a violation is ("low", "medium", "high", "critical")

Return ONLY a valid JSON array, no markdown formatting.

POLICY DOCUMENT:
{policy_text}"""

VIOLATION_CHECK_PROMPT = """You are a corporate expense compliance reviewer. Given this transaction and the applicable policy rules, determine if there are any violations.

TRANSACTION:
- Employee: {employee_name} ({department})
- Vendor: {vendor}
- Category: {category}
- Amount: ${amount:.2f}
- Description: {description}
- Date/Time: {date}
- Receipt Attached: {receipt_attached}
- Pre-approved: {pre_approved}

APPLICABLE POLICY RULES:
{rules_json}

Respond with a JSON object:
{{
  "has_violation": true/false,
  "violations": [
    {{
      "rule_id": "the rule ID violated",
      "severity": "low/medium/high/critical",
      "explanation": "one-sentence explanation of the violation"
    }}
  ],
  "compliance_score": 0.0 to 1.0 (1.0 = fully compliant)
}}

Return ONLY valid JSON, no markdown."""


def _get_client():
    return OpenAI(base_url=CEREBRAS_BASE_URL, api_key=CEREBRAS_API_KEY)


# Deterministic policy limits used when no LLM key is available. Mirrors the
# quantitative rules in data/sample_policy.md so the platform produces real
# compliance findings even offline.
_FALLBACK_RULES = [
    ("Meals & Dining", 75, "meal_limit", "medium", lambda r: True,
     "${amount:.2f} exceeds $75/person client meal limit"),
    ("Travel - Flights", 500, "flight_preapproval", "high", lambda r: not r["pre_approved"],
     "Flight ${amount:.2f} exceeds $500 — requires pre-approval"),
    ("Travel - Hotels", 250, "hotel_nightly", "medium", lambda r: True,
     "${amount:.2f} exceeds $250/night hotel limit"),
    ("Software & SaaS", 200, "software_it_approval", "high", lambda r: not r["pre_approved"],
     "Software purchase ${amount:.2f} exceeds $200 — requires IT sign-off"),
    ("Entertainment", 100, "entertainment_limit", "medium", lambda r: True,
     "${amount:.2f} exceeds $100/person entertainment limit"),
]


def fallback_policy_check(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based policy checking that needs no LLM. Used when CEREBRAS_API_KEY
    is unset, so both the app and the static precompute still surface violations."""
    results = []
    for _, row in df.iterrows():
        violations = []
        for category, limit, rule_id, severity, cond, msg in _FALLBACK_RULES:
            if row["category"] == category and row["amount"] > limit and cond(row):
                violations.append({
                    "rule_id": rule_id, "severity": severity,
                    "explanation": msg.format(amount=row["amount"]),
                })
        if not row["receipt_attached"] and row["amount"] > 25:
            violations.append({
                "rule_id": "receipt_required", "severity": "low",
                "explanation": f"Missing receipt for ${row['amount']:.2f} transaction (required over $25)",
            })
        hour = row.get("hour", 12)
        if hour >= 23 or hour <= 4:
            violations.append({
                "rule_id": "late_night", "severity": "medium",
                "explanation": f"Transaction at {int(hour):02d}:00 — auto-flagged (11PM-5AM policy)",
            })
        results.append({
            "has_violation": bool(violations),
            "violations": violations,
            "compliance_score": max(0.0, 1.0 - 0.25 * len(violations)) if violations else 1.0,
        })
    return pd.DataFrame(results, index=df.index)


def _llm_call(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


class PolicyChecker:
    def __init__(self, policy_text: str):
        self.policy_text = policy_text
        self.client = _get_client()
        self.rules = []

    def extract_rules(self) -> list[dict]:
        text = _llm_call(
            self.client,
            RULE_EXTRACTION_PROMPT.format(policy_text=self.policy_text),
        )
        self.rules = json.loads(text)
        return self.rules

    def check_transaction(self, row: pd.Series) -> dict:
        relevant_rules = [r for r in self.rules if self._rule_matches_category(r, row["category"])]
        general_rules = [r for r in self.rules if r.get("category", "").lower() in ("general", "all", "")]
        all_applicable = relevant_rules + general_rules

        if not all_applicable:
            return {"has_violation": False, "violations": [], "compliance_score": 1.0}

        quick_violations = self._quick_rule_check(row, all_applicable)
        if quick_violations:
            return {
                "has_violation": True,
                "violations": quick_violations,
                "compliance_score": max(0.0, 1.0 - 0.25 * len(quick_violations)),
            }

        return {"has_violation": False, "violations": [], "compliance_score": 1.0}

    def check_transaction_llm(self, row: pd.Series) -> dict:
        relevant_rules = [r for r in self.rules if self._rule_matches_category(r, row["category"])]
        general_rules = [r for r in self.rules if r.get("category", "").lower() in ("general", "all", "")]
        all_applicable = relevant_rules + general_rules

        if not all_applicable:
            return {"has_violation": False, "violations": [], "compliance_score": 1.0}

        prompt = VIOLATION_CHECK_PROMPT.format(
            employee_name=row["employee_name"],
            department=row["department"],
            vendor=row["vendor"],
            category=row["category"],
            amount=row["amount"],
            description=row["description"],
            date=row["date"],
            receipt_attached=row["receipt_attached"],
            pre_approved=row["pre_approved"],
            rules_json=json.dumps(all_applicable, indent=2),
        )

        text = _llm_call(self.client, prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"has_violation": False, "violations": [], "compliance_score": 1.0}

    def check_all(self, df: pd.DataFrame, use_llm_for_ambiguous: bool = True) -> pd.DataFrame:
        results = []
        ambiguous_indices = []

        for idx, row in df.iterrows():
            result = self.check_transaction(row)
            results.append(result)
            if not result["has_violation"] and self._might_have_violation(row):
                ambiguous_indices.append((len(results) - 1, idx, row))

        if use_llm_for_ambiguous and ambiguous_indices:
            for list_idx, df_idx, row in ambiguous_indices[:20]:
                try:
                    llm_result = self.check_transaction_llm(row)
                    if llm_result.get("has_violation"):
                        results[list_idx] = llm_result
                except Exception:
                    pass

        policy_df = pd.DataFrame(results, index=df.index)
        return policy_df

    def _rule_matches_category(self, rule: dict, category: str) -> bool:
        rule_cat = rule.get("category", "").lower()
        cat_lower = category.lower()
        return (
            rule_cat in cat_lower
            or cat_lower in rule_cat
            or any(w in cat_lower for w in rule_cat.split())
        )

    def _quick_rule_check(self, row: pd.Series, rules: list[dict]) -> list[dict]:
        violations = []

        for rule in rules:
            limit = rule.get("limit_amount")
            if limit is None:
                continue
            # Ensure limit is a number — LLM sometimes returns dicts or strings
            if isinstance(limit, dict):
                limit = limit.get("amount", limit.get("value", None))
            try:
                limit = float(limit)
            except (TypeError, ValueError):
                continue

            condition = rule.get("condition", "")

            if "receipt" in rule.get("rule_text", "").lower():
                if not row["receipt_attached"] and row["amount"] > limit:
                    violations.append({
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "medium"),
                        "explanation": f"Missing receipt for ${row['amount']:.2f} transaction (receipts required over ${limit})",
                    })
                continue

            if "pre-approv" in condition or "pre_approv" in condition or "preapprov" in condition:
                if row["amount"] > limit and not row["pre_approved"]:
                    violations.append({
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "high"),
                        "explanation": f"${row['amount']:.2f} exceeds ${limit} limit — requires pre-approval",
                    })
                continue

            if "solo" in rule.get("rule_text", "").lower() and "solo" in row.get("description", "").lower():
                if row["amount"] > limit:
                    violations.append({
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "medium"),
                        "explanation": f"Solo meal ${row['amount']:.2f} exceeds ${limit} limit",
                    })
                continue

            if row["amount"] > limit and self._rule_matches_category(rule, row["category"]):
                if "per_person" in condition or "per person" in condition:
                    if row["amount"] > limit * 4:
                        violations.append({
                            "rule_id": rule["id"],
                            "severity": rule.get("severity", "medium"),
                            "explanation": f"${row['amount']:.2f} likely exceeds ${limit}/person limit even for a group",
                        })
                elif "nightly" in condition:
                    if row["amount"] > limit:
                        violations.append({
                            "rule_id": rule["id"],
                            "severity": rule.get("severity", "medium"),
                            "explanation": f"${row['amount']:.2f} exceeds ${limit}/night hotel limit",
                        })
                else:
                    violations.append({
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "medium"),
                        "explanation": f"${row['amount']:.2f} exceeds ${limit} policy limit",
                    })

        return violations

    def _might_have_violation(self, row: pd.Series) -> bool:
        if row["amount"] > 200:
            return True
        if not row["receipt_attached"] and row["amount"] > 25:
            return True
        if row.get("is_weekend", False) and row["category"] == "Entertainment":
            return True
        if row.get("hour", 12) >= 23 or row.get("hour", 12) <= 4:
            return True
        return False
