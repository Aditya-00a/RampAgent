"""Ensemble anomaly detection for corporate expense transactions."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import ANOMALY_WEIGHTS


class AnomalyDetector:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = pd.DataFrame(index=df.index)
        self.results["anomaly_flags"] = [[] for _ in range(len(df))]
        self.results["anomaly_score"] = 0.0

    def run_all(self) -> pd.DataFrame:
        self._isolation_forest()
        self._zscore_by_category()
        self._duplicate_detection()
        self._temporal_patterns()
        self._velocity_checks()
        self._first_time_vendor()
        self._missing_receipts()
        self._compute_ensemble_score()
        return self.results

    def _isolation_forest(self):
        features = self.df[["amount", "hour", "day_of_week", "category_encoded"]].copy()
        features = features.fillna(0)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        iso = IsolationForest(contamination=0.12, random_state=42, n_estimators=200)
        preds = iso.fit_predict(scaled)
        scores = iso.decision_function(scaled)

        min_s, max_s = scores.min(), scores.max()
        normalized = 1 - (scores - min_s) / (max_s - min_s + 1e-10)

        self.results["iso_forest_score"] = normalized
        for i in range(len(self.df)):
            if preds[i] == -1:
                self.results.at[i, "anomaly_flags"] = self.results.at[i, "anomaly_flags"] + ["isolation_forest"]

    def _zscore_by_category(self):
        scores = np.zeros(len(self.df))
        for cat in self.df["category"].unique():
            mask = self.df["category"] == cat
            if mask.sum() < 3:
                continue
            amounts = self.df.loc[mask, "amount"]
            mean, std = amounts.mean(), amounts.std()
            if std < 1e-10:
                continue
            z = np.abs((amounts - mean) / std)
            scores[mask] = np.clip(z / 5.0, 0, 1)
            for idx in amounts[z > 2.5].index:
                self.results.at[idx, "anomaly_flags"] = self.results.at[idx, "anomaly_flags"] + [
                    f"zscore_outlier (${self.df.at[idx, 'amount']:.0f} vs avg ${mean:.0f})"
                ]

        self.results["zscore_score"] = scores

    def _duplicate_detection(self):
        scores = np.zeros(len(self.df))
        df = self.df

        for i in range(len(df)):
            same_vendor = df["vendor"] == df.iloc[i]["vendor"]
            same_employee = df["employee_name"] == df.iloc[i]["employee_name"]
            amount_close = np.abs(df["amount"] - df.iloc[i]["amount"]) / (df.iloc[i]["amount"] + 1e-10) < 0.05
            time_close = np.abs((df["date"] - df.iloc[i]["date"]).dt.total_seconds()) < 48 * 3600
            not_self = df.index != i

            dupes = same_vendor & same_employee & amount_close & time_close & not_self
            if dupes.any():
                scores[i] = 1.0
                self.results.at[i, "anomaly_flags"] = self.results.at[i, "anomaly_flags"] + [
                    f"potential_duplicate (similar to {dupes.sum()} other txn)"
                ]

        self.results["duplicate_score"] = scores

    def _temporal_patterns(self):
        scores = np.zeros(len(self.df))

        late_night = (self.df["hour"] >= 23) | (self.df["hour"] <= 4)
        weekend = self.df["is_weekend"]

        for i in range(len(self.df)):
            flags = []
            s = 0.0
            if late_night.iloc[i]:
                s += 0.6
                flags.append(f"late_night ({self.df.iloc[i]['hour']:02d}:00)")
            if weekend.iloc[i]:
                s += 0.5
                flags.append("weekend_transaction")
            scores[i] = min(s, 1.0)
            if flags:
                self.results.at[i, "anomaly_flags"] = self.results.at[i, "anomaly_flags"] + flags

        self.results["temporal_score"] = scores

    def _velocity_checks(self):
        scores = np.zeros(len(self.df))
        df = self.df.sort_values("date")

        for emp in df["employee_name"].unique():
            emp_mask = df["employee_name"] == emp
            emp_df = df[emp_mask]
            if len(emp_df) < 5:
                continue

            weekly_spend = emp_df.groupby("week_number")["amount"].sum()
            if len(weekly_spend) < 3:
                continue

            mean_weekly = weekly_spend.mean()
            std_weekly = weekly_spend.std()
            if std_weekly < 1e-10:
                continue

            for week, total in weekly_spend.items():
                if total > mean_weekly + 2 * std_weekly:
                    week_mask = emp_mask & (df["week_number"] == week)
                    ratio = total / mean_weekly
                    for idx in df[week_mask].index:
                        scores[idx] = min((ratio - 1) / 3, 1.0)
                        self.results.at[idx, "anomaly_flags"] = self.results.at[idx, "anomaly_flags"] + [
                            f"velocity_spike ({ratio:.1f}x weekly avg)"
                        ]

        self.results["velocity_score"] = scores

    def _first_time_vendor(self):
        scores = np.zeros(len(self.df))
        df = self.df.sort_values("date")

        for emp in df["employee_name"].unique():
            emp_df = df[df["employee_name"] == emp].sort_values("date")
            seen_vendors = set()
            for idx, row in emp_df.iterrows():
                if row["vendor"] not in seen_vendors and len(seen_vendors) > 3:
                    all_amounts = df[df["vendor"] == row["vendor"]]["amount"]
                    if len(all_amounts) <= 2 and row["amount"] > 100:
                        scores[idx] = 0.6
                        self.results.at[idx, "anomaly_flags"] = self.results.at[idx, "anomaly_flags"] + [
                            f"first_time_vendor ({row['vendor']})"
                        ]
                seen_vendors.add(row["vendor"])

        self.results["first_vendor_score"] = scores

    def _missing_receipts(self):
        for i in range(len(self.df)):
            if not self.df.iloc[i]["receipt_attached"] and self.df.iloc[i]["amount"] > 25:
                self.results.at[i, "anomaly_flags"] = self.results.at[i, "anomaly_flags"] + [
                    "missing_receipt"
                ]

    def _compute_ensemble_score(self):
        w = ANOMALY_WEIGHTS
        self.results["anomaly_score"] = (
            self.results.get("iso_forest_score", 0) * w["isolation_forest"]
            + self.results.get("zscore_score", 0) * w["zscore"]
            + self.results.get("duplicate_score", 0) * w["duplicate"]
            + self.results.get("temporal_score", 0) * w["temporal"]
            + self.results.get("velocity_score", 0) * w["velocity"]
            + self.results.get("first_vendor_score", 0) * w["first_time_vendor"]
        )
        self.results["anomaly_score"] = self.results["anomaly_score"].clip(0, 1)
