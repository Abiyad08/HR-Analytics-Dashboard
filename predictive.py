"""
Predictive layer: trains a Random Forest to estimate attrition risk per
employee and exposes feature importances + per-employee risk scores.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PALETTE = ["#2563EB", "#7C3AED", "#DB2777", "#EA580C", "#16A34A", "#0891B2", "#CA8A04"]

FEATURE_COLUMNS = [
    "Department", "JobLevel", "Age", "Gender", "Location", "Education",
    "TenureYears", "Salary", "PerformanceRating", "EngagementScore",
    "OvertimeHoursMonth", "DistanceFromOfficeKm", "YearsSinceLastPromotion",
    "TrainingHoursYear",
]
CATEGORICAL = ["Department", "JobLevel", "Gender", "Location", "Education"]


def train_attrition_model(df: pd.DataFrame):
    """Trains a Random Forest classifier and returns the model, encoded
    feature frame, predicted risk scores, AUC, and feature importances."""
    data = df.copy()
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        data[col + "_enc"] = le.fit_transform(data[col])
        encoders[col] = le

    feature_cols_enc = [c + "_enc" if c in CATEGORICAL else c for c in FEATURE_COLUMNS]
    X = data[feature_cols_enc]
    y = (data["Attrition"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.sum() > 5 else None
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        auc = None

    risk_scores = model.predict_proba(X)[:, 1]
    data["AttritionRisk"] = risk_scores

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)

    return {
        "model": model,
        "data_with_risk": data,
        "auc": auc,
        "importances": importances,
    }


def feature_importance_chart(importances: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = importances.head(10).sort_values()
    ax.barh(top.index, top.values, color=PALETTE[0])
    ax.set_xlabel("Relative Importance")
    fig.suptitle("Top Drivers of Attrition Risk", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    top_driver = importances.index[0]
    insight = f"'{top_driver}' is the strongest predictor of attrition in this dataset."
    return fig, insight


def risk_distribution_chart(data_with_risk: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(data_with_risk["AttritionRisk"], bins=20, color=PALETTE[2], edgecolor="white")
    ax.axvline(0.5, color="#111827", linestyle="--", alpha=0.5, label="50% risk threshold")
    ax.set_xlabel("Predicted Attrition Risk")
    ax.set_ylabel("Number of Employees")
    ax.legend()
    fig.suptitle("Distribution of Predicted Attrition Risk", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    high_risk = (data_with_risk["AttritionRisk"] > 0.5).sum()
    insight = f"{high_risk} employees ({high_risk/len(data_with_risk):.0%}) are flagged as high risk (>50% predicted probability)."
    return fig, insight


def top_at_risk_employees(data_with_risk: pd.DataFrame, n=10):
    cols = ["EmployeeID", "Department", "JobLevel", "TenureYears", "EngagementScore", "AttritionRisk"]
    cols = [c for c in cols if c in data_with_risk.columns]
    top = data_with_risk.sort_values("AttritionRisk", ascending=False)[cols].head(n).copy()
    top["AttritionRisk"] = (top["AttritionRisk"] * 100).round(1).astype(str) + "%"
    return top.reset_index(drop=True)
