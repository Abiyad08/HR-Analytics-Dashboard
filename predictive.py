"""
predictive.py — Random Forest attrition risk model.
Works with any columns detected by data_prep.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

COLORS = ["#1D4ED8","#6D28D9","#BE185D","#15803D","#C2410C","#0E7490","#B45309","#B91C1C"]

NUM_CONCEPTS  = ["_tenure_years","_age","_yrs_since_promo","_probation_days",
                 "primary_leave_taken","secondary_leave_taken","_on_probation"]
CAT_CONCEPTS  = ["division","business_unit","dept_group","department","designation_main",
                 "employment_type","gender","education","work_location_type","grade"]

def _L(title="", h=420):
    return dict(
        title=dict(text=f"<b>{title}</b>",font=dict(size=15,color="#0f172a",family="Inter"),x=0),
        font=dict(family="Inter,system-ui,sans-serif",size=12,color="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#f8fafc",
        height=h,margin=dict(l=10,r=10,t=56,b=10),colorway=COLORS,
        showlegend=True,
        legend=dict(bgcolor="rgba(255,255,255,0.95)",bordercolor="#cbd5e1",borderwidth=1,
                    font=dict(size=12,color="#1e293b"),orientation="h",
                    yanchor="bottom",y=1.02,xanchor="right",x=1),
        hoverlabel=dict(bgcolor="white",bordercolor="#e2e8f0",font_family="Inter",font_size=12),
        xaxis=dict(tickfont=dict(size=12,color="#334155"),title_font=dict(size=13,color="#1e293b")),
        yaxis=dict(tickfont=dict(size=12,color="#334155"),title_font=dict(size=13,color="#1e293b")),
    )

def _g(col_map, *concepts):
    for c in concepts:
        v = col_map.get(c)
        if v: return v
    return None

def train_model(df: pd.DataFrame, col_map: dict):
    att_col = col_map.get("_attrition")
    if not att_col or att_col not in df.columns: return None
    y = pd.to_numeric(df[att_col], errors="coerce").fillna(0).astype(int)
    if y.sum() < 10: return None

    X_parts, feat_names = [], []

    for concept in NUM_CONCEPTS:
        col = _g(col_map, concept)
        if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            filled = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if df[col].notna().any() else 0)
            X_parts.append(filled.values.reshape(-1, 1))
            feat_names.append(col)

    for concept in CAT_CONCEPTS:
        col = _g(col_map, concept)
        if col and col in df.columns:
            le = LabelEncoder()
            enc = le.fit_transform(df[col].fillna("Unknown").astype(str))
            X_parts.append(enc.reshape(-1, 1))
            feat_names.append(col)

    if not X_parts: return None
    X = np.hstack(X_parts)

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42,
            stratify=y if y.sum() > 10 else None
        )
    except Exception:
        X_tr, X_te, y_tr, y_te = X, X, y, y

    model = RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    try:    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    except: auc = None

    risk = model.predict_proba(X)[:, 1]
    importances = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)

    return {"risk": risk, "auc": auc, "importances": importances,
            "n_features": len(feat_names), "n_train": len(X_tr)}


def feature_importance_chart(importances: pd.Series):
    top = importances.head(15).sort_values()
    colors = [COLORS[0] if v > top.median() else "#93c5fd" for v in top.values]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.3f}" for v in top.values], textposition="outside",
    ))
    fig.update_layout(**_L("Top Attrition Risk Drivers"), xaxis_title="Feature Importance")
    return fig, f"'{importances.index[0]}' is the strongest predictor of attrition in this dataset."


def risk_distribution_chart(risk: np.ndarray):
    fig = px.histogram(x=risk, nbins=40, color_discrete_sequence=["#DB2777"], marginal="box")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#475569",
                  annotation_text="50% threshold", annotation_position="top right")
    fig.update_traces(marker_line_width=0)
    fig.update_layout(**_L("Distribution of Predicted Attrition Risk"),
                      xaxis_title="Risk Score (0–1)", yaxis_title="Employees")
    high = (risk > 0.5).mean()
    return fig, f"{high:.1%} of employees score above 50% predicted attrition risk."


def risk_by_group_chart(df: pd.DataFrame, col_map: dict, risk: np.ndarray, group_col: str):
    if not group_col or group_col not in df.columns: return None, ""
    tmp = df.copy(); tmp["__risk"] = risk
    avg = tmp.groupby(group_col)["__risk"].mean().sort_values(ascending=False)
    n   = tmp.groupby(group_col).size()
    colors = ["#DC2626" if v > 0.4 else "#F97316" if v > 0.25 else "#16A34A"
              for v in avg.values]
    fig = go.Figure(go.Bar(
        x=avg.values * 100, y=avg.index, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.1f}%  (n={n.get(g,0):,})" for v, g in zip(avg.values * 100, avg.index)],
        textposition="outside",
    ))
    fig.update_layout(**_L(f"Average Predicted Attrition Risk by {group_col}"),
                      xaxis_title="Avg Risk Score (%)")
    return fig, f"{avg.idxmax()} has the highest predicted risk ({avg.max():.1%} average)."


def flight_risk_heatmap(df: pd.DataFrame, col_map: dict, risk: np.ndarray, hier: list):
    if len(hier) < 2: return None, ""
    col_a, col_b = hier[0][1], hier[1][1]
    if col_a not in df.columns or col_b not in df.columns: return None, ""
    tmp = df.copy(); tmp["__risk"] = risk
    top_a = tmp[col_a].value_counts().head(8).index
    top_b = tmp[col_b].value_counts().head(10).index
    sub = tmp[tmp[col_a].isin(top_a) & tmp[col_b].isin(top_b)]
    pivot = sub.groupby([col_a, col_b])["__risk"].mean().unstack(fill_value=0) * 100
    fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn_r",
                    zmin=0, zmax=60, aspect="auto")
    fig.update_layout(**_L(f"Flight Risk Heatmap — {col_a} × {col_b}", h=450))
    return fig, "Darker red = higher predicted attrition risk. Prioritise retention actions in these cells."


def top_at_risk_table(df: pd.DataFrame, col_map: dict, risk: np.ndarray, n=20):
    tmp = df.copy()
    tmp["⚠ Attrition Risk"] = (risk * 100).round(1).astype(str) + "%"
    show = ["⚠ Attrition Risk"]
    for concept in ["employee_id","employee_name","division","business_unit","department",
                    "designation_main","designation","_tenure_years","_age"]:
        col = col_map.get(concept)
        if col and col in tmp.columns: show.append(col)
    show = list(dict.fromkeys(show))
    top = tmp.nlargest(n, "__risk" if "__risk" in tmp.columns else "⚠ Attrition Risk")[show]
    top.index = range(1, len(top) + 1)

    # Rename tenure/age cols for display
    renames = {}
    if col_map.get("_tenure_years") in top.columns:
        renames[col_map["_tenure_years"]] = "Tenure (yrs)"
    if col_map.get("_age") in top.columns:
        renames[col_map["_age"]] = "Age"
    return top.rename(columns=renames).round(1)
