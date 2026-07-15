import io

import pandas as pd
import streamlit as st

from analytics import ANALYTICS_REGISTRY, REQUIRED_COLUMNS
from predictive import (
    train_attrition_model,
    feature_importance_chart,
    risk_distribution_chart,
    top_at_risk_employees,
)
from export_utils import fig_to_jpg_bytes, build_pptx

st.set_page_config(page_title="HR Analytics Dashboard", page_icon="📊", layout="wide")

# ---------- HEADER ----------
st.title("📊 HR Analytics Dashboard")
st.caption(
    "Upload an employee dataset, choose the analytics you want to see, "
    "and download charts as JPG or a full PPTX report."
)

# ---------- SIDEBAR: DATA UPLOAD ----------
with st.sidebar:
    st.header("1. Upload your data")
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    use_sample = st.checkbox("Use sample dataset instead", value=uploaded_file is None)

    st.markdown("---")
    st.markdown(
        "**Expected columns:**\n\n"
        + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
    )
    st.markdown("---")
    st.caption("Built by Abiyad · Master of Business Analytics, Macquarie University")

# ---------- LOAD DATA ----------
@st.cache_data
def load_sample():
    return pd.read_excel("sample_hr_data.xlsx")


df = None
if uploaded_file is not None and not use_sample:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
elif use_sample:
    df = load_sample()

if df is None:
    st.info("👈 Upload an Excel file or check 'Use sample dataset' in the sidebar to get started.")
    st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.error(
        f"Your file is missing required columns: {', '.join(missing)}. "
        f"Check the expected columns list in the sidebar, or use the sample dataset."
    )
    st.stop()

# ---------- KPI ROW ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Employees", len(df))
k2.metric("Attrition Rate", f"{(df['Attrition']=='Yes').mean():.1%}")
k3.metric("Avg Salary", f"${df['Salary'].mean():,.0f}")
k4.metric("Avg Engagement", f"{df['EngagementScore'].mean():.0f}/100")

st.markdown("---")

# ---------- SELECTION ----------
st.subheader("2. Choose your analytics")
tab1, tab2 = st.tabs(["📈 Descriptive Analytics", "🔮 Predictive: Attrition Risk"])

selected_outputs = []  # list of dicts: title, fig, insight — used for export

with tab1:
    chosen = st.multiselect(
        "Select one or more visuals to generate",
        options=list(ANALYTICS_REGISTRY.keys()),
        default=["Overall Attrition Rate", "Headcount by Department", "Attrition by Department"],
    )

    if chosen:
        cols = st.columns(2)
        for i, name in enumerate(chosen):
            fig, insight = ANALYTICS_REGISTRY[name](df)
            with cols[i % 2]:
                st.pyplot(fig, use_container_width=True)
                st.caption(f"💡 {insight}")
                st.download_button(
                    f"Download '{name}' as JPG",
                    data=fig_to_jpg_bytes(fig),
                    file_name=f"{name.replace(' ', '_')}.jpg",
                    mime="image/jpeg",
                    key=f"jpg_{name}",
                )
            selected_outputs.append({"title": name, "fig": fig, "insight": insight})
    else:
        st.warning("Select at least one visual above.")

with tab2:
    st.write(
        "Trains a Random Forest model on your data to estimate each employee's "
        "probability of leaving, and surfaces the strongest drivers of attrition."
    )
    run_model = st.button("Run attrition prediction model", type="primary")

    if run_model:
        with st.spinner("Training model..."):
            result = train_attrition_model(df)
        st.session_state["model_result"] = result

    if "model_result" in st.session_state:
        result = st.session_state["model_result"]
        if result["auc"] is not None:
            st.metric("Model AUC (test set)", f"{result['auc']:.2f}")

        c1, c2 = st.columns(2)
        with c1:
            fig_imp, ins_imp = feature_importance_chart(result["importances"])
            st.pyplot(fig_imp, use_container_width=True)
            st.caption(f"💡 {ins_imp}")
            st.download_button(
                "Download 'Feature Importance' as JPG",
                data=fig_to_jpg_bytes(fig_imp),
                file_name="attrition_feature_importance.jpg",
                mime="image/jpeg",
                key="jpg_importance",
            )
            selected_outputs.append({"title": "Top Drivers of Attrition Risk", "fig": fig_imp, "insight": ins_imp})

        with c2:
            fig_risk, ins_risk = risk_distribution_chart(result["data_with_risk"])
            st.pyplot(fig_risk, use_container_width=True)
            st.caption(f"💡 {ins_risk}")
            st.download_button(
                "Download 'Risk Distribution' as JPG",
                data=fig_to_jpg_bytes(fig_risk),
                file_name="attrition_risk_distribution.jpg",
                mime="image/jpeg",
                key="jpg_risk",
            )
            selected_outputs.append({"title": "Distribution of Predicted Attrition Risk", "fig": fig_risk, "insight": ins_risk})

        st.markdown("**Top 10 employees at highest attrition risk**")
        st.dataframe(top_at_risk_employees(result["data_with_risk"]), use_container_width=True)
    else:
        st.info("Click the button above to train the model and see attrition risk predictions.")

# ---------- EXPORT ALL ----------
st.markdown("---")
st.subheader("3. Export your report")

if selected_outputs:
    pptx_bytes = build_pptx(selected_outputs, company_name="HR Analytics Report")
    st.download_button(
        "📥 Download full report as PPTX",
        data=pptx_bytes,
        file_name="hr_analytics_report.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        type="primary",
    )
    st.caption(f"Report will include {len(selected_outputs)} slide(s) based on your current selections above.")
else:
    st.caption("Select some visuals above to enable the PPTX export.")
