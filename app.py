"""
Conglomerate HR Analytics Dashboard
Multi-division, multi-BU, drill-down analytics with cascading filters.
"""

import io
import pandas as pd
import streamlit as st

from data_prep import prepare
from analytics import get_available
from predictive import (train_model, feature_importance_chart,
                        risk_distribution_chart, risk_by_group_chart,
                        flight_risk_heatmap, top_at_risk_table)
from export_utils import fig_to_jpg, build_pptx
from sample_data import generate_sample_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Intelligence Platform",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"],.stMarkdown{font-family:'Inter',system-ui,sans-serif!important}
#MainMenu,footer,header{visibility:hidden}

/* App bg */
.stApp{background:#eef2f7}
.main .block-container{padding:1.2rem 1.8rem 2rem;max-width:1600px}

/* Sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(175deg,#0b1829 0%,#112240 40%,#0d1b2e 100%)!important;
  border-right:1px solid rgba(255,255,255,0.05);
}
[data-testid="stSidebar"] *{color:#94a3b8!important}
[data-testid="stSidebar"] strong,[data-testid="stSidebar"] b{color:#e2e8f0!important}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,0.08)!important}
[data-testid="stSidebar"] .stSelectbox>div>div,
[data-testid="stSidebar"] .stMultiSelect>div>div{
  background:rgba(255,255,255,0.06)!important;
  border-color:rgba(255,255,255,0.12)!important;
  border-radius:8px!important;color:#e2e8f0!important
}
[data-testid="stSidebar"] .stRadio label{color:#94a3b8!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{
  background:rgba(255,255,255,0.05);border-radius:10px;padding:6px
}
[data-testid="stSidebar"] .stTextInput input{
  background:rgba(255,255,255,0.07)!important;
  border-color:rgba(255,255,255,0.15)!important;
  color:#e2e8f0!important;border-radius:8px
}

/* Page header */
.dash-header{
  background:linear-gradient(130deg,#0b1829 0%,#1e3a8a 45%,#4f46e5 80%,#7c3aed 100%);
  border-radius:18px;padding:26px 34px;margin-bottom:20px;
  box-shadow:0 8px 32px rgba(30,58,138,0.30);position:relative;overflow:hidden;
}
.dash-header::after{
  content:"";position:absolute;bottom:-40px;right:-40px;
  width:180px;height:180px;background:rgba(255,255,255,0.04);border-radius:50%;
}
.dash-header h1{color:#fff!important;font-size:1.75rem;font-weight:800;
  margin:0 0 4px;letter-spacing:-.025em}
.dash-header p{color:rgba(255,255,255,0.72)!important;font-size:0.88rem;margin:0}

/* Breadcrumb */
.breadcrumb{
  background:white;border-radius:10px;padding:10px 18px;margin-bottom:14px;
  font-size:0.82rem;color:#475569;border:1px solid #e2e8f0;
  box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.breadcrumb span{color:#2563eb;font-weight:600}

/* KPI cards */
div[data-testid="metric-container"]{
  background:white;border-radius:14px;padding:18px 22px!important;
  box-shadow:0 1px 6px rgba(0,0,0,0.05),0 4px 14px rgba(0,0,0,0.04);
  border:1px solid #e4e8ed;transition:transform .15s,box-shadow .15s;
}
div[data-testid="metric-container"]:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,0.09)}
div[data-testid="metric-container"] label{
  color:#64748b!important;font-size:0.7rem!important;font-weight:700!important;
  text-transform:uppercase!important;letter-spacing:.08em!important
}
div[data-testid="metric-container"] [data-testid="metric-value"]{
  color:#0f172a!important;font-size:1.75rem!important;font-weight:800!important;line-height:1.1!important
}
div[data-testid="metric-container"] [data-testid="metric-delta"]{font-size:0.78rem!important;font-weight:600!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  background:white;border-radius:14px;padding:5px;gap:3px;
  box-shadow:0 1px 4px rgba(0,0,0,0.05);border:1px solid #e4e8ed;flex-wrap:wrap;
}
.stTabs [data-baseweb="tab"]{
  border-radius:9px!important;padding:8px 16px!important;
  font-weight:500!important;font-size:0.82rem!important;
  color:#64748b!important;border:none!important;transition:all .15s;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#1e3a8a,#4f46e5)!important;
  color:white!important;box-shadow:0 3px 10px rgba(30,58,138,0.30)!important;
  font-weight:700!important;
}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#1e3a8a,#4f46e5)!important;
  color:white!important;border:none!important;border-radius:10px!important;
  padding:10px 22px!important;font-weight:600!important;font-size:0.85rem!important;
  transition:all .18s!important;box-shadow:0 3px 10px rgba(30,58,138,0.25)!important;
}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 5px 18px rgba(30,58,138,0.35)!important}
[data-testid="stDownloadButton"]>button{
  background:white!important;color:#1e3a8a!important;
  border:1.5px solid #1e3a8a!important;border-radius:8px!important;
  font-weight:600!important;font-size:0.78rem!important;padding:5px 13px!important;
}
[data-testid="stDownloadButton"]>button:hover{background:#eff6ff!important}

/* Chart card */
.chart-card{
  background:white;border-radius:16px;padding:18px 18px 12px;
  box-shadow:0 1px 4px rgba(0,0,0,0.04),0 4px 14px rgba(0,0,0,0.04);
  border:1px solid #e4e8ed;margin-bottom:6px;
}
/* Insight pill */
.insight{
  background:linear-gradient(135deg,#eff6ff,#eef2ff);
  border-left:3px solid #2563eb;border-radius:0 8px 8px 0;
  padding:8px 14px;margin-top:4px;font-size:0.8rem;
  color:#334155;font-style:italic;line-height:1.55;
}
/* Section label */
.sec-label{
  font-size:0.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:#94a3b8;margin:18px 0 8px;
}
/* Filter summary badge */
.filter-badge{
  display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.72rem;
  font-weight:600;background:#dbeafe;color:#1d4ed8;margin:2px 3px;
}
/* Expander */
[data-testid="stExpander"]{background:white!important;border-radius:12px!important;border:1px solid #e4e8ed!important}
[data-testid="stExpander"] summary{font-weight:600!important;color:#0f172a!important}
/* Multiselect tags */
[data-baseweb="tag"]{background:#dbeafe!important;border-color:#93c5fd!important;color:#1d4ed8!important;border-radius:6px!important}
/* Alerts */
.stAlert{border-radius:12px!important}
/* Progress */
.stProgress>div>div{background:linear-gradient(90deg,#1e3a8a,#4f46e5)!important;border-radius:4px!important}
/* Dataframe */
.stDataFrame{border-radius:12px!important;overflow:hidden!important}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _gs_url(url):
    if "docs.google.com/spreadsheets" in url and "/export" not in url:
        try:
            sid = url.split("/d/")[1].split("/")[0]
            gid = url.split("gid=")[1].split("&")[0].split("#")[0] if "gid=" in url else "0"
            return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"
        except Exception: return url
    return url

@st.cache_data(show_spinner=False)
def _load_url(url):
    r = _gs_url(url.strip())
    return pd.read_csv(r) if (r.lower().endswith(".csv") or "format=csv" in r) else pd.read_excel(r)

@st.cache_data(show_spinner=False)
def _load_file(data: bytes, name: str):
    buf = io.BytesIO(data)
    return pd.read_csv(buf) if name.endswith(".csv") else pd.read_excel(buf)

@st.cache_data(show_spinner=False)
def _sample(): return generate_sample_data(5000)

@st.cache_data(show_spinner=False)
def _prepare(df_hash, df):
    return prepare(df)


def _render(fig, insight, title, cat, idx, selected_outputs):
    """Render a chart card with download + collect for PPTX."""
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d"]})
    if insight:
        st.markdown(f"<div class='insight'>💡 {insight}</div>", unsafe_allow_html=True)
    jpg = fig_to_jpg(fig)
    if jpg:
        st.download_button("⬇ JPG", data=jpg,
                           file_name=f"{title.replace(' ','_')[:40]}.jpg",
                           mime="image/jpeg", key=f"dl_{cat}_{idx}")
    st.markdown("</div>", unsafe_allow_html=True)
    selected_outputs.append({"title": title, "fig": fig, "insight": insight or ""})


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Data source
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 2px 10px'>
      <div style='font-size:1.25rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em'>
        📊 HR Intelligence
      </div>
      <div style='font-size:0.72rem;color:#475569;margin-top:2px'>
        Conglomerate People Analytics
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='sec-label'>Data Source</div>", unsafe_allow_html=True)
    source = st.radio("", ["📁 Upload File","🌐 Load from URL","🎲 Sample Data"],
                      index=2, label_visibility="collapsed")
    uploaded_file = data_url = None
    if source == "📁 Upload File":
        uploaded_file = st.file_uploader("CSV or Excel (.csv / .xlsx / .xls)",
                                         type=["csv","xlsx","xls"], label_visibility="collapsed")
        st.caption("Accepts any HR export — columns auto-detected.")
    elif source == "🌐 Load from URL":
        data_url = st.text_input("", placeholder="https://… or Google Sheets link",
                                 label_visibility="collapsed")
        st.caption("Works with direct .csv/.xlsx links and shared Google Sheets.")

    st.divider()
    # Hierarchy filters rendered after data loads — placeholder
    filter_placeholder = st.container()
    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem;color:#334155;line-height:1.9'>
      <b style='color:#94a3b8'>45+ charts</b> across 7 analytics categories<br>
      <b style='color:#94a3b8'>Cascading filters</b> — drill from Group to Dept<br>
      <b style='color:#94a3b8'>Export</b> any chart as JPG or full PPTX deck
    </div>
    <div style='margin-top:14px;font-size:0.7rem;color:#475569'>
      Built by <b style='color:#64748b'>Abiyad</b><br>
      MBA Analytics · Macquarie University
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
raw_df = None
if source == "📁 Upload File" and uploaded_file:
    with st.spinner("Reading file…"):
        try:
            raw_df = _load_file(uploaded_file.read(), uploaded_file.name)
        except Exception as e:
            st.error(f"❌ {e}")
elif source == "🌐 Load from URL" and data_url:
    with st.spinner("Fetching data…"):
        try:
            raw_df = _load_url(data_url)
        except Exception as e:
            st.error(f"❌ {e}")
elif source == "🎲 Sample Data":
    with st.spinner("Generating conglomerate sample dataset…"):
        raw_df = _sample()

# ── Page header (always visible) ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>HR Intelligence Platform</h1>
  <p>
    Conglomerate-grade people analytics &nbsp;·&nbsp;
    Multi-division hierarchy filters &nbsp;·&nbsp;
    45+ interactive charts &nbsp;·&nbsp;
    JPG & PPTX export
  </p>
</div>""", unsafe_allow_html=True)

if raw_df is None:
    st.markdown("""
    <div style='background:white;border-radius:18px;padding:60px 40px;text-align:center;
                border:2px dashed #e2e8f0;margin-top:8px'>
      <div style='font-size:3rem;margin-bottom:18px'>🏢</div>
      <div style='font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:8px'>
        No data loaded
      </div>
      <div style='color:#64748b;font-size:0.9rem;max-width:500px;margin:0 auto'>
        Use the sidebar to upload a file, paste a URL, or load the built-in
        conglomerate sample dataset (5,000 employees across 5 divisions).
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Analysing data…"):
    df_full, meta = prepare(raw_df)

col_map   = meta["col_map"]
hierarchy = meta["hierarchy"]   # [(label, col), ...]
kpi_full  = meta["kpi"]
is_cong   = meta["is_conglomerate"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Cascading hierarchy filters
# ══════════════════════════════════════════════════════════════════════════════
df_filtered = df_full.copy()
active_filters = {}

with filter_placeholder:
    # Status filter
    st.markdown("<div class='sec-label'>Workforce Filter</div>", unsafe_allow_html=True)
    status_opt = st.radio("Show employees",
                          ["Active Only","All (Active + Former)","Former Only"],
                          index=1, label_visibility="collapsed")
    is_active_col = col_map.get("_is_active","__is_active")
    if status_opt == "Active Only" and is_active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[is_active_col] == 1]
    elif status_opt == "Former Only" and is_active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[is_active_col] == 0]

    if hierarchy:
        st.markdown("<div class='sec-label'>Business Hierarchy</div>", unsafe_allow_html=True)
        for level_label, level_col in hierarchy:
            if level_col not in df_filtered.columns:
                continue
            opts = sorted(df_filtered[level_col].dropna().unique().tolist())
            if not opts:
                continue
            chosen = st.multiselect(level_label, opts, default=[],
                                    placeholder=f"All {level_label}s",
                                    key=f"filt_{level_col}")
            if chosen:
                df_filtered = df_filtered[df_filtered[level_col].isin(chosen)]
                active_filters[level_label] = chosen

    st.markdown(f"<div style='font-size:0.75rem;color:#64748b;padding:6px 0'>"
                f"📋 <b style='color:#94a3b8'>{len(df_filtered):,}</b> records selected"
                f"</div>", unsafe_allow_html=True)


# ── Determine grouping column (level one step down from deepest filter) ──────
def _group_col():
    if not hierarchy: return None
    for i, (lbl, col) in enumerate(hierarchy):
        if lbl not in active_filters:
            return col
    return hierarchy[-1][1]

group_col = _group_col()


# ══════════════════════════════════════════════════════════════════════════════
# BREADCRUMB
# ══════════════════════════════════════════════════════════════════════════════
if active_filters or status_opt != "All (Active + Former)":
    parts = []
    if status_opt != "All (Active + Former)":
        parts.append(f"<span>{status_opt}</span>")
    for lbl, vals in active_filters.items():
        for v in vals:
            parts.append(f"<span class='filter-badge'>{lbl}: {v}</span>")
    st.markdown(f"<div class='breadcrumb'>🔎 Viewing: {' &nbsp;›&nbsp; '.join(parts)} "
                f"&nbsp;|&nbsp; <b>{len(df_filtered):,} employees</b></div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
att_col = col_map.get("_attrition")
ten_col = col_map.get("_tenure_years")
gen_col = col_map.get("gender")
prob_col = col_map.get("_on_probation")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("👥 Total Records", f"{len(df_filtered):,}")
if is_active_col in df_filtered.columns:
    act = int((df_filtered[is_active_col]==1).sum())
    k2.metric("✅ Active", f"{act:,}", f"{act/max(len(df_filtered),1):.0%}")
if att_col and att_col in df_filtered.columns:
    att_r = df_filtered[att_col].mean()
    k3.metric("🔄 Attrition", f"{att_r:.1%}",
              f"{'↑' if att_r>0.20 else '↓'} vs 20% ref")
if ten_col and ten_col in df_filtered.columns:
    k4.metric("📅 Avg Tenure", f"{df_filtered[ten_col].mean():.1f} yrs")
if gen_col and gen_col in df_filtered.columns:
    f_pct = (df_filtered[gen_col]=="Female").mean()
    k5.metric("👩 Female %", f"{f_pct:.1%}")
if prob_col and prob_col in df_filtered.columns and is_active_col in df_filtered.columns:
    active_only = df_filtered[df_filtered[is_active_col]==1]
    prob_pct = active_only[prob_col].mean() if len(active_only) else 0
    k6.metric("⏳ On Probation", f"{prob_pct:.0%}", "of active")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TABS
# ══════════════════════════════════════════════════════════════════════════════
available = get_available(df_filtered, col_map)

if not available:
    st.warning("⚠️ No charts available for this data slice. Try broadening your filters.")
    st.stop()

tab_labels  = list(available.keys()) + ["🔮 Predictive AI"]
tabs        = st.tabs(tab_labels)
selected_outputs = []

PLOTLY_CONFIG = {"displayModeBar": True, "displaylogo": False,
                 "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d"]}

# ── Descriptive tabs ──────────────────────────────────────────────────────────
for tab, (cat_name, charts) in zip(tabs[:-1], available.items()):
    with tab:
        chart_names = [c["name"] for c in charts]
        default     = chart_names[:3]

        chosen = st.multiselect(
            "Select charts to display:",
            options=chart_names, default=default,
            key=f"ms_{cat_name}", placeholder="Choose charts…"
        )

        if not chosen:
            st.markdown("""
            <div style='text-align:center;padding:48px 24px;background:white;
                        border-radius:16px;border:2px dashed #e2e8f0'>
              <div style='font-size:2.5rem;margin-bottom:12px'>📊</div>
              <div style='color:#64748b;font-size:0.9rem'>
                Select charts above to get started
              </div>
            </div>""", unsafe_allow_html=True)
            continue

        fn_lookup = {c["name"]: c for c in charts}
        col1, col2 = st.columns(2, gap="medium")

        for idx, name in enumerate(chosen):
            chart_def = fn_lookup[name]
            fn = chart_def["fn"]
            target = col1 if idx % 2 == 0 else col2

            with target:
                try:
                    with st.spinner(f"Rendering…"):
                        if chart_def.get("needs_hierarchy"):
                            fig, insight = fn(df_filtered, col_map, hierarchy)
                        elif chart_def.get("no_group"):
                            fig, insight = fn(df_filtered, col_map)
                        else:
                            fig, insight = fn(df_filtered, col_map, group_col)

                    if fig is None:
                        st.info(f"Not enough data for '{name}' with current filters.")
                        continue
                    _render(fig, insight, name, cat_name, idx, selected_outputs)

                except Exception as e:
                    st.error(f"Chart error in '{name}': {e}")

# ── Predictive AI tab ─────────────────────────────────────────────────────────
with tabs[-1]:
    att_check = col_map.get("_attrition")
    st.markdown("""
    <div style='background:white;border-radius:16px;padding:22px 26px;
                border:1px solid #e4e8ed;margin-bottom:18px'>
      <div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:8px'>
        🔮 Predictive Attrition Risk Model
      </div>
      <div style='font-size:0.86rem;color:#475569;line-height:1.75'>
        Trains a <b>Random Forest classifier</b> on the currently filtered
        dataset to predict each active employee's probability of leaving.<br>
        Respects your hierarchy filters — compare risk across divisions or
        drill into a single business unit.
      </div>
    </div>""", unsafe_allow_html=True)

    if not att_check or att_check not in df_filtered.columns:
        st.warning("⚠️ No attrition column detected. The model requires a Yes/No or Y/N column indicating whether employees have left.")
    elif len(df_filtered) < 50:
        st.warning("⚠️ Too few records in the current filter to train a meaningful model. Broaden your selection.")
    else:
        if st.button("🚀 Train Attrition Risk Model", type="primary"):
            with st.spinner("Training model…"):
                result = train_model(df_filtered, col_map)
            if result:
                st.session_state["pred_result"]  = result
                st.session_state["pred_df"]      = df_filtered.copy()
                st.session_state["pred_colmap"]  = col_map
                st.session_state["pred_hier"]    = hierarchy
                st.session_state["pred_grpcol"]  = group_col
                st.success(f"✅ Model trained on {result['n_train']:,} samples using {result['n_features']} features.")
            else:
                st.error("Could not train model — not enough usable features. Check that numeric columns are present.")

        if "pred_result" in st.session_state:
            result   = st.session_state["pred_result"]
            pred_df  = st.session_state["pred_df"]
            risk     = result["risk"]
            hier     = st.session_state["pred_hier"]
            g_col    = st.session_state["pred_grpcol"]

            # ── KPIs
            m1, m2, m3, m4 = st.columns(4)
            if result["auc"]:
                m1.metric("Model AUC", f"{result['auc']:.3f}", "1.0=perfect | 0.5=random")
            m2.metric("High-risk (>50%)", f"{(risk>0.5).sum():,}", f"{(risk>0.5).mean():.1%} of data")
            m3.metric("Very high-risk (>70%)", f"{(risk>0.7).sum():,}", f"{(risk>0.7).mean():.1%}")
            m4.metric("Features used", str(result["n_features"]))

            # ── Row 1: Feature importance + Risk distribution
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                fig_imp, ins_imp = feature_importance_chart(result["importances"])
                _render(fig_imp, ins_imp, "Attrition Risk Drivers", "pred", 0, selected_outputs)
            with c2:
                fig_rd, ins_rd = risk_distribution_chart(risk)
                _render(fig_rd, ins_rd, "Risk Score Distribution", "pred", 1, selected_outputs)

            # ── Row 2: Risk by group + Flight risk heatmap
            c3, c4 = st.columns(2, gap="medium")
            with c3:
                if g_col:
                    fig_rg, ins_rg = risk_by_group_chart(pred_df, col_map, risk, g_col)
                    if fig_rg:
                        _render(fig_rg, ins_rg, f"Risk by {g_col}", "pred", 2, selected_outputs)
            with c4:
                fig_fh, ins_fh = flight_risk_heatmap(pred_df, col_map, risk, hier)
                if fig_fh:
                    _render(fig_fh, ins_fh, "Flight Risk Heatmap", "pred", 3, selected_outputs)

            # ── At-risk table
            st.markdown("""
            <div style='font-size:0.95rem;font-weight:700;color:#0f172a;
                        margin:20px 0 10px'>
              🚨 Top 20 Employees at Highest Predicted Risk
            </div>""", unsafe_allow_html=True)

            # Only active employees for the at-risk table
            active_mask = pred_df[col_map.get("_is_active","__is_active")] == 1
            active_df   = pred_df[active_mask]
            active_risk = risk[active_mask.values]

            if len(active_df) > 0:
                tbl = top_at_risk_table(active_df, col_map, active_risk, n=20)
                st.dataframe(
                    tbl.style.background_gradient(subset=["⚠ Attrition Risk"],
                                                  cmap="RdYlGn_r"),
                    use_container_width=True, height=540,
                )
            else:
                st.info("No active employees in current filter.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:64px 24px;background:white;
                        border-radius:16px;border:2px dashed #e2e8f0'>
              <div style='font-size:2.5rem;margin-bottom:14px'>🤖</div>
              <div style='font-size:0.95rem;font-weight:600;color:#0f172a;margin-bottom:6px'>
                Ready to predict
              </div>
              <div style='color:#64748b;font-size:0.85rem'>
                Click the button above to train the model on the current data slice
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:12px'>"
            "📥 Export Report</div>", unsafe_allow_html=True)

if selected_outputs:
    e1, e2, e3 = st.columns([2, 2, 3])
    with e1:
        # Build title from active filters
        report_title = " | ".join(
            f"{lbl}: {', '.join(vals[:2])}" for lbl, vals in list(active_filters.items())[:2]
        ) or "Conglomerate HR Analytics"
        pptx_bytes = build_pptx(selected_outputs, report_title)
        st.download_button(
            "📊 Download PPTX Report",
            data=pptx_bytes,
            file_name="hr_analytics_report.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            type="primary",
        )
    with e2:
        st.markdown(
            f"<div style='padding:10px 0;color:#64748b;font-size:0.85rem'>"
            f"📎 {len(selected_outputs)} chart{'s' if len(selected_outputs)!=1 else ''} "
            f"in report</div>",
            unsafe_allow_html=True,
        )
    with e3:
        st.markdown(
            "<div style='padding:10px 0;color:#94a3b8;font-size:0.8rem'>"
            "Each chart also has an individual ⬇ JPG button below it.</div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<div style='color:#94a3b8;font-size:0.88rem;padding:6px 0'>"
        "Select charts in the tabs above to enable export.</div>",
        unsafe_allow_html=True,
    )
