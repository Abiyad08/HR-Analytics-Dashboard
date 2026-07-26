"""
HR Analytics Platform — Universal edition.
Works for any business: SME, startup, enterprise, conglomerate.
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
st.set_page_config(page_title="HR Analytics Platform",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown {
  font-family: 'Inter', system-ui, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp { background: #f1f5f9; }
.main .block-container { padding: 1.4rem 2rem 2.5rem; max-width: 1600px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(175deg, #0c1f3f 0%, #0f2a52 50%, #0a1a35 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.07);
}
/* All sidebar text — high contrast on dark bg */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stMarkdown { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* Sidebar radio buttons */
[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #94a3b8 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div p { color: #60a5fa !important; font-weight: 600 !important; }

/* Sidebar selects & inputs */
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="multi-select"] div {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.18) !important;
  color: #f1f5f9 !important;
}
[data-testid="stSidebar"] input {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.18) !important;
  color: #f1f5f9 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="multi-select"] span { color: #f1f5f9 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: rgba(255,255,255,0.05); border-radius: 10px; padding: 6px;
}

/* Sidebar slider */
[data-testid="stSidebar"] [data-testid="stSlider"] label { color: #cbd5e1 !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] p { color: #cbd5e1 !important; }

/* Multiselect option list (dropdown) */
[data-testid="stSidebar"] [data-baseweb="popover"] li { color: #0f172a !important; }

/* ── Page header ── */
.dash-header {
  background: linear-gradient(130deg, #0c1f3f 0%, #1e40af 40%, #4338ca 75%, #6d28d9 100%);
  border-radius: 18px; padding: 28px 36px; margin-bottom: 22px;
  box-shadow: 0 8px 32px rgba(30,64,175,0.28); position: relative; overflow: hidden;
}
.dash-header::before {
  content: ""; position: absolute; top: -50px; right: -50px;
  width: 200px; height: 200px; background: rgba(255,255,255,0.05); border-radius: 50%;
}
.dash-header::after {
  content: ""; position: absolute; bottom: -30px; left: 40%;
  width: 120px; height: 120px; background: rgba(255,255,255,0.03); border-radius: 50%;
}
.dash-header h1 {
  color: #ffffff !important; font-size: 1.8rem; font-weight: 800;
  margin: 0 0 5px; letter-spacing: -0.025em;
}
.dash-header p { color: rgba(255,255,255,0.80) !important; font-size: 0.88rem; margin: 0; line-height: 1.6; }

/* ── KPI metric cards ── */
div[data-testid="metric-container"] {
  background: white; border-radius: 14px; padding: 18px 22px !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.05);
  border: 1px solid #dde3ec; transition: transform .16s, box-shadow .16s;
}
div[data-testid="metric-container"]:hover {
  transform: translateY(-2px); box-shadow: 0 6px 22px rgba(0,0,0,0.10);
}
div[data-testid="metric-container"] label {
  color: #475569 !important; font-size: 0.68rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .09em !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
  color: #0f172a !important; font-size: 1.75rem !important;
  font-weight: 800 !important; line-height: 1.15 !important;
}
div[data-testid="metric-container"] [data-testid="metric-delta"] {
  font-size: 0.76rem !important; font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: white; border-radius: 14px; padding: 5px; gap: 3px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05); border: 1px solid #dde3ec; flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 9px !important; padding: 8px 16px !important;
  font-weight: 600 !important; font-size: 0.81rem !important;
  color: #374151 !important; border: none !important; transition: all .15s;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  background: #f1f5f9 !important; color: #1e3a8a !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #1e40af, #4338ca) !important;
  color: white !important;
  box-shadow: 0 3px 10px rgba(30,64,175,0.32) !important;
  font-weight: 700 !important;
}

/* ── Primary buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #1e40af, #4338ca) !important;
  color: #ffffff !important; border: none !important; border-radius: 10px !important;
  padding: 10px 24px !important; font-weight: 700 !important;
  font-size: 0.85rem !important; transition: all .18s !important;
  box-shadow: 0 3px 12px rgba(30,64,175,0.28) !important;
  letter-spacing: 0.01em;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(30,64,175,0.38) !important;
  opacity: 0.95;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
  background: white !important; color: #1e40af !important;
  border: 2px solid #1e40af !important; border-radius: 8px !important;
  font-weight: 700 !important; font-size: 0.79rem !important;
  padding: 6px 14px !important; transition: all .15s !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: #eff6ff !important; color: #1e3a8a !important;
}

/* ── Chart cards ── */
.chart-card {
  background: white; border-radius: 16px; padding: 20px 20px 14px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.05);
  border: 1px solid #dde3ec; margin-bottom: 8px;
}

/* ── Insight pill ── */
.insight {
  background: #f0f7ff;
  border-left: 4px solid #2563eb; border-radius: 0 8px 8px 0;
  padding: 10px 16px; margin-top: 6px; font-size: 0.82rem;
  color: #1e293b; line-height: 1.6; font-style: italic;
}

/* ── Section labels ── */
.sec-label {
  font-size: 0.67rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: #94a3b8; margin: 16px 0 7px;
}

/* ── Breadcrumb bar ── */
.breadcrumb {
  background: white; border-radius: 10px; padding: 10px 18px; margin-bottom: 16px;
  font-size: 0.82rem; color: #334155; border: 1px solid #dde3ec;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.filter-badge {
  display: inline-block; padding: 3px 11px; border-radius: 20px;
  font-size: 0.72rem; font-weight: 700;
  background: #dbeafe; color: #1d4ed8; margin: 2px 3px;
}
.time-badge {
  display: inline-block; padding: 3px 11px; border-radius: 20px;
  font-size: 0.72rem; font-weight: 700;
  background: #d1fae5; color: #065f46; margin: 2px 3px;
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
  background: #dbeafe !important; border-color: #93c5fd !important;
  color: #1d4ed8 !important; border-radius: 6px !important;
}
[data-baseweb="tag"] span { color: #1d4ed8 !important; font-weight: 600; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: white !important; border-radius: 12px !important;
  border: 1px solid #dde3ec !important;
}
[data-testid="stExpander"] summary { font-weight: 700 !important; color: #0f172a !important; }

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; }

/* ── Progress ── */
.stProgress > div > div {
  background: linear-gradient(90deg, #1e40af, #4338ca) !important;
  border-radius: 4px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, #1e40af, #4338ca) !important;
}

/* ── Caption text in main area ── */
.stCaption { color: #475569 !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _gs_url(url):
    if "docs.google.com/spreadsheets" in url and "/export" not in url:
        try:
            sid = url.split("/d/")[1].split("/")[0]
            gid = url.split("gid=")[1].split("&")[0].split("#")[0] if "gid=" in url else "0"
            return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"
        except Exception:
            return url
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
def _sample():
    return generate_sample_data(5000)

def _render(fig, insight, title, cat, idx, selected_outputs):
    """Render one chart card with insight + download, collect for PPTX."""
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d"]})
    if insight:
        st.markdown(f"<div class='insight'>💡 {insight}</div>", unsafe_allow_html=True)
    jpg = fig_to_jpg(fig)
    if jpg:
        st.download_button(
            "⬇ Download JPG", data=jpg,
            file_name=f"{title.replace(' ','_')[:40]}.jpg",
            mime="image/jpeg",
            key=f"dl_{cat}_{idx}_{hash(title) % 9999}"
        )
    st.markdown("</div>", unsafe_allow_html=True)
    selected_outputs.append({"title": title, "fig": fig, "insight": insight or ""})


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 4px 10px'>
      <div style='font-size:1.3rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em'>
        📊 HR Analytics
      </div>
      <div style='font-size:0.72rem;color:#64748b;margin-top:3px'>
        People Intelligence Platform
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='sec-label'>Data Source</div>", unsafe_allow_html=True)
    source = st.radio("", ["📁 Upload File", "🌐 Load from URL", "🎲 Sample Data"],
                      index=2, label_visibility="collapsed")
    uploaded_file = data_url = None
    if source == "📁 Upload File":
        uploaded_file = st.file_uploader(
            "CSV or Excel", type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )
        st.caption("Any HR export accepted — columns are auto-detected.")
    elif source == "🌐 Load from URL":
        data_url = st.text_input(
            "", placeholder="Paste URL (.csv / .xlsx / Google Sheets)",
            label_visibility="collapsed"
        )
        st.caption("Supports direct file links and shared Google Sheets.")

    st.divider()
    filter_placeholder = st.container()   # filters injected here after data loads
    st.divider()

    st.markdown("""
    <div style='font-size:0.76rem;color:#94a3b8;line-height:1.9'>
      ✦ <b style='color:#cbd5e1'>Auto-detects</b> any column structure<br>
      ✦ <b style='color:#cbd5e1'>45+ charts</b> across 7 analytics areas<br>
      ✦ <b style='color:#cbd5e1'>Cascading filters</b> — drill any hierarchy<br>
      ✦ <b style='color:#cbd5e1'>Time range</b> slider for period analysis<br>
      ✦ <b style='color:#cbd5e1'>Export</b> JPG per chart or full PPTX deck
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:0.73rem;line-height:1.8;padding-bottom:4px'>
      <span style='color:#94a3b8'>Built by</span>
      <b style='color:#e2e8f0'> Abiyad Islam</b><br>
      <span style='color:#64748b'>Master of Business Analytics</span><br>
      <span style='color:#64748b'>Macquarie University</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
raw_df = None
if source == "📁 Upload File" and uploaded_file:
    with st.spinner("Reading file…"):
        try:
            raw_df = _load_file(uploaded_file.read(), uploaded_file.name)
        except Exception as e:
            st.error(f"❌ Could not read file: {e}")
elif source == "🌐 Load from URL" and data_url:
    with st.spinner("Fetching from URL…"):
        try:
            raw_df = _load_url(data_url)
        except Exception as e:
            st.error(f"❌ Could not load URL: {e}")
elif source == "🎲 Sample Data":
    with st.spinner("Generating sample dataset…"):
        raw_df = _sample()


# ── Page header (always shown) ────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>📊 HR Analytics Platform</h1>
  <p>
    Upload any employee dataset &nbsp;·&nbsp; Columns auto-detected &nbsp;·&nbsp;
    Works for any business — SME, enterprise, or multi-division group &nbsp;·&nbsp;
    45+ interactive charts &nbsp;·&nbsp; Export to JPG &amp; PPTX
  </p>
</div>
""", unsafe_allow_html=True)

if raw_df is None:
    st.markdown("""
    <div style='background:white;border-radius:18px;padding:64px 40px;text-align:center;
                border:2px dashed #dde3ec;margin-top:8px'>
      <div style='font-size:3.2rem;margin-bottom:18px'>📂</div>
      <div style='font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:10px'>
        No data loaded yet
      </div>
      <div style='color:#475569;font-size:0.9rem;max-width:520px;margin:0 auto;line-height:1.7'>
        Choose a data source in the sidebar.<br>
        Upload a <b>CSV or Excel file</b>, paste a <b>URL</b>, or try the built-in
        <b>sample dataset</b> (5,000 employees, 5 divisions, 15 business units).
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PREPARE
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Analysing dataset…"):
    df_full, meta = prepare(raw_df)

col_map   = meta["col_map"]
hierarchy = meta["hierarchy"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Filters (injected into placeholder)
# ══════════════════════════════════════════════════════════════════════════════
df_filtered    = df_full.copy()
active_filters = {}
time_filter    = None        # (year_from, year_to) or None

with filter_placeholder:
    # ── 1. Workforce status ──────────────────────────────────────────────────
    st.markdown("<div class='sec-label'>Workforce Status</div>", unsafe_allow_html=True)
    is_active_col = col_map.get("_is_active", "__is_active")
    status_opt = st.radio(
        "",
        ["Active Only", "All (Active + Former)", "Former Only"],
        index=1, label_visibility="collapsed"
    )
    if status_opt == "Active Only" and is_active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[is_active_col] == 1]
    elif status_opt == "Former Only" and is_active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[is_active_col] == 0]

    # ── 2. Time frame ────────────────────────────────────────────────────────
    hire_year_col = col_map.get("_hire_year")
    if hire_year_col and hire_year_col in df_filtered.columns:
        years = sorted(df_filtered[hire_year_col].dropna().astype(int).unique())
        if len(years) > 1:
            st.markdown("<div class='sec-label'>Time Frame (Hire Year)</div>",
                        unsafe_allow_html=True)
            y_min, y_max = int(years[0]), int(years[-1])
            selected_range = st.slider(
                "", min_value=y_min, max_value=y_max,
                value=(y_min, y_max),
                key="year_slider", label_visibility="collapsed"
            )
            if selected_range != (y_min, y_max):
                df_filtered = df_filtered[
                    df_filtered[hire_year_col].between(selected_range[0], selected_range[1])
                ]
                time_filter = selected_range

    # ── 3. Business hierarchy cascading filters ──────────────────────────────
    if hierarchy:
        st.markdown("<div class='sec-label'>Business Hierarchy</div>",
                    unsafe_allow_html=True)
        for level_label, level_col in hierarchy:
            if level_col not in df_filtered.columns:
                continue
            opts = sorted(df_filtered[level_col].dropna().unique().tolist())
            if not opts:
                continue
            chosen = st.multiselect(
                level_label, opts, default=[],
                placeholder=f"All {level_label}s",
                key=f"filt_{level_col}"
            )
            if chosen:
                df_filtered = df_filtered[df_filtered[level_col].isin(chosen)]
                active_filters[level_label] = chosen

    st.markdown(
        f"<div style='font-size:0.76rem;color:#94a3b8;padding:8px 2px 2px'>"
        f"<b style='color:#cbd5e1'>{len(df_filtered):,}</b> records match current filters"
        f"</div>",
        unsafe_allow_html=True
    )


# ── Determine auto grouping column ────────────────────────────────────────────
def _group_col():
    if not hierarchy:
        return None
    for lbl, col in hierarchy:
        if lbl not in active_filters:
            return col
    return hierarchy[-1][1]

group_col = _group_col()


# ══════════════════════════════════════════════════════════════════════════════
# BREADCRUMB
# ══════════════════════════════════════════════════════════════════════════════
has_active_filter = (active_filters or time_filter or status_opt != "All (Active + Former)")
if has_active_filter:
    parts = []
    if status_opt != "All (Active + Former)":
        parts.append(f"<span class='filter-badge'>{status_opt}</span>")
    if time_filter:
        parts.append(f"<span class='time-badge'>📅 {time_filter[0]}–{time_filter[1]}</span>")
    for lbl, vals in active_filters.items():
        display_vals = vals[:3]
        suffix = f" +{len(vals)-3}" if len(vals) > 3 else ""
        parts.append(f"<span class='filter-badge'>{lbl}: {', '.join(display_vals)}{suffix}</span>")
    st.markdown(
        f"<div class='breadcrumb'>🔎 &nbsp;"
        f"{'&nbsp; '.join(parts)}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;<b>{len(df_filtered):,} employees</b>"
        f"</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
att_col  = col_map.get("_attrition")
ten_col  = col_map.get("_tenure_years")
gen_col  = col_map.get("gender")
prob_col = col_map.get("_on_probation")

kpi_cols = st.columns(6)
kpi_data = [("👥 Total Records", f"{len(df_filtered):,}", None)]

if is_active_col in df_filtered.columns:
    act = int((df_filtered[is_active_col] == 1).sum())
    kpi_data.append(("✅ Active", f"{act:,}", f"{act/max(len(df_filtered),1):.0%} of total"))

if att_col and att_col in df_filtered.columns:
    att_r = df_filtered[att_col].mean()
    delta_str = "▲ Above 20% ref" if att_r > 0.20 else "▼ Below 20% ref"
    kpi_data.append(("🔄 Attrition Rate", f"{att_r:.1%}", delta_str))

if ten_col and ten_col in df_filtered.columns:
    kpi_data.append(("📅 Avg Tenure", f"{df_filtered[ten_col].mean():.1f} yrs", None))

if gen_col and gen_col in df_filtered.columns:
    f_pct = (df_filtered[gen_col] == "Female").mean()
    kpi_data.append(("👩 Female", f"{f_pct:.1%}", None))

if prob_col and prob_col in df_filtered.columns and is_active_col in df_filtered.columns:
    act_df   = df_filtered[df_filtered[is_active_col] == 1]
    prob_pct = act_df[prob_col].mean() if len(act_df) else 0
    kpi_data.append(("⏳ On Probation", f"{prob_pct:.0%}", "of active employees"))

for i, (label, value, delta) in enumerate(kpi_data[:6]):
    with kpi_cols[i]:
        st.metric(label, value, delta)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TABS
# ══════════════════════════════════════════════════════════════════════════════
available = get_available(df_filtered, col_map)

if not available:
    st.warning("⚠️ No analytics available for this data slice. Try broadening your filters.")
    st.stop()

tab_labels   = list(available.keys()) + ["🔮 Predictive AI"]
tabs         = st.tabs(tab_labels)
selected_outputs = []

# ── Descriptive tabs ──────────────────────────────────────────────────────────
for tab, (cat_name, charts) in zip(tabs[:-1], available.items()):
    with tab:
        chart_names = [c["name"] for c in charts]

        col_sel, col_info = st.columns([3, 1])
        with col_sel:
            chosen = st.multiselect(
                "Select charts:",
                options=chart_names,
                default=chart_names[:3],
                key=f"ms_{cat_name}",
                placeholder="Choose one or more charts to display…"
            )
        with col_info:
            st.markdown(
                f"<div style='padding:10px 0;font-size:0.8rem;color:#475569'>"
                f"<b>{len(charts)}</b> charts available in this category</div>",
                unsafe_allow_html=True
            )

        if not chosen:
            st.markdown("""
            <div style='text-align:center;padding:52px 24px;background:white;
                        border-radius:16px;border:2px dashed #dde3ec;margin-top:8px'>
              <div style='font-size:2.5rem;margin-bottom:12px'>📊</div>
              <div style='font-size:0.95rem;font-weight:600;color:#0f172a;margin-bottom:4px'>
                Select charts from the list above
              </div>
              <div style='color:#64748b;font-size:0.85rem'>
                You can select multiple charts at once
              </div>
            </div>""", unsafe_allow_html=True)
            continue

        fn_lookup = {c["name"]: c for c in charts}
        col1, col2 = st.columns(2, gap="medium")

        for idx, name in enumerate(chosen):
            chart_def = fn_lookup[name]
            fn        = chart_def["fn"]
            target    = col1 if idx % 2 == 0 else col2

            with target:
                try:
                    with st.spinner(f"Building chart…"):
                        if chart_def.get("needs_hierarchy"):
                            fig, insight = fn(df_filtered, col_map, hierarchy)
                        elif chart_def.get("no_group"):
                            fig, insight = fn(df_filtered, col_map)
                        else:
                            fig, insight = fn(df_filtered, col_map, group_col)

                    if fig is None:
                        st.info(f"Not enough data to render '{name}' with current filters.")
                        continue

                    _render(fig, insight, name, cat_name, idx, selected_outputs)

                except Exception as e:
                    st.error(f"Error in '{name}': {e}")


# ── Predictive AI tab ─────────────────────────────────────────────────────────
with tabs[-1]:
    att_check = col_map.get("_attrition")

    st.markdown("""
    <div style='background:white;border-radius:16px;padding:22px 28px;
                border:1px solid #dde3ec;margin-bottom:20px'>
      <div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:8px'>
        🔮 Predictive Attrition Risk Model
      </div>
      <div style='font-size:0.875rem;color:#334155;line-height:1.8'>
        Trains a <b>Random Forest classifier</b> on the currently selected data slice to
        estimate each employee's probability of leaving. Automatically uses every numeric
        and categorical column it can find. The model fully respects your active filters —
        change the filters and retrain to compare risk across any segment of your workforce.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not att_check or att_check not in df_filtered.columns:
        st.warning("⚠️ No attrition column detected. The model needs a column showing whether "
                   "each employee has left (Yes/No, Y/N, 1/0, True/False, etc.).")
    elif len(df_filtered) < 50:
        st.warning("⚠️ Too few records in this filter slice to train a reliable model. "
                   "Broaden your filters to include more employees.")
    else:
        if st.button("🚀 Train Attrition Risk Model", type="primary"):
            with st.spinner("Training model on selected data slice…"):
                result = train_model(df_filtered, col_map)
            if result:
                st.session_state.update({
                    "pred_result": result,
                    "pred_df":     df_filtered.copy(),
                    "pred_colmap": col_map,
                    "pred_hier":   hierarchy,
                    "pred_grpcol": group_col,
                })
                st.success(
                    f"✅ Model trained — {result['n_train']:,} training samples, "
                    f"{result['n_features']} features used."
                )
            else:
                st.error("Could not train model. Ensure the dataset has numeric feature columns.")

        if "pred_result" in st.session_state:
            result  = st.session_state["pred_result"]
            pred_df = st.session_state["pred_df"]
            risk    = result["risk"]
            hier    = st.session_state["pred_hier"]
            g_col   = st.session_state["pred_grpcol"]

            m1, m2, m3, m4 = st.columns(4)
            if result["auc"]:
                m1.metric("Model AUC", f"{result['auc']:.3f}",
                          "1.0 = perfect  |  0.5 = random guess")
            m2.metric("High-risk (> 50%)", f"{(risk > 0.5).sum():,}",
                      f"{(risk > 0.5).mean():.1%} of employees")
            m3.metric("Very high-risk (> 70%)", f"{(risk > 0.7).sum():,}",
                      f"{(risk > 0.7).mean():.1%} of employees")
            m4.metric("Features used", str(result["n_features"]))

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                fig_imp, ins_imp = feature_importance_chart(result["importances"])
                _render(fig_imp, ins_imp, "Attrition Risk Drivers", "pred", 10, selected_outputs)
            with c2:
                fig_rd, ins_rd = risk_distribution_chart(risk)
                _render(fig_rd, ins_rd, "Risk Score Distribution", "pred", 11, selected_outputs)

            c3, c4 = st.columns(2, gap="medium")
            with c3:
                if g_col:
                    fig_rg, ins_rg = risk_by_group_chart(pred_df, col_map, risk, g_col)
                    if fig_rg:
                        _render(fig_rg, ins_rg, f"Avg Risk by {g_col}", "pred", 12, selected_outputs)
            with c4:
                fig_fh, ins_fh = flight_risk_heatmap(pred_df, col_map, risk, hier)
                if fig_fh:
                    _render(fig_fh, ins_fh, "Flight Risk Heatmap", "pred", 13, selected_outputs)

            st.markdown(
                "<div style='font-size:0.95rem;font-weight:700;color:#0f172a;"
                "margin:24px 0 10px'>🚨 Top 20 Employees at Highest Predicted Risk</div>",
                unsafe_allow_html=True
            )
            act_mask  = pred_df[col_map.get("_is_active","__is_active")] == 1
            act_df    = pred_df[act_mask]
            act_risk  = risk[act_mask.values]

            if len(act_df) > 0:
                tbl = top_at_risk_table(act_df, col_map, act_risk, n=20)
                st.dataframe(
                    tbl.style.background_gradient(subset=["⚠ Attrition Risk"], cmap="RdYlGn_r"),
                    use_container_width=True, height=560,
                )
            else:
                st.info("No active employees in the current filter slice.")

        else:
            st.markdown("""
            <div style='text-align:center;padding:64px 24px;background:white;
                        border-radius:16px;border:2px dashed #dde3ec'>
              <div style='font-size:2.8rem;margin-bottom:16px'>🤖</div>
              <div style='font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:6px'>
                Ready to predict
              </div>
              <div style='color:#475569;font-size:0.86rem'>
                Configure your filters above, then click the button to train the model
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:14px'>"
    "📥 Export Report</div>",
    unsafe_allow_html=True
)

if selected_outputs:
    e1, e2, e3 = st.columns([2, 2, 4])
    with e1:
        filter_parts = [f"{lbl}: {', '.join(v[:2])}" for lbl, v in list(active_filters.items())[:2]]
        if time_filter:
            filter_parts.insert(0, f"{time_filter[0]}–{time_filter[1]}")
        report_title = " | ".join(filter_parts) if filter_parts else "HR Analytics Report"
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
            f"<div style='padding:10px 0;color:#334155;font-size:0.86rem'>"
            f"📎 <b>{len(selected_outputs)}</b> chart{'s' if len(selected_outputs)!=1 else ''} "
            f"included in report</div>",
            unsafe_allow_html=True
        )
    with e3:
        st.markdown(
            "<div style='padding:10px 0;color:#64748b;font-size:0.82rem'>"
            "Each individual chart also has its own ⬇ Download JPG button below it.</div>",
            unsafe_allow_html=True
        )
else:
    st.markdown(
        "<div style='color:#64748b;font-size:0.88rem;padding:8px 0'>"
        "Select charts in the tabs above and they will be included in the PPTX export.</div>",
        unsafe_allow_html=True
    )
