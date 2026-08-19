"""HR Analytics Platform — clean, professional, market-ready."""
import io
import pandas as pd
import streamlit as st

from data_prep    import prepare
from analytics    import get_available
from insights     import generate as gen_insights
from predictive   import (train_model, feature_importance_chart,
                          risk_distribution_chart, risk_by_group_chart,
                          flight_risk_heatmap, top_at_risk_table)
from export_utils import fig_to_jpg, build_pptx, build_excel
from sample_data  import generate_sample_data
from create_template import build_blank_template, build_sample_50

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Analytics", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset ── */
*, html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── App shell ── */
.stApp { background: #f4f6f9; }
.main .block-container { padding: 0 2rem 3rem; max-width: 1540px; }

/* ════════════════════════════════════════
   SIDEBAR — GitHub Dark inspired
   ════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: #0d1117 !important;
  border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([data-baseweb]),
[data-testid="stSidebar"] div:not([data-baseweb]),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption { color: #8b949e !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b { color: #e6edf3 !important; }
[data-testid="stSidebar"] hr { border-color: #21262d !important; margin: 12px 0; }

/* Sidebar radio */
[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #8b949e !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] ~ div p { color: #e6edf3 !important; font-weight: 600 !important; }

/* Sidebar selects */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="multi-select"] > div {
  background: #161b22 !important; border: 1px solid #30363d !important;
  border-radius: 6px !important; color: #e6edf3 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="multi-select"] span { color: #e6edf3 !important; }
[data-testid="stSidebar"] input {
  background: #161b22 !important; border: 1px solid #30363d !important;
  color: #e6edf3 !important; border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #161b22; border: 1px dashed #30363d; border-radius: 8px; padding: 4px;
}
[data-testid="stSidebar"] [data-testid="stSlider"] p,
[data-testid="stSidebar"] [data-testid="stSlider"] label { color: #8b949e !important; }

/* ════════════════════════════════════════
   PAGE HEADER — minimal dark bar
   ════════════════════════════════════════ */
.page-header {
  background: #0d1117;
  padding: 18px 32px;
  margin: 0 -2rem 28px -2rem;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid #21262d;
}
.page-header-left { display: flex; align-items: center; gap: 12px; }
.page-logo { font-size: 1.1rem; font-weight: 800; color: #5046e4; letter-spacing: -0.02em; }
.page-title { font-size: 0.95rem; font-weight: 600; color: #e6edf3; }
.page-subtitle { font-size: 0.78rem; color: #484f58; margin-top: 1px; }
.page-divider { width: 1px; height: 28px; background: #21262d; }

/* ════════════════════════════════════════
   KPI CARDS
   ════════════════════════════════════════ */
.kpi-card {
  background: #ffffff; border: 1px solid #e5e7eb;
  border-radius: 10px; padding: 18px 20px;
  transition: border-color 0.15s ease;
}
.kpi-card:hover { border-color: #5046e4; }
.kpi-label {
  font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 8px;
}
.kpi-value {
  font-size: 1.85rem; font-weight: 800; color: #111827 !important;
  line-height: 1; margin-bottom: 8px; letter-spacing: -0.02em;
}
.kpi-pill {
  display: inline-flex; align-items: center; gap: 3px;
  font-size: 0.72rem; font-weight: 600;
  padding: 3px 9px; border-radius: 20px;
}
.kpi-pill.good { background: #d1fae5; color: #065f46; }
.kpi-pill.bad  { background: #fee2e2; color: #991b1b; }
.kpi-pill.neu  { background: #f3f4f6; color: #4b5563; }

/* ════════════════════════════════════════
   TABS — underline style
   ════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid #e5e7eb !important;
  border-radius: 0 !important; padding: 0 !important; gap: 0 !important;
  box-shadow: none !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 0 !important; padding: 10px 20px !important;
  font-weight: 500 !important; font-size: 0.84rem !important;
  color: #6b7280 !important; background: transparent !important;
  border: none !important; border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important; transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) { color: #374151 !important; }
.stTabs [aria-selected="true"] {
  color: #5046e4 !important; background: transparent !important;
  border-bottom: 2px solid #5046e4 !important;
  font-weight: 600 !important; box-shadow: none !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 24px !important; }

/* ════════════════════════════════════════
   CHART CARDS
   ════════════════════════════════════════ */
.chart-wrap {
  background: #ffffff; border: 1px solid #e5e7eb;
  border-radius: 12px; overflow: hidden; margin-bottom: 16px;
}
.chart-head {
  padding: 14px 18px 0; display: flex;
  align-items: center; justify-content: space-between;
}
.chart-name { font-size: 0.84rem; font-weight: 600; color: #111827; }
.chart-foot {
  padding: 8px 18px 14px; font-size: 0.79rem;
  color: #6b7280; font-style: italic;
  border-top: 1px solid #f3f4f6; margin-top: 4px; line-height: 1.55;
}

/* ════════════════════════════════════════
   INSIGHT CARDS (Summary tab)
   ════════════════════════════════════════ */
.ins-card {
  background: #ffffff; border: 1px solid #e5e7eb;
  border-left: 3px solid #5046e4;
  border-radius: 0 10px 10px 0; padding: 14px 16px;
  display: flex; gap: 12px; align-items: flex-start;
  height: 100%;
}
.ins-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 1px; }
.ins-head { font-weight: 600; color: #111827; font-size: 0.855rem; margin-bottom: 4px; }
.ins-body { color: #6b7280; font-size: 0.78rem; line-height: 1.55; }

/* ════════════════════════════════════════
   BUTTONS
   ════════════════════════════════════════ */
.stButton > button {
  background: #5046e4 !important; color: #ffffff !important;
  border: none !important; border-radius: 8px !important;
  padding: 9px 20px !important; font-weight: 600 !important;
  font-size: 0.85rem !important; transition: background 0.15s !important;
  box-shadow: none !important; letter-spacing: 0.01em;
}
.stButton > button:hover { background: #4338ca !important; transform: none !important; }
[data-testid="stDownloadButton"] > button {
  background: transparent !important; color: #5046e4 !important;
  border: 1.5px solid #5046e4 !important; border-radius: 7px !important;
  font-weight: 600 !important; font-size: 0.78rem !important;
  padding: 5px 13px !important; box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover { background: #eef2ff !important; }

/* ════════════════════════════════════════
   FILTER BADGES
   ════════════════════════════════════════ */
.filter-row {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
  padding: 8px 14px; margin-bottom: 20px; font-size: 0.8rem; color: #374151;
}
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 9px; border-radius: 20px; font-size: 0.71rem; font-weight: 600;
}
.badge-blue   { background: #eef2ff; color: #4338ca; }
.badge-green  { background: #ecfdf5; color: #065f46; }
.badge-slate  { background: #f1f5f9; color: #475569; }

/* ════════════════════════════════════════
   HEALTH BAR
   ════════════════════════════════════════ */
.health-wrap { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px 18px; }
.health-track { background: #f3f4f6; border-radius: 99px; height: 6px; margin: 10px 0; }
.health-fill  { height: 100%; border-radius: 99px; background: #5046e4; }
.field-chip   { display: inline-block; padding: 2px 9px; border-radius: 5px; font-size: 0.69rem; font-weight: 600; margin: 2px; }
.chip-ok  { background: #ecfdf5; color: #065f46; }
.chip-mis { background: #fef2f2; color: #991b1b; }

/* ════════════════════════════════════════
   SECTION TITLE
   ════════════════════════════════════════ */
.sec-title {
  font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.09em; color: #9ca3af; margin: 0 0 8px;
}

/* ════════════════════════════════════════
   EXPANDER
   ════════════════════════════════════════ */
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #e5e7eb !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { font-weight: 600 !important; font-size: 0.875rem !important; color: #111827 !important; }

/* ════════════════════════════════════════
   MULTISELECT TAGS
   ════════════════════════════════════════ */
[data-baseweb="tag"] { background: #eef2ff !important; border-color: #c7d2fe !important; color: #4338ca !important; border-radius: 5px !important; }
[data-baseweb="tag"] span { color: #4338ca !important; font-weight: 600; }

/* ════════════════════════════════════════
   MISC
   ════════════════════════════════════════ */
hr { border-color: #e5e7eb !important; }
.stAlert { border-radius: 8px !important; }
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
.stCaption { color: #6b7280 !important; font-size: 0.79rem !important; }
/* Force dark text in main area */
.main .stMarkdown p, .main .stMarkdown span { color: #374151; }
</style>
""", unsafe_allow_html=True)

# ── Utilities ─────────────────────────────────────────────────────────────────
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
def _blank_template(): return build_blank_template()

@st.cache_data(show_spinner=False)
def _sample_template(): return build_sample_50()

def _kpi(label, value, delta=None, delta_type="neu"):
    pill = f"<span class='kpi-pill {delta_type}'>{delta}</span>" if delta else ""
    return (f"<div class='kpi-card'>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"{pill}</div>")

def _render_chart(fig, insight, title, cat, idx, outputs):
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.markdown(f"<div class='chart-head'><span class='chart-name'>{title}</span></div>",
                unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d"],
        "toImageButtonOptions": {"format":"png","filename":title,"scale":2},
    })
    if insight:
        st.markdown(f"<div class='chart-foot'>💡 {insight}</div>", unsafe_allow_html=True)
    jpg = fig_to_jpg(fig)
    if jpg:
        st.download_button("⬇ JPG", data=jpg,
                           file_name=f"{title.replace(' ','_')[:40]}.jpg",
                           mime="image/jpeg",
                           key=f"dl_{cat}_{idx}_{abs(hash(title))%9999}")
    st.markdown("</div>", unsafe_allow_html=True)
    outputs.append({"title": title, "fig": fig, "insight": insight or ""})

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='padding:20px 4px 16px'>
      <div style='font-size:1.15rem;font-weight:800;color:#e6edf3;letter-spacing:-0.025em'>
        ◈ HR Analytics
      </div>
      <div style='font-size:0.72rem;color:#484f58;margin-top:3px'>
        People Intelligence Platform
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Data source
    st.markdown("<div class='sec-title'>Data Source</div>", unsafe_allow_html=True)
    source = st.radio("", ["📁 Upload File","🌐 URL / Google Sheets","◈ Sample Data"],
                      index=2, label_visibility="collapsed")
    uploaded_file = data_url = None
    if source == "📁 Upload File":
        uploaded_file = st.file_uploader("CSV or Excel (.csv .xlsx .xls)",
                                         type=["csv","xlsx","xls"],
                                         label_visibility="collapsed")
    elif source == "🌐 URL / Google Sheets":
        data_url = st.text_input("", placeholder="Paste link here…",
                                 label_visibility="collapsed")
    st.divider()

    # Filters (populated after data loads)
    filter_zone   = st.container()
    settings_zone = st.container()
    st.divider()

    # Export (populated after charts selected)
    export_zone = st.container()

    # Attribution
    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem;line-height:1.9;padding:2px 0'>
      <div style='color:#e6edf3;font-weight:600'>Abiyad Islam</div>
      <div style='color:#484f58'>Master of Business Analytics</div>
      <div style='color:#484f58'>Macquarie University</div>
    </div>""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
raw_df = None
if source == "📁 Upload File" and uploaded_file:
    with st.spinner("Reading file…"):
        try: raw_df = _load_file(uploaded_file.read(), uploaded_file.name)
        except Exception as e: st.error(f"Could not read file: {e}")
elif source == "🌐 URL / Google Sheets" and data_url:
    with st.spinner("Fetching data…"):
        try: raw_df = _load_url(data_url)
        except Exception as e: st.error(f"Could not load URL: {e}")
elif source == "◈ Sample Data":
    with st.spinner("Loading sample dataset…"): raw_df = _sample()

# ── Page header ───────────────────────────────────────────────────────────────
dataset_label = (uploaded_file.name if uploaded_file
                 else "URL" if data_url
                 else "Sample Dataset — 5,000 employees")
records_label = f"{len(raw_df):,} records" if raw_df is not None else ""

st.markdown(f"""
<div class='page-header'>
  <div class='page-header-left'>
    <span class='page-logo'>◈</span>
    <div class='page-divider'></div>
    <div>
      <div class='page-title'>HR Analytics Platform</div>
      <div class='page-subtitle'>{dataset_label}{" · " + records_label if records_label else ""}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

if raw_df is None:
    st.markdown("""
    <div style='max-width:540px;margin:80px auto;text-align:center'>
      <div style='font-size:2.5rem;margin-bottom:20px;opacity:.4'>◈</div>
      <div style='font-size:1.15rem;font-weight:700;color:#111827;margin-bottom:10px'>
        No data loaded</div>
      <div style='color:#6b7280;font-size:0.9rem;line-height:1.7'>
        Upload a <strong>CSV or Excel file</strong>, paste a <strong>URL</strong>,
        or try the built-in <strong>sample dataset</strong> using the sidebar.
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Prepare data ──────────────────────────────────────────────────────────────
with st.spinner("Analysing…"):
    df_full, meta = prepare(raw_df)
col_map   = meta["col_map"]
hierarchy = meta["hierarchy"]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════════════════════════
df_filtered    = df_full.copy()
active_filters = {}
time_filter    = None
active_col     = col_map.get("_is_active","__is_active")

with filter_zone:
    st.markdown("<div class='sec-title'>Workforce</div>", unsafe_allow_html=True)
    status_opt = st.radio("",
                          ["Active employees only","All records","Former employees only"],
                          index=1, label_visibility="collapsed")
    if status_opt == "Active employees only" and active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[active_col]==1]
    elif status_opt == "Former employees only" and active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[active_col]==0]

    hire_year_col = col_map.get("_hire_year")
    if hire_year_col and hire_year_col in df_filtered.columns:
        years = sorted(df_filtered[hire_year_col].dropna().astype(int).unique())
        if len(years) > 1:
            st.markdown("<div class='sec-title' style='margin-top:12px'>Hire Year Range</div>",
                        unsafe_allow_html=True)
            y_min, y_max = int(years[0]), int(years[-1])
            sel = st.slider("", y_min, y_max, (y_min, y_max),
                            key="yr_slider", label_visibility="collapsed")
            if sel != (y_min, y_max):
                df_filtered = df_filtered[df_filtered[hire_year_col].between(sel[0],sel[1])]
                time_filter = sel

    if hierarchy:
        st.markdown("<div class='sec-title' style='margin-top:12px'>Business Hierarchy</div>",
                    unsafe_allow_html=True)
        for lbl, col in hierarchy:
            if col not in df_filtered.columns: continue
            opts = sorted(df_filtered[col].dropna().unique().tolist())
            if not opts: continue
            chosen = st.multiselect(lbl, opts, default=[],
                                    placeholder=f"All {lbl}s", key=f"filt_{col}")
            if chosen:
                df_filtered = df_filtered[df_filtered[col].isin(chosen)]
                active_filters[lbl] = chosen

    n_filt = len(df_filtered)
    st.markdown(f"<div style='font-size:0.74rem;color:#484f58;padding:10px 0 2px'>"
                f"<b style='color:#8b949e'>{n_filt:,}</b> records selected</div>",
                unsafe_allow_html=True)
    if (active_filters or time_filter):
        if st.button("Clear filters", key="clear_all"):
            st.rerun()

with settings_zone:
    st.divider()
    st.markdown("<div class='sec-title'>Settings</div>", unsafe_allow_html=True)
    custom_benchmark = st.slider("Attrition target (%)", 5, 40, 15, 1,
                                 key="bench_slider")
    with st.expander("Column mapping", expanded=False):
        st.caption("Override auto-detected column assignments.")
        override_concepts = ["department","job_level","salary","attrition","gender",
                             "hire_date","exit_date","tenure_years","employment_type",
                             "designation_main","grade","primary_leave_taken"]
        all_raw = ["(auto)"] + sorted(raw_df.columns.tolist())
        for concept in override_concepts:
            current = col_map.get(concept,"")
            idx = all_raw.index(current) if current in all_raw else 0
            picked = st.selectbox(concept.replace("_"," ").title(), all_raw,
                                  index=idx, key=f"cmap_{concept}")
            if picked != "(auto)": col_map[concept] = picked

# Apply custom benchmark
import analytics as _am, insights as _im
_am.ATTRITION_BENCHMARK = custom_benchmark
_im.ATTRITION_BENCHMARK = custom_benchmark

# Group col
def _group_col():
    if not hierarchy: return None
    for lbl, col in hierarchy:
        if lbl not in active_filters: return col
    return hierarchy[-1][1]
group_col = _group_col()

# ── Filter badge row ──────────────────────────────────────────────────────────
badges = []
if status_opt != "All records":
    short = {"Active employees only":"Active only","Former employees only":"Former only"}
    badges.append(f"<span class='badge badge-slate'>{short.get(status_opt, status_opt)}</span>")
if time_filter:
    badges.append(f"<span class='badge badge-green'>📅 {time_filter[0]}–{time_filter[1]}</span>")
for lbl, vals in active_filters.items():
    disp = vals[:2]; extra = f" +{len(vals)-2}" if len(vals)>2 else ""
    badges.append(f"<span class='badge badge-blue'>{lbl}: {', '.join(disp)}{extra}</span>")

if badges:
    st.markdown(
        f"<div class='filter-row'>🔎 &nbsp;{'  '.join(badges)}"
        f" &nbsp;·&nbsp; <strong>{n_filt:,} employees</strong></div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
att_col = col_map.get("_attrition"); ten_col = col_map.get("_tenure_years")
gen_col = col_map.get("gender");     sal_col = next((col_map.get(c) for c in ["salary"] if col_map.get(c)), None)

def _dt(v, ref, invert=False):
    """Return delta type for KPI pill."""
    if v is None or ref is None: return "neu", ""
    better = v < ref if not invert else v > ref
    return ("good" if better else "bad"), f"{'↓' if v < ref else '↑'} {abs(v-ref):.1f}"

cards, n_kpi = [], 0
cards.append(_kpi("Total Records", f"{len(df_filtered):,}"))
if active_col in df_filtered.columns:
    act = int((df_filtered[active_col]==1).sum())
    pct = act/max(len(df_filtered),1)*100
    cards.append(_kpi("Active Employees", f"{act:,}",
                       f"{pct:.0f}% of records", "good" if pct > 70 else "neu"))
if att_col and att_col in df_filtered.columns:
    r = df_filtered[att_col].mean()*100
    dt, dv = _dt(r, custom_benchmark)
    cards.append(_kpi("Attrition Rate", f"{r:.1f}%",
                       f"{'↑' if r>custom_benchmark else '↓'} {abs(r-custom_benchmark):.1f}pp vs {custom_benchmark}% target",
                       "bad" if r>custom_benchmark else "good"))
if ten_col and ten_col in df_filtered.columns:
    cards.append(_kpi("Avg Tenure", f"{df_filtered[ten_col].mean():.1f} yrs"))
if gen_col and gen_col in df_filtered.columns:
    f_pct = (df_filtered[gen_col]=="Female").mean()*100
    cards.append(_kpi("Female Representation", f"{f_pct:.1f}%",
                       "At target" if f_pct >= 30 else f"{30-f_pct:.0f}pp below 30% target",
                       "good" if f_pct >= 30 else "neu"))
if sal_col and sal_col in df_filtered.columns:
    cards.append(_kpi("Median Pay", f"{df_filtered[sal_col].median():,.0f}"))

n_kpi = min(len(cards), 6)
kpi_cols = st.columns(n_kpi, gap="medium")
for i, card_html in enumerate(cards[:n_kpi]):
    with kpi_cols[i]:
        st.markdown(card_html, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS — TAB NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
available      = get_available(df_filtered, col_map)
ai_insights    = gen_insights(df_filtered, col_map)
selected_outputs = []

if not available:
    st.warning("No analytics available for this data slice — try broadening your filters.")
    st.stop()

# Tab labels: Summary first, then categories, then Predictive
tab_labels = (["Summary"] +
              [cat.split(" ",1)[1] if " " in cat else cat for cat in available.keys()] +
              ["Predictive"])
tabs = st.tabs(tab_labels)

# ── TAB 0 — Summary ───────────────────────────────────────────────────────────
with tabs[0]:
    # ── AI Insights grid ──────────────────────────────────────────────────────
    if ai_insights:
        st.markdown("<div class='sec-title'>AI-Generated Insights</div>",
                    unsafe_allow_html=True)
        rows = [ai_insights[i:i+2] for i in range(0, min(len(ai_insights), 8), 2)]
        for row in rows:
            c1, c2 = st.columns(2, gap="medium")
            for col_obj, (emoji, headline, detail) in zip([c1, c2], row):
                with col_obj:
                    st.markdown(
                        f"<div class='ins-card'>"
                        f"<div class='ins-icon'>{emoji}</div>"
                        f"<div><div class='ins-head'>{headline}</div>"
                        f"<div class='ins-body'>{detail}</div></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

    # ── Data quality ──────────────────────────────────────────────────────────
    KEY_CONCEPTS = ["department","job_level","salary","attrition","gender","hire_date",
                    "tenure_years","location","age","ethnicity","training_hours",
                    "absence_days","designation_main","grade","district","education",
                    "performance_rating","engagement_score","recruitment_source"]
    found   = [c for c in KEY_CONCEPTS if col_map.get(c)]
    missing = [c for c in KEY_CONCEPTS if not col_map.get(c)]
    score   = len(found)/len(KEY_CONCEPTS)

    st.markdown("<div class='sec-title'>Data Quality</div>", unsafe_allow_html=True)
    _fc = " ".join(f'<span class="field-chip chip-ok">&#10003; {c.replace("_"," ")}</span>' for c in found)
    _mc = " ".join(f'<span class="field-chip chip-mis">&#10007; {c.replace("_"," ")}</span>' for c in missing)
    _cover = f"{score:.0%} coverage · {len(raw_df):,} rows · {len(raw_df.columns)} columns · {len(hierarchy)} hierarchy levels"
    _hw = (
        "<div class='health-wrap'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-size:0.84rem;font-weight:600;color:#111827'>{len(found)}/{len(KEY_CONCEPTS)} HR fields detected</span>"
        f"<span style='font-size:0.78rem;color:#6b7280'>{_cover}</span>"
        "</div>"
        f"<div class='health-track'><div class='health-fill' style='width:{score*100:.0f}%'></div></div>"
        f"<div style='margin-top:8px'>{_fc} {_mc}</div></div>"
    )
    st.markdown(_hw, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

    # ── Period comparison ─────────────────────────────────────────────────────
    hire_year_col_pc = col_map.get("_hire_year")
    if hire_year_col_pc and hire_year_col_pc in df_filtered.columns:
        yrs = sorted(df_filtered[hire_year_col_pc].dropna().astype(int).unique())
        if len(yrs) >= 4:
            with st.expander("Compare two time periods", expanded=False):
                pc1, pc2 = st.columns(2, gap="large")
                with pc1:
                    st.markdown("**Period A**")
                    p_a = st.slider("A", yrs[0], yrs[-1], (yrs[0], yrs[len(yrs)//2]), key="pa")
                with pc2:
                    st.markdown("**Period B**")
                    p_b = st.slider("B", yrs[0], yrs[-1], (yrs[len(yrs)//2]+1, yrs[-1]), key="pb")
                pa_df = df_filtered[df_filtered[hire_year_col_pc].between(p_a[0],p_a[1])]
                pb_df = df_filtered[df_filtered[hire_year_col_pc].between(p_b[0],p_b[1])]

                def _cmp(da, db, lbl, fn):
                    try: va,vb = fn(da),fn(db); delta = round(vb-va,2) if isinstance(va,(int,float)) else None
                    except: return lbl,"—","—",None
                    return lbl,va,vb,delta

                comps = [("Records", len(pa_df), len(pb_df), None)]
                if att_col and att_col in df_filtered.columns:
                    comps.append(_cmp(pa_df,pb_df,"Attrition %",lambda d:round(d[att_col].mean()*100,1)))
                if ten_col and ten_col in df_filtered.columns:
                    comps.append(_cmp(pa_df,pb_df,"Avg Tenure",lambda d:round(d[ten_col].mean(),1)))
                if gen_col and gen_col in df_filtered.columns:
                    comps.append(_cmp(pa_df,pb_df,"Female %",lambda d:round((d[gen_col]=="Female").mean()*100,1)))

                c_cols = st.columns(len(comps), gap="medium")
                for i,(lbl,va,vb,delta) in enumerate(comps):
                    with c_cols[i]:
                        d_type = ("bad" if delta and delta>0 and "Att" in lbl else
                                  "good" if delta and delta>0 else "neu")
                        d_str  = f"{'▲' if delta and delta>0 else '▼'} {abs(delta)}" if delta else ""
                        st.markdown(
                            f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div>"
                            f"<div style='display:flex;gap:6px;align-items:baseline;margin-bottom:6px'>"
                            f"<span style='font-size:1.25rem;font-weight:800;color:#5046e4'>{va}</span>"
                            f"<span style='font-size:0.78rem;color:#9ca3af'>→</span>"
                            f"<span style='font-size:1.25rem;font-weight:800;color:#111827'>{vb}</span>"
                            f"</div><span class='kpi-pill {d_type}'>{d_str}</span></div>",
                            unsafe_allow_html=True)
                st.caption(f"A: {p_a[0]}–{p_a[1]} ({len(pa_df):,})   B: {p_b[0]}–{p_b[1]} ({len(pb_df):,})")

    # ── Quick charts ──────────────────────────────────────────────────────────
    st.markdown("<div class='sec-title' style='margin-top:4px'>Quick Charts</div>",
                unsafe_allow_html=True)
    from analytics import attrition_gauge_overview, headcount_active_vs_former
    qc1, qc2 = st.columns(2, gap="medium")
    with qc1:
        try:
            fig,ins = attrition_gauge_overview(df_filtered, col_map, group_col)
            if fig: _render_chart(fig, ins, "Attrition Rate", "summary", 0, selected_outputs)
        except: pass
    with qc2:
        try:
            fig,ins = headcount_active_vs_former(df_filtered, col_map, group_col)
            if fig: _render_chart(fig, ins, "Headcount — Active vs Former", "summary", 1, selected_outputs)
        except: pass

# ── CATEGORY TABS ─────────────────────────────────────────────────────────────
for tab, (cat_name, charts) in zip(tabs[1:-1], available.items()):
    with tab:
        chart_names = [c["name"] for c in charts]

        # Select All / Clear / multiselect in one clean row
        b_all, b_clr, sel_col = st.columns([1,1,10])
        with b_all:
            if st.button("All", key=f"all_{cat_name}"):
                st.session_state[f"ms_{cat_name}"] = chart_names; st.rerun()
        with b_clr:
            if st.button("Clear", key=f"clr_{cat_name}"):
                st.session_state[f"ms_{cat_name}"] = []; st.rerun()
        with sel_col:
            chosen = st.multiselect("", options=chart_names,
                                    default=chart_names[:3],
                                    key=f"ms_{cat_name}",
                                    placeholder=f"Select charts to display ({len(charts)} available)…",
                                    label_visibility="collapsed")

        if not chosen:
            st.markdown(
                "<div style='padding:56px 24px;text-align:center;background:#fff;"
                "border:1px solid #e5e7eb;border-radius:12px;margin-top:8px'>"
                "<div style='font-size:2rem;margin-bottom:12px;opacity:.3'>◈</div>"
                "<div style='font-weight:600;color:#374151;margin-bottom:4px'>"
                "Select charts above</div>"
                "<div style='color:#9ca3af;font-size:0.84rem'>"
                "Use the dropdown to pick one or more charts to display side by side</div>"
                "</div>", unsafe_allow_html=True)
            continue

        fn_lookup = {c["name"]: c for c in charts}
        col1, col2 = st.columns(2, gap="medium")
        for idx, name in enumerate(chosen):
            cd = fn_lookup[name]
            with (col1 if idx%2==0 else col2):
                try:
                    with st.spinner(""):
                        if cd.get("needs_hierarchy"): fig,ins = cd["fn"](df_filtered,col_map,hierarchy)
                        elif cd.get("no_group"):      fig,ins = cd["fn"](df_filtered,col_map)
                        else:                         fig,ins = cd["fn"](df_filtered,col_map,group_col)
                    if fig is None:
                        st.info(f"Not enough data for '{name}' with current filters.")
                    else:
                        _render_chart(fig, ins, name, cat_name, idx, selected_outputs)
                except Exception as e:
                    st.error(f"Error in '{name}': {e}")

# ── PREDICTIVE TAB ────────────────────────────────────────────────────────────
with tabs[-1]:
    att_check = col_map.get("_attrition")
    st.markdown(
        "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:10px;"
        "padding:20px 24px;margin-bottom:20px'>"
        "<div style='font-weight:700;color:#111827;margin-bottom:6px'>"
        "Predictive Attrition Risk Model</div>"
        "<div style='font-size:0.875rem;color:#6b7280;line-height:1.7'>"
        "Trains a <strong>Random Forest classifier</strong> on the current filtered "
        "dataset to predict each employee's likelihood of leaving. "
        "Change your filters and retrain to compare risk across any segment."
        "</div></div>", unsafe_allow_html=True)

    if not att_check or att_check not in df_filtered.columns:
        st.warning("No attrition column detected. The model needs a Yes/No or 1/0 column "
                   "indicating whether each employee has left.")
    elif len(df_filtered) < 50:
        st.warning("Too few records in the current filter slice. Broaden your selection.")
    else:
        if st.button("Train Model", type="primary"):
            with st.spinner("Training…"):
                result = train_model(df_filtered, col_map)
            if result:
                st.session_state.update({
                    "pred_result": result, "pred_df": df_filtered.copy(),
                    "pred_colmap": col_map, "pred_hier": hierarchy,
                    "pred_grpcol": group_col,
                })
                st.success(f"Model ready — {result['n_train']:,} training samples, "
                           f"{result['n_features']} features, AUC {result['auc']:.3f}")
            else:
                st.error("Could not train — insufficient usable feature columns.")

        if "pred_result" in st.session_state:
            res = st.session_state["pred_result"]
            pdf = st.session_state["pred_df"]
            risk = res["risk"]
            hier2 = st.session_state["pred_hier"]
            g2    = st.session_state["pred_grpcol"]

            # Model KPIs
            m_cards = [
                _kpi("Model AUC",       f"{res['auc']:.3f}", "1.0 = perfect", "good" if res['auc']>.7 else "neu"),
                _kpi("High risk > 50%", f"{(risk>.5).sum():,}", f"{(risk>.5).mean():.1%}", "bad" if (risk>.5).mean()>.2 else "neu"),
                _kpi("Very high > 70%", f"{(risk>.7).sum():,}", f"{(risk>.7).mean():.1%}", "bad" if (risk>.7).mean()>.1 else "neu"),
                _kpi("Features used",   str(res["n_features"])),
            ]
            mc = st.columns(4, gap="medium")
            for i,c in enumerate(m_cards):
                with mc[i]: st.markdown(c, unsafe_allow_html=True)
            st.markdown("<div style='margin:16px 0'></div>", unsafe_allow_html=True)

            c1,c2 = st.columns(2, gap="medium")
            with c1:
                fi,ii = feature_importance_chart(res["importances"])
                _render_chart(fi,ii,"Attrition Risk Drivers","pred",10,selected_outputs)
            with c2:
                fr,ir = risk_distribution_chart(risk)
                _render_chart(fr,ir,"Risk Score Distribution","pred",11,selected_outputs)
            c3,c4 = st.columns(2, gap="medium")
            with c3:
                if g2:
                    fg,ig = risk_by_group_chart(pdf,col_map,risk,g2)
                    if fg: _render_chart(fg,ig,f"Avg Risk by {g2}","pred",12,selected_outputs)
            with c4:
                fh,ih = flight_risk_heatmap(pdf,col_map,risk,hier2)
                if fh: _render_chart(fh,ih,"Flight Risk Heatmap","pred",13,selected_outputs)

            st.markdown("<div style='font-weight:600;color:#111827;font-size:0.9rem;"
                        "margin:24px 0 12px'>Top 20 Employees at Highest Risk</div>",
                        unsafe_allow_html=True)
            am = pdf[col_map.get("_is_active","__is_active")]==1
            as_, ar = pdf[am], risk[am.values]
            if len(as_):
                tbl = top_at_risk_table(as_,col_map,ar,n=20)
                st.dataframe(
                    tbl.style.background_gradient(subset=["⚠ Attrition Risk"], cmap="RdYlGn_r"),
                    use_container_width=True, height=540
                )
        else:
            st.markdown(
                "<div style='padding:72px 24px;text-align:center;background:#fff;"
                "border:1px solid #e5e7eb;border-radius:12px'>"
                "<div style='font-size:2.5rem;margin-bottom:16px;opacity:.25'>◈</div>"
                "<div style='font-weight:600;color:#374151;margin-bottom:6px'>"
                "Model not trained yet</div>"
                "<div style='color:#9ca3af;font-size:0.84rem'>"
                "Click the button above to train the model on the current data slice"
                "</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — EXPORT (populated after charts rendered)
# ══════════════════════════════════════════════════════════════════════════════
with export_zone:
    if selected_outputs:
        st.markdown("<div class='sec-title'>Export</div>", unsafe_allow_html=True)
        parts = [f"{lbl}: {', '.join(v[:1])}" for lbl,v in list(active_filters.items())[:1]]
        if time_filter: parts.insert(0, f"{time_filter[0]}–{time_filter[1]}")
        report_title = " | ".join(parts) if parts else "HR Analytics Report"

        pptx_data = build_pptx(selected_outputs, report_title, insights_list=ai_insights)
        st.download_button("📊 Download PPTX",
                           data=pptx_data,
                           file_name="hr_analytics_report.pptx",
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
        try:
            xl_data = build_excel(df_filtered, col_map, ai_insights)
            st.download_button("📥 Download Excel",
                               data=xl_data,
                               file_name="hr_data_export.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception: pass
        st.caption(f"{len(selected_outputs)} chart{'s' if len(selected_outputs)!=1 else ''} ready to export")
