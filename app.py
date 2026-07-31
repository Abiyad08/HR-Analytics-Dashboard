"""HR Analytics Platform — universal, any business size or type."""
import io
import pandas as pd
import streamlit as st

from data_prep    import prepare
from analytics    import get_available
from insights     import generate as generate_insights
from predictive   import (train_model, feature_importance_chart,
                          risk_distribution_chart, risk_by_group_chart,
                          flight_risk_heatmap, top_at_risk_table)
from export_utils import fig_to_jpg, build_pptx, build_excel
from sample_data  import generate_sample_data

st.set_page_config(page_title="HR Analytics Platform", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*,html,body,[class*="css"]{font-family:'Inter',system-ui,sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#eef2f7}
.main .block-container{padding:1.4rem 2.2rem 3rem;max-width:1600px}
/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(170deg,#0b1d3a 0%,#0e2647 55%,#071422 100%)!important;border-right:1px solid rgba(255,255,255,0.07)}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] div,[data-testid="stSidebar"] label,[data-testid="stSidebar"] small{color:#c9d8e8!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong,[data-testid="stSidebar"] b{color:#eef4fb!important}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,0.10)!important}
[data-testid="stSidebar"] [data-baseweb="radio"] label p{color:#a8bfd4!important}
[data-testid="stSidebar"] [data-baseweb="select"]>div,[data-testid="stSidebar"] [data-baseweb="multi-select"]>div{background:rgba(255,255,255,0.10)!important;border-color:rgba(255,255,255,0.22)!important;border-radius:9px!important}
[data-testid="stSidebar"] [data-baseweb="select"] span,[data-testid="stSidebar"] [data-baseweb="multi-select"] span{color:#eef4fb!important}
[data-testid="stSidebar"] input{background:rgba(255,255,255,0.10)!important;border-color:rgba(255,255,255,0.22)!important;color:#eef4fb!important;border-radius:9px!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:rgba(255,255,255,0.07);border-radius:10px;padding:4px}
/* Header */
.dash-header{background:linear-gradient(128deg,#0b1d3a 0%,#1e40af 42%,#4338ca 76%,#6d28d9 100%);border-radius:18px;padding:28px 38px 24px;margin-bottom:22px;box-shadow:0 8px 32px rgba(30,64,175,0.28);position:relative;overflow:hidden}
.dash-header::before{content:"";position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:rgba(255,255,255,0.04);border-radius:50%}
.dash-header h1{color:#ffffff!important;font-size:1.8rem;font-weight:800;margin:0 0 6px;letter-spacing:-.025em}
.dash-header p{color:rgba(255,255,255,0.80)!important;font-size:0.875rem;margin:0;line-height:1.65}
/* KPI Cards */
.kpi-card{background:#ffffff;border-radius:14px;padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.07),0 4px 14px rgba(0,0,0,0.06);border:1px solid #dde4ee;transition:transform .16s,box-shadow .16s;min-height:96px}
.kpi-card:hover{transform:translateY(-2px);box-shadow:0 6px 22px rgba(0,0,0,0.11)}
.kpi-label{font-size:0.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#4b5563;margin-bottom:8px}
.kpi-value{font-size:1.65rem;font-weight:800;color:#0f172a!important;line-height:1.1;margin-bottom:4px}
.kpi-delta{font-size:0.73rem;font-weight:600;margin-top:2px}
.kpi-delta.good{color:#059669}.kpi-delta.bad{color:#dc2626}.kpi-delta.neu{color:#6b7280}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#ffffff;border-radius:14px;padding:5px 6px;gap:3px;box-shadow:0 1px 4px rgba(0,0,0,0.06);border:1px solid #dde4ee;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{border-radius:9px!important;padding:9px 17px!important;font-weight:600!important;font-size:0.80rem!important;color:#374151!important;background:transparent!important;border:none!important;transition:all .15s}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]){background:#f1f5f9!important;color:#1e40af!important}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1e40af,#4338ca)!important;color:#ffffff!important;font-weight:700!important;box-shadow:0 3px 10px rgba(30,64,175,0.30)!important}
/* Buttons */
.stButton>button{background:linear-gradient(135deg,#1e40af,#4338ca)!important;color:#ffffff!important;border:none!important;border-radius:10px!important;padding:10px 24px!important;font-weight:700!important;font-size:0.85rem!important;box-shadow:0 3px 12px rgba(30,64,175,0.28)!important;transition:all .18s!important}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(30,64,175,0.38)!important}
[data-testid="stDownloadButton"]>button{background:#ffffff!important;color:#1e40af!important;border:2px solid #1e40af!important;border-radius:8px!important;font-weight:700!important;font-size:0.79rem!important;padding:6px 14px!important;transition:all .14s!important}
[data-testid="stDownloadButton"]>button:hover{background:#eff6ff!important}
/* Chart card */
.chart-card{background:#ffffff;border-radius:16px;padding:20px 20px 14px;box-shadow:0 1px 4px rgba(0,0,0,0.06),0 4px 14px rgba(0,0,0,0.06);border:1px solid #dde4ee;margin-bottom:8px}
/* Insight pill */
.insight{background:#f0f7ff;border-left:4px solid #2563eb;border-radius:0 8px 8px 0;padding:10px 16px;margin-top:8px;font-size:0.82rem;color:#1e293b;line-height:1.6;font-style:italic}
/* AI insight cards */
.ai-card{background:#ffffff;border-radius:12px;padding:14px 18px;border:1px solid #dde4ee;box-shadow:0 1px 4px rgba(0,0,0,0.04);display:flex;gap:14px;align-items:flex-start;height:100%}
.ai-emoji{font-size:1.35rem;line-height:1;flex-shrink:0;margin-top:2px}
.ai-headline{font-weight:700;color:#0f172a;font-size:0.87rem;margin-bottom:3px}
.ai-detail{color:#374151;font-size:0.79rem;line-height:1.55}
/* Breadcrumb */
.breadcrumb{background:#ffffff;border-radius:10px;padding:10px 18px;margin-bottom:16px;font-size:0.82rem;color:#374151;border:1px solid #dde4ee;box-shadow:0 1px 3px rgba(0,0,0,0.04);display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.filter-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;background:#dbeafe;color:#1d4ed8;margin:1px}
.time-badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.72rem;font-weight:700;background:#d1fae5;color:#065f46;margin:1px}
/* Section labels */
.sec-label{font-size:0.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.10em;color:#94a3b8;margin:16px 0 6px}
/* Data health */
.health-bar{background:#e2e8f0;border-radius:99px;height:7px;overflow:hidden;margin:6px 0}
.health-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#059669,#34d399)}
.field-tag{display:inline-block;padding:3px 9px;border-radius:6px;font-size:0.70rem;font-weight:600;margin:2px}
.field-found{background:#d1fae5;color:#065f46}.field-missing{background:#fee2e2;color:#991b1b}
/* Compare cards */
.compare-card{background:#ffffff;border-radius:12px;padding:14px 16px;border:1px solid #dde4ee;text-align:center}
/* Misc */
[data-baseweb="tag"]{background:#dbeafe!important;border-color:#93c5fd!important;color:#1d4ed8!important;border-radius:6px!important}
[data-baseweb="tag"] span{color:#1d4ed8!important;font-weight:600}
[data-testid="stExpander"]{background:#ffffff!important;border-radius:12px!important;border:1px solid #dde4ee!important}
[data-testid="stExpander"] summary{font-weight:700!important;color:#0f172a!important;font-size:0.90rem!important}
.stAlert{border-radius:12px!important}
.stDataFrame{border-radius:12px!important;overflow:hidden!important}
.stCaption{color:#4b5563!important;font-size:0.80rem!important}
hr{border-color:#e2e8f0!important}
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

def _kpi_card(icon_label, value, delta=None):
    delta_html = ""
    if delta:
        cls = ("bad"  if any(x in str(delta) for x in ["▲","Above","↑","worsening"]) else
               "good" if any(x in str(delta) for x in ["✅","↓","Within","Below","improving"]) else "neu")
        delta_html = f"<div class='kpi-delta {cls}'>{delta}</div>"
    return (f"<div class='kpi-card'>"
            f"<div class='kpi-label'>{icon_label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"{delta_html}</div>")

def _render(fig, insight, title, cat, idx, outputs):
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar":True,"displaylogo":False,
        "modeBarButtonsToRemove":["select2d","lasso2d","autoScale2d"],
        "toImageButtonOptions":{"format":"png","filename":title,"scale":2},
    })
    if insight:
        st.markdown(f"<div class='insight'>💡 {insight}</div>", unsafe_allow_html=True)
    jpg = fig_to_jpg(fig)
    if jpg:
        st.download_button("⬇ JPG", data=jpg,
                           file_name=f"{title.replace(' ','_')[:40]}.jpg",
                           mime="image/jpeg",
                           key=f"dl_{cat}_{idx}_{abs(hash(title))%9999}")
    st.markdown("</div>", unsafe_allow_html=True)
    outputs.append({"title":title,"fig":fig,"insight":insight or ""})

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 4px 12px'>
      <div style='font-size:1.25rem;font-weight:800;color:#eef4fb;letter-spacing:-.02em'>📊 HR Analytics</div>
      <div style='font-size:0.71rem;color:#64748b;margin-top:3px'>People Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='sec-label'>Data Source</div>", unsafe_allow_html=True)
    source = st.radio("", ["📁 Upload File","🌐 Load from URL","🎲 Sample Data"],
                      index=2, label_visibility="collapsed")
    uploaded_file = data_url = None
    if source == "📁 Upload File":
        uploaded_file = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"],
                                         label_visibility="collapsed")
        st.caption("Any HR export — columns auto-detected.")
    elif source == "🌐 Load from URL":
        data_url = st.text_input("", placeholder="Paste .csv/.xlsx or Google Sheets link",
                                 label_visibility="collapsed")
        st.caption("Supports direct links & shared Google Sheets.")
    st.divider()
    filter_zone   = st.container()
    st.divider()
    settings_zone = st.container()
    st.divider()
    st.markdown("""
    <div style='font-size:0.74rem;color:#6b7280;line-height:2.0'>
      ✦ <b style='color:#94a3b8'>40 charts</b> across 6 HR categories<br>
      ✦ <b style='color:#94a3b8'>Auto-detects</b> any column structure<br>
      ✦ <b style='color:#94a3b8'>AI insights</b> auto-generated from data<br>
      ✦ <b style='color:#94a3b8'>Period comparison</b> — compare any two windows<br>
      ✦ <b style='color:#94a3b8'>Export</b> PPTX deck + Excel workbook
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.74rem;line-height:1.9'>
      <span style='color:#6b7280'>Built by</span>
      <b style='color:#e2e8f0'> Abiyad Islam</b><br>
      <span style='color:#4b5563'>Master of Business Analytics</span><br>
      <span style='color:#4b5563'>Macquarie University</span>
    </div>""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
raw_df = None
if source == "📁 Upload File" and uploaded_file:
    with st.spinner("Reading file…"):
        try: raw_df = _load_file(uploaded_file.read(), uploaded_file.name)
        except Exception as e: st.error(f"❌ {e}")
elif source == "🌐 Load from URL" and data_url:
    with st.spinner("Fetching data…"):
        try: raw_df = _load_url(data_url)
        except Exception as e: st.error(f"❌ {e}")
elif source == "🎲 Sample Data":
    with st.spinner("Generating sample dataset…"): raw_df = _sample()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>📊 HR Analytics Platform</h1>
  <p>Upload any employee dataset · Columns auto-detected · Works for any business ·
  40 interactive charts · AI insights · PPTX &amp; Excel export</p>
</div>""", unsafe_allow_html=True)

if raw_df is None:
    st.markdown("""
    <div style='background:#fff;border-radius:18px;padding:64px 40px;text-align:center;border:2px dashed #dde4ee'>
      <div style='font-size:3rem;margin-bottom:16px'>📂</div>
      <div style='font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:10px'>No data loaded yet</div>
      <div style='color:#374151;font-size:0.9rem;max-width:500px;margin:0 auto;line-height:1.7'>
        Upload a <b>CSV or Excel</b>, paste a <b>URL</b>, or try the built-in
        <b>sample dataset</b> (5,000 employees across 5 divisions).
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Prepare ───────────────────────────────────────────────────────────────────
with st.spinner("Analysing dataset…"):
    df_full, meta = prepare(raw_df)
col_map   = meta["col_map"]
hierarchy = meta["hierarchy"]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════════════════
df_filtered    = df_full.copy()
active_filters = {}
time_filter    = None
active_col     = col_map.get("_is_active","__is_active")

with filter_zone:
    st.markdown("<div class='sec-label'>Workforce Status</div>", unsafe_allow_html=True)
    status_opt = st.radio("", ["✅ Active Only","👥 All Records","📋 Former Only"],
                          index=1, label_visibility="collapsed")
    if "Active Only" in status_opt and active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[active_col]==1]
    elif "Former Only" in status_opt and active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[active_col]==0]

    hire_year_col = col_map.get("_hire_year")
    if hire_year_col and hire_year_col in df_filtered.columns:
        years = sorted(df_filtered[hire_year_col].dropna().astype(int).unique())
        if len(years) > 1:
            st.markdown("<div class='sec-label'>Time Frame — Hire Year</div>",
                        unsafe_allow_html=True)
            y_min, y_max = int(years[0]), int(years[-1])
            sel = st.slider("", y_min, y_max, (y_min, y_max),
                            key="yr_slider", label_visibility="collapsed")
            if sel != (y_min, y_max):
                df_filtered = df_filtered[df_filtered[hire_year_col].between(sel[0],sel[1])]
                time_filter = sel

    if hierarchy:
        st.markdown("<div class='sec-label'>Business Hierarchy</div>",
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
    st.markdown(f"<div style='font-size:0.75rem;color:#64748b;padding:8px 2px 0'>"
                f"<b style='color:#94a3b8'>{n_filt:,}</b> records match filters</div>",
                unsafe_allow_html=True)
    if (active_filters or time_filter) and st.button("✕ Clear All Filters", key="clear_all"):
        st.rerun()

# Settings (benchmark + column mapping)
with settings_zone:
    st.markdown("<div class='sec-label'>Settings</div>", unsafe_allow_html=True)
    custom_benchmark = st.slider("Attrition benchmark (%)", 5, 40, 15, 1,
                                 key="benchmark_slider")
    st.caption("Adjusts the benchmark line on all attrition charts.")
    with st.expander("🔧 Column Mapping Override", expanded=False):
        st.caption("Manually correct any auto-detection errors.")
        key_concepts_override = ["department","job_level","salary","attrition","gender",
                                 "hire_date","exit_date","tenure_years","location",
                                 "education","designation_main","grade","district",
                                 "employment_type","primary_leave_taken","secondary_leave_taken"]
        all_raw_cols = ["(auto)"] + sorted(raw_df.columns.tolist())
        for concept in key_concepts_override:
            current = col_map.get(concept,"")
            idx = all_raw_cols.index(current) if current in all_raw_cols else 0
            chosen_col = st.selectbox(concept.replace("_"," ").title(),
                                      options=all_raw_cols, index=idx,
                                      key=f"cmap_{concept}")
            if chosen_col != "(auto)":
                col_map[concept] = chosen_col

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

# Breadcrumb
filters_on = active_filters or time_filter or ("All" not in status_opt)
if filters_on:
    parts = []
    if "All" not in status_opt: parts.append(f"<span class='filter-badge'>{status_opt.split(' ',1)[1]}</span>")
    if time_filter: parts.append(f"<span class='time-badge'>📅 {time_filter[0]}–{time_filter[1]}</span>")
    for lbl, vals in active_filters.items():
        disp = vals[:3]; extra = f" +{len(vals)-3}" if len(vals)>3 else ""
        parts.append(f"<span class='filter-badge'>{lbl}: {', '.join(disp)}{extra}</span>")
    st.markdown(f"<div class='breadcrumb'>🔎 {'  '.join(parts)}"
                f"  &nbsp;|&nbsp; <b style='color:#0f172a'>{n_filt:,} employees</b></div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW — custom HTML (guaranteed dark text, no white-on-white)
# ══════════════════════════════════════════════════════════════════════════════
att_col = col_map.get("_attrition"); ten_col = col_map.get("_tenure_years")
gen_col = col_map.get("gender");     prob_col = col_map.get("_on_probation")
sal_col = next((col_map.get(c) for c in ["salary"] if col_map.get(c)), None)

cards = [_kpi_card("👥 Total Records", f"{len(df_filtered):,}")]
if active_col in df_filtered.columns:
    act = int((df_filtered[active_col]==1).sum())
    cards.append(_kpi_card("✅ Active", f"{act:,}", f"{act/max(len(df_filtered),1):.0%} of records"))
if att_col and att_col in df_filtered.columns:
    r = df_filtered[att_col].mean()
    cards.append(_kpi_card("🔄 Attrition", f"{r:.1%}",
                           f"▲ Above {custom_benchmark}% ref" if r>custom_benchmark/100 else f"✅ Within target"))
if ten_col and ten_col in df_filtered.columns:
    cards.append(_kpi_card("📅 Avg Tenure", f"{df_filtered[ten_col].mean():.1f} yrs"))
if gen_col and gen_col in df_filtered.columns:
    f_pct = (df_filtered[gen_col]=="Female").mean()
    cards.append(_kpi_card("👩 Female", f"{f_pct:.1%}",
                           "✅ At 30% target" if f_pct>=0.30 else "Below 30% target"))
if sal_col and sal_col in df_filtered.columns:
    cards.append(_kpi_card("💰 Median Pay", f"{df_filtered[sal_col].median():,.0f}"))

n_cards = min(len(cards), 6)
kpi_cols = st.columns(n_cards, gap="small")
for i, card_html in enumerate(cards[:n_cards]):
    with kpi_cols[i]:
        st.markdown(card_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-GENERATED INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
ai_insights = generate_insights(df_filtered, col_map)
if ai_insights:
    with st.expander(f"🧠 {len(ai_insights)} AI-Generated Insights", expanded=True):
        rows = [ai_insights[i:i+2] for i in range(0, len(ai_insights), 2)]
        for row in rows:
            ai_cols = st.columns(2, gap="medium")
            for j, (emoji, headline, detail) in enumerate(row):
                with ai_cols[j]:
                    st.markdown(
                        f"<div class='ai-card'>"
                        f"<div class='ai-emoji'>{emoji}</div>"
                        f"<div><div class='ai-headline'>{headline}</div>"
                        f"<div class='ai-detail'>{detail}</div></div>"
                        f"</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA HEALTH
# ══════════════════════════════════════════════════════════════════════════════
KEY_CONCEPTS = ["department","job_level","salary","attrition","performance_rating",
                "engagement_score","gender","hire_date","tenure_years","location",
                "recruitment_source","age","ethnicity","training_hours","absence_days",
                "designation_main","grade","district","education"]
found   = [c for c in KEY_CONCEPTS if col_map.get(c)]
missing = [c for c in KEY_CONCEPTS if not col_map.get(c)]
score   = len(found)/len(KEY_CONCEPTS)

with st.expander(f"📋 Data Health — {score:.0%} of key HR fields detected ({len(found)}/{len(KEY_CONCEPTS)})", expanded=False):
    st.markdown(f"<div class='health-bar'><div class='health-fill' style='width:{score*100:.0f}%'></div></div>",
                unsafe_allow_html=True)
    st.caption(f"Dataset: {len(raw_df):,} rows × {len(raw_df.columns)} cols  |  "
               f"Hierarchy levels: {len(hierarchy)}  |  "
               f"Derived fields: {sum(1 for k in col_map if k.startswith('_'))}")
    fh = "".join(f"<span class='field-tag field-found'>✓ {c.replace('_',' ')}</span>" for c in found)
    fm = "".join(f"<span class='field-tag field-missing'>✗ {c.replace('_',' ')}</span>" for c in missing)
    st.markdown(f"<div style='margin:10px 0 4px;font-weight:600;color:#0f172a'>Detected:</div>{fh}", unsafe_allow_html=True)
    if missing:
        st.markdown(f"<div style='margin:10px 0 4px;font-weight:600;color:#0f172a'>Not found (charts hidden):</div>{fm}", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
with st.expander(f"🗂 Data Preview — {len(df_filtered):,} rows × {len(df_filtered.columns)} columns", expanded=False):
    view_cols = [c for c in df_filtered.columns if not c.startswith("__")]
    st.dataframe(df_filtered[view_cols].head(50), use_container_width=True, height=280)
    with st.expander("Column statistics", expanded=False):
        stats = df_filtered[view_cols].describe(include="all").T.reset_index()
        stats.columns = ["Column"] + list(stats.columns[1:])
        st.dataframe(stats, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# PERIOD COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
hire_year_col_pc = col_map.get("_hire_year")
if hire_year_col_pc and hire_year_col_pc in df_filtered.columns:
    yrs = sorted(df_filtered[hire_year_col_pc].dropna().astype(int).unique())
    if len(yrs) >= 4:
        with st.expander("🔀 Period Comparison — compare two time windows side by side", expanded=False):
            st.caption("Select two hire-year ranges to compare key metrics between them.")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown("**🔵 Period A**")
                p_a = st.slider("Period A", yrs[0], yrs[-1], (yrs[0], yrs[len(yrs)//2]), key="pa")
            with pc2:
                st.markdown("**🟣 Period B**")
                p_b = st.slider("Period B", yrs[0], yrs[-1], (yrs[len(yrs)//2]+1, yrs[-1]), key="pb")
            pa_df = df_filtered[df_filtered[hire_year_col_pc].between(p_a[0],p_a[1])]
            pb_df = df_filtered[df_filtered[hire_year_col_pc].between(p_b[0],p_b[1])]

            def _cmp(da, db, label, fn):
                try: va,vb = fn(da), fn(db)
                except: return label,"—","—",None
                delta = round(vb-va,2) if isinstance(va,(int,float)) else None
                return label, va, vb, delta

            comp = [("Employees", len(pa_df), len(pb_df), None)]
            if att_col and att_col in df_filtered.columns:
                comp.append(_cmp(pa_df,pb_df,"Attrition %",lambda d:round(d[att_col].mean()*100,1)))
            if ten_col and ten_col in df_filtered.columns:
                comp.append(_cmp(pa_df,pb_df,"Avg Tenure",lambda d:round(d[ten_col].mean(),1)))
            if gen_col and gen_col in df_filtered.columns:
                comp.append(_cmp(pa_df,pb_df,"Female %",lambda d:round((d[gen_col]=="Female").mean()*100,1)))

            comp_cols = st.columns(len(comp))
            for i,(lbl,va,vb,delta) in enumerate(comp):
                with comp_cols[i]:
                    da = "bad" if delta and delta>0 and "Att" in lbl else "good" if delta and delta>0 else "neu"
                    d_str = f"{'▲' if delta and delta>0 else '▼'} {abs(delta)}" if delta else ""
                    st.markdown(
                        f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div>"
                        f"<div style='display:flex;gap:6px;align-items:baseline'>"
                        f"<span style='font-size:1.1rem;font-weight:700;color:#1e40af'>{va}</span>"
                        f"<span style='font-size:0.8rem;color:#94a3b8'>vs</span>"
                        f"<span style='font-size:1.1rem;font-weight:700;color:#7c3aed'>{vb}</span>"
                        f"</div><div class='kpi-delta {da}'>{d_str}</div></div>",
                        unsafe_allow_html=True)
            st.caption(f"🔵 A: {p_a[0]}–{p_a[1]} ({len(pa_df):,} employees)  "
                       f"🟣 B: {p_b[0]}–{p_b[1]} ({len(pb_df):,} employees)")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TABS
# ══════════════════════════════════════════════════════════════════════════════
available = get_available(df_filtered, col_map)
if not available:
    st.warning("⚠️ No analytics available. Try broadening your filters.")
    st.stop()

tab_labels   = list(available.keys()) + ["🔮 Predict"]
tabs         = st.tabs(tab_labels)
selected_outputs = []

for tab, (cat_name, charts) in zip(tabs[:-1], available.items()):
    with tab:
        chart_names = [c["name"] for c in charts]
        # Select All / Clear All / multiselect
        c_all, c_clr, c_sel, c_cnt = st.columns([1,1,6,1])
        with c_all:
            if st.button("All", key=f"all_{cat_name}", help="Select all"):
                st.session_state[f"ms_{cat_name}"] = chart_names; st.rerun()
        with c_clr:
            if st.button("Clear", key=f"clr_{cat_name}", help="Deselect all"):
                st.session_state[f"ms_{cat_name}"] = []; st.rerun()
        with c_sel:
            chosen = st.multiselect("", options=chart_names, default=chart_names[:3],
                                    key=f"ms_{cat_name}", placeholder="Choose charts…",
                                    label_visibility="collapsed")
        with c_cnt:
            st.markdown(f"<div style='padding:10px 0;font-size:0.78rem;color:#374151;text-align:right'>"
                        f"<b>{len(charts)}</b></div>", unsafe_allow_html=True)

        if not chosen:
            st.markdown("""<div style='text-align:center;padding:48px 24px;background:#fff;
                border-radius:16px;border:2px dashed #dde4ee;margin-top:8px'>
              <div style='font-size:2.5rem;margin-bottom:12px'>📊</div>
              <div style='font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:4px'>
                Select charts above</div>
              <div style='color:#374151;font-size:0.84rem'>Multiple charts can be compared side by side</div>
            </div>""", unsafe_allow_html=True)
            continue

        fn_lookup = {c["name"]:c for c in charts}
        col1, col2 = st.columns(2, gap="medium")
        for idx, name in enumerate(chosen):
            cd = fn_lookup[name]; fn = cd["fn"]
            with (col1 if idx%2==0 else col2):
                try:
                    with st.spinner(""):
                        if cd.get("needs_hierarchy"): fig,ins = fn(df_filtered,col_map,hierarchy)
                        elif cd.get("no_group"):      fig,ins = fn(df_filtered,col_map)
                        else:                         fig,ins = fn(df_filtered,col_map,group_col)
                    if fig is None:
                        st.info(f"Not enough data for '{name}' with current filters.")
                    else:
                        _render(fig,ins,name,cat_name,idx,selected_outputs)
                except Exception as e:
                    st.error(f"Error in '{name}': {e}")

# ── Predictive AI tab ─────────────────────────────────────────────────────────
with tabs[-1]:
    att_check = col_map.get("_attrition")
    st.markdown("""<div style='background:#fff;border-radius:16px;padding:22px 28px;
        border:1px solid #dde4ee;margin-bottom:20px'>
      <div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:8px'>
        🔮 Predictive Attrition Risk Model</div>
      <div style='font-size:0.875rem;color:#374155;line-height:1.8'>
        Trains a <b>Random Forest classifier</b> on the current filtered slice to predict
        each employee's probability of leaving. Respects all active filters — retrain after
        changing filters to compare risk across segments.
      </div></div>""", unsafe_allow_html=True)

    if not att_check or att_check not in df_filtered.columns:
        st.warning("⚠️ No attrition column detected. Model needs a Yes/No or 1/0 column.")
    elif len(df_filtered) < 50:
        st.warning("⚠️ Too few records. Broaden your filters.")
    else:
        if st.button("🚀 Train Attrition Risk Model", type="primary"):
            with st.spinner("Training model…"):
                result = train_model(df_filtered, col_map)
            if result:
                st.session_state.update({"pred_result":result,"pred_df":df_filtered.copy(),
                                         "pred_colmap":col_map,"pred_hier":hierarchy,
                                         "pred_grpcol":group_col})
                st.success(f"✅ Trained — {result['n_train']:,} samples, "
                           f"{result['n_features']} features, AUC {result['auc']:.3f}")
            else:
                st.error("Could not train — insufficient feature columns.")

        if "pred_result" in st.session_state:
            res=st.session_state["pred_result"]; pdf=st.session_state["pred_df"]
            risk=res["risk"]; hier2=st.session_state["pred_hier"]; g2=st.session_state["pred_grpcol"]

            p_cards = [
                _kpi_card("📐 Model AUC",     f"{res['auc']:.3f}", "1.0=perfect | 0.5=random"),
                _kpi_card("🚨 High-risk >50%", f"{(risk>0.5).sum():,}", f"{(risk>0.5).mean():.1%}"),
                _kpi_card("⚠️ Very high >70%", f"{(risk>0.7).sum():,}", f"{(risk>0.7).mean():.1%}"),
                _kpi_card("🔧 Features",        str(res["n_features"])),
            ]
            p_cols = st.columns(4, gap="small")
            for i,ch in enumerate(p_cards):
                with p_cols[i]: st.markdown(ch, unsafe_allow_html=True)
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

            c1,c2 = st.columns(2,gap="medium")
            with c1:
                fi,ii = feature_importance_chart(res["importances"])
                _render(fi,ii,"Attrition Risk Drivers","pred",10,selected_outputs)
            with c2:
                fr,ir = risk_distribution_chart(risk)
                _render(fr,ir,"Risk Score Distribution","pred",11,selected_outputs)
            c3,c4 = st.columns(2,gap="medium")
            with c3:
                if g2:
                    fg,ig = risk_by_group_chart(pdf,col_map,risk,g2)
                    if fg: _render(fg,ig,f"Avg Risk by {g2}","pred",12,selected_outputs)
            with c4:
                fh,ih = flight_risk_heatmap(pdf,col_map,risk,hier2)
                if fh: _render(fh,ih,"Flight Risk Heatmap","pred",13,selected_outputs)

            st.markdown("<div style='font-size:0.95rem;font-weight:700;color:#0f172a;margin:24px 0 10px'>"
                        "🚨 Top 20 Employees at Highest Predicted Risk</div>", unsafe_allow_html=True)
            am = pdf[col_map.get("_is_active","__is_active")]==1
            as_,ar = pdf[am], risk[am.values]
            if len(as_):
                tbl = top_at_risk_table(as_,col_map,ar,n=20)
                st.dataframe(tbl.style.background_gradient(subset=["⚠ Attrition Risk"],
                             cmap="RdYlGn_r"), use_container_width=True, height=560)
            else:
                st.info("No active employees in current filter slice.")
        else:
            st.markdown("""<div style='text-align:center;padding:64px 24px;background:#fff;
                border-radius:16px;border:2px dashed #dde4ee'>
              <div style='font-size:2.8rem;margin-bottom:16px'>🤖</div>
              <div style='font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:6px'>
                Ready to predict</div>
              <div style='color:#374151;font-size:0.85rem'>
                Set your filters, then click the button above
              </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:14px'>"
            "📥 Export Report</div>", unsafe_allow_html=True)
if selected_outputs:
    e1,e2,e3 = st.columns([2,2,4])
    with e1:
        parts = [f"{lbl}: {', '.join(v[:2])}" for lbl,v in list(active_filters.items())[:2]]
        if time_filter: parts.insert(0,f"{time_filter[0]}–{time_filter[1]}")
        report_title = " | ".join(parts) if parts else "HR Analytics Report"
        st.download_button("📊 Download PPTX",
                           data=build_pptx(selected_outputs,report_title,insights_list=ai_insights),
                           file_name="hr_analytics_report.pptx",
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           type="primary")
    with e2:
        try:
            xl = build_excel(df_filtered, col_map, ai_insights)
            st.download_button("📥 Download Excel", data=xl,
                               file_name="hr_data_export.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as ex:
            st.caption(f"Excel: {ex}")
    with e3:
        st.markdown(f"<div style='padding:10px 0;color:#374155;font-size:0.84rem'>"
                    f"📎 PPTX: <b>{len(selected_outputs)}</b> chart slides + editable summary slide. "
                    f"Excel: 3 sheets (data, KPIs, column map).</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='color:#94a3b8;font-size:0.88rem;padding:8px 0'>"
                "Select charts above to enable export.</div>", unsafe_allow_html=True)
