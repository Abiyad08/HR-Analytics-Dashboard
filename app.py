"""
HR Analytics Platform — universal, any business size or type.
"""
import io
import pandas as pd
import streamlit as st

from data_prep    import prepare
from analytics    import get_available
from insights     import generate as generate_insights
from predictive   import (train_model, feature_importance_chart,
                          risk_distribution_chart, risk_by_group_chart,
                          flight_risk_heatmap, top_at_risk_table)
from export_utils import fig_to_jpg, build_pptx
from sample_data  import generate_sample_data

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="HR Analytics Platform", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif!important; }
#MainMenu,footer,header{visibility:hidden}

/* ── APP BG ── */
.stApp{background:#eef2f7}
.main .block-container{padding:1.4rem 2.2rem 3rem;max-width:1600px}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
  background:linear-gradient(170deg,#0b1d3a 0%,#0e2647 55%,#071422 100%)!important;
  border-right:1px solid rgba(255,255,255,0.07)
}
/* Force ALL sidebar text to be readable on dark bg */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small{ color:#c9d8e8!important }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b{ color:#eef4fb!important }
[data-testid="stSidebar"] hr{ border-color:rgba(255,255,255,0.10)!important }
/* Radio buttons */
[data-testid="stSidebar"] [data-baseweb="radio"] label p{ color:#a8bfd4!important }
/* Selects */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="multi-select"] > div{
  background:rgba(255,255,255,0.10)!important;
  border-color:rgba(255,255,255,0.22)!important;
  border-radius:9px!important
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="multi-select"] span{ color:#eef4fb!important }
/* Text input */
[data-testid="stSidebar"] input{
  background:rgba(255,255,255,0.10)!important;
  border-color:rgba(255,255,255,0.22)!important;
  color:#eef4fb!important; border-radius:9px!important
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{
  background:rgba(255,255,255,0.07);border-radius:10px;padding:4px
}

/* ── PAGE HEADER ── */
.dash-header{
  background:linear-gradient(128deg,#0b1d3a 0%,#1e40af 42%,#4338ca 76%,#6d28d9 100%);
  border-radius:18px;padding:28px 38px 24px;margin-bottom:22px;
  box-shadow:0 8px 32px rgba(30,64,175,0.28);position:relative;overflow:hidden
}
.dash-header::before{content:"";position:absolute;top:-60px;right:-60px;
  width:220px;height:220px;background:rgba(255,255,255,0.04);border-radius:50%}
.dash-header h1{color:#ffffff!important;font-size:1.8rem;font-weight:800;margin:0 0 6px;letter-spacing:-.025em}
.dash-header p{color:rgba(255,255,255,0.80)!important;font-size:0.875rem;margin:0;line-height:1.65}

/* ── CUSTOM KPI CARDS (replaces st.metric so we control ALL colors) ── */
.kpi-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:20px}
.kpi-card{
  background:#ffffff;border-radius:14px;padding:18px 20px;
  box-shadow:0 1px 4px rgba(0,0,0,0.07),0 4px 14px rgba(0,0,0,0.06);
  border:1px solid #dde4ee;transition:transform .16s,box-shadow .16s;
}
.kpi-card:hover{transform:translateY(-2px);box-shadow:0 6px 22px rgba(0,0,0,0.11)}
.kpi-label{font-size:0.67rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.09em;color:#4b5563;margin-bottom:8px}
.kpi-value{font-size:1.75rem;font-weight:800;color:#0f172a;line-height:1.1;margin-bottom:4px}
.kpi-delta{font-size:0.73rem;font-weight:600;margin-top:2px}
.kpi-delta.good{color:#059669}
.kpi-delta.bad {color:#dc2626}
.kpi-delta.neu {color:#6b7280}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{
  background:#ffffff;border-radius:14px;padding:5px 6px;gap:3px;
  box-shadow:0 1px 4px rgba(0,0,0,0.06);border:1px solid #dde4ee;flex-wrap:wrap
}
.stTabs [data-baseweb="tab"]{
  border-radius:9px!important;padding:9px 17px!important;
  font-weight:600!important;font-size:0.81rem!important;
  color:#374151!important;background:transparent!important;
  border:none!important;transition:all .14s
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]){
  background:#f1f5f9!important;color:#1e40af!important
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#1e40af,#4338ca)!important;
  color:#ffffff!important;font-weight:700!important;
  box-shadow:0 3px 10px rgba(30,64,175,0.30)!important
}
/* Tab panel background & text */
.stTabs [data-baseweb="tab-panel"]{background:transparent!important}

/* ── BUTTONS ── */
.stButton>button{
  background:linear-gradient(135deg,#1e40af,#4338ca)!important;
  color:#ffffff!important;border:none!important;border-radius:10px!important;
  padding:10px 24px!important;font-weight:700!important;font-size:0.85rem!important;
  box-shadow:0 3px 12px rgba(30,64,175,0.28)!important;transition:all .18s!important
}
.stButton>button:hover{transform:translateY(-1px)!important;
  box-shadow:0 6px 20px rgba(30,64,175,0.38)!important}
[data-testid="stDownloadButton"]>button{
  background:#ffffff!important;color:#1e40af!important;
  border:2px solid #1e40af!important;border-radius:8px!important;
  font-weight:700!important;font-size:0.79rem!important;
  padding:6px 14px!important;transition:all .14s!important
}
[data-testid="stDownloadButton"]>button:hover{background:#eff6ff!important}

/* ── CHART CARD ── */
.chart-card{
  background:#ffffff;border-radius:16px;padding:20px 20px 14px;
  box-shadow:0 1px 4px rgba(0,0,0,0.06),0 4px 14px rgba(0,0,0,0.06);
  border:1px solid #dde4ee;margin-bottom:8px
}

/* ── INSIGHT PILL ── */
.insight{
  background:#f0f7ff;border-left:4px solid #2563eb;
  border-radius:0 8px 8px 0;padding:10px 16px;margin-top:8px;
  font-size:0.82rem;color:#1e293b;line-height:1.6;font-style:italic
}

/* ── AUTO-INSIGHT CARDS ── */
.ai-card{
  background:#ffffff;border-radius:12px;padding:14px 18px;
  border:1px solid #dde4ee;box-shadow:0 1px 4px rgba(0,0,0,0.04);
  display:flex;gap:14px;align-items:flex-start;margin-bottom:0;height:100%
}
.ai-emoji{font-size:1.35rem;line-height:1;flex-shrink:0;margin-top:2px}
.ai-headline{font-weight:700;color:#0f172a;font-size:0.87rem;margin-bottom:3px}
.ai-detail{color:#374151;font-size:0.79rem;line-height:1.55}

/* ── BREADCRUMB ── */
.breadcrumb{
  background:#ffffff;border-radius:10px;padding:10px 18px;margin-bottom:16px;
  font-size:0.82rem;color:#374151;border:1px solid #dde4ee;
  box-shadow:0 1px 3px rgba(0,0,0,0.04);display:flex;align-items:center;
  gap:6px;flex-wrap:wrap
}
.filter-badge{display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:0.72rem;font-weight:700;background:#dbeafe;color:#1d4ed8;margin:1px}
.time-badge{display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:0.72rem;font-weight:700;background:#d1fae5;color:#065f46;margin:1px}

/* ── SECTION LABELS ── */
.sec-label{font-size:0.66rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.10em;color:#94a3b8;margin:16px 0 6px}

/* ── DATA HEALTH ── */
.health-bar{background:#e2e8f0;border-radius:99px;height:7px;overflow:hidden;margin:6px 0}
.health-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#059669,#34d399)}
.field-tag{display:inline-block;padding:3px 9px;border-radius:6px;
  font-size:0.70rem;font-weight:600;margin:2px 2px}
.field-found{background:#d1fae5;color:#065f46}
.field-missing{background:#fee2e2;color:#991b1b}

/* ── MULTISELECT TAGS ── */
[data-baseweb="tag"]{background:#dbeafe!important;border-color:#93c5fd!important;
  color:#1d4ed8!important;border-radius:6px!important}
[data-baseweb="tag"] span{color:#1d4ed8!important;font-weight:600}

/* ── EXPANDER ── */
[data-testid="stExpander"]{background:#ffffff!important;border-radius:12px!important;
  border:1px solid #dde4ee!important}
[data-testid="stExpander"] summary{font-weight:700!important;color:#0f172a!important;
  font-size:0.90rem!important}
[data-testid="stExpander"] summary p{color:#0f172a!important}

/* ── MISC ── */
.stAlert{border-radius:12px!important}
.stDataFrame{border-radius:12px!important;overflow:hidden!important}
.stCaption{color:#374151!important;font-size:0.80rem!important}
.stProgress>div>div{background:linear-gradient(90deg,#1e40af,#4338ca)!important;border-radius:4px!important}
hr{border-color:#e2e8f0!important}
/* Force all main-area text to be dark */
.main p,.main span:not([class*="badge"]):not([class*="pill"]):not([class*="tag"]),
.main div:not([class*="card"]):not([class*="header"]):not([class*="sidebar"]):not([class*="badge"]){
  color:#1e293b
}
</style>
""", unsafe_allow_html=True)

# ── Data loading helpers ───────────────────────────────────────────────────────
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

# ── Custom KPI card (no Streamlit metric = no white-on-white) ─────────────────
def _kpi_card(icon_label, value, delta=None):
    delta_html = ""
    if delta:
        cls = "bad" if any(x in delta for x in ["▲","Above","↑"]) \
              else "good" if any(x in delta for x in ["✅","↓","Within","Below"]) \
              else "neu"
        delta_html = f"<div class='kpi-delta {cls}'>{delta}</div>"
    return (f"<div class='kpi-card'>"
            f"<div class='kpi-label'>{icon_label}</div>"
            f"<div class='kpi-value'>{value}</div>"
            f"{delta_html}</div>")

def _render(fig, insight, title, cat, idx, outputs):
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True, "displaylogo": False,
        "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d"],
        "toImageButtonOptions": {"format":"png","filename":title,"scale":2}
    })
    if insight:
        st.markdown(f"<div class='insight'>💡 {insight}</div>", unsafe_allow_html=True)
    jpg = fig_to_jpg(fig)
    if jpg:
        st.download_button("⬇ Download JPG", data=jpg,
                           file_name=f"{title.replace(' ','_')[:40]}.jpg",
                           mime="image/jpeg",
                           key=f"dl_{cat}_{idx}_{abs(hash(title))%9999}")
    st.markdown("</div>", unsafe_allow_html=True)
    outputs.append({"title": title, "fig": fig, "insight": insight or ""})

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 4px 12px'>
      <div style='font-size:1.25rem;font-weight:800;color:#eef4fb;letter-spacing:-.02em'>
        📊 HR Analytics
      </div>
      <div style='font-size:0.71rem;color:#64748b;margin-top:3px'>
        People Intelligence Platform
      </div>
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
        st.caption("Direct links & shared Google Sheets supported.")

    st.divider()
    filter_zone = st.container()
    st.divider()

    st.markdown("""
    <div style='font-size:0.74rem;color:#6b7280;line-height:2.0'>
      ✦ <b style='color:#94a3b8'>Auto-detects</b> any column structure<br>
      ✦ <b style='color:#94a3b8'>33+ charts</b> across 7 HR categories<br>
      ✦ <b style='color:#94a3b8'>AI insights</b> auto-generated from data<br>
      ✦ <b style='color:#94a3b8'>Cascading filters</b> for any hierarchy<br>
      ✦ <b style='color:#94a3b8'>Time range</b> slider — any period<br>
      ✦ <b style='color:#94a3b8'>Export</b> JPG per chart or full PPTX deck
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("""
    <div style='font-size:0.74rem;line-height:1.9'>
      <span style='color:#6b7280'>Built by</span>
      <b style='color:#e2e8f0'> Abiyad Islam</b><br>
      <span style='color:#4b5563'>Master of Business Analytics</span><br>
      <span style='color:#4b5563'>Macquarie University</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
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

# ── Header (always visible) ───────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <h1>📊 HR Analytics Platform</h1>
  <p>Upload any employee dataset &nbsp;·&nbsp; Columns auto-detected — works for any business
  &nbsp;·&nbsp; 33+ interactive charts &nbsp;·&nbsp; AI-generated insights
  &nbsp;·&nbsp; Export to JPG &amp; PPTX</p>
</div>""", unsafe_allow_html=True)

if raw_df is None:
    st.markdown("""
    <div style='background:#fff;border-radius:18px;padding:64px 40px;text-align:center;
                border:2px dashed #dde4ee'>
      <div style='font-size:3rem;margin-bottom:16px'>📂</div>
      <div style='font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:10px'>
        No data loaded yet</div>
      <div style='color:#374151;font-size:0.9rem;max-width:500px;margin:0 auto;line-height:1.7'>
        Choose a data source in the sidebar — upload a <b>CSV or Excel</b>, paste a <b>URL</b>,
        or try the built-in <b>sample dataset</b> (5,000 employees, 5 divisions).
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PREPARE
# ══════════════════════════════════════════════════════════════════════════════
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
    # Status
    st.markdown("<div class='sec-label'>Workforce Status</div>", unsafe_allow_html=True)
    status_opt = st.radio("", ["✅ Active Only","👥 All Records","📋 Former Only"],
                          index=1, label_visibility="collapsed")
    if "Active Only" in status_opt and active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[active_col]==1]
    elif "Former Only" in status_opt and active_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[active_col]==0]

    # Time frame
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

    # Hierarchy filters
    if hierarchy:
        st.markdown("<div class='sec-label'>Business Hierarchy</div>", unsafe_allow_html=True)
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
    st.markdown(f"<div style='font-size:0.75rem;color:#6b7280;padding:8px 2px 0'>"
                f"<b style='color:#94a3b8'>{n_filt:,}</b> records match filters</div>",
                unsafe_allow_html=True)
    has_active = bool(active_filters or time_filter)
    if has_active and st.button("✕ Clear All Filters", key="clear_all"):
        st.rerun()

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
    if time_filter:             parts.append(f"<span class='time-badge'>📅 {time_filter[0]}–{time_filter[1]}</span>")
    for lbl, vals in active_filters.items():
        disp  = vals[:3]; extra = f" +{len(vals)-3}" if len(vals)>3 else ""
        parts.append(f"<span class='filter-badge'>{lbl}: {', '.join(disp)}{extra}</span>")
    st.markdown(f"<div class='breadcrumb'>🔎 {'  '.join(parts)}"
                f"  &nbsp;|&nbsp; <b style='color:#0f172a'>{n_filt:,} employees</b></div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW — custom HTML, dark text guaranteed
# ══════════════════════════════════════════════════════════════════════════════
att_col = col_map.get("_attrition")
ten_col = col_map.get("_tenure_years")
gen_col = col_map.get("gender")
prob_col= col_map.get("_on_probation")
sal_col = next((col_map.get(c) for c in ["salary"] if col_map.get(c)), None)
act_df  = df_filtered[df_filtered[active_col]==1] if active_col in df_filtered.columns else df_filtered

cards = []
cards.append(_kpi_card("👥 Total Records", f"{len(df_filtered):,}"))

if active_col in df_filtered.columns:
    act = int((df_filtered[active_col]==1).sum())
    cards.append(_kpi_card("✅ Active Employees", f"{act:,}",
                           f"{act/max(len(df_filtered),1):.0%} of records"))

if att_col and att_col in df_filtered.columns:
    r = df_filtered[att_col].mean()
    cards.append(_kpi_card("🔄 Attrition Rate", f"{r:.1%}",
                           f"▲ Above 15% ref" if r>0.15 else f"✅ Within target"))

if ten_col and ten_col in df_filtered.columns:
    cards.append(_kpi_card("📅 Avg Tenure", f"{df_filtered[ten_col].mean():.1f} yrs"))

if gen_col and gen_col in df_filtered.columns:
    f_pct = (df_filtered[gen_col]=="Female").mean()
    cards.append(_kpi_card("👩 Female Representation", f"{f_pct:.1%}",
                           "✅ At 30% target" if f_pct>=0.30 else "Below 30% target"))

if sal_col and sal_col in df_filtered.columns:
    cards.append(_kpi_card("💰 Median Salary", f"${df_filtered[sal_col].median():,.0f}"))

n_cards   = min(len(cards), 6)
col_widths = [1]*n_cards
cols_kpi  = st.columns(col_widths)
for i, card_html in enumerate(cards[:n_cards]):
    with cols_kpi[i]:
        st.markdown(card_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-GENERATED INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
ai_insights = generate_insights(df_filtered, col_map)
if ai_insights:
    with st.expander(f"🧠 {len(ai_insights)} AI-Generated Insights — click to read", expanded=True):
        n_cols = 2
        rows   = [ai_insights[i:i+n_cols] for i in range(0, len(ai_insights), n_cols)]
        for row in rows:
            ai_cols = st.columns(n_cols, gap="medium")
            for j, (emoji, headline, detail) in enumerate(row):
                with ai_cols[j]:
                    st.markdown(
                        f"<div class='ai-card'>"
                        f"<div class='ai-emoji'>{emoji}</div>"
                        f"<div><div class='ai-headline'>{headline}</div>"
                        f"<div class='ai-detail'>{detail}</div></div>"
                        f"</div>", unsafe_allow_html=True
                    )

# ══════════════════════════════════════════════════════════════════════════════
# DATA HEALTH PANEL
# ══════════════════════════════════════════════════════════════════════════════
KEY_CONCEPTS = ["department","job_level","salary","attrition","performance_rating",
                "engagement_score","gender","hire_date","tenure_years","location",
                "recruitment_source","age","ethnicity","training_hours","absence_days",
                "designation_main","grade","district","education"]
found   = [c for c in KEY_CONCEPTS if col_map.get(c)]
missing = [c for c in KEY_CONCEPTS if not col_map.get(c)]
score   = len(found)/len(KEY_CONCEPTS)

with st.expander(f"📋 Data Health — {score:.0%} of key HR fields detected ({len(found)}/{len(KEY_CONCEPTS)})",
                 expanded=False):
    st.markdown(f"<div class='health-bar'>"
                f"<div class='health-fill' style='width:{score*100:.0f}%'></div></div>",
                unsafe_allow_html=True)
    st.caption(f"Dataset: {len(raw_df):,} rows × {len(raw_df.columns)} cols  |  "
               f"Hierarchy levels: {len(hierarchy)}  |  "
               f"Derived fields: {sum(1 for k in col_map if k.startswith('_'))}")
    found_html   = "".join(f"<span class='field-tag field-found'>✓ {c.replace('_',' ')}</span>" for c in found)
    missing_html = "".join(f"<span class='field-tag field-missing'>✗ {c.replace('_',' ')}</span>" for c in missing)
    st.markdown(f"<div style='margin:10px 0 4px;font-weight:600;color:#0f172a'>Detected:</div>{found_html}",
                unsafe_allow_html=True)
    if missing:
        st.markdown(f"<div style='margin:10px 0 4px;font-weight:600;color:#0f172a'>"
                    f"Not found (related charts are hidden):</div>{missing_html}",
                    unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TABS
# ══════════════════════════════════════════════════════════════════════════════
available = get_available(df_filtered, col_map)
if not available:
    st.warning("⚠️ No analytics available for this filter slice. Try broadening your selection.")
    st.stop()

tab_labels   = list(available.keys()) + ["🔮 Predictive AI"]
tabs         = st.tabs(tab_labels)
selected_outputs = []

for tab, (cat_name, charts) in zip(tabs[:-1], available.items()):
    with tab:
        chart_names = [c["name"] for c in charts]
        c_sel, c_info = st.columns([4,1])
        with c_sel:
            chosen = st.multiselect("Select charts to display:", options=chart_names,
                                    default=chart_names[:3], key=f"ms_{cat_name}",
                                    placeholder="Choose charts…")
        with c_info:
            st.markdown(f"<div style='padding:10px 0;font-size:0.79rem;color:#374151'>"
                        f"<b>{len(charts)}</b> available</div>", unsafe_allow_html=True)

        if not chosen:
            st.markdown("""
            <div style='text-align:center;padding:52px 24px;background:#fff;
                        border-radius:16px;border:2px dashed #dde4ee;margin-top:8px'>
              <div style='font-size:2.5rem;margin-bottom:12px'>📊</div>
              <div style='font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:4px'>
                Select charts above</div>
              <div style='color:#374151;font-size:0.84rem'>
                Multiple charts can be selected and compared side by side</div>
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
                        _render(fig, ins, name, cat_name, idx, selected_outputs)
                except Exception as e:
                    st.error(f"Error in '{name}': {e}")

# ── Predictive AI tab ─────────────────────────────────────────────────────────
with tabs[-1]:
    att_check = col_map.get("_attrition")
    st.markdown("""
    <div style='background:#fff;border-radius:16px;padding:22px 28px;
                border:1px solid #dde4ee;margin-bottom:20px'>
      <div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:8px'>
        🔮 Predictive Attrition Risk Model</div>
      <div style='font-size:0.875rem;color:#374151;line-height:1.8'>
        Trains a <b>Random Forest classifier</b> on your current filtered data to predict
        each employee's probability of leaving. All active filters are respected —
        retrain after changing filters to compare risk across segments.
      </div>
    </div>""", unsafe_allow_html=True)

    if not att_check or att_check not in df_filtered.columns:
        st.warning("⚠️ No attrition column detected. The model needs a column showing "
                   "whether each employee has left — e.g. Yes/No, Y/N, 1/0.")
    elif len(df_filtered)<50:
        st.warning("⚠️ Too few records in this slice. Broaden your filters.")
    else:
        if st.button("🚀 Train Attrition Risk Model", type="primary"):
            with st.spinner("Training model…"):
                result = train_model(df_filtered, col_map)
            if result:
                st.session_state.update({"pred_result":result,"pred_df":df_filtered.copy(),
                                         "pred_colmap":col_map,"pred_hier":hierarchy,
                                         "pred_grpcol":group_col})
                st.success(f"✅ Model trained — {result['n_train']:,} samples, "
                           f"{result['n_features']} features, AUC {result['auc']:.3f}")
            else:
                st.error("Could not train model — insufficient feature columns.")

        if "pred_result" in st.session_state:
            result=st.session_state["pred_result"]; pred_df=st.session_state["pred_df"]
            risk=result["risk"]; hier=st.session_state["pred_hier"]; g_col=st.session_state["pred_grpcol"]

            # Metric row for predictive — also custom HTML
            pred_cards = [
                _kpi_card("📐 Model AUC",   f"{result['auc']:.3f}", "1.0=perfect | 0.5=random"),
                _kpi_card("🚨 High-risk >50%", f"{(risk>0.5).sum():,}", f"{(risk>0.5).mean():.1%} of workforce"),
                _kpi_card("⚠️ Very high >70%", f"{(risk>0.7).sum():,}", f"{(risk>0.7).mean():.1%} of workforce"),
                _kpi_card("🔧 Features used", str(result["n_features"])),
            ]
            pred_cols = st.columns(4)
            for i,ch in enumerate(pred_cards):
                with pred_cols[i]: st.markdown(ch, unsafe_allow_html=True)
            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

            c1,c2 = st.columns(2,gap="medium")
            with c1:
                fi,ii = feature_importance_chart(result["importances"])
                _render(fi,ii,"Attrition Risk Drivers","pred",10,selected_outputs)
            with c2:
                fr,ir = risk_distribution_chart(risk)
                _render(fr,ir,"Risk Score Distribution","pred",11,selected_outputs)
            c3,c4 = st.columns(2,gap="medium")
            with c3:
                if g_col:
                    fg,ig = risk_by_group_chart(pred_df,col_map,risk,g_col)
                    if fg: _render(fg,ig,f"Avg Risk by {g_col}","pred",12,selected_outputs)
            with c4:
                fh,ih = flight_risk_heatmap(pred_df,col_map,risk,hier)
                if fh: _render(fh,ih,"Flight Risk Heatmap","pred",13,selected_outputs)

            st.markdown("<div style='font-size:0.95rem;font-weight:700;color:#0f172a;"
                        "margin:24px 0 10px'>🚨 Top 20 Employees at Highest Predicted Risk</div>",
                        unsafe_allow_html=True)
            am = pred_df[col_map.get("_is_active","__is_active")]==1
            as_ = pred_df[am]; ar = risk[am.values]
            if len(as_):
                tbl = top_at_risk_table(as_,col_map,ar,n=20)
                st.dataframe(tbl.style.background_gradient(subset=["⚠ Attrition Risk"],
                             cmap="RdYlGn_r"), use_container_width=True, height=560)
            else:
                st.info("No active employees in current filter slice.")
        else:
            st.markdown("""
            <div style='text-align:center;padding:64px 24px;background:#fff;
                        border-radius:16px;border:2px dashed #dde4ee'>
              <div style='font-size:2.8rem;margin-bottom:16px'>🤖</div>
              <div style='font-size:0.95rem;font-weight:700;color:#0f172a;margin-bottom:6px'>
                Ready to predict</div>
              <div style='color:#374151;font-size:0.85rem'>
                Set your filters above, then click the button to train
              </div>
            </div>""", unsafe_allow_html=True)

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
        st.download_button("📊 Download PPTX Report",
                           data=build_pptx(selected_outputs,report_title),
                           file_name="hr_analytics_report.pptx",
                           mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                           type="primary")
    with e2:
        st.markdown(f"<div style='padding:10px 0;color:#374151;font-size:0.85rem'>"
                    f"📎 <b>{len(selected_outputs)}</b> chart"
                    f"{'s' if len(selected_outputs)!=1 else ''} in report</div>",
                    unsafe_allow_html=True)
    with e3:
        st.markdown("<div style='padding:10px 0;color:#64748b;font-size:0.82rem'>"
                    "Each chart also has its own ⬇ JPG button below it.</div>",
                    unsafe_allow_html=True)
else:
    st.markdown("<div style='color:#64748b;font-size:0.88rem;padding:8px 0'>"
                "Select charts in the tabs above to enable export.</div>",
                unsafe_allow_html=True)
