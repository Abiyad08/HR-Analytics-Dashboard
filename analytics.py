"""
analytics.py  —  55 Plotly charts covering:
  Conglomerate Overview · Workforce · People Flow & Attrition ·
  Demographics & Diversity · Leave Management · Career & Grades · Predictive
All charts accept (df, col_map, group_col) where group_col is the current
hierarchy level to group/colour by (Division, BU, DeptGroup, Dept).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design system ──────────────────────────────────────────────────────────
COLORS = ["#1D4ED8","#6D28D9","#BE185D","#15803D","#C2410C","#0E7490",
          "#B45309","#B91C1C","#0F766E","#7C3AED","#374151","#92400E"]
C = {"blue":"#1D4ED8","purple":"#6D28D9","pink":"#BE185D","green":"#15803D",
     "orange":"#C2410C","teal":"#0E7490","red":"#B91C1C","amber":"#B45309"}

def _L(title="", h=420):
    return dict(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=15,color="#0f172a",family="Inter"),x=0,pad=dict(l=4)),
        font=dict(family="Inter,system-ui,sans-serif",size=12,color="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8fafc",
        height=h, margin=dict(l=10,r=10,t=56,b=10,pad=2),
        colorway=COLORS,
        showlegend=True,
        hoverlabel=dict(bgcolor="white",bordercolor="#e2e8f0",
                        font_family="Inter,sans-serif",font_size=12,namelength=-1),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",bordercolor="#cbd5e1",
            borderwidth=1,font=dict(size=12,color="#1e293b"),
            orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
            itemsizing="constant",itemwidth=30,
        ),
        xaxis=dict(showgrid=True,gridcolor="#e8ecf0",linecolor="#cbd5e1",
                   tickfont=dict(size=12,color="#334155"),title_font=dict(size=13,color="#1e293b")),
        yaxis=dict(showgrid=True,gridcolor="#e8ecf0",linecolor="rgba(0,0,0,0)",
                   tickfont=dict(size=12,color="#334155"),title_font=dict(size=13,color="#1e293b"),
                   zeroline=False),
    )

def _safe(df, col):
    return df[col] if col and col in df.columns else None

def _g(col_map, *concepts):
    for c in concepts:
        v = col_map.get(c)
        if v: return v
    return None

def _top_n(series, n=10):
    return series.value_counts().head(n).index.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# 1 · CONGLOMERATE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def headcount_sunburst(df, col_map, hierarchy):
    """Full hierarchy sunburst — Division > BU > DeptGroup > Dept."""
    path = [h[1] for h in hierarchy if h[1] in df.columns]
    if not path: return None, ""
    data = df.dropna(subset=[path[0]])
    fig = px.sunburst(data, path=path, color=path[0],
                      color_discrete_sequence=COLORS)
    fig.update_traces(textinfo="label+percent parent",
                      hovertemplate="<b>%{label}</b><br>%{value} employees (%{percentParent:.0%} of parent)<extra></extra>")
    fig.update_layout(**_L("Workforce Sunburst — Full Business Hierarchy", 520))
    top = data[path[0]].value_counts().idxmax()
    return fig, f"{top} is the largest division ({data[path[0]].value_counts().max()} employees)."

def conglomerate_heatmap(df, col_map, group_col):
    """Division × metric heatmap comparing key indicators."""
    if not group_col or group_col not in df.columns: return None, ""
    att = col_map.get("_attrition"); ten = col_map.get("_tenure_years")
    gen = _g(col_map,"gender"); prob = col_map.get("_on_probation")
    metrics, labels = [], []
    if att:
        r = df.groupby(group_col)[att].mean()*100
        metrics.append(r); labels.append("Attrition %")
    if ten:
        r = df.groupby(group_col)[ten].mean()
        metrics.append(r); labels.append("Avg Tenure (yrs)")
    if gen:
        r = df.groupby(group_col)[gen].apply(lambda x: (x=="Female").mean()*100)
        metrics.append(r); labels.append("Female %")
    if prob:
        r = df.groupby(group_col)[prob].mean()*100
        metrics.append(r); labels.append("On Probation %")
    if not metrics: return None, ""
    pivot = pd.DataFrame({l: m for l,m in zip(labels,metrics)}).fillna(0)
    norm = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-9)
    fig = go.Figure(go.Heatmap(
        z=norm.values.T, x=pivot.index.astype(str), y=labels,
        text=[[f"{pivot[l][r]:.1f}" for r in pivot.index] for l in labels],
        texttemplate="%{text}", colorscale="RdYlGn", showscale=False,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(**_L(f"Conglomerate Scorecard by {group_col}", h=max(300, len(labels)*70+100)))
    return fig, "Green = better performance; red = attention needed. Normalized across all business units for comparability."

def headcount_bar_comparison(df, col_map, group_col):
    if not group_col or group_col not in df.columns: return None, ""
    active = col_map.get("_is_active","__is_active")
    if active in df.columns:
        grp = df.groupby(group_col)[active].agg(["sum","count"]).reset_index()
        grp.columns = [group_col,"Active","Total"]
        grp["Left"] = grp["Total"] - grp["Active"]
        grp = grp.sort_values("Total", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=grp[group_col],x=grp["Active"],orientation="h",
                             name="Active",marker_color=C["green"],marker_line_width=0))
        fig.add_trace(go.Bar(y=grp[group_col],x=grp["Left"],orientation="h",
                             name="Left",marker_color="#fca5a5",marker_line_width=0))
        fig.update_layout(**_L(f"Headcount by {group_col} — Active vs Former"),
                          barmode="stack",yaxis_title="",xaxis_title="Employees")
        total = grp["Total"].sum()
        return fig, f"Total workforce (all time): {total:,}. Active: {grp['Active'].sum():,} | Former: {grp['Left'].sum():,}."
    counts = df[group_col].value_counts().sort_values(ascending=True)
    fig = px.bar(x=counts.values,y=counts.index,orientation="h",
                 color=counts.values,color_continuous_scale=[[0,"#bfdbfe"],[1,C["blue"]]],
                 text=counts.values)
    fig.update_traces(textposition="outside",marker_line_width=0)
    fig.update_layout(**_L(f"Headcount by {group_col}"),coloraxis_showscale=False)
    return fig, f"Largest group: {counts.idxmax()} ({counts.max():,})."

def employment_type_mix(df, col_map, group_col):
    emp = _g(col_map,"employment_type")
    if not emp or emp not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col], df[emp], normalize="index")*100
        ct = ct.reindex(df[group_col].value_counts().head(12).index)
        fig = px.bar(ct.reset_index(), x=group_col,
                     y=[c for c in ct.columns], barmode="stack",
                     color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Employment Type Mix by {group_col}"),
                          yaxis_title="% of Workforce",xaxis_title="")
        perm = (df[emp].str.upper()=="PERMANENT").mean()*100
        return fig, f"{perm:.0f}% of all records are permanent employees."
    counts = df[emp].value_counts()
    fig = px.pie(values=counts.values,names=counts.index,hole=0.55,
                 color_discrete_sequence=COLORS)
    fig.update_layout(**_L("Employment Type Distribution",380),showlegend=True)
    return fig, f"Largest category: {counts.idxmax()} ({counts.max():,} employees)."

def management_level_comparison(df, col_map, group_col):
    mgmt = _g(col_map,"designation_main")
    if not mgmt or mgmt not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col], df[mgmt], normalize="index")*100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(),x=group_col,y=list(ct.columns),
                     barmode="stack",color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Management Level Mix by {group_col}"),
                          yaxis_title="% of Employees",xaxis_title="")
        return fig, "A healthy hierarchy has a wide base of Executives tapering up through Managers to Directors."
    counts = df[mgmt].value_counts().sort_values()
    fig = px.funnel(x=counts.values,y=counts.index,color_discrete_sequence=[C["purple"]])
    fig.update_layout(**_L("Management Hierarchy Funnel",380))
    return fig, f"Ratio of Executives to Managers: {counts.get('Executive / Officer',0):,} : {counts.get('Manager',0):,}."

# ══════════════════════════════════════════════════════════════════════════════
# 2 · WORKFORCE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def org_treemap(df, col_map, hierarchy):
    path = [h[1] for h in hierarchy[:3] if h[1] in df.columns]
    if not path: return None, ""
    fig = px.treemap(df.dropna(subset=[path[0]]),path=path,color=path[0],
                     color_discrete_sequence=COLORS)
    fig.update_traces(textinfo="label+value+percent parent",
                      hovertemplate="<b>%{label}</b><br>%{value} employees<extra></extra>")
    fig.update_layout(**_L("Organisational Structure Treemap",520))
    return fig, "Size = headcount. Click any block to drill deeper into that business unit."

def designation_treemap(df, col_map, group_col):
    desg = _g(col_map,"designation"); mgmt = _g(col_map,"designation_main")
    if not desg or desg not in df.columns: return None, ""
    path = [p for p in [group_col, mgmt, desg] if p and p in df.columns]
    if len(path) < 2: path = [desg]
    top_desgs = df[desg].value_counts().head(30).index
    sub = df[df[desg].isin(top_desgs)].dropna(subset=[path[0]])
    fig = px.treemap(sub,path=path,color=path[0] if len(path)>1 else desg,
                     color_discrete_sequence=COLORS)
    fig.update_layout(**_L("Designation Distribution Treemap",500))
    return fig, f"{df[desg].nunique():,} unique designations across {len(df):,} employees — top 30 shown."

def grade_distribution(df, col_map, group_col):
    grade = _g(col_map,"grade")
    if not grade or grade not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        top_grades = df[grade].value_counts().head(15).index
        sub = df[df[grade].isin(top_grades)]
        ct = pd.crosstab(sub[grade],sub[group_col])
        fig = px.imshow(ct,text_auto=True,color_continuous_scale="Blues",aspect="auto")
        fig.update_layout(**_L(f"Grade Distribution: {grade} × {group_col}",h=500),
                          coloraxis_showscale=False)
        return fig, f"{df[grade].nunique()} distinct grades. Darker = more employees in that grade/division."
    counts = df[grade].value_counts().head(20).sort_values()
    fig = px.bar(x=counts.values,y=counts.index,orientation="h",
                 color=counts.values,color_continuous_scale=[[0,"#bfdbfe"],[1,C["purple"]]],
                 text=counts.values)
    fig.update_traces(textposition="outside",marker_line_width=0)
    fig.update_layout(**_L("Top 20 Grade Codes by Headcount"),coloraxis_showscale=False,
                      xaxis_title="Employees",yaxis_title="")
    return fig, f"Top grade: {df[grade].value_counts().idxmax()} ({df[grade].value_counts().max():,} employees)."

def span_of_control(df, col_map):
    sup = _g(col_map,"super_code","supervisor")
    if not sup or sup not in df.columns: return None, ""
    spans = df[df[sup].notna() & (df[sup].astype(str).str.strip()!="")].groupby(sup).size()
    if len(spans) < 3: return None, ""
    fig = px.histogram(x=spans.values,nbins=20,color_discrete_sequence=[C["teal"]])
    fig.add_vline(x=spans.mean(),line_dash="dash",line_color=C["red"],
                  annotation_text=f"Mean {spans.mean():.1f}",annotation_position="top right")
    fig.update_traces(marker_line_width=1,marker_line_color="white")
    fig.update_layout(**_L("Span of Control Distribution"),
                      xaxis_title="Direct Reports per Manager",yaxis_title="Managers")
    return fig, f"Average span of control: {spans.mean():.1f}. {(spans>10).mean():.0%} of managers have >10 direct reports."

def unique_roles_per_division(df, col_map, group_col):
    desg = _g(col_map,"designation")
    if not desg or not group_col or desg not in df.columns or group_col not in df.columns:
        return None, ""
    roles = df.groupby(group_col)[desg].nunique().sort_values(ascending=False)
    headcount = df.groupby(group_col).size()
    ratio = (headcount / roles).round(1)
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=roles.index,y=roles.values,name="Unique Designations",
                         marker_color=C["purple"],marker_line_width=0))
    fig.add_trace(go.Scatter(x=ratio.index,y=ratio.values,mode="markers+lines",
                             name="Avg employees/role",marker_size=8,
                             line_color=C["orange"]),secondary_y=True)
    fig.update_layout(**_L(f"Role Complexity by {group_col}"))
    fig.update_yaxes(title_text="Unique Designations",secondary_y=False)
    fig.update_yaxes(title_text="Avg Employees per Role",secondary_y=True)
    return fig, f"More unique designations = more complex structure. Ratio close to 1 = every role is unique."

# ══════════════════════════════════════════════════════════════════════════════
# 3 · PEOPLE FLOW & ATTRITION
# ══════════════════════════════════════════════════════════════════════════════

def joiners_leavers_timeline(df, col_map, group_col):
    hire_q = col_map.get("_hire_quarter"); exit_q = col_map.get("_exit_quarter")
    if not hire_q or hire_q not in df.columns: return None, ""
    hires = df.dropna(subset=[hire_q]).groupby(hire_q).size().rename("Joiners")
    fig = go.Figure()
    if group_col and group_col in df.columns:
        top_groups = df[group_col].value_counts().head(5).index
        for i,grp in enumerate(top_groups):
            sub = df[df[group_col]==grp]
            h = sub.dropna(subset=[hire_q]).groupby(hire_q).size()
            fig.add_trace(go.Scatter(x=h.index,y=h.values,mode="lines",
                                     name=f"{grp} Joiners",
                                     line=dict(color=COLORS[i],width=2)))
            if exit_q and exit_q in df.columns:
                e = sub.dropna(subset=[exit_q]).groupby(exit_q).size()
                fig.add_trace(go.Scatter(x=e.index,y=e.values,mode="lines",
                                         name=f"{grp} Leavers",
                                         line=dict(color=COLORS[i],width=1.5,dash="dot")))
    else:
        fig.add_trace(go.Scatter(x=hires.index,y=hires.values,mode="lines+markers",
                                 fill="tozeroy",fillcolor="rgba(37,99,235,0.1)",
                                 line=dict(color=C["blue"],width=2.5),name="Joiners"))
        if exit_q and exit_q in df.columns:
            exits = df.dropna(subset=[exit_q]).groupby(exit_q).size().rename("Leavers")
            fig.add_trace(go.Scatter(x=exits.index,y=exits.values,mode="lines",
                                     fill="tozeroy",fillcolor="rgba(220,38,38,0.08)",
                                     line=dict(color=C["red"],width=2),name="Leavers"))
    fig.update_layout(**_L("Joiners vs Leavers Over Time (Quarterly)"),
                      xaxis_title="Quarter",yaxis_title="Employees")
    peak_hire = hires.idxmax()
    return fig, f"Hiring peaked in {peak_hire} ({hires.max()} joiners). Dotted lines = departures per quarter."

def attrition_gauge_by_group(df, col_map, group_col):
    att = col_map.get("_attrition")
    if not att or att not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        rates = df.groupby(group_col)[att].mean()*100
        rates = rates.sort_values(ascending=False)
        overall = df[att].mean()*100
        colors = [C["red"] if v>overall*1.3 else C["orange"] if v>overall else C["green"]
                  for v in rates.values]
        fig = go.Figure(go.Bar(
            x=rates.values, y=rates.index, orientation="h",
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in rates.values], textposition="outside",
        ))
        fig.add_vline(x=overall,line_dash="dash",line_color="#475569",
                      annotation_text=f"Overall {overall:.1f}%",annotation_position="bottom right")
        fig.update_layout(**_L(f"Attrition Rate by {group_col}"),
                          xaxis_title="Attrition Rate (%)",yaxis_title="")
        return fig, f"Overall attrition rate: {overall:.1f}%. {rates.idxmax()} has the highest at {rates.max():.1f}%."
    overall = df[att].mean()*100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",value=overall,
        number=dict(suffix="%",font_size=46),
        gauge=dict(axis=dict(range=[0,60]),
                   bar=dict(color=C["red"] if overall>30 else C["orange"] if overall>15 else C["green"]),
                   steps=[dict(range=[0,15],color="#d1fae5"),
                          dict(range=[15,30],color="#fef9c3"),
                          dict(range=[30,60],color="#fee2e2")],
                   threshold=dict(line=dict(color="#1e293b",width=3),value=20)),
        title=dict(text="Overall Attrition Rate",font_size=14),
    ))
    fig.update_layout(font=dict(family="Inter,sans-serif"),paper_bgcolor="rgba(0,0,0,0)",
                      height=300,margin=dict(l=30,r=30,t=30,b=20))
    return fig, f"{df[att].sum():,} employees ({overall:.1f}%) have left across all time."

def attrition_by_tenure(df, col_map, group_col):
    att = col_map.get("_attrition"); ten = col_map.get("_tenure_years")
    if not att or not ten or att not in df.columns or ten not in df.columns: return None, ""
    bins=[0,0.5,1,2,3,5,8,12,50]
    labels=["<6m","6m-1y","1-2y","2-3y","3-5y","5-8y","8-12y","12y+"]
    data = df.dropna(subset=[ten]).copy()
    data["TBand"] = pd.cut(data[ten],bins=bins,labels=labels)
    if group_col and group_col in df.columns:
        top5 = df[group_col].value_counts().head(5).index
        sub = data[data[group_col].isin(top5)]
        rates = sub.groupby(["TBand",group_col],observed=True)[att].mean()*100
        rates = rates.reset_index()
        rates.columns = ["TBand",group_col,"AttrRate"]
        fig = px.line(rates,x="TBand",y="AttrRate",color=group_col,
                      markers=True,color_discrete_sequence=COLORS)
    else:
        rates = data.groupby("TBand",observed=True)[att].mean()*100
        counts = data.groupby("TBand",observed=True).size()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rates.index.astype(str),y=rates.values,
                             marker_color=[C["red"] if v>rates.mean() else C["green"] for v in rates],
                             marker_line_width=0,name="Attrition %"))
    fig.update_layout(**_L("Attrition Rate by Tenure Band"),
                      xaxis_title="Tenure",yaxis_title="Attrition Rate (%)")
    peak = data.groupby("TBand",observed=True)[att].mean().idxmax()
    return fig, f"Highest attrition is at {peak} tenure — the most critical retention window."

def survival_curve(df, col_map, group_col):
    att = col_map.get("_attrition"); ten = col_map.get("_tenure_years")
    if not att or not ten or att not in df.columns or ten not in df.columns: return None, ""
    data = df.dropna(subset=[ten]).copy(); data["_att"] = data[att]
    fig = go.Figure()
    def km(sub):
        times = sorted(sub[sub["_att"]==1][ten].unique())[:40]
        s = [1.0]; tv = [0]
        for t in times:
            at_risk = (sub[ten] >= t).sum()
            if at_risk == 0: continue
            events = ((sub[ten].round(1)==round(t,1)) & (sub["_att"]==1)).sum()
            s.append(s[-1]*(1-events/at_risk)); tv.append(t)
        return tv, s
    if group_col and group_col in df.columns:
        for i,grp in enumerate(df[group_col].value_counts().head(6).index):
            sub = data[data[group_col]==grp]
            if len(sub) < 20: continue
            tv, sv = km(sub)
            fig.add_trace(go.Scatter(x=tv,y=[v*100 for v in sv],mode="lines",
                                     name=grp,line=dict(color=COLORS[i],width=2)))
    else:
        tv, sv = km(data)
        fig.add_trace(go.Scatter(x=tv,y=[v*100 for v in sv],mode="lines",
                                 fill="tozeroy",fillcolor="rgba(220,38,38,0.08)",
                                 line=dict(color=C["red"],width=2.5),name="All"))
    fig.update_layout(**_L("Employee Survival Curve"),
                      xaxis_title="Years of Service",yaxis_title="% Still Employed")
    return fig, "Steeper early drop = high early attrition. Compare curves to identify which divisions retain better."

def probation_analysis(df, col_map, group_col):
    prob_days = col_map.get("_probation_days"); prob_flag = col_map.get("_on_probation")
    hire_col = _g(col_map,"hire_date")
    if not prob_days and not prob_flag: return None, ""
    if prob_days and prob_days in df.columns:
        data = df.dropna(subset=[prob_days]).copy()
        data = data[data[prob_days].between(0,730)]  # cap at 2 years
        if group_col and group_col in df.columns:
            fig = px.violin(data,y=prob_days,x=group_col,color=group_col,
                            box=True,points="outliers",color_discrete_sequence=COLORS)
            fig.update_layout(**_L(f"Time to Confirmation (Days) by {group_col}"))
            fig.update_layout(showlegend=False,yaxis_title="Days in Probation")
        else:
            fig = px.histogram(data,x=prob_days,nbins=30,
                               color_discrete_sequence=[C["amber"]])
            fig.update_layout(**_L("Distribution of Probation Duration (Days)"),
                              xaxis_title="Days in Probation")
        med = data[prob_days].median()
        return fig, f"Median time to confirmation: {med:.0f} days ({med/30:.1f} months)."
    if prob_flag and prob_flag in df.columns and group_col and group_col in df.columns:
        pct = df[df[col_map.get("_is_active","__is_active")]==1].groupby(group_col)[prob_flag].mean()*100
        fig = px.bar(x=pct.values,y=pct.index,orientation="h",
                     color=pct.values,color_continuous_scale=[[0,C["green"]],[0.5,C["amber"]],[1,C["red"]]],
                     text=[f"{v:.0f}%" for v in pct.values])
        fig.update_traces(textposition="outside",marker_line_width=0)
        fig.update_layout(**_L(f"% Active Employees Still on Probation by {group_col}"),
                          coloraxis_showscale=False,xaxis_title="% on Probation")
        return fig, f"High probation rates may indicate recent hiring waves or slow confirmation processes."
    return None, ""

def net_headcount_change(df, col_map, group_col):
    hire_q = col_map.get("_hire_quarter"); exit_q = col_map.get("_exit_quarter")
    if not hire_q or hire_q not in df.columns: return None, ""
    recent_hire = df.dropna(subset=[hire_q])
    recent_hire = recent_hire[recent_hire[hire_q] >= "2015Q1"]
    hires = recent_hire.groupby(hire_q).size()
    exits_s = pd.Series(dtype=int)
    if exit_q and exit_q in df.columns:
        recent_exit = df.dropna(subset=[exit_q])
        recent_exit = recent_exit[recent_exit[exit_q] >= "2015Q1"]
        exits_s = recent_exit.groupby(exit_q).size()
    all_q = sorted(set(hires.index) | set(exits_s.index))
    net = pd.Series({q: hires.get(q,0) - exits_s.get(q,0) for q in all_q})
    cumnet = net.cumsum()
    colors = [C["green"] if v>=0 else C["red"] for v in net.values]
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        subplot_titles=["Quarterly Net Change","Cumulative Headcount"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Bar(x=net.index,y=net.values,marker_color=colors,
                         name="Net Change",marker_line_width=0),row=1,col=1)
    fig.add_trace(go.Scatter(x=cumnet.index,y=cumnet.values,mode="lines",
                              fill="tozeroy",fillcolor="rgba(37,99,235,0.1)",
                              line=dict(color=C["blue"],width=2),name="Cumulative"),row=2,col=1)
    fig.update_layout(**_L("Net Headcount Change (Quarterly)",h=500))
    return fig, f"Peak growth quarter: {net.idxmax()} (+{net.max()} net). Net change 2015–now: {int(cumnet.iloc[-1]):+,}."

def attrition_heatmap_by_hire_year(df, col_map, group_col):
    att = col_map.get("_attrition"); hire_y = col_map.get("_hire_year")
    if not att or not hire_y: return None, ""
    grp = group_col or _g(col_map,"department","division")
    if not grp or grp not in df.columns: return None, ""
    data = df.dropna(subset=[hire_y]).copy()
    data = data[data[hire_y].between(2010,2026)]
    top = df[grp].value_counts().head(10).index
    sub = data[data[grp].isin(top)]
    pivot = sub.groupby([grp,hire_y])[att].mean()*100
    pivot = pivot.unstack(fill_value=0)
    fig = px.imshow(pivot,text_auto=".0f",color_continuous_scale="RdYlGn_r",
                    zmin=0,zmax=80,aspect="auto")
    fig.update_layout(**_L(f"Attrition Rate: {grp} × Hire Year Cohort",h=450))
    return fig, "Each cell = % of employees hired that year who later left. Darker = higher attrition for that cohort."

# ══════════════════════════════════════════════════════════════════════════════
# 4 · DEMOGRAPHICS & DIVERSITY
# ══════════════════════════════════════════════════════════════════════════════

def gender_overview(df, col_map, group_col):
    gen = _g(col_map,"gender")
    if not gen or gen not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        female_pct = df.groupby(group_col)[gen].apply(
            lambda x: (x=="Female").mean()*100
        ).sort_values(ascending=False)
        colors = [C["pink"] if v>female_pct.mean() else "#93c5fd" for v in female_pct.values]
        fig = go.Figure(go.Bar(
            x=female_pct.values,y=female_pct.index,orientation="h",
            marker_color=colors,marker_line_width=0,
            text=[f"{v:.1f}%" for v in female_pct.values],textposition="outside",
        ))
        fig.add_vline(x=female_pct.mean(),line_dash="dash",line_color="#475569",
                      annotation_text=f"Avg {female_pct.mean():.1f}%")
        fig.update_layout(**_L(f"Female Representation by {group_col}"),
                          xaxis_title="% Female Employees",yaxis_title="")
        return fig, f"Overall female representation: {(df[gen]=='Female').mean():.1%}. Target benchmark is typically 30–50%."
    counts = df[gen].value_counts()
    fig = px.pie(values=counts.values,names=counts.index,hole=0.6,
                 color_discrete_sequence=[C["blue"],C["pink"],"#94a3b8"])
    fig.update_traces(textposition="outside",textinfo="percent+label")
    fig.update_layout(**_L("Gender Breakdown",380),showlegend=False)
    f_pct = (df[gen]=="Female").mean()
    return fig, f"Female: {f_pct:.1%} | Male: {1-f_pct:.1%}."

def gender_by_management_level(df, col_map):
    gen = _g(col_map,"gender"); mgmt = _g(col_map,"designation_main")
    if not gen or not mgmt or gen not in df.columns or mgmt not in df.columns: return None, ""
    pivot = pd.crosstab(df[mgmt],df[gen],normalize="index")*100
    order = ["Executive / Officer","Assistant Manager","Manager","General Manager","Director"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    fig = px.bar(pivot.reset_index(),x=mgmt,y=pivot.columns.tolist(),
                 barmode="stack",color_discrete_sequence=[C["blue"],C["pink"],"#94a3b8"])
    fig.update_layout(**_L("Gender Balance by Management Level"),
                      yaxis_title="% at Level",xaxis_title="")
    return fig, "If female % shrinks at senior levels, this indicates a 'leaky pipeline' — a key D&I metric to track."

def religion_distribution(df, col_map, group_col):
    rel = _g(col_map,"religion")
    if not rel or rel not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col],df[rel],normalize="index")*100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(),x=group_col,y=list(ct.columns),
                     barmode="stack",color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Religion Distribution by {group_col}"),
                          yaxis_title="% of Employees",xaxis_title="")
    else:
        counts = df[rel].value_counts()
        fig = px.pie(values=counts.values,names=counts.index,hole=0.55,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textinfo="percent+label",textposition="outside")
        fig.update_layout(**_L("Religion Distribution",380),showlegend=False)
    return fig, f"Dominant religion: {df[rel].value_counts().idxmax()} ({df[rel].value_counts().iloc[0]/len(df):.0%})."

def district_heatmap(df, col_map, group_col):
    dist = _g(col_map,"district")
    if not dist or dist not in df.columns: return None, ""
    top_dists = df[dist].value_counts().head(20).index
    sub = df[df[dist].isin(top_dists)]
    if group_col and group_col in df.columns:
        ct = pd.crosstab(sub[dist],sub[group_col])
        ct = ct.reindex(top_dists)
        fig = px.imshow(ct,text_auto=True,color_continuous_scale="Blues",aspect="auto")
        fig.update_layout(**_L(f"Geographic Spread — District × {group_col}",h=500),
                          coloraxis_showscale=False)
    else:
        counts = df[dist].value_counts().head(20).sort_values()
        fig = px.bar(x=counts.values,y=counts.index,orientation="h",
                     color=counts.values,color_continuous_scale=[[0,"#bfdbfe"],[1,C["teal"]]],
                     text=counts.values)
        fig.update_traces(textposition="outside",marker_line_width=0)
        fig.update_layout(**_L("Top 20 Employee Districts of Origin"),
                          coloraxis_showscale=False,xaxis_title="Employees")
    return fig, f"Employees come from {df[dist].nunique()} districts. Top origin: {df[dist].value_counts().idxmax()}."

def work_location_analysis(df, col_map, group_col):
    wl = _g(col_map,"work_location_type","location")
    if not wl or wl not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col],df[wl],normalize="index")*100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(),x=group_col,y=list(ct.columns),
                     barmode="stack",color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Work Location Type by {group_col}"),
                          yaxis_title="% of Workforce",xaxis_title="")
    else:
        counts = df[wl].value_counts()
        fig = px.pie(values=counts.values,names=counts.index,hole=0.5,
                     color_discrete_sequence=COLORS)
        fig.update_layout(**_L("Work Location Distribution",380),showlegend=True)
    return fig, f"Most employees are {df[wl].value_counts().idxmax()}."

def age_distribution(df, col_map, group_col):
    age = col_map.get("_age")
    if not age or age not in df.columns: return None, ""
    data = df.dropna(subset=[age])
    data = data[data[age].between(18,65)]
    if group_col and group_col in df.columns:
        top5 = df[group_col].value_counts().head(5).index
        sub = data[data[group_col].isin(top5)]
        fig = px.violin(sub,y=age,x=group_col,color=group_col,
                        box=True,color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Age Distribution by {group_col}"))
        fig.update_layout(showlegend=False,yaxis_title="Age",xaxis_title="")
    else:
        fig = px.histogram(data,x=age,nbins=30,color_discrete_sequence=[C["amber"]],marginal="box")
        fig.update_layout(**_L("Age Distribution"))
    return fig, f"Avg age: {data[age].mean():.0f} years. Range: {data[age].min():.0f}–{data[age].max():.0f}."

def education_breakdown(df, col_map, group_col):
    edu = _g(col_map,"education")
    if not edu or edu not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        top_edu = df[edu].value_counts().head(8).index
        sub = df[df[edu].isin(top_edu)]
        ct = pd.crosstab(sub[group_col],sub[edu],normalize="index")*100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(),x=group_col,y=list(ct.columns),
                     barmode="stack",color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Education Level by {group_col}"),
                          yaxis_title="% of Employees",xaxis_title="")
    else:
        counts = df[edu].value_counts().head(12).sort_values()
        fig = px.bar(x=counts.values,y=counts.index,orientation="h",
                     color=counts.values,color_continuous_scale=[[0,"#bfdbfe"],[1,C["purple"]]],
                     text=counts.values)
        fig.update_traces(textposition="outside",marker_line_width=0)
        fig.update_layout(**_L("Education Level Distribution"),coloraxis_showscale=False)
    return fig, f"Most common qualification: {df[edu].value_counts().idxmax()} ({df[edu].value_counts().iloc[0]:,} employees)."

def marital_status(df, col_map, group_col):
    mar = _g(col_map,"marital_status")
    if not mar or mar not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col],df[mar],normalize="index")*100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(),x=group_col,y=list(ct.columns),
                     barmode="stack",color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Marital Status by {group_col}"),
                          yaxis_title="% of Employees",xaxis_title="")
    else:
        counts = df[mar].value_counts()
        fig = px.pie(values=counts.values,names=counts.index,hole=0.5,
                     color_discrete_sequence=COLORS)
        fig.update_layout(**_L("Marital Status Distribution",380),showlegend=True)
    return fig, f"Married: {(df[mar]=='Married').mean():.0%} | Single: {(df[mar]=='Single').mean():.0%}."

# ══════════════════════════════════════════════════════════════════════════════
# 5 · LEAVE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def primary_leave_utilisation(df, col_map, group_col):
    taken = _g(col_map,"primary_leave_taken")
    entitled = _g(col_map,"leave_entitled")
    if not taken or taken not in df.columns: return None, ""
    data = df.dropna(subset=[taken]).copy()
    data[taken] = pd.to_numeric(data[taken],errors="coerce")
    if group_col and group_col in df.columns:
        avg_taken = data.groupby(group_col)[taken].mean().sort_values()
        if entitled and entitled in df.columns:
            data[entitled] = pd.to_numeric(data[entitled],errors="coerce")
            avg_ent = data.groupby(group_col)[entitled].mean()
            util = (avg_taken / avg_ent * 100).fillna(0)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=avg_taken.values,y=avg_taken.index,orientation="h",
                                 name="Avg Days Taken",marker_color=C["teal"],marker_line_width=0))
            fig.add_trace(go.Bar(x=avg_ent.reindex(avg_taken.index).values,y=avg_taken.index,
                                 orientation="h",name="Avg Entitled",
                                 marker_color="#e2e8f0",marker_line_width=0))
            fig.update_layout(**_L(f"Privileged Leave: Taken vs Entitled by {group_col}"),
                              barmode="overlay",xaxis_title="Days")
        else:
            colors = [C["green"] if v<avg_taken.mean() else C["orange"] for v in avg_taken.values]
            fig = go.Figure(go.Bar(x=avg_taken.values,y=avg_taken.index,orientation="h",
                                   marker_color=colors,marker_line_width=0,
                                   text=[f"{v:.1f}d" for v in avg_taken.values],
                                   textposition="outside"))
            fig.update_layout(**_L(f"Avg Privileged Leave Taken by {group_col}"),
                              xaxis_title="Days/Year")
        return fig, f"Overall avg privileged leave taken: {data[taken].mean():.1f} days/year."
    fig = px.histogram(data,x=taken,nbins=30,color_discrete_sequence=[C["teal"]])
    fig.update_layout(**_L("Privileged Leave Days Taken Distribution"),xaxis_title="Days Taken")
    return fig, f"Avg: {data[taken].mean():.1f} days. {(data[taken]==0).mean():.0%} took zero leave."

def secondary_leave_analysis(df, col_map, group_col):
    taken = _g(col_map,"secondary_leave_taken")
    if not taken or taken not in df.columns: return None, ""
    data = df.dropna(subset=[taken]).copy()
    data[taken] = pd.to_numeric(data[taken],errors="coerce").fillna(0)
    if group_col and group_col in df.columns:
        avg = data.groupby(group_col)[taken].mean().sort_values()
        colors = [C["red"] if v>avg.mean()*1.3 else C["orange"] if v>avg.mean() else C["green"]
                  for v in avg.values]
        fig = go.Figure(go.Bar(x=avg.values,y=avg.index,orientation="h",
                               marker_color=colors,marker_line_width=0,
                               text=[f"{v:.1f}d" for v in avg.values],textposition="outside"))
        fig.add_vline(x=avg.mean(),line_dash="dash",line_color="#475569",
                      annotation_text=f"Avg {avg.mean():.1f}d")
        fig.update_layout(**_L(f"Avg Sick/Casual Leave Taken by {group_col}"),
                          xaxis_title="Days/Year",yaxis_title="")
        return fig, f"High sick leave ({avg.idxmax()}: {avg.max():.1f}d/yr) may signal workload or health concerns."
    fig = px.histogram(data,x=taken,nbins=25,color_discrete_sequence=[C["orange"]],marginal="box")
    fig.update_layout(**_L("Sick/Casual Leave Distribution"),xaxis_title="Days Taken")
    return fig, f"Avg sick/casual leave: {data[taken].mean():.1f} days/year."

def leave_forfeiture(df, col_map, group_col):
    forf = next((c for c in df.columns if "forfeit" in c.lower() or "forfit" in c.lower()), None)
    if not forf: return None, ""
    data = df.copy(); data[forf] = pd.to_numeric(data[forf],errors="coerce").fillna(0)
    if group_col and group_col in df.columns:
        avg = data.groupby(group_col)[forf].mean().sort_values(ascending=False)
        total = data.groupby(group_col)[forf].sum()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=avg.index,y=avg.values,name="Avg Days Forfeited",
                             marker_color=C["red"],marker_line_width=0))
        fig.add_trace(go.Scatter(x=total.index,y=total.values,mode="markers",
                                 name="Total Days Lost",marker_size=10,marker_color=C["orange"]),
                      secondary_y=True)
        fig.update_layout(**_L(f"Leave Forfeiture by {group_col}"))
        fig.update_yaxes(title_text="Avg Days Forfeited",secondary_y=False)
        fig.update_yaxes(title_text="Total Days Lost",secondary_y=True)
        return fig, f"Total leave days forfeited across org: {data[forf].sum():,.0f} — a direct wellbeing and compensation cost."
    total = data[forf].sum()
    high = (data[forf]>0).mean()
    fig = px.histogram(data[data[forf]>0],x=forf,nbins=20,color_discrete_sequence=[C["red"]])
    fig.update_layout(**_L("Leave Forfeiture Distribution (employees who forfeited leave)"))
    return fig, f"Total forfeited days: {total:,.0f}. {high:.0%} of employees forfeited at least 1 leave day."

# ══════════════════════════════════════════════════════════════════════════════
# 6 · CAREER & GRADE PROGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def promotion_rate(df, col_map, group_col):
    promo_col = _g(col_map,"last_promotion_date")
    if not promo_col or promo_col not in df.columns: return None, ""
    df2 = df.copy()
    df2[promo_col] = pd.to_datetime(df2[promo_col],errors="coerce")
    valid_promo = df2[promo_col].notna() & (df2[promo_col].dt.year > 1950)
    df2["__has_promo"] = valid_promo.astype(int)
    if group_col and group_col in df.columns:
        rates = df2.groupby(group_col)["__has_promo"].mean()*100
        rates = rates.sort_values()
        fig = go.Figure(go.Bar(
            x=rates.values,y=rates.index,orientation="h",
            marker_color=[C["green"] if v>rates.mean() else C["orange"] for v in rates.values],
            marker_line_width=0,
            text=[f"{v:.0f}%" for v in rates.values],textposition="outside",
        ))
        fig.add_vline(x=rates.mean(),line_dash="dash",line_color="#475569",
                      annotation_text=f"Avg {rates.mean():.0f}%")
        fig.update_layout(**_L(f"% Employees with Promotion Record by {group_col}"),
                          xaxis_title="% with Promotion History",yaxis_title="")
        return fig, f"Overall: {valid_promo.mean():.0%} of employees have at least one promotion recorded."
    yr = df2.loc[valid_promo, promo_col].dt.year.value_counts().sort_index()
    yr = yr[yr.index >= 2010]
    fig = px.bar(x=yr.index,y=yr.values,color=yr.values,
                 color_continuous_scale=[[0,"#bfdbfe"],[1,C["purple"]]],text=yr.values)
    fig.update_traces(textposition="outside",marker_line_width=0)
    fig.update_layout(**_L("Promotions Per Year"),coloraxis_showscale=False,
                      xaxis_title="Year",yaxis_title="Promotions")
    return fig, f"Most promotions in: {yr.idxmax()} ({yr.max():,}). Total promotions on record: {int(valid_promo.sum()):,}."

def tenure_distribution(df, col_map, group_col):
    ten = col_map.get("_tenure_years")
    if not ten or ten not in df.columns: return None, ""
    data = df.dropna(subset=[ten]); data = data[data[ten].between(0,40)]
    if group_col and group_col in df.columns:
        top6 = df[group_col].value_counts().head(6).index
        sub = data[data[group_col].isin(top6)]
        fig = px.violin(sub,y=ten,x=group_col,color=group_col,
                        box=True,points=False,color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Tenure Distribution by {group_col}"))
        fig.update_layout(showlegend=False,yaxis_title="Years of Service",xaxis_title="")
    else:
        fig = px.histogram(data,x=ten,nbins=30,color_discrete_sequence=[C["blue"]],marginal="box")
        fig.add_vline(x=data[ten].mean(),line_dash="dash",line_color=C["red"],
                      annotation_text=f"Mean {data[ten].mean():.1f}y",annotation_position="top right")
        fig.update_layout(**_L("Tenure Distribution"),xaxis_title="Years of Service")
    return fig, f"Avg tenure: {data[ten].mean():.1f} years. {(data[ten]<1).mean():.0%} have less than 1 year service."

def yrs_since_promotion(df, col_map, group_col):
    ysp = col_map.get("_yrs_since_promo")
    if not ysp or ysp not in df.columns: return None, ""
    data = df.dropna(subset=[ysp]); data = data[data[ysp].between(0,20)]
    if group_col and group_col in df.columns:
        avg = data.groupby(group_col)[ysp].median().sort_values(ascending=False)
        colors = [C["red"] if v>5 else C["orange"] if v>3 else C["green"] for v in avg.values]
        fig = go.Figure(go.Bar(x=avg.values,y=avg.index,orientation="h",
                               marker_color=colors,marker_line_width=0,
                               text=[f"{v:.1f}y" for v in avg.values],textposition="outside"))
        fig.update_layout(**_L(f"Median Years Since Last Promotion by {group_col}"),
                          xaxis_title="Years",yaxis_title="")
        return fig, f"Longest promotion gap: {avg.idxmax()} ({avg.max():.1f} years). Consider career development interventions."
    fig = px.histogram(data,x=ysp,nbins=25,color_discrete_sequence=[C["orange"]],marginal="box")
    fig.update_layout(**_L("Years Since Last Promotion"),xaxis_title="Years")
    return fig, f"Median time since last promotion: {data[ysp].median():.1f} years."


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

CHART_REGISTRY = {
    "🏢 Conglomerate Overview": [
        {"name":"Workforce Sunburst — Full Hierarchy",    "fn":headcount_sunburst,        "needs_hierarchy":True,  "requires":[]},
        {"name":"Conglomerate KPI Scorecard (Heatmap)",  "fn":conglomerate_heatmap,      "requires":[]},
        {"name":"Headcount — Active vs Former",           "fn":headcount_bar_comparison,  "requires":[]},
        {"name":"Employment Type Mix",                    "fn":employment_type_mix,       "requires":["employment_type"]},
        {"name":"Management Level Comparison",            "fn":management_level_comparison,"requires":["designation_main"]},
    ],
    "👥 Workforce Structure": [
        {"name":"Org Treemap (3-Level Drill)",            "fn":org_treemap,               "needs_hierarchy":True,  "requires":[]},
        {"name":"Designation Distribution Treemap",       "fn":designation_treemap,       "requires":["designation"]},
        {"name":"Grade Code Distribution",                "fn":grade_distribution,        "requires":["grade"]},
        {"name":"Span of Control",                        "fn":span_of_control,           "no_group":True, "requires":["super_code"]},
        {"name":"Role Complexity per Division",           "fn":unique_roles_per_division, "requires":["designation"]},
    ],
    "📅 People Flow & Attrition": [
        {"name":"Joiners vs Leavers Timeline",            "fn":joiners_leavers_timeline,  "requires":["_hire_quarter"]},
        {"name":"Net Headcount Change (Quarterly)",       "fn":net_headcount_change,      "requires":["_hire_quarter"]},
        {"name":"Attrition Rate by Business Unit",        "fn":attrition_gauge_by_group,  "requires":["_attrition"]},
        {"name":"Attrition by Tenure Band",               "fn":attrition_by_tenure,       "requires":["_attrition","_tenure_years"]},
        {"name":"Employee Survival Curve",                "fn":survival_curve,            "requires":["_attrition","_tenure_years"]},
        {"name":"Attrition by Hire Cohort (Heatmap)",    "fn":attrition_heatmap_by_hire_year,"requires":["_attrition","_hire_year"]},
        {"name":"Probation Status & Duration",            "fn":probation_analysis,        "requires":["_on_probation"]},
    ],
    "🌍 Demographics & Diversity": [
        {"name":"Gender Representation",                  "fn":gender_overview,           "requires":["gender"]},
        {"name":"Gender Pipeline by Management Level",   "fn":gender_by_management_level,"no_group":True,"requires":["gender","designation_main"]},
        {"name":"Religion Distribution",                  "fn":religion_distribution,     "requires":["religion"]},
        {"name":"Geographic Origin (District)",           "fn":district_heatmap,          "requires":["district"]},
        {"name":"Work Location Type",                     "fn":work_location_analysis,    "requires":["work_location_type"]},
        {"name":"Age Distribution",                       "fn":age_distribution,          "requires":["_age"]},
        {"name":"Education Level",                        "fn":education_breakdown,       "requires":["education"]},
        {"name":"Marital Status",                         "fn":marital_status,            "requires":["marital_status"]},
    ],
    "📋 Leave Management": [
        {"name":"Privileged Leave Utilisation",           "fn":primary_leave_utilisation, "requires":["primary_leave_taken"]},
        {"name":"Sick / Casual Leave Analysis",           "fn":secondary_leave_analysis,  "requires":["secondary_leave_taken"]},
        {"name":"Leave Forfeiture Analysis",              "fn":leave_forfeiture,          "requires":[]},
    ],
    "🎯 Career & Progression": [
        {"name":"Promotion Rate by Division",             "fn":promotion_rate,            "requires":["last_promotion_date"]},
        {"name":"Tenure Distribution",                    "fn":tenure_distribution,       "requires":["_tenure_years"]},
        {"name":"Years Since Last Promotion",             "fn":yrs_since_promotion,       "requires":["_yrs_since_promo"]},
    ],
}


def get_available(df, col_map):
    available = {}
    for cat, charts in CHART_REGISTRY.items():
        usable = []
        for c in charts:
            reqs = c.get("requires", [])
            if all((col_map.get(r) and col_map[r] in df.columns) or
                   (r.startswith("__") and r in df.columns)
                   for r in reqs):
                usable.append(c)
        if usable:
            available[cat] = usable
    return available
