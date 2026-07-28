"""
analytics.py — 50+ Plotly charts across 7 HR analytics categories.
Universal design: works for any business size and type.
All charts: (df, col_map, group_col) → (fig | None, insight_str)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design tokens ──────────────────────────────────────────────────────────────
COLORS = [
    "#2563EB","#7C3AED","#DB2777","#059669","#D97706",
    "#0891B2","#DC2626","#9333EA","#0F766E","#EA580C",
    "#4F46E5","#BE185D",
]
C = {
    "blue":"#2563EB", "indigo":"#4F46E5", "purple":"#7C3AED",
    "pink":"#DB2777", "green":"#059669", "teal":"#0F766E",
    "orange":"#D97706", "red":"#DC2626", "sky":"#0891B2",
}
GOOD  = "#059669"
BAD   = "#DC2626"
WARN  = "#D97706"
BENCH_LINE = dict(line_dash="dot", line_color="#64748B", line_width=1.5)
ATTRITION_BENCHMARK = 15   # %

def _L(title="", h=430, legend_below=False):
    leg = dict(
        bgcolor="rgba(255,255,255,0.96)", bordercolor="#e2e8f0", borderwidth=1,
        font=dict(size=12, color="#1e293b"), itemsizing="constant",
    )
    if legend_below:
        leg.update(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5)
    else:
        leg.update(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#0f172a", family="Inter"),
            x=0, pad=dict(l=4),
        ),
        font=dict(family="Inter,system-ui,sans-serif", size=12, color="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        height=h,
        margin=dict(l=10, r=16, t=58, b=10, pad=4),
        colorway=COLORS,
        showlegend=True,
        legend=leg,
        hoverlabel=dict(
            bgcolor="white", bordercolor="#e2e8f0",
            font_family="Inter,sans-serif", font_size=12, namelength=-1,
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#e8ecf0", gridwidth=1,
            linecolor="#cbd5e1", linewidth=1,
            tickfont=dict(size=12, color="#374151"),
            title_font=dict(size=13, color="#1e293b"),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#e8ecf0", gridwidth=1,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=12, color="#374151"),
            title_font=dict(size=13, color="#1e293b"),
            zeroline=False,
        ),
    )

def _g(col_map, *concepts):
    for c in concepts:
        v = col_map.get(c)
        if v: return v
    return None

def _att_color(val, mean):
    if val > mean * 1.25: return BAD
    if val > mean: return WARN
    return GOOD

# ══════════════════════════════════════════════════════════════════════════════
# 1 · BUSINESS OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def headcount_sunburst(df, col_map, hierarchy):
    path = [h[1] for h in hierarchy if h[1] in df.columns]
    if not path: return None, ""
    data = df.dropna(subset=[path[0]])
    # Limit to 2 levels for readability when there are many segments
    path_display = path[:2] if len(data[path[0]].unique()) > 6 else path
    fig = px.sunburst(data, path=path_display, color=path_display[0],
                      color_discrete_sequence=COLORS)
    fig.update_traces(
        textinfo="label+percent parent",
        insidetextfont=dict(size=13, color="white"),
        texttemplate="<b>%{label}</b><br>%{percentParent:.0%}",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value:,} employees<br>%{percentParent:.0%} of parent<extra></extra>",
    )
    note = " (top 2 levels shown for readability)" if len(path_display) < len(path) else ""
    fig.update_layout(**_L(f"Workforce Sunburst — Click to Explore{note}", 560))
    top = data[path[0]].value_counts().idxmax()
    pct = data[path[0]].value_counts().max() / len(data)
    return fig, f"{top} is the largest group ({pct:.0%} of total). Click any segment to explore the hierarchy."

def kpi_scorecard_heatmap(df, col_map, group_col):
    if not group_col or group_col not in df.columns: return None, ""
    att  = col_map.get("_attrition"); ten = col_map.get("_tenure_years")
    gen  = _g(col_map, "gender");    prob = col_map.get("_on_probation")
    sal  = _g(col_map, "salary")
    metrics, labels = [], []
    if att:   metrics.append(df.groupby(group_col)[att].mean()*100); labels.append("Attrition %")
    if ten:   metrics.append(df.groupby(group_col)[ten].mean()); labels.append("Avg Tenure (yrs)")
    if gen:   metrics.append(df.groupby(group_col)[gen].apply(lambda x:(x=="Female").mean()*100)); labels.append("Female %")
    if prob:  metrics.append(df.groupby(group_col)[prob].mean()*100); labels.append("On Probation %")
    if sal:   metrics.append(df.groupby(group_col)[sal].median()/1000); labels.append("Median Salary (K)")
    if not metrics: return None, ""
    pivot = pd.DataFrame({l: m for l, m in zip(labels, metrics)}).fillna(0)
    norm  = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-9)
    fig = go.Figure(go.Heatmap(
        z=norm.values.T, x=pivot.index.astype(str), y=labels,
        text=[[f"{pivot[l][r]:.1f}" for r in pivot.index] for l in labels],
        texttemplate="<b>%{text}</b>", textfont=dict(size=12),
        colorscale="RdYlGn", showscale=True,
        colorbar=dict(title="Score", thickness=12, len=0.8, tickfont=dict(size=11)),
        hovertemplate="<b>%{x}</b><br>%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(**_L(f"Business Unit KPI Scorecard (Heatmap)", h=max(320, len(labels)*72+110)))
    return fig, "Green = above average; red = needs attention. Scores normalised across all units for fair comparison."

def headcount_active_vs_former(df, col_map, group_col):
    if not group_col or group_col not in df.columns: return None, ""
    active_col = col_map.get("_is_active", "__is_active")
    if active_col not in df.columns: return None, ""
    grp = df.groupby(group_col)[active_col].agg(["sum","count"]).reset_index()
    grp.columns = [group_col, "Active", "Total"]
    grp["Former"] = grp["Total"] - grp["Active"]
    grp["Attrition %"] = (grp["Former"] / grp["Total"] * 100).round(1)
    grp = grp.sort_values("Total", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=grp[group_col], x=grp["Active"], orientation="h",
                         name="Active", marker_color=GOOD, marker_line_width=0,
                         hovertemplate="<b>%{y}</b><br>Active: %{x:,}<extra></extra>"))
    fig.add_trace(go.Bar(y=grp[group_col], x=grp["Former"], orientation="h",
                         name="Former", marker_color="#fca5a5", marker_line_width=0,
                         hovertemplate="<b>%{y}</b><br>Former: %{x:,}<extra></extra>"))
    fig.update_layout(**_L(f"Headcount by {group_col} — Active vs Former"),
                      barmode="stack", yaxis_title="", xaxis_title="Total Employees")
    ttl = int(grp["Total"].sum()); act = int(grp["Active"].sum())
    return fig, f"All-time workforce: {ttl:,}. Currently active: {act:,} ({act/ttl:.0%}). Former: {ttl-act:,}."

def employment_type_mix(df, col_map, group_col):
    emp = _g(col_map, "employment_type")
    if not emp or emp not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col], df[emp], normalize="index") * 100
        ct = ct.reindex(df[group_col].value_counts().head(12).index)
        fig = px.bar(ct.reset_index(), x=group_col,
                     y=[c for c in ct.columns], barmode="stack",
                     color_discrete_sequence=COLORS,
                     labels={"value": "% of Employees", group_col: ""})
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}%<extra></extra>")
        fig.update_layout(**_L(f"Employment Type Mix by {group_col}"), yaxis_title="% of Workforce")
        top_type = df[emp].value_counts().idxmax()
        top_pct  = df[emp].value_counts().iloc[0] / len(df)
        return fig, f"Dominant employment type: {top_type} ({top_pct:.0%}). Review contractual balance for compliance risk."
    counts = df[emp].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                 color_discrete_sequence=COLORS)
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      hovertemplate="<b>%{label}</b><br>%{value:,} employees (%{percent})<extra></extra>")
    fig.update_layout(**_L("Employment Type Distribution", 420), showlegend=True)
    return fig, f"Largest category: {counts.idxmax()} ({counts.iloc[0]:,} employees, {counts.iloc[0]/len(df):.0%})."

def management_level_comparison(df, col_map, group_col):
    mgmt = _g(col_map, "designation_main")
    if not mgmt or mgmt not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col], df[mgmt], normalize="index") * 100
        ct = ct.reindex(df[group_col].value_counts().head(12).index)
        fig = px.bar(ct.reset_index(), x=group_col, y=list(ct.columns),
                     barmode="stack", color_discrete_sequence=COLORS,
                     labels={"value": "% of Employees", group_col: ""})
        fig.update_layout(**_L(f"Management Level Mix by {group_col}"), yaxis_title="% of Workforce")
        return fig, "A healthy structure narrows from a wide base of individual contributors up to Directors."
    counts = df[mgmt].value_counts().sort_values()
    fig = px.funnel(x=counts.values, y=counts.index, color_discrete_sequence=[C["indigo"]])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} employees<extra></extra>")
    fig.update_layout(**_L("Management Hierarchy", 400))
    return fig, f"Largest management band: {counts.idxmax()} ({counts.max():,}). Review pyramid for leadership pipeline gaps."

# ══════════════════════════════════════════════════════════════════════════════
# 2 · WORKFORCE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def org_treemap(df, col_map, hierarchy):
    path = [h[1] for h in hierarchy[:3] if h[1] in df.columns]
    if not path: return None, ""
    fig = px.treemap(df.dropna(subset=[path[0]]), path=path, color=path[0],
                     color_discrete_sequence=COLORS)
    fig.update_traces(textinfo="label+value+percent parent",
                      textfont=dict(size=12),
                      hovertemplate="<b>%{label}</b><br>%{value} employees<br>%{percentParent:.0%} of parent<extra></extra>")
    fig.update_layout(**_L("Organisational Treemap — Click to Drill Down", 530))
    return fig, "Size = headcount. Click any block to drill deeper. Hover for exact employee counts."

def designation_treemap(df, col_map, group_col):
    desg = _g(col_map, "designation"); mgmt = _g(col_map, "designation_main")
    if not desg or desg not in df.columns: return None, ""
    path = [p for p in [group_col, mgmt, desg] if p and p in df.columns]
    if len(path) < 2: path = [desg]
    top = df[desg].value_counts().head(30).index
    sub = df[df[desg].isin(top)].dropna(subset=[path[0]])
    fig = px.treemap(sub, path=path, color=path[0] if len(path) > 1 else desg,
                     color_discrete_sequence=COLORS)
    fig.update_layout(**_L(f"Top 30 Role Distribution Treemap", 510))
    return fig, f"{df[desg].nunique():,} unique designations — top 30 shown. Role diversity may indicate org complexity."

def grade_distribution(df, col_map, group_col):
    grade = _g(col_map, "grade")
    if not grade or grade not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        top_g = df[grade].value_counts().head(15).index
        sub   = df[df[grade].isin(top_g)]
        ct    = pd.crosstab(sub[grade], sub[group_col])
        fig   = px.imshow(ct, text_auto=True, color_continuous_scale="Blues",
                          aspect="auto")
        fig.update_layout(**_L(f"Grade Distribution: {grade} × {group_col}", h=520),
                          coloraxis_showscale=False)
        return fig, f"{df[grade].nunique()} distinct grades. Darker = more employees. Review grade equity across units."
    counts = df[grade].value_counts().head(20).sort_values()
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h",
        marker=dict(color=counts.values, colorscale="Blues", showscale=False),
        text=counts.values, textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,} employees<extra></extra>",
    ))
    fig.update_layout(**_L("Grade Code Distribution"), xaxis_title="Employees", yaxis_title="")
    return fig, f"Top grade: {df[grade].value_counts().idxmax()} ({df[grade].value_counts().max():,} employees)."

def span_of_control(df, col_map):
    sup = _g(col_map, "super_code", "supervisor")
    if not sup or sup not in df.columns: return None, ""
    valid = df[sup].notna() & (df[sup].astype(str).str.strip() != "")
    spans = df[valid].groupby(sup).size()
    if len(spans) < 3: return None, ""
    mean_span = spans.mean()
    fig = px.histogram(x=spans.values, nbins=max(10, int(spans.max())),
                       color_discrete_sequence=[C["sky"]])
    fig.add_vline(x=mean_span, **BENCH_LINE,
                  annotation_text=f"Mean {mean_span:.1f}",
                  annotation_font=dict(size=12, color="#374151"))
    fig.add_vline(x=8, line_dash="dot", line_color=BAD, line_width=1,
                  annotation_text="⚠ 8+ (risk)", annotation_font_color=BAD, annotation_font_size=11)
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    fig.update_layout(**_L("Span of Control Distribution"))
    fig.update_layout(xaxis_title="Direct Reports per Manager", yaxis_title="Number of Managers",
                      showlegend=False)
    wide = (spans > 8).mean()
    return fig, f"Average span: {mean_span:.1f} direct reports. {wide:.0%} of managers have >8 reports — potential burnout risk."

def role_complexity(df, col_map, group_col):
    desg = _g(col_map, "designation")
    if not desg or not group_col or desg not in df.columns or group_col not in df.columns:
        return None, ""
    roles  = df.groupby(group_col)[desg].nunique().sort_values(ascending=False)
    hc     = df.groupby(group_col).size()
    ratio  = (hc / roles).round(1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=roles.index, y=roles.values, name="Unique Roles",
                         marker_color=C["purple"], marker_line_width=0,
                         hovertemplate="<b>%{x}</b><br>%{y} unique roles<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, mode="markers+lines",
                              name="Avg employees/role", marker_size=9, marker_color=C["orange"],
                              line_color=C["orange"],
                              hovertemplate="<b>%{x}</b><br>%{y:.1f} employees per role<extra></extra>"),
                  secondary_y=True)
    fig.update_layout(**_L(f"Role Complexity by {group_col}"))
    fig.update_yaxes(title_text="Unique Designations", secondary_y=False)
    fig.update_yaxes(title_text="Avg Employees per Role", secondary_y=True, showgrid=False)
    return fig, "Many unique roles + low employees/role = high complexity. Consider role consolidation where ratio < 2."

def headcount_by_location(df, col_map, group_col):
    loc = _g(col_map, "location", "work_location_type")
    if not loc or loc not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col], df[loc], normalize="index") * 100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(), x=group_col, y=list(ct.columns),
                     barmode="stack", color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Location Mix by {group_col}"), yaxis_title="% of Employees")
    else:
        counts = df[loc].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(**_L("Workforce Location Distribution", 420))
    return fig, f"Most common location: {df[loc].value_counts().idxmax()} ({df[loc].value_counts().iloc[0]/len(df):.0%} of workforce)."

# ══════════════════════════════════════════════════════════════════════════════
# 3 · PEOPLE FLOW & ATTRITION
# ══════════════════════════════════════════════════════════════════════════════

def joiners_leavers_timeline(df, col_map, group_col):
    hire_q = col_map.get("_hire_quarter"); exit_q = col_map.get("_exit_quarter")
    hire_y = col_map.get("_hire_year")
    if not hire_q or hire_q not in df.columns: return None, ""
    hires = df.dropna(subset=[hire_q]).groupby(hire_q).size().rename("Joiners")
    fig   = go.Figure()
    if group_col and group_col in df.columns:
        top5 = df[group_col].value_counts().head(5).index
        for i, grp in enumerate(top5):
            sub = df[df[group_col] == grp]
            h   = sub.dropna(subset=[hire_q]).groupby(hire_q).size()
            fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines", name=f"{grp}",
                                     line=dict(color=COLORS[i], width=2.5),
                                     hovertemplate=f"<b>{grp}</b><br>%{{x}}: %{{y}} joiners<extra></extra>"))
            if exit_q and exit_q in df.columns:
                e = sub.dropna(subset=[exit_q]).groupby(exit_q).size()
                fig.add_trace(go.Scatter(x=e.index, y=e.values, mode="lines",
                                         name=f"{grp} (exits)",
                                         line=dict(color=COLORS[i], width=1.5, dash="dot"),
                                         hovertemplate=f"<b>{grp} exits</b><br>%{{x}}: %{{y}}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=hires.index, y=hires.values, mode="lines+markers",
                                 fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
                                 line=dict(color=C["blue"], width=2.5), marker_size=4,
                                 name="Joiners"))
        if exit_q and exit_q in df.columns:
            exits = df.dropna(subset=[exit_q]).groupby(exit_q).size()
            fig.add_trace(go.Scatter(x=exits.index, y=exits.values, mode="lines",
                                     fill="tozeroy", fillcolor="rgba(220,38,38,0.07)",
                                     line=dict(color=BAD, width=2), name="Leavers"))
    fig.update_layout(**_L("Joiners vs Leavers — Quarterly Trend"),
                      xaxis_title="Quarter", yaxis_title="Employees")
    peak = hires.idxmax()
    return fig, f"Hiring peaked in {peak} ({hires.max()} joiners). Solid lines = joiners; dotted = leavers."

def net_headcount_change(df, col_map, group_col):
    hire_q = col_map.get("_hire_quarter"); exit_q = col_map.get("_exit_quarter")
    if not hire_q or hire_q not in df.columns: return None, ""
    hires_df = df.dropna(subset=[hire_q])
    hires_df = hires_df[hires_df[hire_q] >= "2010Q1"]
    hires    = hires_df.groupby(hire_q).size()
    exits_s  = pd.Series(dtype=int)
    if exit_q and exit_q in df.columns:
        exits_df = df.dropna(subset=[exit_q])
        exits_df = exits_df[exits_df[exit_q] >= "2010Q1"]
        exits_s  = exits_df.groupby(exit_q).size()
    all_q = sorted(set(hires.index) | set(exits_s.index))
    net   = pd.Series({q: hires.get(q, 0) - exits_s.get(q, 0) for q in all_q})
    cumnet= net.cumsum()
    colors = [GOOD if v >= 0 else BAD for v in net.values]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Quarterly Net Change", "Cumulative Growth"],
                        vertical_spacing=0.1)
    fig.add_trace(go.Bar(x=net.index, y=net.values, marker_color=colors,
                         marker_line_width=0, name="Net Change",
                         hovertemplate="<b>%{x}</b><br>Net: %{y:+,}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cumnet.index, y=cumnet.values, mode="lines",
                              fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
                              line=dict(color=C["blue"], width=2.5), name="Cumulative",
                              hovertemplate="<b>%{x}</b><br>Cumulative: %{y:,}<extra></extra>"), row=2, col=1)
    fig.update_layout(**_L("Net Headcount Change Over Time", h=520))
    return fig, f"Peak growth: {net.idxmax()} (+{net.max():,}). Net change 2010-present: {int(cumnet.iloc[-1]):+,} employees."

def attrition_gauge_overview(df, col_map, group_col):
    att = col_map.get("_attrition")
    if not att or att not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        rates  = df.groupby(group_col)[att].mean() * 100
        rates  = rates.sort_values(ascending=False)
        overall = df[att].mean() * 100
        colors  = [_att_color(v, overall) for v in rates.values]
        fig = go.Figure(go.Bar(
            x=rates.values, y=rates.index, orientation="h",
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in rates.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Attrition: %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=overall, **BENCH_LINE,
                      annotation_text=f"Overall {overall:.1f}%",
                      annotation_position="bottom right", annotation_font_size=12)
        fig.add_vline(x=ATTRITION_BENCHMARK, line_dash="dot", line_color=BAD, line_width=1,
                      annotation_text=f"⚠ {ATTRITION_BENCHMARK}% benchmark",
                      annotation_font_color=BAD, annotation_font_size=11)
        fig.update_layout(**_L(f"Attrition Rate by {group_col}"),
                          xaxis_title="Attrition Rate (%)", yaxis_title="")
        return fig, f"Overall attrition: {overall:.1f}%. {rates.idxmax()} is highest at {rates.max():.1f}% — {rates.max()-ATTRITION_BENCHMARK:+.1f}pp vs {ATTRITION_BENCHMARK}% benchmark."
    overall = df[att].mean() * 100
    gauge_color = BAD if overall > 25 else WARN if overall > ATTRITION_BENCHMARK else GOOD
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall,
        number=dict(suffix="%", font=dict(size=52, color="#0f172a"), valueformat=".1f"),
        delta=dict(reference=ATTRITION_BENCHMARK, valueformat=".1f", suffix="pp vs 15% benchmark",
                   increasing=dict(color=BAD), decreasing=dict(color=GOOD)),
        gauge=dict(
            axis=dict(range=[0, 50], tickwidth=1, tickfont_size=12),
            bar=dict(color=gauge_color, thickness=0.28),
            bgcolor="#f1f5f9",
            steps=[
                dict(range=[0, ATTRITION_BENCHMARK], color="#d1fae5"),
                dict(range=[ATTRITION_BENCHMARK, 25], color="#fef9c3"),
                dict(range=[25, 50], color="#fee2e2"),
            ],
            threshold=dict(line=dict(color="#1e293b", width=2.5), thickness=0.8, value=ATTRITION_BENCHMARK),
        ),
        title=dict(text="Overall Attrition Rate", font=dict(size=15, color="#374151")),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=340,
                      margin=dict(l=30, r=30, t=40, b=20),
                      font=dict(family="Inter,sans-serif"))
    n = int(df[att].sum())
    return fig, f"{n:,} employees ({overall:.1f}%) have left. {'⚠ Above' if overall>ATTRITION_BENCHMARK else '✅ Below'} the {ATTRITION_BENCHMARK}% industry benchmark."

def attrition_by_tenure(df, col_map, group_col):
    att = col_map.get("_attrition"); ten = col_map.get("_tenure_years")
    if not att or not ten or att not in df.columns or ten not in df.columns: return None, ""
    bins   = [0, 0.5, 1, 2, 3, 5, 8, 12, 50]
    labels = ["<6m","6m-1y","1-2y","2-3y","3-5y","5-8y","8-12y","12y+"]
    data   = df.dropna(subset=[ten]).copy()
    data["TBand"] = pd.cut(data[ten], bins=bins, labels=labels)
    if group_col and group_col in df.columns:
        top5 = df[group_col].value_counts().head(5).index
        sub  = data[data[group_col].isin(top5)]
        rates= sub.groupby(["TBand", group_col], observed=True)[att].mean() * 100
        rates= rates.reset_index(); rates.columns = ["TBand", group_col, "Rate"]
        fig  = px.line(rates, x="TBand", y="Rate", color=group_col,
                       markers=True, color_discrete_sequence=COLORS,
                       labels={"Rate": "Attrition Rate (%)", "TBand": "Tenure Band"})
        fig.update_traces(line_width=2.5, marker_size=8)
    else:
        rates  = data.groupby("TBand", observed=True)[att].mean() * 100
        counts = data.groupby("TBand", observed=True).size()
        colors = [_att_color(v, rates.mean()) for v in rates.values]
        fig = go.Figure(go.Bar(
            x=rates.index.astype(str), y=rates.values,
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in rates.values], textposition="outside",
            customdata=counts.values,
            hovertemplate="<b>%{x}</b><br>Attrition: %{y:.1f}%<br>Sample: %{customdata:,}<extra></extra>",
        ))
        fig.add_hline(y=ATTRITION_BENCHMARK, **BENCH_LINE,
                      annotation_text=f"{ATTRITION_BENCHMARK}% benchmark",
                      annotation_font_size=11)
    fig.update_layout(**_L("Attrition Rate by Tenure Band"),
                      xaxis_title="Years of Service", yaxis_title="Attrition Rate (%)")
    peak = data.groupby("TBand", observed=True)[att].mean().idxmax()
    return fig, f"Attrition peaks at {peak} tenure — the most critical window for retention interventions."

def survival_curve(df, col_map, group_col):
    att = col_map.get("_attrition"); ten = col_map.get("_tenure_years")
    if not att or not ten or att not in df.columns or ten not in df.columns: return None, ""
    data = df.dropna(subset=[ten]).copy()
    data["_att"] = data[att]
    def km(sub):
        times = sorted(sub[sub["_att"] == 1][ten].unique())[:50]
        s, tv = [1.0], [0]
        for t in times:
            at_risk = (sub[ten] >= t).sum()
            if at_risk == 0: continue
            events = ((sub[ten].round(1) == round(t, 1)) & (sub["_att"] == 1)).sum()
            s.append(s[-1] * (1 - events / at_risk)); tv.append(t)
        return tv, s
    fig = go.Figure()
    if group_col and group_col in df.columns:
        for i, grp in enumerate(df[group_col].value_counts().head(6).index):
            sub = data[data[group_col] == grp]
            if len(sub) < 20: continue
            tv, sv = km(sub)
            fig.add_trace(go.Scatter(x=tv, y=[v*100 for v in sv], mode="lines",
                                     name=grp, line=dict(color=COLORS[i], width=2.5),
                                     hovertemplate=f"<b>{grp}</b><br>Tenure: %{{x:.1f}}y<br>Still employed: %{{y:.1f}}%<extra></extra>"))
    else:
        tv, sv = km(data)
        fig.add_trace(go.Scatter(x=tv, y=[v*100 for v in sv], mode="lines",
                                 fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                                 line=dict(color=BAD, width=2.5), name="All Employees",
                                 hovertemplate="Tenure: %{x:.1f}y<br>Still employed: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=80, **BENCH_LINE, annotation_text="80% retention target",
                  annotation_font_size=11, annotation_font_color="#64748B")
    fig.update_layout(**_L("Employee Survival Curve (Kaplan-Meier)"),
                      xaxis_title="Years of Service", yaxis_title="% Still Employed")
    return fig, "Steeper early drop = high early-stage attrition. Flat curves = strong retention. Compare groups to find best-retaining units."

def flight_risk_heatmap(df, col_map, group_col):
    att = col_map.get("_attrition"); hire_y = col_map.get("_hire_year")
    grp = group_col or _g(col_map, "division", "department")
    if not att or not grp or grp not in df.columns or not hire_y or hire_y not in df.columns:
        return None, ""
    data = df.dropna(subset=[hire_y]).copy()
    data = data[data[hire_y].between(2010, 2026)]
    top  = data[grp].value_counts().head(10).index
    sub  = data[data[grp].isin(top)]
    pivot= sub.groupby([grp, hire_y])[att].mean().unstack(fill_value=0) * 100
    fig  = px.imshow(pivot, text_auto=".0f",
                     color_continuous_scale=[[0, "#f0fdf4"],[0.3, "#fef9c3"],[0.6, "#fed7aa"],[1, "#fee2e2"]],
                     zmin=0, zmax=60, aspect="auto")
    fig.update_traces(textfont=dict(size=12))
    fig.update_layout(**_L(f"Attrition Rate: {grp} × Hire Year Cohort", h=460))
    return fig, "Each cell = % of employees hired that year who later left. Red = high-attrition cohorts needing investigation."

def probation_analysis(df, col_map, group_col):
    prob_days = col_map.get("_probation_days"); prob_flag = col_map.get("_on_probation")
    if not prob_days and not prob_flag: return None, ""
    if prob_days and prob_days in df.columns:
        data = df.dropna(subset=[prob_days]).copy()
        data = data[data[prob_days].between(0, 730)]
        if group_col and group_col in df.columns:
            fig = px.violin(data, y=prob_days, x=group_col, color=group_col,
                            box=True, points="outliers", color_discrete_sequence=COLORS)
            fig.update_layout(**_L(f"Time to Confirmation by {group_col}"))
            fig.update_layout(showlegend=False, yaxis_title="Days in Probation", xaxis_title="")
        else:
            fig = px.histogram(data, x=prob_days, nbins=30, color_discrete_sequence=[C["orange"]])
            med = data[prob_days].median()
            fig.add_vline(x=med, **BENCH_LINE,
                          annotation_text=f"Median {med:.0f}d", annotation_font_size=12)
            fig.update_layout(**_L("Probation Duration Distribution"), xaxis_title="Days in Probation")
        return fig, f"Median time to confirmation: {data[prob_days].median():.0f} days ({data[prob_days].median()/30:.1f} months)."
    if prob_flag and prob_flag in df.columns and group_col and group_col in df.columns:
        active_col = col_map.get("_is_active", "__is_active")
        act = df[df[active_col] == 1] if active_col in df.columns else df
        pct = act.groupby(group_col)[prob_flag].mean() * 100
        pct = pct.sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=pct.values, y=pct.index, orientation="h",
            marker_color=[_att_color(v, pct.mean()) for v in pct.values],
            marker_line_width=0,
            text=[f"{v:.0f}%" for v in pct.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>On probation: %{x:.0f}%<extra></extra>",
        ))
        fig.add_vline(x=pct.mean(), **BENCH_LINE,
                      annotation_text=f"Avg {pct.mean():.0f}%", annotation_font_size=11)
        fig.update_layout(**_L("% Active Employees Still on Probation by Unit"),
                          xaxis_title="% on Probation")
        return fig, "High probation % may indicate recent bulk hiring or slow confirmation processes."
    return None, ""

# ══════════════════════════════════════════════════════════════════════════════
# 4 · DEMOGRAPHICS & DIVERSITY
# ══════════════════════════════════════════════════════════════════════════════

def gender_overview(df, col_map, group_col):
    gen = _g(col_map, "gender")
    if not gen or gen not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        f_pct = df.groupby(group_col)[gen].apply(lambda x: (x=="Female").mean()*100).sort_values()
        overall = (df[gen] == "Female").mean() * 100
        colors  = [BAD if v < 15 else WARN if v < 30 else GOOD for v in f_pct.values]
        fig = go.Figure(go.Bar(
            x=f_pct.values, y=f_pct.index, orientation="h",
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in f_pct.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Female: %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=overall, **BENCH_LINE,
                      annotation_text=f"Overall {overall:.1f}%", annotation_font_size=12)
        fig.add_vline(x=30, line_dash="dot", line_color=GOOD, line_width=1.5,
                      annotation_text="30% target", annotation_font_color=GOOD, annotation_font_size=11)
        fig.update_layout(**_L(f"Female Representation by {group_col}"),
                          xaxis_title="% Female", yaxis_title="")
        return fig, f"Overall female representation: {overall:.1f}%. Green = at or above 30% target."
    counts = df[gen].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, hole=0.62,
                 color_discrete_sequence=[C["blue"], C["pink"], "#94a3b8"])
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      hovertemplate="<b>%{label}</b><br>%{value:,} employees (%{percent})<extra></extra>")
    fig.update_layout(**_L("Gender Breakdown", 420), showlegend=True)
    f_pct = (df[gen] == "Female").mean()
    return fig, f"Female: {f_pct:.1%} | Male: {1-f_pct:.1%}."

def gender_pipeline(df, col_map):
    gen = _g(col_map, "gender"); mgmt = _g(col_map, "designation_main")
    dept = _g(col_map, "department", "division")
    grp  = mgmt or dept
    if not gen or not grp or gen not in df.columns or grp not in df.columns: return None, ""
    pivot = pd.crosstab(df[grp], df[gen], normalize="index") * 100
    order = ["Executive / Officer","Assistant Manager","Manager","General Manager","Director",
             "Junior","Mid","Senior","Lead","Manager"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    if len(pivot) == 0: pivot = pd.crosstab(df[grp], df[gen], normalize="index") * 100
    fig = px.bar(pivot.reset_index(), x=grp, y=pivot.columns.tolist(),
                 barmode="stack", color_discrete_sequence=[C["blue"], C["pink"], "#94a3b8"],
                 labels={"value":"% at Level", grp:""},
                 text_auto=".0f")
    fig.update_traces(textposition="inside", textfont_size=11)
    fig.update_layout(**_L("Gender Pipeline by Level/Category"), yaxis_title="% of Employees",
                      legend_title="Gender")
    return fig, "A narrowing female share at senior levels signals a 'leaky pipeline' — a key D&I metric to monitor and address."

def age_distribution(df, col_map, group_col):
    age = col_map.get("_age")
    if not age or age not in df.columns: return None, ""
    data = df.dropna(subset=[age]); data = data[data[age].between(18, 65)]
    if group_col and group_col in df.columns:
        top5 = df[group_col].value_counts().head(5).index
        sub  = data[data[group_col].isin(top5)]
        fig  = px.violin(sub, y=age, x=group_col, color=group_col,
                         box=True, points=False, color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Age Distribution by {group_col}"))
        fig.update_layout(showlegend=False, yaxis_title="Age", xaxis_title="")
    else:
        bins   = [18,25,35,45,55,65]; labels = ["18-24","25-34","35-44","45-54","55-64"]
        data   = data.copy()
        data["AgeBand"] = pd.cut(data[age], bins=bins, labels=labels, right=False)
        counts = data["AgeBand"].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=counts.index.astype(str), y=counts.values,
            marker=dict(color=counts.values, colorscale="Blues", showscale=False),
            text=counts.values, textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:,} employees<extra></extra>",
        ))
        fig.update_layout(**_L("Age Band Distribution"), yaxis_title="Employees", xaxis_title="Age Band",
                          showlegend=False)
    return fig, f"Average age: {data[age].mean():.0f} years. Spread: {data[age].min():.0f}–{data[age].max():.0f}."

def religion_distribution(df, col_map, group_col):
    rel = _g(col_map, "religion")
    if not rel or rel not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col], df[rel], normalize="index") * 100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(), x=group_col, y=list(ct.columns),
                     barmode="stack", color_discrete_sequence=COLORS, labels={"value":"% of Employees"})
        fig.update_layout(**_L(f"Religion Distribution by {group_col}"), yaxis_title="% of Employees")
    else:
        counts = df[rel].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textinfo="percent+label", textposition="outside")
        fig.update_layout(**_L("Religion Distribution", 420))
    return fig, f"Dominant faith: {df[rel].value_counts().idxmax()} ({df[rel].value_counts().iloc[0]/len(df):.0%})."

def district_heatmap(df, col_map, group_col):
    dist = _g(col_map, "district")
    if not dist or dist not in df.columns: return None, ""
    top_d = df[dist].value_counts().head(20).index
    sub   = df[df[dist].isin(top_d)]
    if group_col and group_col in df.columns:
        ct  = pd.crosstab(sub[dist], sub[group_col])
        ct  = ct.reindex(top_d)
        fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(**_L(f"District Origin × {group_col}", h=540), coloraxis_showscale=False)
    else:
        counts = df[dist].value_counts().head(20).sort_values()
        fig = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker=dict(color=counts.values, colorscale="Blues", showscale=False),
            text=counts.values, textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,} employees<extra></extra>",
        ))
        fig.update_layout(**_L("Top 20 Employee Districts"), xaxis_title="Employees", showlegend=False)
    return fig, f"Employees from {df[dist].nunique()} districts. Top origin: {df[dist].value_counts().idxmax()}."

def education_breakdown(df, col_map, group_col):
    edu = _g(col_map, "education")
    if not edu or edu not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        top_e = df[edu].value_counts().head(8).index
        sub   = df[df[edu].isin(top_e)]
        ct    = pd.crosstab(sub[group_col], sub[edu], normalize="index") * 100
        ct    = ct.reindex(df[group_col].value_counts().head(10).index)
        fig   = px.bar(ct.reset_index(), x=group_col, y=list(ct.columns),
                       barmode="stack", color_discrete_sequence=COLORS, labels={"value":"% Employees"})
        fig.update_layout(**_L(f"Education Level by {group_col}"), yaxis_title="% of Workforce")
    else:
        counts = df[edu].value_counts().head(12).sort_values()
        fig = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker=dict(color=counts.values, colorscale="Purples", showscale=False),
            text=counts.values, textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:,} employees<extra></extra>",
        ))
        fig.update_layout(**_L("Education Level Distribution"), showlegend=False)
    return fig, f"Most common qualification: {df[edu].value_counts().idxmax()} ({df[edu].value_counts().iloc[0]:,} employees)."

def marital_status(df, col_map, group_col):
    mar = _g(col_map, "marital_status")
    if not mar or mar not in df.columns: return None, ""
    if group_col and group_col in df.columns:
        ct = pd.crosstab(df[group_col], df[mar], normalize="index") * 100
        ct = ct.reindex(df[group_col].value_counts().head(10).index)
        fig = px.bar(ct.reset_index(), x=group_col, y=list(ct.columns),
                     barmode="stack", color_discrete_sequence=COLORS, labels={"value":"% Employees"})
        fig.update_layout(**_L(f"Marital Status by {group_col}"), yaxis_title="% of Workforce")
    else:
        counts = df[mar].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textinfo="percent+label", textposition="outside")
        fig.update_layout(**_L("Marital Status Distribution", 400))
    return fig, f"Married: {(df[mar]=='Married').mean():.0%} | Single: {(df[mar]=='Single').mean():.0%}."

# ══════════════════════════════════════════════════════════════════════════════
# 5 · LEAVE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def primary_leave_utilisation(df, col_map, group_col):
    taken    = _g(col_map, "primary_leave_taken")
    entitled = _g(col_map, "leave_entitled")
    if not taken or taken not in df.columns: return None, ""
    data     = df.dropna(subset=[taken]).copy()
    data[taken] = pd.to_numeric(data[taken], errors="coerce")
    if group_col and group_col in df.columns:
        avg_t = data.groupby(group_col)[taken].mean().sort_values()
        if entitled and entitled in data.columns:
            data[entitled] = pd.to_numeric(data[entitled], errors="coerce")
            avg_e = data.groupby(group_col)[entitled].mean()
            util  = (avg_t / avg_e * 100).fillna(0)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=avg_e.reindex(avg_t.index).values, x=avg_t.index,
                                 name="Entitled", marker_color="#e2e8f0", marker_line_width=0))
            fig.add_trace(go.Bar(y=avg_t.values, x=avg_t.index,
                                 name="Taken", marker_color=C["teal"], marker_line_width=0,
                                 text=[f"{v:.1f}d" for v in avg_t.values], textposition="outside"))
            fig.update_layout(**_L(f"Leave Entitlement vs Taken by {group_col}"),
                              barmode="overlay", yaxis_title="Avg Days", xaxis_title="")
        else:
            colors = [BAD if v > avg_t.mean() * 1.2 else WARN if v > avg_t.mean() else GOOD
                      for v in avg_t.values]
            fig = go.Figure(go.Bar(
                x=avg_t.values, y=avg_t.index, orientation="h",
                marker_color=colors, marker_line_width=0,
                text=[f"{v:.1f}d" for v in avg_t.values], textposition="outside",
                hovertemplate="<b>%{y}</b><br>Avg leave taken: %{x:.1f} days<extra></extra>",
            ))
            fig.add_vline(x=avg_t.mean(), **BENCH_LINE,
                          annotation_text=f"Avg {avg_t.mean():.1f}d")
            fig.update_layout(**_L(f"Avg Privileged Leave by {group_col}"), xaxis_title="Days/Year")
        return fig, f"Avg privileged leave taken: {data[taken].mean():.1f} days/year across the organisation."
    fig = px.histogram(data, x=taken, nbins=30, color_discrete_sequence=[C["teal"]])
    fig.add_vline(x=data[taken].mean(), **BENCH_LINE,
                  annotation_text=f"Mean {data[taken].mean():.1f}d")
    fig.update_layout(**_L("Privileged Leave Days Taken"), xaxis_title="Days", showlegend=False)
    return fig, f"Avg: {data[taken].mean():.1f} days. {(data[taken]==0).mean():.0%} took zero leave — a burnout risk signal."

def secondary_leave_analysis(df, col_map, group_col):
    taken = _g(col_map, "secondary_leave_taken")
    if not taken or taken not in df.columns: return None, ""
    data  = df.dropna(subset=[taken]).copy()
    data[taken] = pd.to_numeric(data[taken], errors="coerce").fillna(0)
    if group_col and group_col in df.columns:
        avg = data.groupby(group_col)[taken].mean().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=avg.values, y=avg.index, orientation="h",
            marker_color=[_att_color(v, avg.mean()) for v in avg.values],
            marker_line_width=0,
            text=[f"{v:.1f}d" for v in avg.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg sick leave: %{x:.1f} days<extra></extra>",
        ))
        fig.add_vline(x=avg.mean(), **BENCH_LINE,
                      annotation_text=f"Avg {avg.mean():.1f}d", annotation_font_size=11)
        fig.update_layout(**_L(f"Avg Sick/Casual Leave by {group_col}"), xaxis_title="Days/Year")
        return fig, f"High sick leave in {avg.idxmax()} ({avg.max():.1f}d/yr) may signal workload or health issues."
    fig = px.histogram(data, x=taken, nbins=25, color_discrete_sequence=[C["orange"]], marginal="box")
    fig.update_layout(**_L("Sick / Casual Leave Distribution"), xaxis_title="Days Taken", showlegend=False)
    return fig, f"Avg sick/casual leave: {data[taken].mean():.1f} days/year."

def leave_forfeiture(df, col_map, group_col):
    forf = next((c for c in df.columns if "forfeit" in c.lower() or "forfit" in c.lower()), None)
    if not forf: return None, ""
    data     = df.copy(); data[forf] = pd.to_numeric(data[forf], errors="coerce").fillna(0)
    if group_col and group_col in df.columns:
        avg   = data.groupby(group_col)[forf].mean().sort_values(ascending=False)
        total = data.groupby(group_col)[forf].sum()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=avg.index, y=avg.values, name="Avg Days Forfeited",
                             marker_color=BAD, marker_line_width=0,
                             hovertemplate="<b>%{x}</b><br>Avg forfeited: %{y:.1f}d<extra></extra>"))
        fig.add_trace(go.Scatter(x=total.index, y=total.values, mode="markers",
                                 name="Total Days Lost", marker_size=12, marker_color=WARN,
                                 hovertemplate="<b>%{x}</b><br>Total lost: %{y:,}d<extra></extra>"),
                      secondary_y=True)
        fig.update_yaxes(title_text="Avg Days Forfeited", secondary_y=False)
        fig.update_yaxes(title_text="Total Days Lost", secondary_y=True, showgrid=False)
        fig.update_layout(**_L(f"Leave Forfeiture by {group_col}"))
        return fig, f"Total leave forfeited: {data[forf].sum():,.0f} days — a direct wellbeing and compensation cost."
    total = data[forf].sum()
    fig = px.histogram(data[data[forf] > 0], x=forf, nbins=20, color_discrete_sequence=[BAD])
    fig.update_layout(**_L("Leave Forfeiture Distribution"), showlegend=False)
    return fig, f"Total forfeited leave: {total:,.0f} days. {(data[forf]>0).mean():.0%} of employees forfeited at least 1 day."

# ══════════════════════════════════════════════════════════════════════════════
# 6 · CAREER & PROGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def promotion_rate(df, col_map, group_col):
    promo_col = _g(col_map, "last_promotion_date")
    if not promo_col or promo_col not in df.columns: return None, ""
    df2  = df.copy()
    df2[promo_col] = pd.to_datetime(df2[promo_col], errors="coerce")
    valid = df2[promo_col].notna() & (df2[promo_col].dt.year > 1950)
    df2["__has_promo"] = valid.astype(int)
    if group_col and group_col in df.columns:
        rates  = df2.groupby(group_col)["__has_promo"].mean() * 100
        rates  = rates.sort_values()
        colors = [GOOD if v >= rates.mean() else WARN for v in rates.values]
        fig = go.Figure(go.Bar(
            x=rates.values, y=rates.index, orientation="h",
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.0f}%" for v in rates.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>% with promotion: %{x:.0f}%<extra></extra>",
        ))
        fig.add_vline(x=rates.mean(), **BENCH_LINE,
                      annotation_text=f"Avg {rates.mean():.0f}%", annotation_font_size=12)
        fig.update_layout(**_L(f"% Employees with Promotion Record by {group_col}"),
                          xaxis_title="% with Any Promotion History")
        return fig, f"Overall: {valid.mean():.0%} of employees have a promotion on record."
    yr  = df2.loc[valid, promo_col].dt.year.value_counts().sort_index()
    yr  = yr[yr.index >= 2010]
    fig = go.Figure(go.Bar(
        x=yr.index, y=yr.values,
        marker=dict(color=yr.values, colorscale="Purples", showscale=False),
        text=yr.values, textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:,} promotions<extra></extra>",
    ))
    fig.update_layout(**_L("Promotions Per Year"), xaxis_title="Year", yaxis_title="Promotions",
                      showlegend=False)
    return fig, f"Most promotions: {yr.idxmax()} ({yr.max():,}). Total on record: {int(valid.sum()):,}."

def tenure_distribution(df, col_map, group_col):
    ten = col_map.get("_tenure_years")
    if not ten or ten not in df.columns: return None, ""
    data = df.dropna(subset=[ten]); data = data[data[ten].between(0, 40)]
    if group_col and group_col in df.columns:
        top6 = df[group_col].value_counts().head(6).index
        sub  = data[data[group_col].isin(top6)]
        fig  = px.violin(sub, y=ten, x=group_col, color=group_col,
                         box=True, points=False, color_discrete_sequence=COLORS)
        fig.update_layout(**_L(f"Tenure Distribution by {group_col}"))
        fig.update_layout(showlegend=False, yaxis_title="Years of Service", xaxis_title="")
    else:
        fig = px.histogram(data, x=ten, nbins=30, color_discrete_sequence=[C["blue"]], marginal="box")
        fig.add_vline(x=data[ten].mean(), **BENCH_LINE,
                      annotation_text=f"Mean {data[ten].mean():.1f}y",
                      annotation_font_size=12)
        fig.update_layout(**_L("Tenure Distribution"), xaxis_title="Years of Service",
                          showlegend=False)
    return fig, f"Avg tenure: {data[ten].mean():.1f} years. {(data[ten]<1).mean():.0%} have < 1 year service."

def yrs_since_promotion(df, col_map, group_col):
    ysp = col_map.get("_yrs_since_promo")
    if not ysp or ysp not in df.columns: return None, ""
    data = df.dropna(subset=[ysp]); data = data[data[ysp].between(0, 20)]
    if group_col and group_col in df.columns:
        avg    = data.groupby(group_col)[ysp].median().sort_values(ascending=False)
        colors = [BAD if v > 5 else WARN if v > 3 else GOOD for v in avg.values]
        fig = go.Figure(go.Bar(
            x=avg.values, y=avg.index, orientation="h",
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}y" for v in avg.values], textposition="outside",
            hovertemplate="<b>%{y}</b><br>Median yrs since promo: %{x:.1f}<extra></extra>",
        ))
        fig.add_vline(x=3, line_dash="dot", line_color=WARN, line_width=1.5,
                      annotation_text="3y warning threshold",
                      annotation_font_color=WARN, annotation_font_size=11)
        fig.update_layout(**_L(f"Median Years Since Last Promotion by {group_col}"),
                          xaxis_title="Years", yaxis_title="")
        return fig, f"Longest gap: {avg.idxmax()} ({avg.max():.1f} years). Red bars (>5y) = high flight risk."
    fig = px.histogram(data, x=ysp, nbins=25, color_discrete_sequence=[C["orange"]], marginal="box")
    fig.add_vline(x=data[ysp].median(), **BENCH_LINE,
                  annotation_text=f"Median {data[ysp].median():.1f}y")
    fig.update_layout(**_L("Years Since Last Promotion"), xaxis_title="Years", showlegend=False)
    return fig, f"Median time since last promotion: {data[ysp].median():.1f} years. {(data[ysp]>4).mean():.0%} waiting 4+ years."

def salary_vs_tenure(df, col_map, group_col):
    sal = _g(col_map, "salary"); ten = col_map.get("_tenure_years")
    if not sal or not ten or sal not in df.columns or ten not in df.columns: return None, ""
    dept = group_col
    data = df.dropna(subset=[sal, ten])
    fig  = px.scatter(data, x=ten, y=sal, color=dept if dept and dept in data.columns else None,
                      trendline="lowess", trendline_options=dict(frac=0.3),
                      opacity=0.55, color_discrete_sequence=COLORS,
                      labels={ten: "Years of Service", sal: "Salary"},
                      hover_data=[c for c in [_g(col_map,"employee_id"), dept] if c and c in data.columns])
    fig.update_traces(marker_size=5)
    fig.update_layout(**_L("Salary vs. Tenure"), yaxis_tickformat=",")
    corr = data[sal].corr(data[ten])
    return fig, f"Salary-tenure correlation: {corr:.2f}. LOWESS trend shows how pay grows with experience."

def salary_by_group(df, col_map, group_col):
    sal = _g(col_map, "salary")
    if not sal or not group_col or sal not in df.columns or group_col not in df.columns: return None, ""
    data    = df.dropna(subset=[sal])
    overall = data[sal].median()
    dept_med= data.groupby(group_col)[sal].median().sort_values()
    pct_diff= ((dept_med - overall) / overall * 100)
    colors  = [GOOD if v >= 0 else BAD for v in pct_diff.values]
    fig = go.Figure(go.Bar(
        x=pct_diff.values, y=pct_diff.index, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.1f}%" for v in pct_diff.values], textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:+.1f}% vs company median<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#374151", line_width=1.5)
    fig.update_layout(**_L(f"Median Salary vs. Company Median by {group_col}"),
                      xaxis_title="% Difference from Company Median")
    return fig, f"Company median pay: {overall:,.0f}. Green = above median, red = below."

# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

CHART_REGISTRY = {
    "🏢 Overview": [
        {"name": "Workforce Sunburst — Full Hierarchy",   "fn": headcount_sunburst,        "needs_hierarchy": True, "requires": []},
        {"name": "Business Unit KPI Scorecard (Heatmap)",           "fn": kpi_scorecard_heatmap,     "requires": []},
        {"name": "Headcount — Active vs Former",          "fn": headcount_active_vs_former,"requires": []},
        {"name": "Employment Type Mix",                   "fn": employment_type_mix,       "requires": ["employment_type"]},
        {"name": "Management Level Hierarchy",            "fn": management_level_comparison,"requires": ["designation_main"]},
        {"name": "Workforce by Location",                 "fn": headcount_by_location,     "requires": []},
    ],
    "👥 Structure": [
        {"name": "Org Treemap — Drill Down",              "fn": org_treemap,               "needs_hierarchy": True, "requires": []},
        {"name": "Role Distribution Treemap",             "fn": designation_treemap,       "requires": ["designation"]},
        {"name": "Grade / Band Distribution",             "fn": grade_distribution,        "requires": ["grade"]},
        {"name": "Span of Control",                       "fn": span_of_control,           "no_group": True, "requires": ["super_code"]},
        {"name": "Role Complexity by Unit",               "fn": role_complexity,           "requires": ["designation"]},
        {"name": "Salary vs. Tenure",                     "fn": salary_vs_tenure,          "requires": ["salary","_tenure_years"]},
        {"name": "Salary by Business Unit",               "fn": salary_by_group,           "requires": ["salary"]},
    ],
    "📅 Attrition": [
        {"name": "Joiners vs Leavers Timeline",           "fn": joiners_leavers_timeline,  "requires": ["_hire_quarter"]},
        {"name": "Net Headcount Change (Quarterly)",      "fn": net_headcount_change,      "requires": ["_hire_quarter"]},
        {"name": "Attrition Rate Overview",               "fn": attrition_gauge_overview,  "requires": ["_attrition"]},
        {"name": "Attrition by Tenure Band",              "fn": attrition_by_tenure,       "requires": ["_attrition","_tenure_years"]},
        {"name": "Employee Survival Curve",               "fn": survival_curve,            "requires": ["_attrition","_tenure_years"]},
        {"name": "Attrition by Hire Cohort (Heatmap)",   "fn": flight_risk_heatmap,       "requires": ["_attrition","_hire_year"]},
        {"name": "Probation Status & Duration",           "fn": probation_analysis,        "requires": ["_on_probation"]},
    ],
    "🌍 Diversity": [
        {"name": "Gender Representation",                 "fn": gender_overview,           "requires": ["gender"]},
        {"name": "Gender Pipeline by Level",              "fn": gender_pipeline,           "no_group": True, "requires": ["gender"]},
        {"name": "Age Distribution",                      "fn": age_distribution,          "requires": ["_age"]},
        {"name": "Education Level",                       "fn": education_breakdown,       "requires": ["education"]},
        {"name": "Religion Distribution",                 "fn": religion_distribution,     "requires": ["religion"]},
        {"name": "Geographic Origin (District)",          "fn": district_heatmap,          "requires": ["district"]},
        {"name": "Marital Status",                        "fn": marital_status,            "requires": ["marital_status"]},
    ],
    "📋 Leave": [
        {"name": "Privileged Leave Utilisation",          "fn": primary_leave_utilisation, "requires": ["primary_leave_taken"]},
        {"name": "Sick / Casual Leave Analysis",          "fn": secondary_leave_analysis,  "requires": ["secondary_leave_taken"]},
        {"name": "Leave Forfeiture Analysis",             "fn": leave_forfeiture,          "requires": []},
    ],
    "🎯 Career": [
        {"name": "Promotion Rate by Unit",                "fn": promotion_rate,            "requires": ["last_promotion_date"]},
        {"name": "Tenure Distribution",                   "fn": tenure_distribution,       "requires": ["_tenure_years"]},
        {"name": "Years Since Last Promotion",            "fn": yrs_since_promotion,       "requires": ["_yrs_since_promo"]},
    ],
}


def get_available(df, col_map):
    available = {}
    for cat, charts in CHART_REGISTRY.items():
        usable = []
        for c in charts:
            reqs = c.get("requires", [])
            ok   = all(
                (col_map.get(r) and col_map[r] in df.columns)
                or (r.startswith("__") and r in df.columns)
                or (r in df.columns)
                for r in reqs
            )
            if ok:
                usable.append(c)
        if usable:
            available[cat] = usable
    return available
