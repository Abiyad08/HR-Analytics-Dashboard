"""
insights.py — Generates automatic narrative insights from the prepared HR dataframe.
Returns a list of (emoji, headline, detail) tuples.
"""

import pandas as pd
import numpy as np

ATTRITION_BENCHMARK = 15   # industry reference %
FEMALE_TARGET       = 30   # minimum representation target %
PROBATION_WARN      = 20   # % active on probation — high threshold
TENURE_WARN_LOW     = 2.0  # average tenure years below this = instability
PROMO_STALE         = 4.0  # years without promotion = flight risk


def _g(col_map, *concepts):
    for c in concepts:
        v = col_map.get(c)
        if v: return v
    return None


def generate(df: pd.DataFrame, col_map: dict) -> list[tuple]:
    insights = []

    att_col    = col_map.get("_attrition")
    ten_col    = col_map.get("_tenure_years")
    gen_col    = _g(col_map, "gender")
    prob_col   = col_map.get("_on_probation")
    active_col = col_map.get("_is_active", "__is_active")
    sal_col    = _g(col_map, "salary")
    promo_col  = _g(col_map, "last_promotion_date")
    ysp_col    = col_map.get("_yrs_since_promo")
    age_col    = col_map.get("_age")
    hire_col   = _g(col_map, "hire_date")
    leave_col  = _g(col_map, "primary_leave_taken")
    dept_col   = _g(col_map, "division", "department", "business_unit", "dept_group")

    active_df = df[df[active_col] == 1] if active_col in df.columns else df

    # ── 1. Headcount ────────────────────────────────────────────────────────
    total  = len(df)
    active = len(active_df)
    if active_col in df.columns:
        insights.append((
            "👥",
            f"{active:,} active employees out of {total:,} total records",
            f"Historical attrition has removed {total-active:,} employees "
            f"({(total-active)/total:.0%} of all-time workforce)."
        ))
    else:
        insights.append(("👥", f"{total:,} employee records in this dataset", ""))

    # ── 2. Attrition ────────────────────────────────────────────────────────
    if att_col and att_col in df.columns:
        rate = df[att_col].mean() * 100
        diff = rate - ATTRITION_BENCHMARK
        emoji = "🚨" if rate > 25 else "⚠️" if rate > ATTRITION_BENCHMARK else "✅"
        label = "critically high" if rate > 30 else "above benchmark" if rate > ATTRITION_BENCHMARK else "within target"
        insights.append((
            emoji,
            f"Attrition rate is {rate:.1f}% — {label}",
            f"{diff:+.1f}pp vs the {ATTRITION_BENCHMARK}% industry benchmark. "
            f"Review exit interview data to identify the root causes driving departures."
        ))

    # ── 3. Tenure ────────────────────────────────────────────────────────────
    if ten_col and ten_col in df.columns:
        avg_ten = df[ten_col].mean()
        low_pct = (df[ten_col] < 1).mean() * 100
        emoji   = "⚠️" if avg_ten < TENURE_WARN_LOW else "📅"
        insights.append((
            emoji,
            f"Average tenure is {avg_ten:.1f} years",
            f"{low_pct:.0f}% of employees have less than 1 year of service — "
            f"{'high volume of recent hires or early attrition concern.' if low_pct > 25 else 'relatively stable workforce entry pattern.'}"
        ))

    # ── 4. Gender ────────────────────────────────────────────────────────────
    if gen_col and gen_col in df.columns:
        f_pct = (df[gen_col] == "Female").mean() * 100
        emoji = "✅" if f_pct >= FEMALE_TARGET else "⚠️" if f_pct >= 15 else "🚨"
        insights.append((
            emoji,
            f"Female representation is {f_pct:.1f}%",
            f"{'At or above' if f_pct >= FEMALE_TARGET else 'Below'} the {FEMALE_TARGET}% D&I target. "
            f"{'Consider targeted hiring and retention programs for female talent.' if f_pct < FEMALE_TARGET else 'Continue monitoring pipeline at senior levels.'}"
        ))

    # ── 5. Probation ────────────────────────────────────────────────────────
    if prob_col and prob_col in df.columns and active_col in df.columns:
        prob_pct = active_df[prob_col].mean() * 100 if prob_col in active_df.columns else 0
        if prob_pct > 0:
            emoji = "⚠️" if prob_pct > PROBATION_WARN else "ℹ️"
            insights.append((
                emoji,
                f"{prob_pct:.0f}% of active employees are on probation",
                f"{int(active_df[prob_col].sum())} employees yet to be confirmed. "
                f"{'Consider accelerating confirmation reviews.' if prob_pct > PROBATION_WARN else 'Probation pipeline appears manageable.'}"
            ))

    # ── 6. Promotion staleness ───────────────────────────────────────────────
    if ysp_col and ysp_col in df.columns:
        stale_pct = (df[ysp_col] > PROMO_STALE).mean() * 100
        median_ysp= df[ysp_col].median()
        emoji = "⚠️" if stale_pct > 30 else "🎯"
        insights.append((
            emoji,
            f"{stale_pct:.0f}% of employees haven't been promoted in {PROMO_STALE:.0f}+ years",
            f"Median time since last promotion is {median_ysp:.1f} years. "
            f"{'Career stagnation is a leading attrition predictor — review promotion pipelines.' if stale_pct > 30 else 'Promotion cadence appears reasonable.'}"
        ))

    # ── 7. Largest group ─────────────────────────────────────────────────────
    if dept_col and dept_col in df.columns:
        vc   = df[dept_col].value_counts()
        top  = vc.idxmax(); top_n = vc.max()
        insights.append((
            "🏢",
            f"{top} is the largest business unit ({top_n:,} employees, {top_n/total:.0%})",
            f"Workforce spans {df[dept_col].nunique()} {dept_col.lower().replace('_',' ')} groups."
        ))

    # ── 8. Salary ─────────────────────────────────────────────────────────────
    if sal_col and sal_col in df.columns:
        med_sal = df[sal_col].median()
        q10 = df[sal_col].quantile(0.1); q90 = df[sal_col].quantile(0.9)
        sal_range = q90 - q10
        ratio = q90 / q10 if q10 > 0 else 0
        insights.append((
            "💰",
            f"Median compensation: {med_sal:,.0f}",
            f"Middle 80% pay range: {q10:,.0f}–{q90:,.0f} "
            f"(ratio {ratio:.1f}×). "
            f"{'High pay spread — review equity across levels and departments.' if ratio > 4 else 'Pay distribution appears proportionate.'}"
        ))

    # ── 9. Leave utilisation ──────────────────────────────────────────────────
    if leave_col and leave_col in df.columns:
        leave_data = pd.to_numeric(df[leave_col], errors="coerce").dropna()
        if len(leave_data) > 10:
            zero_pct = (leave_data == 0).mean() * 100
            avg_leave= leave_data.mean()
            emoji = "⚠️" if zero_pct > 30 else "📋"
            insights.append((
                emoji,
                f"Avg privileged leave taken: {avg_leave:.1f} days/year",
                f"{zero_pct:.0f}% of employees took zero leave. "
                f"{'High zero-leave rate is a burnout risk indicator.' if zero_pct > 30 else 'Leave utilisation appears healthy.'}"
            ))

    # ── 10. Age/retirement risk ───────────────────────────────────────────────
    if age_col and age_col in df.columns:
        age_data   = df[age_col].dropna()
        near_retire= (age_data >= 55).mean() * 100
        avg_age    = age_data.mean()
        if near_retire > 10:
            insights.append((
                "📊",
                f"{near_retire:.0f}% of employees are aged 55+ — succession planning needed",
                f"Average workforce age: {avg_age:.0f} years. "
                f"Consider knowledge transfer programmes for critical roles."
            ))

    return insights
