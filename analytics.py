"""
Core analytics functions. Each function takes the HR dataframe and returns
a matplotlib Figure plus a short text insight, so the app layer stays thin.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
PALETTE = ["#2563EB", "#7C3AED", "#DB2777", "#EA580C", "#16A34A", "#0891B2", "#CA8A04"]


def _style_fig(fig, title):
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------- HEADCOUNT ----------
def headcount_by_department(df: pd.DataFrame):
    counts = df["Department"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(counts.index, counts.values, color=PALETTE[0])
    ax.set_xlabel("Number of Employees")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
    _style_fig(fig, "Headcount by Department")
    insight = f"{counts.idxmax()} is the largest department with {counts.max()} employees ({counts.max()/len(df):.0%} of total headcount)."
    return fig, insight


def headcount_by_location(df: pd.DataFrame):
    counts = df["Location"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           colors=PALETTE, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    _style_fig(fig, "Workforce Distribution by Location")
    insight = f"{counts.idxmax()} hosts the largest share of the workforce ({counts.max()/len(df):.0%})."
    return fig, insight


# ---------- ATTRITION ----------
def attrition_overview(df: pd.DataFrame):
    counts = df["Attrition"].value_counts()
    rate = counts.get("Yes", 0) / len(df)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = ["#DC2626" if x == "Yes" else "#16A34A" for x in counts.index]
    ax.bar(counts.index, counts.values, color=colors, width=0.5)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Employees")
    _style_fig(fig, f"Overall Attrition Rate: {rate:.1%}")
    insight = f"Overall attrition rate is {rate:.1%} ({counts.get('Yes', 0)} of {len(df)} employees have left)."
    return fig, insight


def attrition_by_department(df: pd.DataFrame):
    rate = df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(rate.index, rate.values * 100, color=PALETTE[2])
    ax.set_ylabel("Attrition Rate (%)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i, v in enumerate(rate.values):
        ax.text(i, v * 100 + 0.5, f"{v:.0%}", ha="center", fontsize=9)
    _style_fig(fig, "Attrition Rate by Department")
    insight = f"{rate.idxmax()} has the highest attrition rate at {rate.max():.1%}, vs. company average of {(df['Attrition']=='Yes').mean():.1%}."
    return fig, insight


def attrition_by_tenure(df: pd.DataFrame):
    bins = [0, 1, 2, 5, 10, 100]
    labels = ["<1 yr", "1-2 yrs", "2-5 yrs", "5-10 yrs", "10+ yrs"]
    df = df.copy()
    df["TenureBand"] = pd.cut(df["TenureYears"], bins=bins, labels=labels)
    rate = df.groupby("TenureBand", observed=True)["Attrition"].apply(lambda x: (x == "Yes").mean())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rate.index.astype(str), rate.values * 100, marker="o", color=PALETTE[1], linewidth=2.5, markersize=8)
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_xlabel("Tenure")
    _style_fig(fig, "Attrition Rate by Tenure Band")
    insight = f"Employees with {rate.idxmax()} tenure show the highest attrition risk ({rate.max():.1%})."
    return fig, insight


# ---------- SALARY ----------
def salary_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["Salary"], bins=25, color=PALETTE[4], edgecolor="white")
    ax.axvline(df["Salary"].median(), color="#DC2626", linestyle="--", label=f"Median: ${df['Salary'].median():,.0f}")
    ax.set_xlabel("Salary ($)")
    ax.set_ylabel("Number of Employees")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend()
    _style_fig(fig, "Salary Distribution")
    insight = f"Median salary is ${df['Salary'].median():,.0f}; range spans ${df['Salary'].min():,.0f} to ${df['Salary'].max():,.0f}."
    return fig, insight


def salary_by_department_level(df: pd.DataFrame):
    pivot = df.pivot_table(values="Salary", index="Department", columns="JobLevel", aggfunc="mean")
    order = ["Junior", "Mid", "Senior", "Lead", "Manager"]
    pivot = pivot[[c for c in order if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="Blues", ax=ax, cbar_kws={"label": "Avg Salary ($)"})
    _style_fig(fig, "Average Salary by Department & Level")
    insight = "Heatmap reveals where pay bands are highest/lowest across departments and seniority — useful for spotting pay equity gaps."
    return fig, insight


def gender_pay_gap(df: pd.DataFrame):
    avg = df.groupby("Gender")["Salary"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(avg.index, avg.values, color=PALETTE[3])
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    for i, v in enumerate(avg.values):
        ax.text(i, v + 500, f"${v:,.0f}", ha="center", fontsize=9)
    _style_fig(fig, "Average Salary by Gender")
    gap = (avg.max() - avg.min()) / avg.max()
    insight = f"Pay gap between highest and lowest average-salary gender groups is {gap:.1%}."
    return fig, insight


# ---------- PERFORMANCE & ENGAGEMENT ----------
def engagement_vs_attrition(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for status, color in [("No", "#16A34A"), ("Yes", "#DC2626")]:
        subset = df[df["Attrition"] == status]
        ax.hist(subset["EngagementScore"], bins=15, alpha=0.6, label=f"Attrition: {status}", color=color)
    ax.set_xlabel("Engagement Score")
    ax.set_ylabel("Number of Employees")
    ax.legend()
    _style_fig(fig, "Engagement Score vs. Attrition")
    avg_stay = df[df["Attrition"] == "No"]["EngagementScore"].mean()
    avg_leave = df[df["Attrition"] == "Yes"]["EngagementScore"].mean()
    insight = f"Employees who left averaged {avg_leave:.0f} engagement score vs. {avg_stay:.0f} for those who stayed."
    return fig, insight


def performance_distribution(df: pd.DataFrame):
    counts = df["PerformanceRating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color=PALETTE[5])
    ax.set_xlabel("Performance Rating (1-5)")
    ax.set_ylabel("Number of Employees")
    _style_fig(fig, "Performance Rating Distribution")
    insight = f"Most employees ({counts.max()}) are rated {counts.idxmax()}/5; mean rating is {df['PerformanceRating'].mean():.1f}."
    return fig, insight


# ---------- DIVERSITY ----------
def diversity_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    gender_counts = df["Gender"].value_counts()
    axes[0].pie(gender_counts.values, labels=gender_counts.index, autopct="%1.0f%%",
                colors=PALETTE, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("Gender Mix", fontsize=11)

    edu_counts = df["Education"].value_counts()
    axes[1].barh(edu_counts.index, edu_counts.values, color=PALETTE[1])
    axes[1].set_title("Education Level", fontsize=11)
    axes[1].set_xlabel("Employees")

    _style_fig(fig, "Workforce Diversity Snapshot")
    insight = f"Gender mix is led by {gender_counts.idxmax()} ({gender_counts.max()/len(df):.0%}); most common education level is {edu_counts.idxmax()}."
    return fig, insight


def age_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["Age"], bins=20, color=PALETTE[6], edgecolor="white")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Employees")
    _style_fig(fig, "Age Distribution")
    insight = f"Average employee age is {df['Age'].mean():.0f}; workforce spans {df['Age'].min()}-{df['Age'].max()} years."
    return fig, insight


# Registry used by the app to build the menu dynamically
ANALYTICS_REGISTRY = {
    "Headcount by Department": headcount_by_department,
    "Workforce by Location": headcount_by_location,
    "Overall Attrition Rate": attrition_overview,
    "Attrition by Department": attrition_by_department,
    "Attrition by Tenure": attrition_by_tenure,
    "Salary Distribution": salary_distribution,
    "Salary by Department & Level (Heatmap)": salary_by_department_level,
    "Gender Pay Gap": gender_pay_gap,
    "Engagement vs. Attrition": engagement_vs_attrition,
    "Performance Rating Distribution": performance_distribution,
    "Diversity Snapshot": diversity_overview,
    "Age Distribution": age_distribution,
}

REQUIRED_COLUMNS = [
    "Department", "JobLevel", "Age", "Gender", "Location", "Education",
    "TenureYears", "Salary", "PerformanceRating", "EngagementScore",
    "OvertimeHoursMonth", "DistanceFromOfficeKm", "YearsSinceLastPromotion",
    "TrainingHoursYear", "Attrition",
]
