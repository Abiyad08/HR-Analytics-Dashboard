"""
data_prep.py
Cleans, enriches, and standardises any HR dataset.
Detects conglomerate hierarchy (Division → BU → DeptGroup → Dept) when present.
Returns analysis-ready DataFrame + metadata dict.
"""

import re
import pandas as pd
import numpy as np

TODAY = pd.Timestamp("2026-06-30")

# ── Column concept aliases ─────────────────────────────────────────────────
_ALIASES = {
    "employee_id":        ["empcode","employeeid","emp_id","id","staffid","worker_id","employee_number","empno"],
    "employee_name":      ["employeename","name","fullname","emp_name","staff_name"],
    "division":           ["division","business_division","company","subsidiary","entity","business_group"],
    "business_unit":      ["deptunit","business_unit","bu","businessunit","sub_division","dept_unit","business_area"],
    "dept_group":         ["departmentgroup","dept_group","deptgroup","department_group","dept_category","dept_cluster"],
    "department":         ["deptname","department","dept","team","division_dept","function","cost_center"],
    "sub_department":     ["subdepartment","sub_department","sub_dept","subdept","department_sub"],
    "designation":        ["desgname","designation","job_title","jobtitle","title","role","position"],
    "designation_main":   ["desgmainname","management_level","mgmt_level","job_category","job_band","level_name"],
    "grade":              ["grade","pay_grade","paygrade","band","job_grade","salary_grade"],
    "employment_type":    ["jobtypedetails","employment_type","emp_type","contract_type","job_type","employmenttype"],
    "employment_category":["jobtypecategorydetails","jobtypecategorydeatils","employee_category","emp_category","mgmt_category"],
    "location":           ["location","office","site","work_site","branch","office_location"],
    "work_location_type": ["worklocation","work_location","location_type","office_type"],
    "district":           ["districtname","district","city_district","home_district"],
    "gender":             ["gender","sex"],
    "religion":           ["religion","faith"],
    "marital_status":     ["maritalstatus","marital_status","marital"],
    "birth_date":         ["birthdate","birth_date","dob","date_of_birth"],
    "hire_date":          ["joiningdate","hire_date","start_date","joining_date","date_joined","date_hired","doj"],
    "exit_date":          ["leftdate","exit_date","leaving_date","termination_date","date_left","end_date","left_date"],
    "confirmation_date":  ["confirmdate","confirmation_date","confirm_date","confirmed_date"],
    "confirmation_due":   ["confduedate","confirmation_due","confirm_due_date","probation_end"],
    "last_promotion_date":["lstpromotiondate","last_promotion_date","promotion_date","last_promoted"],
    "supervisor":         ["supervisor","manager","supervisor_id","managerid","line_manager","reports_to"],
    "super_code":         ["supercode","supervisor_code","manager_code","mgr_code"],
    "active_flag":        ["active","is_active","employment_status","employee_status","status"],
    "education":          ["degree","education","qualification","highest_degree","educational_level"],
    "edu_subject":        ["subject","edu_subject","field_of_study","specialisation"],
    "institute":          ["institute","university","college","institution","school"],
    "primary_leave_taken":["preleavedaystaken","primary_leave_taken","pl_taken","annual_leave_taken","leave_taken"],
    "primary_leave_bal":  ["preleavebalance","primary_leave_balance","pl_balance","annual_leave_balance"],
    "secondary_leave_taken":["secleavedaystaken","secondary_leave_taken","sl_taken","sick_leave_taken","casual_leave_taken"],
    "secondary_leave_bal":["secleavebalance","secondary_leave_balance","sl_balance"],
    "leave_entitled":     ["preentitled","preleaveentitled","leave_entitled","annual_entitlement"],
    "salary":             ["salary","annualsalary","base_salary","base_pay","monthlyincome","compensation","ctc"],
    "performance_rating": ["performancerating","performance_rating","rating","appraisal","performance_score"],
    "assessment_score":   ["assessmentscore","written_score","writtenScore","writtenscores","exam_score"],
    "appraisal_2019":     ["apprisalpoint2019","appraisal_2019","performance_2019","rating_2019"],
    "appraisal_2020":     ["apprisalpoint2020","appraisal_2020","performance_2020","rating_2020"],
    "appraisal_2021":     ["apprisalpoint2021","appraisal_2021","performance_2021","rating_2021"],
    "letter_head":        ["letterhead","letter_head","company_name","entity_name","business_name"],
    "native_country":     ["nativecountry","native_country","nationality","country"],
    # Performance & KPI
    "kpi_score":          ["kpiscore","kpi_score","kpiachievement","kpi_achievement","kpi_points",
                           "apprisalpoint","appraisal_score","performancepct","performance_pct"],
    "target_achievement": ["targetachievement","target_achievement","achievementpct","achievement_pct",
                           "performance_achievement","achievement_percentage","target_pct","score_pct"],
    "kpi_submitted":      ["kpisubmitted","kpi_submitted","kpi_status","submitted_kpi","kpisubmission"],
    # Business productivity
    "revenue":            ["revenue","sales_revenue","gross_revenue","topline","sales","total_sales"],
    "gross_profit":       ["grossprofit","gross_profit","gp","gp_amount","profit_contribution","netprofit"],
    "incentive":          ["incentive","incentiveamount","bonus","bonus_amount","variable_pay_amount",
                           "incentive_amount","performance_bonus","achievementbonus"],
    "application_count":  ["applications","applicationnumber","application_count","total_applications",
                           "candidate_applications","applicants"],
    "resume_reviewed":    ["resumereviewed","resume_reviewed","shortlisted","cv_reviewed","screened"],
    "assessment_count":   ["assessmentcall","assessment_call","assessment_count","called_for_assessment"],
    "interviewed_count":  ["interviewed","interview_count","total_interviewed","candidates_interviewed"],
}

def _norm(s):
    return re.sub(r"[^a-z0-9]","", str(s).lower())

def detect_columns(df: pd.DataFrame) -> dict:
    """Map DataFrame columns to HR concepts. Returns {concept: actual_col}."""
    norm_map = {_norm(c): c for c in df.columns}
    col_map, used = {}, set()
    for concept, aliases in _ALIASES.items():
        for alias in aliases:
            n = _norm(alias)
            if n in norm_map and norm_map[n] not in used:
                col_map[concept] = norm_map[n]
                used.add(norm_map[n])
                break
    return col_map

def _get(col_map, *concepts):
    for c in concepts:
        if col_map.get(c): return col_map[c]
    return None

# ── Value normalisers ──────────────────────────────────────────────────────
def _norm_gender(s):
    s = str(s).strip().lower()
    if s in ("female","f","fem","female","women","woman"): return "Female"
    if s in ("male","m","men","man"): return "Male"
    return str(s).strip().title() if s not in ("nan","none","") else None

def _norm_religion(s):
    s = str(s).strip().lower()
    mapping = {"islam":"Islam","muslim":"Islam","hindu":"Hindu","hinduism":"Hindu",
               "buddhist":"Buddhist","buddhism":"Buddhist","christian":"Christian",
               "christain":"Christian","christianity":"Christian"}
    return mapping.get(s, str(s).strip().title() if s not in ("nan","none","") else None)

def _norm_marital(s):
    s = str(s).strip().lower()
    mapping = {"married":"Married","marrried":"Married","single":"Single",
               "unmarried":"Single","divorced":"Divorced","widowed":"Widowed"}
    return mapping.get(s, str(s).strip().title() if s not in ("nan","none","") else None)

def _detect_attrition(col: pd.Series) -> pd.Series:
    """Convert any attrition/active column to binary 1=left, 0=active."""
    s = col.astype(str).str.strip().str.lower()
    # Active flag: Y=active(0), N=left(1)
    if set(s.dropna().unique()) <= {"y","n","yes","no","true","false","1","0","active","left","terminated"}:
        return s.map({"n":1,"no":1,"false":1,"0":1,"left":1,"terminated":1,
                      "y":0,"yes":0,"true":0,"1":0,"active":0}).fillna(0).astype(int)
    return pd.Series(np.zeros(len(col), dtype=int), index=col.index)

# ── Main prep function ─────────────────────────────────────────────────────
def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Cleans & enriches the dataframe.
    Returns (enriched_df, metadata_dict).
    metadata contains: col_map, hierarchy_cols, kpi dict, is_conglomerate bool.
    """
    df = df.copy()
    col_map = detect_columns(df)

    # ── Dates ──────────────────────────────────────────────────────────────
    for concept in ["hire_date","exit_date","birth_date","confirmation_date",
                    "confirmation_due","last_promotion_date"]:
        col = _get(col_map, concept)
        if col and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            # Exclude sentinel dates (1900-01-01 etc.)
            if "1900" in str(df[col].mode().iloc[0]) if len(df[col].dropna()) else False:
                df.loc[df[col].dt.year < 1950, col] = pd.NaT

    # ── Active / Attrition flag ────────────────────────────────────────────
    active_col = _get(col_map, "active_flag")
    exit_col   = _get(col_map, "exit_date")
    if active_col:
        df["__attrition"] = _detect_attrition(df[active_col])
    elif exit_col:
        df["__attrition"] = df[exit_col].notna().astype(int)
    else:
        df["__attrition"] = 0
    col_map["_attrition"] = "__attrition"

    # ── Active mask ────────────────────────────────────────────────────────
    if active_col:
        s = df[active_col].astype(str).str.strip().str.upper()
        df["__is_active"] = s.isin(["Y","YES","TRUE","1","ACTIVE"]).astype(int)
    else:
        df["__is_active"] = (df["__attrition"] == 0).astype(int)
    col_map["_is_active"] = "__is_active"

    # ── Tenure ─────────────────────────────────────────────────────────────
    hire_col = _get(col_map, "hire_date")
    if hire_col:
        end_date = df[exit_col].where(df[exit_col].notna(), TODAY) if exit_col else TODAY
        df["__tenure_years"] = ((end_date - df[hire_col]).dt.days / 365.25).clip(0, 60)
        col_map["_tenure_years"] = "__tenure_years"

    # ── Age ────────────────────────────────────────────────────────────────
    bdate_col = _get(col_map, "birth_date")
    if bdate_col:
        df["__age"] = ((TODAY - df[bdate_col]).dt.days / 365.25).clip(18, 80)
        col_map["_age"] = "__age"

    # ── Generation from age ────────────────────────────────────────────────
    age_col = col_map.get("_age")
    if age_col and age_col in df.columns:
        age_num = pd.to_numeric(df[age_col], errors="coerce")
        df["__generation"] = pd.cut(
            age_num, bins=[0, 27, 43, 59, 150],
            labels=["Gen Z (<=27)","Millennial (28-43)","Gen X (44-59)","Boomer (60+)"]
        ).astype(str).replace("nan", None)
        col_map["_generation"] = "__generation"

    # ── Years since promotion ──────────────────────────────────────────────
    promo_col = _get(col_map, "last_promotion_date")
    if promo_col:
        valid = df[promo_col].notna() & (df[promo_col] > pd.Timestamp("1950-01-01"))
        df["__yrs_since_promo"] = np.where(valid,
            (TODAY - df[promo_col]).dt.days / 365.25, np.nan)
        col_map["_yrs_since_promo"] = "__yrs_since_promo"

    # ── Probation status ───────────────────────────────────────────────────
    conf_col = _get(col_map, "confirmation_date")
    if conf_col and hire_col:
        df["__on_probation"] = (
            (df["__is_active"] == 1) & df[conf_col].isna()
        ).astype(int)
        valid = df[conf_col].notna() & df[hire_col].notna()
        df["__probation_days"] = np.where(valid,
            (df[conf_col] - df[hire_col]).dt.days, np.nan)
        col_map["_on_probation"]   = "__on_probation"
        col_map["_probation_days"] = "__probation_days"

    # ── Normalise categorical values ───────────────────────────────────────
    gen_col = _get(col_map, "gender")
    if gen_col: df[gen_col] = df[gen_col].map(_norm_gender)

    rel_col = _get(col_map, "religion")
    if rel_col: df[rel_col] = df[rel_col].map(_norm_religion)

    mar_col = _get(col_map, "marital_status")
    if mar_col: df[mar_col] = df[mar_col].map(_norm_marital)

    # ── Hire year / quarter ────────────────────────────────────────────────
    if hire_col:
        df["__hire_year"] = df[hire_col].dt.year
        df["__hire_quarter"] = df[hire_col].dt.to_period("Q").astype(str)
        col_map["_hire_year"]    = "__hire_year"
        col_map["_hire_quarter"] = "__hire_quarter"
    if exit_col:
        df["__exit_year"] = df[exit_col].dt.year
        df["__exit_quarter"] = df[exit_col].dt.to_period("Q").astype(str)
        col_map["_exit_year"]    = "__exit_year"
        col_map["_exit_quarter"] = "__exit_quarter"

    # ── Detect conglomerate hierarchy ──────────────────────────────────────
    hierarchy = []
    for level, concept in [("Division","division"),("Business Unit","business_unit"),
                            ("Department Group","dept_group"),("Department","department")]:
        col = _get(col_map, concept)
        if col and df[col].nunique() > 1:
            hierarchy.append((level, col))

    # Fallback: if only one level, treat it as Division
    if not hierarchy:
        dept = _get(col_map, "department")
        if dept: hierarchy.append(("Department", dept))

    is_conglomerate = len(hierarchy) >= 3

    # ── KPIs ───────────────────────────────────────────────────────────────
    active_mask = df["__is_active"] == 1
    att_mask    = df["__attrition"] == 1
    kpi = {
        "total":          len(df),
        "active":         int(active_mask.sum()),
        "left":           int(att_mask.sum()),
        "attrition_rate": float(att_mask.mean()),
        "avg_tenure":     float(df["__tenure_years"].mean()) if "__tenure_years" in df else None,
        "on_probation":   int(df["__on_probation"].sum()) if "__on_probation" in df else 0,
        "is_conglomerate":is_conglomerate,
        "hierarchy":      hierarchy,
    }

    return df, {"col_map": col_map, "hierarchy": hierarchy,
                "is_conglomerate": is_conglomerate, "kpi": kpi}
