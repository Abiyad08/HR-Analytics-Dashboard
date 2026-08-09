"""
Generates a realistic conglomerate HR dataset in memory.
Structure mirrors a large multi-division group (5 divisions, 15 BUs, 60+ depts).
"""

import datetime as dt
import numpy as np
import pandas as pd

TODAY = dt.date(2026, 6, 30)

STRUCTURE = {
    "Pharmaceuticals": {
        "Pharma Sales & Marketing": ["Institutional Sales","Retail Sales","Trade Marketing","Digital Marketing","Regulatory Affairs"],
        "Pharma Manufacturing":     ["Production - Solid","Production - Liquid","Quality Control","Packaging","Engineering - Pharma"],
        "Pharma R&D":               ["Formulation R&D","Analytical Lab","Clinical Affairs"],
    },
    "Agribusiness": {
        "Crop Protection":          ["Crop Sales","Agro Marketing","Technical Services - Crop","Distribution - Agro"],
        "Seeds & Fertilizers":      ["Seed Sales","Fertilizer Sales","Agronomy","Supply Chain - Agro"],
        "Animal Health":            ["Vet Sales","Livestock Services","Animal Nutrition"],
    },
    "Consumer Brands": {
        "Food & Beverage":          ["F&B Sales","Brand Management - F&B","Trade Marketing - F&B","F&B Production"],
        "Home & Personal Care":     ["HPC Sales","Brand Management - HPC","HPC Manufacturing"],
        "Retail Operations":        ["Modern Trade","Traditional Trade","E-commerce"],
    },
    "Logistics & Supply Chain": {
        "ACI Logistics":            ["Warehousing","Fleet Management","Last Mile Delivery","Customs & Compliance"],
        "Procurement & SCM":        ["Strategic Procurement","Vendor Management","Import & Export"],
    },
    "Support & Corporate": {
        "Human Resources":          ["Talent Acquisition","HR Operations","Learning & Development","Payroll & Benefits","HR Business Partners"],
        "Finance & Accounts":       ["Management Accounts","Treasury","Internal Audit","Tax & Compliance","Financial Planning"],
        "Information Technology":   ["IT Infrastructure","Software Development","Cybersecurity","IT Support"],
        "Legal & Compliance":       ["Corporate Legal","Regulatory Compliance","Company Secretariat"],
        "Administration":           ["Facilities","Admin Services","Corporate Affairs"],
    },
}

MGMT_LEVELS     = ["Executive / Officer","Assistant Manager","Manager","General Manager","Director"]
MGMT_PROBS      = [0.62, 0.18, 0.12, 0.06, 0.02]
EMP_TYPES       = ["Permanent","Probationary","Graded","Contractual"]
EMP_TYPE_PROBS  = [0.72, 0.16, 0.08, 0.04]
WORK_LOCS       = ["Field","Office","Factory","Remote","Head Office"]
WORK_LOC_PROBS  = [0.38, 0.32, 0.16, 0.07, 0.07]
GENDERS         = ["Male","Female"]
GENDER_PROBS    = [0.88, 0.12]
RELIGIONS       = ["Islam","Hindu","Christian","Buddhist"]
RELIGION_PROBS  = [0.88, 0.09, 0.02, 0.01]
MARITAL         = ["Married","Single","Divorced"]
MARITAL_PROBS   = [0.68, 0.30, 0.02]
DEGREES         = ["BSc","BA","BBA","MBA","MSc","MA","HSC","SSC","PhD","Diploma"]
DEGREE_PROBS    = [0.22, 0.12, 0.20, 0.14, 0.10, 0.06, 0.08, 0.04, 0.02, 0.02]
DISTRICTS = [
    "Dhaka","Chittagong","Sylhet","Rajshahi","Khulna","Barisal","Comilla",
    "Narayanganj","Gazipur","Mymensingh","Rangpur","Jessore","Bogra","Noakhali",
    "Cox's Bazar","Tangail","Dinajpur","Faridpur","Jamalpur","Pabna",
]
DIST_PROBS = [0.22,0.12,0.07,0.06,0.06,0.04,0.05,0.05,0.05,0.04,
              0.03,0.03,0.03,0.03,0.02,0.02,0.02,0.02,0.02,0.02]
SUPERVISORS = [f"EMP{i:05d}" for i in range(1, 200)]

DIVISION_SALARY_BASE = {
    "Pharmaceuticals": 42000, "Agribusiness": 35000, "Consumer Brands": 38000,
    "Logistics & Supply Chain": 30000, "Support & Corporate": 45000,
}
LEVEL_MULT = {"Executive / Officer": 1.0, "Assistant Manager": 1.6,
              "Manager": 2.4, "General Manager": 3.8, "Director": 5.5}
GRADES = {
    "Executive / Officer": ["G1","G2","G3","G4"],
    "Assistant Manager":   ["M1","M2"],
    "Manager":             ["M3","M4"],
    "General Manager":     ["GM1","GM2"],
    "Director":            ["D1","D2"],
}


def generate_sample_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []

    for i in range(1, n + 1):
        # Pick hierarchy
        div  = np.random.choice(list(STRUCTURE.keys()),
                                p=[0.28,0.22,0.20,0.14,0.16])
        bu   = np.random.choice(list(STRUCTURE[div].keys()))
        dept = np.random.choice(STRUCTURE[div][bu])

        # Role
        mgmt  = np.random.choice(MGMT_LEVELS, p=MGMT_PROBS)
        grade = np.random.choice(GRADES[mgmt])
        desg  = f"{dept} {mgmt.split('/')[0].strip()}"

        # Employment
        emp_type = np.random.choice(EMP_TYPES, p=EMP_TYPE_PROBS)
        work_loc = np.random.choice(WORK_LOCS, p=WORK_LOC_PROBS)

        # Demographics
        gender   = np.random.choice(GENDERS,   p=GENDER_PROBS)
        religion = np.random.choice(RELIGIONS,  p=RELIGION_PROBS)
        marital  = np.random.choice(MARITAL,    p=MARITAL_PROBS)
        district = np.random.choice(DISTRICTS,  p=DIST_PROBS)
        degree   = np.random.choice(DEGREES,    p=DEGREE_PROBS)
        age_at_join = int(np.clip(np.random.normal(27, 5), 20, 50))

        # Dates
        max_tenure = min(TODAY.year - (TODAY.year - age_at_join + 22), 25)
        tenure_y   = round(min(np.random.exponential(3.8), max(0.1, max_tenure)), 2)
        hire_date  = TODAY - dt.timedelta(days=int(tenure_y * 365.25))
        birth_date = hire_date - dt.timedelta(days=age_at_join * 365)

        # Attrition
        risk = 0.07
        if mgmt == "Executive / Officer": risk += 0.14
        if emp_type == "Probationary":    risk += 0.18
        if emp_type == "Contractual":     risk += 0.12
        if tenure_y < 1:                  risk += 0.15
        if tenure_y > 10:                 risk -= 0.05
        if mgmt in ("Director","General Manager"): risk -= 0.08
        risk = float(np.clip(risk, 0.03, 0.85))
        is_active  = int(np.random.binomial(1, 1 - risk))

        left_date = confirm_date = last_promo = None
        if not is_active:
            exit_after = np.random.uniform(0.05, tenure_y)
            left_dt    = hire_date + dt.timedelta(days=int(exit_after * 365.25))
            left_date  = min(left_dt, TODAY)

        # Probation / Confirmation
        if emp_type == "Probationary" and is_active:
            # Still on probation — no confirm date
            confirm_date = None
        else:
            prob_days = int(np.clip(np.random.normal(150, 40), 60, 365))
            confirm_date = hire_date + dt.timedelta(days=prob_days)
            if confirm_date > TODAY: confirm_date = None  # not yet confirmed

        # Promotion
        if tenure_y > 2 and np.random.random() < 0.65:
            promo_back = np.random.uniform(0.3, min(tenure_y - 0.5, 8))
            last_promo = TODAY - dt.timedelta(days=int(promo_back * 365.25))

        # Salary / Compensation
        base  = DIVISION_SALARY_BASE[div] * LEVEL_MULT[mgmt]
        salary = int(round(np.random.normal(base, base * 0.10), -2))

        # Leave
        leave_entitled   = 20
        leave_taken      = round(min(max(0, np.random.normal(12, 5)), leave_entitled), 1) if is_active else 0
        leave_bal        = round(leave_entitled - leave_taken, 1)
        sec_leave_taken  = round(min(max(0, np.random.normal(5, 3)), 14), 1) if is_active else 0
        sec_leave_bal    = round(14 - sec_leave_taken, 1)

        # Appraisal scores
        base_ap = np.random.normal(72, 15)
        ap19 = round(np.clip(base_ap + np.random.normal(0, 5), 0, 100), 1) if hire_date.year <= 2019 else 0
        ap20 = round(np.clip(base_ap + np.random.normal(0, 5), 0, 100), 1) if hire_date.year <= 2020 else 0
        ap21 = round(np.clip(base_ap + np.random.normal(0, 5), 0, 100), 1) if hire_date.year <= 2021 else 0

        rows.append({
            "EmpCode":               f"EMP{i:05d}",
            "EmployeeName":          f"Employee {i:05d}",
            "Division":              div,
            "DeptUnit":              bu,
            "DepartmentGroup":       f"{bu} — {dept.split()[0]}",
            "DeptName":              dept,
            "DesgMainName":          mgmt,
            "DesgName":              desg,
            "Grade":                 grade,
            "JobTypeDetails":        emp_type,
            "WorkLocation":          work_loc,
            "Gender":                gender,
            "Religion":              religion,
            "MaritalStatus":         marital,
            "DistrictName":          district,
            "Degree":                degree,
            "BirthDate":             birth_date.isoformat(),
            "JoiningDate":           hire_date.isoformat(),
            "LeftDate":              left_date.isoformat() if left_date else "",
            "Active":                "Y" if is_active else "N",
            "ConfirmDate":           confirm_date.isoformat() if confirm_date else "",
            "LstPromotionDate":      last_promo.isoformat() if last_promo else "",
            "SuperCode":             np.random.choice(SUPERVISORS),
            "Salary":                salary,
            "PreLeaveEntitled":      leave_entitled,
            "PreLeaveDAYSTAKEN":     leave_taken,
            "PreLeaveBalance":       leave_bal,
            "SECLeaveDAYSTAKEN":     sec_leave_taken,
            "SecLeaveBalance":       sec_leave_bal,
            "ApprisalPoint2019":     ap19,
            "ApprisalPoint2020":     ap20,
            "ApprisalPoint2021":     ap21,
        })

    df = pd.DataFrame(rows)
    # Replace empty strings with NaN for date cols so pd.to_datetime works
    for col in ["LeftDate","ConfirmDate","LstPromotionDate"]:
        df[col] = df[col].replace("", pd.NA)
    return df


if __name__ == "__main__":
    df = generate_sample_data(5000)
    active = (df["Active"] == "Y").sum()
    print(f"✅ {len(df):,} records | Active: {active:,} | Left: {len(df)-active:,} | "
          f"Attrition: {(len(df)-active)/len(df):.1%}")
    df.to_excel("sample_hr_data.xlsx", index=False)
    print("   Saved sample_hr_data.xlsx")
