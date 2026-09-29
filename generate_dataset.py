"""
generate_dataset.py
-------------------
Generates the synthetic/adapted educational scholarship dataset:
'data/scholarship_students.csv'

Integrates with existing educational records in 'data/' (e.g. Tamil Nadu student performance records)
to synthesize realistic socioeconomic, academic, and demographic attributes for scholarship evaluation.

CRITICAL RESPONSIBLE AI DESIGN:
1. Target 'eligible' is calculated based ONLY on legitimate merit-cum-means criteria:
   - Academic merit (academic_score, attendance)
   - Financial need (family_income, first_generation_student)
   - Extracurricular involvement
2. 'gender' is strictly NOT used in determining eligibility.
3. Realistic correlations (e.g., between school type, location, and extracurricular access)
   are modeled to demonstrate proxy effects and subgroup disparities in fairness audits.
"""

import os
import numpy as np
import pandas as pd

def generate_scholarship_dataset(num_records=1200, output_path="data/scholarship_students.csv", seed=42):
    np.random.seed(seed)
    
    # Check if raw data exists in data/ folder for realistic student names & background
    raw_tn_path = "data/student_performance_tamilnadu.csv"
    raw_perf_path = "data/Student Performance records.csv"
    
    names_pool = []
    base_academics = []
    
    if os.path.exists(raw_tn_path):
        try:
            df_tn = pd.read_csv(raw_tn_path)
            if 'Student_Name' in df_tn.columns:
                names_pool = df_tn['Student_Name'].dropna().tolist()
            if 'Higher_Secondary_School_GPA' in df_tn.columns and 'Secondary_School_GPA' in df_tn.columns:
                # normalize to 0-100
                gpa_series = (df_tn['Higher_Secondary_School_GPA'].clip(40, 100) + df_tn['Secondary_School_GPA'].clip(40, 100)) / 2.0
                base_academics = gpa_series.dropna().tolist()
        except Exception as e:
            print(f"Notice: Could not fully parse {raw_tn_path}: {e}")
            
    # Default name pools if needed
    first_names = [
        "Aarav", "Aditi", "Ananya", "Arjun", "Bhavya", "Deepak", "Divya", "Ganesh", "Gayathri",
        "Harish", "Ishaan", "Janani", "Karthik", "Kavya", "Keerthana", "Madhavan", "Meera",
        "Manoj", "Naveen", "Nisha", "Pranav", "Pooja", "Rahul", "Rohit", "Sandhya", "Sanjay",
        "Sneha", "Surya", "Swetha", "Vignesh", "Vijay", "Varun", "Yogesh", "Bala", "Dhanush"
    ]
    last_names = [
        "Sundaram", "Raman", "Natarajan", "Krishnan", "Subramanian", "Iyer", "Pillai", "Reddy",
        "Balaji", "Venkatesh", "Murugan", "Shankar", "Kumar", "Rao", "Nair", "Patel", "Sharma"
    ]
    
    records = []
    school_types = ["Government", "Private Aided", "Private Unaided"]
    locations = ["Rural", "Semi-Urban", "Urban"]
    
    for i in range(num_records):
        student_id = f"SCH-{2024000 + i + 1}"
        
        if names_pool and i < len(names_pool):
            name = names_pool[i]
        else:
            name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
            
        age = int(np.random.choice([18, 19, 20, 21, 22, 23, 24], p=[0.25, 0.35, 0.20, 0.10, 0.05, 0.03, 0.02]))
        
        # Gender split: approximately 50/50
        gender = np.random.choice(["Female", "Male"], p=[0.49, 0.51])
        
        # Location distribution
        location = np.random.choice(locations, p=[0.38, 0.34, 0.28])
        
        # School type correlated with location (realistic demographic distribution)
        if location == "Rural":
            school_type = np.random.choice(school_types, p=[0.60, 0.30, 0.10])
        elif location == "Semi-Urban":
            school_type = np.random.choice(school_types, p=[0.35, 0.45, 0.20])
        else: # Urban
            school_type = np.random.choice(school_types, p=[0.15, 0.35, 0.50])
            
        # First generation student status (higher probability in rural & government school backgrounds)
        if school_type == "Government" or location == "Rural":
            first_gen_prob = 0.58
        elif school_type == "Private Aided":
            first_gen_prob = 0.36
        else:
            first_gen_prob = 0.18
        first_generation_student = int(np.random.rand() < first_gen_prob)
        
        # Family income distribution (log-normal realistic income in INR)
        # Rural & Government school students tend to have lower household incomes
        if school_type == "Government":
            base_income = np.random.lognormal(mean=11.6, sigma=0.55) # ~110k INR
        elif school_type == "Private Aided":
            base_income = np.random.lognormal(mean=12.2, sigma=0.60) # ~200k INR
        else:
            base_income = np.random.lognormal(mean=12.9, sigma=0.65) # ~400k INR
            
        family_income = int(np.clip(base_income, 50000, 1000000))
        # Round income to nearest 5,000 for clean realism
        family_income = int(round(family_income / 5000.0) * 5000)
        
        # Academic score (40 to 100)
        if base_academics and i < len(base_academics):
            raw_score = base_academics[i] + np.random.normal(0, 4)
        else:
            raw_score = np.random.normal(74, 13)
        academic_score = float(np.clip(raw_score, 40.0, 99.5))
        academic_score = round(academic_score, 1)
        
        # Attendance (50 to 100%)
        # Slightly correlated with academic engagement
        raw_att = 68.0 + 0.25 * (academic_score - 40) + np.random.normal(0, 6)
        attendance = float(np.clip(raw_att, 50.0, 99.5))
        attendance = round(attendance, 1)
        
        # Extracurricular score (0 to 100)
        # Socioeconomic proxy effect: Urban/private students may have had slightly more access
        # to coaching, clubs, and sports infrastructure
        if school_type == "Private Unaided":
            raw_extra = np.random.normal(68, 16)
        elif school_type == "Private Aided":
            raw_extra = np.random.normal(56, 17)
        else: # Government
            raw_extra = np.random.normal(48, 18)
        extracurricular_score = float(np.clip(raw_extra, 5.0, 98.0))
        extracurricular_score = round(extracurricular_score, 1)
        
        # Previous scholarship
        prev_scholarship_prob = 0.22 if academic_score > 75 else 0.08
        previous_scholarship = int(np.random.rand() < prev_scholarship_prob)
        
        # Financial Need Calculation (0 to 100 index)
        # Inversely related to income, boosted by first-gen status & family burden
        # Income component: 100 at 50,000 INR, down to 0 at 1,000,000 INR
        income_need = max(0.0, min(100.0, (1.0 - (family_income - 50000) / 950000.0) * 100.0))
        first_gen_bonus = 12.0 if first_generation_student == 1 else 0.0
        rural_factor = 5.0 if location == "Rural" else (2.0 if location == "Semi-Urban" else 0.0)
        
        raw_financial_need = 0.82 * income_need + first_gen_bonus + rural_factor + np.random.normal(0, 3)
        financial_need = float(np.clip(raw_financial_need, 5.0, 98.0))
        financial_need = round(financial_need, 1)
        
        # ELIGIBILITY / TARGET DETERMINATION
        # Based on transparent Merit-cum-Means scholarship policy:
        # Academic Merit (38%) + Financial Need (38%) + Attendance (14%) + Extracurriculars (10%)
        # Note: GENDER IS COMPLETELY EXCLUDED.
        merit_component = (academic_score / 100.0) * 38.0
        need_component = (financial_need / 100.0) * 38.0
        attendance_component = (attendance / 100.0) * 14.0
        extra_component = (extracurricular_score / 100.0) * 10.0
        first_gen_equity = 3.0 if first_generation_student == 1 else 0.0
        
        composite_score = merit_component + need_component + attendance_component + extra_component + first_gen_equity + np.random.normal(0, 2.5)
        
        # Threshold chosen so approx ~35% students qualify (competitive scholarship selection quota)
        # Composite score mean is ~73.8; threshold at 77.0 yields ~35% eligible
        eligible = 1 if composite_score >= 77.0 else 0
        
        records.append({
            "student_id": student_id,
            "name": name,
            "age": age,
            "gender": gender,
            "location": location,
            "school_type": school_type,
            "family_income": family_income,
            "academic_score": academic_score,
            "attendance": attendance,
            "extracurricular_score": extracurricular_score,
            "first_generation_student": first_generation_student,
            "previous_scholarship": previous_scholarship,
            "financial_need": financial_need,
            "composite_score": round(composite_score, 2),
            "eligible": eligible
        })
        
    df = pd.DataFrame(records)
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Generated dataset successfully saved to: {output_path}")
    print(f"Total Records: {len(df)}")
    print(f"Eligibility Rate: {df['eligible'].mean() * 100:.2f}%")
    print(f"Gender Breakdown:\n{df['gender'].value_counts(normalize=True) * 100}")
    print(f"Location Breakdown:\n{df['location'].value_counts(normalize=True) * 100}")
    print(f"School Type Breakdown:\n{df['school_type'].value_counts(normalize=True) * 100}")
    return df

if __name__ == "__main__":
    generate_scholarship_dataset()
