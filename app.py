"""
app.py
------
Fairness and Transparency in AI-Based Scholarship Selection
A Responsible AI Decision-Support Web Application
Authors: AI Development Team
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from utils.helpers import (
    load_data, load_model_bundle, format_currency_inr,
    load_review_decisions, save_review_decision,
    load_student_appeals, submit_student_appeal,
    get_custom_css
)
from fairness_analysis import (
    compute_fairness_audit, plot_selection_rates,
    plot_error_rates, get_fairness_interpretation
)
from explainability import (
    plot_global_importance, plot_shap_summary,
    explain_single_student_by_features, plot_individual_shap,
    generate_plain_language_explanation
)

# Page configuration
st.set_page_config(
    page_title="Responsible AI Scholarship DSS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Load data and model bundle
@st.cache_data
def get_cached_data():
    return load_data()

@st.cache_resource
def get_cached_bundle():
    return load_model_bundle()

try:
    df_students = get_cached_data()
    bundle = get_cached_bundle()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# =========================================================================
# Sidebar Navigation
# =========================================================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h2 style="margin: 0; color: #1e3a8a; font-size: 20px;"> Responsible AI DSS</h2>
    <p style="margin: 4px 0 0 0; color: #64748b; font-size: 12px; font-weight: 500;">
        Scholarship Selection System<br>
        <span style="color: #0284c7; font-weight: 600;">Decision-Support Only</span>
    </p>
</div>
""", unsafe_allow_html=True)

nav_options = [
    "Project Overview",
    "Student Data",
    "AI Assessment",
    "Fairness Analysis",
    "Explainability",
    "Human Review",
    "Governance & Accountability"
]

selected_page = st.sidebar.radio("Navigation", nav_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 11.5px; color: #475569; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;">
    <strong>Project:</strong> Responsible AI DSS<br>
    <strong>Version:</strong> 1.0.0<br>
</div>
""", unsafe_allow_html=True)



# =========================================================================
# PAGE 1:  Project Overview
# =========================================================================
if selected_page == "Project Overview":
    st.markdown("""
    <div class="academic-banner">
        <h1>Fairness and Transparency in AI-Based Scholarship Selection</h1>
        <p>A Responsible AI Study of Automated Scholarship Selection Systems — Decision-Support System Architecture</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader(" Project Objective & Background")
        st.write("""
        Scholarships provide transformative educational opportunities for students confronting economic hardship.
        However, traditional manual evaluation of thousands of scholarship applications creates substantial administrative burden.
        While Artificial Intelligence can expedite application screening, **deploying autonomous AI in high-stakes educational decisions introduces severe risks of algorithmic bias, proxy discrimination, and lack of transparency**.
        
        This project establishes a **Responsible AI Decision-Support System (DSS)** that balances algorithmic efficiency with strict human oversight, algorithmic fairness audits, transparent explanations, and accountable governance.
        """)

        st.markdown("""
        ####  The Four Responsible AI Pillars
        1. **Fairness and Non-Discrimination**: Continuous auditing across demographic and socioeconomic subgroups (Gender, School Type, Location) using Microsoft Fairlearn to identify disparate impact and error rate skews.
        2. **Transparency and Explainability**: Explaining individual and global AI recommendations using SHAP (SHapley Additive exPlanations) in plain, accessible language rather than black-box scores.
        3. **Accountability and Governance**: Institutional oversight aligned with the **NIST AI Risk Management Framework (RMF)** and the **EU AI Act** (High-Risk Classification for Educational Access).
        4. **Human Oversight and Control**: Absolute human-in-the-loop governance. **AI recommendations are non-binding advisory inputs**; authorized human scholarship committees hold sole final authority.
        """)

    with col2:
        st.subheader(" End-to-End Workflow")
        st.markdown("""
        ```mermaid
        graph TD
            A[Scholarship Application] --> B[Data Preprocessing]
            B --> C[AI Decision Support]
            C --> D[Fairness & SHAP Audit]
            D --> E[Human Review Committee]
            E -->|Approved / Overridden| F[Final Selection Decision]
            F --> G[Student Grievance & Appeal]
            G -->|Correction| E
        ```
        """)

        st.info("""
        **System Specification:**
        - **Model:** Random Forest Classifier (Depth-constrained, balanced)
        - **Target:** Merit-cum-Means Recommendation (`eligible`)
        - **Protected Attributes:** `gender`, `school_type`, `location`
        - **Auditing Suite:** Microsoft Fairlearn & SHAP
        - **Human Control:** Persistent Reviewer Override Queue
        """)

    st.markdown("---")
    st.subheader(" Regulatory & Ethical Alignment")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown("""
        **NIST AI RMF 1.0**
        - **Govern:** Clear institutional RACI matrix
        - **Map:** Contextual risks of proxy bias
        - **Measure:** Fairlearn MetricFrame parity
        - **Manage:** Human override & audit logging
        """)
    with rc2:
        st.markdown("""
        **EU AI Act (Annex III)**
        - Classified under **High-Risk AI Systems** (Education & Vocational Training)
        - Requires technical robustness, explainability, high quality data, and human oversight.
        """)
    with rc3:
        st.markdown("""
        **ISO/IEC 42001**
        - Artificial Intelligence Management System (AIMS) standard for continuous monitoring, logging, and redress.
        """)

# =========================================================================
# PAGE 2:  Student Data
# =========================================================================
elif selected_page == "Student Data":
    st.title(" Student Scholarship Dataset")
    st.write("Explore the educational records, applicant socioeconomic profiles, and scholarship criteria.")

    # High-level metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    total_students = len(df_students)
    avg_academic = df_students['academic_score'].mean()
    avg_income = df_students['family_income'].mean()
    overall_elig = df_students['eligible'].mean() * 100

    col_m1.metric("Total Applicants", f"{total_students:,}")
    col_m2.metric("Avg Academic Score", f"{avg_academic:.1f} / 100")
    col_m3.metric("Avg Family Income", format_currency_inr(int(avg_income)))
    col_m4.metric("Recommendation Rate", f"{overall_elig:.1f}%")

    st.markdown("---")
    st.subheader(" Filter & Inspect Records")

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        gender_filter = st.selectbox("Gender", ["All"] + list(df_students['gender'].unique()))
    with fc2:
        school_filter = st.selectbox("School Type", ["All"] + list(df_students['school_type'].unique()))
    with fc3:
        loc_filter = st.selectbox("Location", ["All"] + list(df_students['location'].unique()))
    with fc4:
        elig_filter = st.selectbox("AI Eligibility", ["All", "Recommended (1)", "Not Recommended (0)"])

    filtered_df = df_students.copy()
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
    if school_filter != "All":
        filtered_df = filtered_df[filtered_df['school_type'] == school_filter]
    if loc_filter != "All":
        filtered_df = filtered_df[filtered_df['location'] == loc_filter]
    if elig_filter == "Recommended (1)":
        filtered_df = filtered_df[filtered_df['eligible'] == 1]
    elif elig_filter == "Not Recommended (0)":
        filtered_df = filtered_df[filtered_df['eligible'] == 0]

    st.write(f"Showing **{len(filtered_df)}** of **{len(df_students)}** student records:")
    
    # Prettify table for display
    display_df = filtered_df.copy()
    display_df['family_income_formatted'] = display_df['family_income'].apply(format_currency_inr)
    display_cols = [
        'student_id', 'name', 'age', 'gender', 'location', 'school_type',
        'family_income_formatted', 'academic_score', 'attendance',
        'extracurricular_score', 'first_generation_student', 'financial_need', 'eligible'
    ]
    st.dataframe(
        display_df[display_cols].rename(columns={
            'family_income_formatted': 'family_income',
            'first_generation_student': 'first_gen'
        }),
        use_container_width=True,
        height=380
    )

    # Download CSV
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_scholarship_students.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader(" Socioeconomic & Academic Distributions")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        fig_acad, ax_acad = plt.subplots(figsize=(6, 3.8))
        ax_acad.hist(df_students[df_students['eligible'] == 1]['academic_score'], bins=20, alpha=0.75,
                     label='Recommended', color='#2b8a3e', edgecolor='black')
        ax_acad.hist(df_students[df_students['eligible'] == 0]['academic_score'], bins=20, alpha=0.75,
                     label='Not Recommended', color='#c92a2a', edgecolor='black')
        ax_acad.set_title("Academic Score by Recommendation Status", fontsize=11, fontweight='bold')
        ax_acad.set_xlabel("Academic Score (out of 100)")
        ax_acad.set_ylabel("Number of Applicants")
        ax_acad.legend()
        ax_acad.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_acad)
        plt.close(fig_acad)

    with d_col2:
        fig_inc, ax_inc = plt.subplots(figsize=(6, 3.8))
        ax_inc.hist(df_students[df_students['eligible'] == 1]['financial_need'], bins=20, alpha=0.75,
                    label='Recommended', color='#1971c2', edgecolor='black')
        ax_inc.hist(df_students[df_students['eligible'] == 0]['financial_need'], bins=20, alpha=0.75,
                    label='Not Recommended', color='#fa5252', edgecolor='black')
        ax_inc.set_title("Financial Need Index by Recommendation Status", fontsize=11, fontweight='bold')
        ax_inc.set_xlabel("Financial Need Index (0 to 100)")
        ax_inc.set_ylabel("Number of Applicants")
        ax_inc.legend()
        ax_inc.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig_inc)
        plt.close(fig_inc)

# =========================================================================
# PAGE 3:  AI Assessment
# =========================================================================
elif selected_page == "AI Assessment":
    st.title(" AI Application Assessment Simulator")
    st.write("Enter student application details to evaluate recommendation probability and decision-support guidance.")

    # Form
    with st.form("student_assessment_form"):
        st.subheader(" Student Application Form")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_name = st.text_input("Applicant Name", value="Jane Doe")
            applicant_id = st.text_input("Application Reference ID", value="APP-NEW-2024")
            age = st.slider("Age", min_value=17, max_value=26, value=20)
            gender = st.selectbox("Gender (Used for Audit Only, NOT Model Training)", ["Female", "Male"])
            
        with c2:
            location = st.selectbox("Location / Region", ["Rural", "Semi-Urban", "Urban"])
            school_type = st.selectbox("Secondary School Background", ["Government", "Private Aided", "Private Unaided"])
            family_income = st.slider("Annual Family Income (INR)", min_value=50000, max_value=1000000, value=140000, step=10000)
            first_generation = st.checkbox("First-Generation College Student", value=True)
            
        with c3:
            academic_score = st.slider("Academic Score / GPA (0 - 100)", min_value=40.0, max_value=100.0, value=87.5, step=0.5)
            attendance = st.slider("Attendance Consistency (%)", min_value=50.0, max_value=100.0, value=93.0, step=0.5)
            extracurricular_score = st.slider("Extracurricular & Leadership Score (0 - 100)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
            prev_scholarship = st.checkbox("Previous Scholarship Recipient", value=False)

        submit_assessment = st.form_submit_button(" Assess Application with AI DSS", use_container_width=True)

    # Process Form
    if submit_assessment or 'last_assessment' in st.session_state:
        if submit_assessment:
            # Calculate financial need index
            income_need = max(0.0, min(100.0, (1.0 - (family_income - 50000) / 950000.0) * 100.0))
            first_gen_bonus = 12.0 if first_generation else 0.0
            rural_factor = 5.0 if location == "Rural" else (2.0 if location == "Semi-Urban" else 0.0)
            calculated_financial_need = round(min(100.0, 0.82 * income_need + first_gen_bonus + rural_factor), 1)

            applicant_dict = {
                'academic_score': academic_score,
                'attendance': attendance,
                'family_income': family_income,
                'financial_need': calculated_financial_need,
                'extracurricular_score': extracurricular_score,
                'first_generation_student': 1 if first_generation else 0,
                'school_type': school_type,
                'location': location
            }

            # Run SHAP / model explanation
            exp = explain_single_student_by_features(bundle, applicant_dict, student_id=applicant_id)
            st.session_state['last_assessment'] = {
                'name': applicant_name,
                'id': applicant_id,
                'gender': gender,
                'dict': applicant_dict,
                'exp': exp
            }

        assessed = st.session_state['last_assessment']
        exp = assessed['exp']
        prob_pct = exp['probability'] * 100
        is_rec = (exp['prediction'] == 1)

        st.markdown("---")
        st.subheader(" Decision-Support Assessment Results")

        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            st.markdown(f"### Candidate: **{assessed['name']}**")
            st.caption(f"Ref ID: {assessed['id']} | Region: {assessed['dict']['location']}")
            
            if is_rec:
                st.markdown("""
                <div class="badge-recommended">
                     AI Recommendation: RECOMMENDED FOR CONSIDERATION
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="badge-not-recommended">
                     AI Recommendation: NOT RECOMMENDED
                </div>
                """, unsafe_allow_html=True)

            st.metric("Recommendation Confidence", f"{prob_pct:.1f}%")
            st.write(f"**Demonstrated Financial Need:** `{assessed['dict']['financial_need']} / 100`")
            st.write(f"**Family Income:** `{format_currency_inr(assessed['dict']['family_income'])}`")

        with res_c2:
            st.markdown("""
            <div class="dss-alert">
                <strong> AI Recommendation — Human Review Required</strong><br>
                This assessment is generated as a decision-support advisory signal based on merit-cum-means criteria.
                <strong>Do NOT treat this output as a final scholarship award.</strong>
                In accordance with institutional Responsible AI governance, an authorized scholarship committee must evaluate the candidate's documentation.
            </div>
            """, unsafe_allow_html=True)

            # Quick button to send directly to human review queue
            if st.button(" Send Candidate to Human Review Queue", key="send_to_queue"):
                st.session_state['review_student_override'] = {
                    'student_id': assessed['id'],
                    'student_name': assessed['name'],
                    'gender': assessed['gender'],
                    'academic_score': assessed['dict']['academic_score'],
                    'family_income': assessed['dict']['family_income'],
                    'financial_need': assessed['dict']['financial_need'],
                    'school_type': assessed['dict']['school_type'],
                    'location': assessed['dict']['location'],
                    'ai_recommendation': exp['recommendation_text'],
                    'ai_confidence': prob_pct
                }
                st.success(f"Candidate {assessed['name']} ({assessed['id']}) queued for committee review! Navigate to the ' Human Review' tab to take action.")

        st.markdown("---")
        st.subheader(" AI Factor Attribution for this Candidate")
        shap_c1, shap_c2 = st.columns([3, 2])
        with shap_c1:
            fig_ind = plot_individual_shap(exp)
            st.pyplot(fig_ind)
            plt.close(fig_ind)
        with shap_c2:
            st.markdown(generate_plain_language_explanation(exp))

# =========================================================================
# PAGE 4:  Fairness Analysis
# =========================================================================
elif selected_page == "Fairness Analysis":
    st.title(" Algorithmic Fairness & Subgroup Auditing")
    st.write("""
    Audit the AI scholarship recommendation model across protected demographic and socioeconomic groups.
    Implemented using **Microsoft Fairlearn** and the **EEOC Four-Fifths (80%) Rule**.
    """)

    group_var = st.selectbox(
        "Select Subgroup Attribute to Audit:",
        ["Gender", "School Type", "Location"],
        index=0
    )
    col_map = {"Gender": "gender", "School Type": "school_type", "Location": "location"}
    sens_col = col_map[group_var]

    # Compute audit
    audit = compute_fairness_audit(bundle, sensitive_col=sens_col)

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Selection Rate", f"{audit['overall_selection_rate']*100:.1f}%")
    m2.metric("Statistical Parity Diff", f"{audit['statistical_parity_difference']*100:.1f}%")
    
    di_val = audit['disparate_impact_ratio']
    di_status = "PASS (≥ 0.80)" if audit['is_four_fifths_compliant'] else "FLAG (< 0.80)"
    m3.metric("Disparate Impact Ratio", f"{di_val:.2f}", delta=di_status)
    m4.metric("FPR Disparity", f"{audit['fpr_difference']*100:.1f}%")

    st.markdown("---")
    st.subheader(f" Detailed MetricFrame Table by {group_var}")

    # Display Fairlearn Table
    tbl = audit['group_metrics_df'].copy()
    format_dict = {
        'Selection Rate': '{:.2%}',
        'Accuracy': '{:.2%}',
        'False Positive Rate': '{:.2%}',
        'False Negative Rate': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}'
    }
    st.dataframe(tbl.style.format(format_dict), use_container_width=True)

    st.markdown("---")
    st.subheader(f" Subgroup Fairness Visualizations ({group_var})")
    
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        fig_sr = plot_selection_rates(audit)
        st.pyplot(fig_sr)
        plt.close(fig_sr)

    with chart_c2:
        fig_err = plot_error_rates(audit)
        st.pyplot(fig_err)
        plt.close(fig_err)

    st.markdown("---")
    # Comprehensive Plain-Language Interpretation
    st.markdown(get_fairness_interpretation(audit))

# =========================================================================
# PAGE 5:  Explainability
# =========================================================================
elif selected_page == "Explainability":
    st.title(" Model Explainability & Transparency (SHAP)")
    st.write("Understand global feature importance and inspect individual student prediction drivers using SHAP.")

    tab_global, tab_local, tab_whatif = st.tabs([
        " Global Feature Importance",
        " Individual Student Explanation",
        " Interactive Counterfactual (What-If)"
    ])

    with tab_global:
        st.subheader(" Global Explanation: What Drives Scholarship Recommendations?")
        st.write("SHAP evaluates how much each feature alters the model's recommendation probability across the test dataset.")
        
        gc1, gc2 = st.columns([1, 1])
        with gc1:
            fig_glob = plot_global_importance(bundle)
            st.pyplot(fig_glob)
            plt.close(fig_glob)
        with gc2:
            fig_beeswarm = plot_shap_summary(bundle)
            st.pyplot(fig_beeswarm)
            plt.close(fig_beeswarm)

        st.info("""
        **Key Global Explainability Insights:**
        - **Academic Score** and **Financial Need Index** dominate the recommendation decisions, strictly reflecting the merit-cum-means policy.
        - **Family Income** has a negative directional SHAP value (higher income reduces eligibility, lower income boosts it).
        - **Attendance** and **Extracurricular Activities** act as secondary balancing factors.
        """)

    with tab_local:
        st.subheader(" Individual Student Transparency")
        st.write("Select a student from the test cohort to inspect the exact SHAP attribution waterfall and plain-language explanation.")

        df_test = bundle['df_test_audit'].copy()
        student_choices = [f"{row['student_id']} - {row['name']} ({row['gender']})" for _, row in df_test.iterrows()]
        
        selected_choice = st.selectbox("Select Student Applicant:", student_choices, index=0)
        selected_id = selected_choice.split(" - ")[0]
        
        # Retrieve candidate raw data
        cand_row = df_students[df_students['student_id'] == selected_id].iloc[0]
        cand_dict = {
            'academic_score': cand_row['academic_score'],
            'attendance': cand_row['attendance'],
            'family_income': cand_row['family_income'],
            'financial_need': cand_row['financial_need'],
            'extracurricular_score': cand_row['extracurricular_score'],
            'first_generation_student': cand_row['first_generation_student'],
            'school_type': cand_row['school_type'],
            'location': cand_row['location']
        }

        explanation = explain_single_student_by_features(bundle, cand_dict, student_id=selected_id)

        ind_col1, ind_col2 = st.columns([3, 2])
        with ind_col1:
            fig_ind = plot_individual_shap(explanation)
            st.pyplot(fig_ind)
            plt.close(fig_ind)
        with ind_col2:
            st.markdown(generate_plain_language_explanation(explanation))

    with tab_whatif:
        st.subheader(" Counterfactual / 'What-If' Simulation")
        st.write("Adjust an applicant's parameters to see how changes in their academic or financial profile alter the AI recommendation.")

        w_col1, w_col2 = st.columns(2)
        with w_col1:
            w_acad = st.slider("Simulated Academic Score", 40.0, 100.0, 75.0, 0.5, key="w_acad")
            w_inc = st.slider("Simulated Family Income (INR)", 50000, 1000000, 200000, 25000, key="w_inc")
            w_att = st.slider("Simulated Attendance (%)", 50.0, 100.0, 85.0, 1.0, key="w_att")
            w_extra = st.slider("Simulated Extracurriculars", 0.0, 100.0, 60.0, 1.0, key="w_extra")
            w_first = st.checkbox("First-Generation Student", value=True, key="w_first")
            w_loc = st.selectbox("Location", ["Rural", "Semi-Urban", "Urban"], key="w_loc")
            w_school = st.selectbox("School Type", ["Government", "Private Aided", "Private Unaided"], key="w_school")

        # Compute simulated need
        inc_need = max(0.0, min(100.0, (1.0 - (w_inc - 50000) / 950000.0) * 100.0))
        sim_need = round(min(100.0, 0.82 * inc_need + (12.0 if w_first else 0.0) + (5.0 if w_loc == "Rural" else 0.0)), 1)

        sim_dict = {
            'academic_score': w_acad,
            'attendance': w_att,
            'family_income': w_inc,
            'financial_need': sim_need,
            'extracurricular_score': w_extra,
            'first_generation_student': 1 if w_first else 0,
            'school_type': w_school,
            'location': w_loc
        }

        sim_exp = explain_single_student_by_features(bundle, sim_dict, student_id="SIMULATED-APPLICANT")

        with w_col2:
            st.markdown("### Simulated AI Outcome")
            if sim_exp['prediction'] == 1:
                st.markdown('<div class="badge-recommended">Recommended for Consideration</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-not-recommended">Not Recommended</div>', unsafe_allow_html=True)
            
            st.metric("Recommendation Probability", f"{sim_exp['probability']*100:.1f}%")
            st.write(f"**Calculated Need Index:** `{sim_need} / 100`")
            
            fig_sim = plot_individual_shap(sim_exp)
            st.pyplot(fig_sim)
            plt.close(fig_sim)

# =========================================================================
# PAGE 6:  Human Review
# =========================================================================
elif selected_page == "Human Review":
    st.title(" Scholarship Committee Human Review & Override")
    st.markdown("""
    <div class="dss-alert">
        <strong>Mandatory Human Oversight Protocol:</strong>
        The AI recommendation is not the final scholarship decision.
        A scholarship committee or authorized human reviewer remains strictly responsible for the final scholarship determination.
        Reviewers may affirm the AI recommendation, override it with written justification, or request further documentary verification.
    </div>
    """, unsafe_allow_html=True)

    # Check if a student was pushed from Page 3
    override_candidate = st.session_state.get('review_student_override', None)

    col_q, col_action = st.columns([1, 1])

    with col_q:
        st.subheader(" Select Applicant for Review")
        
        # Combine queue options
        df_queue = bundle['df_test_audit'].head(25).copy()
        queue_options = []
        if override_candidate:
            queue_options.append(f"{override_candidate['student_id']} - {override_candidate['student_name']} (Queued from Simulator)")
        for _, r in df_queue.iterrows():
            queue_options.append(f"{r['student_id']} - {r['name']} ({r['school_type']})")

        selected_queue_item = st.selectbox("Select Candidate from Triage Queue:", queue_options, index=0)
        
        if "(Queued from Simulator)" in selected_queue_item and override_candidate:
            curr_cand = override_candidate
            c_id = curr_cand['student_id']
            c_name = curr_cand['student_name']
            c_acad = curr_cand['academic_score']
            c_inc = curr_cand['family_income']
            c_need = curr_cand['financial_need']
            c_rec = curr_cand['ai_recommendation']
            c_conf = curr_cand['ai_confidence']
            c_school = curr_cand['school_type']
            c_loc = curr_cand['location']
        else:
            c_id = selected_queue_item.split(" - ")[0].strip()
            cand_full = df_students[df_students['student_id'] == c_id].iloc[0]
            c_name = cand_full['name']
            c_acad = cand_full['academic_score']
            c_inc = cand_full['family_income']
            c_need = cand_full['financial_need']
            c_school = cand_full['school_type']
            c_loc = cand_full['location']
            
            # Predict
            s_dict = {
                'academic_score': c_acad, 'attendance': cand_full['attendance'],
                'family_income': c_inc, 'financial_need': c_need,
                'extracurricular_score': cand_full['extracurricular_score'],
                'first_generation_student': cand_full['first_generation_student'],
                'school_type': c_school, 'location': c_loc
            }
            c_exp = explain_single_student_by_features(bundle, s_dict, student_id=c_id)
            c_rec = c_exp['recommendation_text']
            c_conf = c_exp['probability'] * 100

        # Applicant Summary Card
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 18px; margin-top: 10px;">
            <h4 style="margin: 0 0 10px 0; color: #1e293b;">Applicant Profile: {c_name} ({c_id})</h4>
            <p style="margin: 4px 0; font-size: 13.5px;"><strong>Academic Score:</strong> {c_acad}/100</p>
            <p style="margin: 4px 0; font-size: 13.5px;"><strong>Family Income:</strong> {format_currency_inr(c_inc)}</p>
            <p style="margin: 4px 0; font-size: 13.5px;"><strong>Financial Need Index:</strong> {c_need}/100</p>
            <p style="margin: 4px 0; font-size: 13.5px;"><strong>School & Region:</strong> {c_school} | {c_loc}</p>
            <hr style="margin: 10px 0;">
            <p style="margin: 4px 0; font-size: 13.5px;"><strong>AI Advisory Recommendation:</strong> <code>{c_rec}</code></p>
            <p style="margin: 4px 0; font-size: 13.5px;"><strong>AI Model Confidence:</strong> {c_conf:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col_action:
        st.subheader(" Record Human Committee Decision")
        with st.form("human_review_form"):
            reviewer_name = st.text_input("Reviewer Full Name", value="Jane Smith")
            reviewer_role = st.selectbox(
                "Reviewer Role / Designation",
                [
                    "Scholarship Committee Chairperson",
                    "Dean of Academic Affairs",
                    "Financial Aid Officer",
                    "Faculty Ethics Ombudsman"
                ]
            )
            
            human_decision = st.radio(
                "Official Human Determination:",
                [
                    "Accept AI Recommendation (Award Scholarship)",
                    "Override AI Recommendation (Grant Discretionary Scholarship)",
                    "Override AI Recommendation (Decline Scholarship)",
                    "Send for Further Verification / Committee Hearing"
                ],
                index=0
            )

            reviewer_comments = st.text_area(
                "Mandatory Justification / Review Notes:",
                placeholder="Explain the rationale for this determination, documentary evidence examined, or context regarding any override..."
            )

            submit_decision = st.form_submit_button(" Record Final Human Decision & Update Audit Trail", use_container_width=True)

        if submit_decision:
            if not reviewer_comments.strip():
                st.warning(" Please provide reviewer notes/justification before recording the decision.")
            else:
                save_review_decision(
                    student_id=c_id,
                    student_name=c_name,
                    ai_rec=c_rec,
                    ai_conf=c_conf,
                    human_dec=human_decision,
                    reviewer_name=reviewer_name,
                    reviewer_role=reviewer_role,
                    comments=reviewer_comments
                )
                st.success(f" Final decision recorded by human reviewer for {c_name} ({c_id})! Decision logged to institutional audit trail.")

    st.markdown("---")
    st.subheader(" Institutional Human Review Audit Log")
    decisions_df = load_review_decisions()
    if len(decisions_df) > 0:
        st.dataframe(decisions_df, use_container_width=True)
        csv_log = decisions_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Review Audit Trail (CSV)", data=csv_log, file_name="scholarship_review_audit_trail.csv", mime="text/csv")
    else:
        st.info("No human review decisions recorded yet. Use the form above to record committee determinations.")

# =========================================================================
# PAGE 7:  Governance & Accountability
# =========================================================================
elif selected_page == "Governance & Accountability":
    st.title(" Responsible AI Governance, Accountability & Redress")
    st.write("Institutional accountability framework, regulatory compliance mappings, and student grievance redressal.")

    st.subheader(" 8-Stage Responsible AI Governance Lifecycle")
    st.markdown("""
    ```mermaid
    graph LR
        S1[1. AI System] --> S2[2. Continuous Monitoring]
        S2 --> S3[3. Fairness Audits]
        S3 --> S4[4. Human Review]
        S4 --> S5[5. Final Decision]
        S5 --> S6[6. Transparent Notification]
        S6 --> S7[7. Appeal & Redress]
        S7 --> S8[8. Corrective Action]
        S8 -.-> S1
    ```
    """)

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("""
        ####  Governance Policy Commitments
        1. **Human-in-the-Loop Supremacy:** Autonomous AI scholarship rejections are strictly prohibited. Every non-recommendation can be contested.
        2. **Periodic Bias Audits:** Algorithmic performance and fairness metrics (Statistical Parity & Equalized Odds) are recalculated each admission cycle.
        3. **Data Protection & Privacy:** Student socioeconomic records comply with data protection regulations (Digital Personal Data Protection Act / GDPR).
        4. **Model Documentation:** Maintenance of Model Cards describing training constraints, performance bounds, and known proxy limitations.
        """)

    with g_col2:
        st.markdown("""
        ####  Institutional RACI Responsibility Matrix
        - **Scholarship Committee:** *Accountable (A)* for final student selection determinations.
        - **AI Engineering Team:** *Responsible (R)* for model pipeline maintenance and Fairlearn metrics generation.
        - **Institutional Ethics Board:** *Consulted (C)* on fairness threshold adjustments and disparate impact flags.
        - **Student Applicant Body:** *Informed (I)* with transparent, plain-language decision explanations and appeal avenues.
        """)

    st.markdown("---")
    st.subheader(" Student Appeal & Grievance Redressal System")
    st.write("Students dissatisfied with an AI-assisted scholarship determination have an institutional right to lodge a formal grievance.")

    app_c1, app_c2 = st.columns([1, 1])

    with app_c1:
        st.markdown("#### Submit a Scholarship Appeal")
        with st.form("appeal_submission_form"):
            app_stud_id = st.text_input("Applicant Student ID", value="SCH-2024045")
            app_stud_name = st.text_input("Applicant Name", value="John Doe")
            app_ground = st.selectbox(
                "Primary Ground of Appeal:",
                [
                    "Discrepancy in Family Income Assessment / Recent Job Loss",
                    "Extenuating Medical or Personal Circumstances Affecting Attendance",
                    "Unrecorded Academic or Extracurricular Achievements",
                    "Systemic / Regional Hardship Consideration",
                    "Other Documented Disadvantaged Status"
                ]
            )
            app_evidence = st.text_area(
                "Description of Supporting Documents Provided:",
                placeholder="List medical certificates, updated income tax returns, government tahsildar certificates..."
            )
            submit_appeal_btn = st.form_submit_button(" Lodge Official Student Appeal", use_container_width=True)

        if submit_appeal_btn:
            if not app_evidence.strip():
                st.warning(" Please provide documentary evidence details for the appeal.")
            else:
                new_app_id = submit_student_appeal(app_stud_id, app_stud_name, app_ground, app_evidence)
                st.success(f" Appeal successfully registered! Appeal Reference ID: **{new_app_id}**. Assigned to Human Ombudsman for expedited review.")

    with app_c2:
        st.markdown("#### Active Grievances & Redress Status")
        appeals_df = load_student_appeals()
        st.dataframe(appeals_df, use_container_width=True, height=320)

