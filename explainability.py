"""
explainability.py
-----------------
Implements SHAP (SHapley Additive exPlanations) for global and local transparency.

Features:
1. Global feature importance (Mean |SHAP| values)
2. SHAP summary beeswarm plot
3. Individual student prediction breakdown (SHAP contribution waterfall / bar plot)
4. Plain-Language explanations converting technical weights into clear, actionable prose
   suitable for scholarship committees and student applicants.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

_CACHED_EXPLAINER = None
_CACHED_SHAP_TEST = None

def get_explainer(model):
    global _CACHED_EXPLAINER
    if _CACHED_EXPLAINER is None:
        _CACHED_EXPLAINER = shap.TreeExplainer(model)
    return _CACHED_EXPLAINER

def get_test_shap_values(bundle):
    """
    Computes or retrieves cached SHAP values for the test set.
    """
    global _CACHED_SHAP_TEST
    if _CACHED_SHAP_TEST is None:
        explainer = get_explainer(bundle['rf_model'])
        X_test_df = bundle['X_test_df']
        sv = explainer.shap_values(X_test_df)
        
        # Binary classification handling: extract values for positive class (1 = Eligible)
        if isinstance(sv, list):
            class1_sv = sv[1]
        elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
            class1_sv = sv[:, :, 1]
        else:
            class1_sv = sv
            
        _CACHED_SHAP_TEST = (explainer, class1_sv)
    return _CACHED_SHAP_TEST

def plot_global_importance(bundle, top_n=8):
    """
    Plots horizontal bar chart of mean absolute SHAP values across all features.
    """
    explainer, class1_sv = get_test_shap_values(bundle)
    feature_names = bundle['all_feature_names']
    
    mean_abs_shap = np.mean(np.abs(class1_sv), axis=0)
    importance_df = pd.DataFrame({
        'Feature': [f.replace('num__', '').replace('cat__', '') for f in feature_names],
        'Mean_SHAP': mean_abs_shap
    }).sort_values(by='Mean_SHAP', ascending=True)
    
    # Prettify labels
    clean_labels = {
        'academic_score': 'Academic Score (GPA)',
        'financial_need': 'Financial Need Index',
        'family_income': 'Family Income (Socioeconomic)',
        'attendance': 'Class Attendance (%)',
        'extracurricular_score': 'Extracurricular Activities',
        'first_generation_student': 'First-Gen Student Status',
        'school_type_Private Aided': 'School: Private Aided',
        'school_type_Private Unaided': 'School: Private Unaided',
        'location_Semi-Urban': 'Location: Semi-Urban',
        'location_Urban': 'Location: Urban'
    }
    importance_df['Display_Name'] = importance_df['Feature'].map(lambda x: clean_labels.get(x, x))
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(importance_df['Display_Name'].tail(top_n),
                   importance_df['Mean_SHAP'].tail(top_n),
                   color='#2b5c8f', edgecolor='black', alpha=0.88, height=0.6)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_title("Global Feature Importance (Mean |SHAP| Impact on Scholarship Recommendation)",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value (|Impact|)", fontsize=10)
    ax.set_xlim(0, max(importance_df['Mean_SHAP']) * 1.22)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_shap_summary(bundle):
    """
    Renders standard SHAP summary beeswarm plot.
    """
    explainer, class1_sv = get_test_shap_values(bundle)
    X_test_df = bundle['X_test_df'].copy()
    
    # Clean column names for the plot
    clean_cols = [c.replace('num__', '').replace('cat__', '') for c in X_test_df.columns]
    X_test_df.columns = clean_cols
    
    fig = plt.figure(figsize=(8, 4.8))
    shap.summary_plot(class1_sv, X_test_df, show=False, max_display=8)
    plt.title("SHAP Feature Impact Distribution (Directional Effect on Eligibility)",
              fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()
    return fig

def explain_single_student_by_features(bundle, student_raw_dict, student_id="APPLICANT"):
    """
    Explains an individual student given their raw input dictionary.
    """
    preprocessor = bundle['preprocessor']
    rf_model = bundle['rf_model']
    explainer = get_explainer(rf_model)
    
    # Create single-row DataFrame
    input_df = pd.DataFrame([student_raw_dict])
    
    # Ensure correct columns
    num_cols = bundle['numerical_features']
    cat_cols = bundle['categorical_features']
    raw_features_df = input_df[num_cols + cat_cols]
    
    # Transform
    X_trans = preprocessor.transform(raw_features_df)
    X_df = pd.DataFrame(X_trans, columns=bundle['all_feature_names'])
    
    # Predict
    prob = float(rf_model.predict_proba(X_df)[0, 1])
    pred = int(rf_model.predict(X_df)[0])
    
    # SHAP values
    sv = explainer.shap_values(X_df)
    if isinstance(sv, list):
        class1_sv = sv[1][0]
    elif isinstance(sv, np.ndarray) and len(sv.shape) == 3:
        class1_sv = sv[0, :, 1]
    else:
        class1_sv = sv[0]
        
    expected_val = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
    
    # Assemble feature impacts
    clean_names = [f.replace('num__', '').replace('cat__', '') for f in bundle['all_feature_names']]
    
    impacts = []
    for clean_name, val, raw_f in zip(clean_names, class1_sv, bundle['all_feature_names']):
        # Find raw value
        if clean_name in student_raw_dict:
            raw_v = student_raw_dict[clean_name]
        else:
            raw_v = "Present" if "1" in clean_name else "N/A"
            
        impacts.append({
            'feature': clean_name,
            'shap_value': float(val),
            'raw_value': raw_v,
            'direction': 'Increases Recommendation' if val > 0 else 'Decreases Recommendation'
        })
        
    impacts_df = pd.DataFrame(impacts).sort_values(by='shap_value', key=abs, ascending=False)
    
    explanation = {
        'student_id': student_id,
        'prediction': pred,
        'probability': prob,
        'recommendation_text': "Recommended for Consideration" if pred == 1 else "Not Recommended",
        'base_value': expected_val,
        'impacts_df': impacts_df,
        'student_raw': student_raw_dict
    }
    
    return explanation

def plot_individual_shap(explanation):
    """
    Renders a clear waterfall / horizontal bar chart showing individual feature contributions.
    """
    df = explanation['impacts_df'].copy()
    top_df = df.head(7).iloc[::-1] # Reverse to display top at top
    
    clean_labels = {
        'academic_score': f"Academic Score ({explanation['student_raw'].get('academic_score', '')})",
        'financial_need': f"Financial Need Index ({explanation['student_raw'].get('financial_need', '')})",
        'family_income': f"Family Income (₹{explanation['student_raw'].get('family_income', 0):,})",
        'attendance': f"Attendance ({explanation['student_raw'].get('attendance', '')}%)",
        'extracurricular_score': f"Extracurriculars ({explanation['student_raw'].get('extracurricular_score', '')})",
        'first_generation_student': f"First-Gen Student ({'Yes' if explanation['student_raw'].get('first_generation_student') == 1 else 'No'})",
        'school_type_Private Aided': "School: Private Aided",
        'school_type_Private Unaided': "School: Private Unaided",
        'location_Semi-Urban': "Location: Semi-Urban",
        'location_Urban': "Location: Urban"
    }
    
    labels = [clean_labels.get(f, f) for f in top_df['feature']]
    values = top_df['shap_value'].values
    colors = ['#28a745' if v > 0 else '#dc3545' for v in values]
    
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(labels, values, color=colors, edgecolor='black', alpha=0.88, height=0.55)
    
    for bar in bars:
        w = bar.get_width()
        x_pos = w + (0.005 if w >= 0 else -0.005)
        ha_align = 'left' if w >= 0 else 'right'
        ax.annotate(f"{w:+.3f}", xy=(x_pos, bar.get_y() + bar.get_height() / 2),
                    ha=ha_align, va='center', fontsize=9, fontweight='bold')
        
    ax.axvline(0, color='black', linewidth=1.2)
    max_abs = max(0.15, np.max(np.abs(values)) * 1.35)
    ax.set_xlim(-max_abs, max_abs)
    
    ax.set_title(f"SHAP Feature Attribution: {explanation['student_id']}\n(Green: Favors Recommendation | Red: Lowers Recommendation)",
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("SHAP Impact Score on Recommendation Probability", fontsize=10)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig

def generate_plain_language_explanation(explanation):
    """
    Translates SHAP mathematical attributions into human-understandable prose.
    """
    df = explanation['impacts_df']
    pos_factors = df[df['shap_value'] > 0.005].sort_values(by='shap_value', ascending=False)
    neg_factors = df[df['shap_value'] < -0.005].sort_values(by='shap_value', ascending=True)
    raw = explanation['student_raw']
    prob_pct = explanation['probability'] * 100
    
    lines = []
    lines.append(f"**AI Recommendation**: `{explanation['recommendation_text']}` (Confidence Score: **{prob_pct:.1f}%**)")
    lines.append("")
    lines.append("###  Key Factors Increasing Recommendation:")
    if len(pos_factors) > 0:
        for _, row in pos_factors.head(3).iterrows():
            f = row['feature']
            val = row['shap_value']
            if f == 'academic_score':
                lines.append(f"- **Academic Excellence**: High score of **{raw.get('academic_score', '')}/100** contributed strongly (+{val:.2f}) toward recommendation.")
            elif f == 'financial_need':
                lines.append(f"- **Demonstrated Financial Need**: Need index of **{raw.get('financial_need', '')}/100** contributed positively (+{val:.2f}) based on equity criteria.")
            elif f == 'family_income':
                lines.append(f"- **Socioeconomic Eligibility**: Household income of **₹{raw.get('family_income', 0):,}** supported qualification (+{val:.2f}).")
            elif f == 'attendance':
                lines.append(f"- **Consistent Attendance**: Attendance rate of **{raw.get('attendance', '')}%** positively influenced the recommendation (+{val:.2f}).")
            elif f == 'first_generation_student':
                lines.append(f"- **First-Generation College Goer**: Verified first-generation status provided affirmative support (+{val:.2f}).")
            elif f == 'extracurricular_score':
                lines.append(f"- **Holistic Profile**: Active extracurricular rating of **{raw.get('extracurricular_score', '')}/100** boosted the score (+{val:.2f}).")
            else:
                lines.append(f"- **Background Attribute ({f})**: Contributed positively (+{val:.2f}).")
    else:
        lines.append("- *No significant positive factors identified for this applicant.*")
        
    lines.append("")
    lines.append("###  Factors Decreasing Recommendation:")
    if len(neg_factors) > 0:
        for _, row in neg_factors.head(3).iterrows():
            f = row['feature']
            val = row['shap_value']
            if f == 'family_income':
                lines.append(f"- **Higher Family Income**: Household income of **₹{raw.get('family_income', 0):,}** reduced need-based weighting ({val:.2f}).")
            elif f == 'financial_need':
                lines.append(f"- **Lower Financial Need Index**: Financial need score of **{raw.get('financial_need', '')}/100** lowered priority ({val:.2f}).")
            elif f == 'academic_score':
                lines.append(f"- **Academic Score**: Score of **{raw.get('academic_score', '')}/100** was below the competitive threshold ({val:.2f}).")
            elif f == 'extracurricular_score':
                lines.append(f"- **Extracurricular Participation**: Score of **{raw.get('extracurricular_score', '')}/100** lowered overall composite ranking ({val:.2f}).")
            elif f == 'attendance':
                lines.append(f"- **Attendance Gaps**: Attendance record of **{raw.get('attendance', '')}%** reduced consistency rating ({val:.2f}).")
            elif f == 'first_generation_student':
                lines.append(f"- **Non-First-Generation**: Did not receive first-generation equity weighting ({val:.2f}).")
            else:
                lines.append(f"- **Background Attribute ({f})**: Lowered recommendation score ({val:.2f}).")
    else:
        lines.append("- *No major negative factors identified for this applicant.*")
        
    lines.append("")
    lines.append(">  **Institutional Notice**: *These factors represent AI feature contributions to aid human reviewers. Final scholarship awards require verification of supporting documentation by the Scholarship Committee.*")
    
    return "\n".join(lines)

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    import joblib
    bundle = joblib.load("models/scholarship_model.pkl")
    print("Testing single student explanation...")
    sample = {
        'academic_score': 88.5,
        'attendance': 92.0,
        'family_income': 120000,
        'financial_need': 78.0,
        'extracurricular_score': 70.0,
        'first_generation_student': 1,
        'school_type': 'Government',
        'location': 'Rural'
    }
    exp = explain_single_student_by_features(bundle, sample, student_id="STU-DEMO-01")
    print("Probability:", exp['probability'])
    print(generate_plain_language_explanation(exp))
