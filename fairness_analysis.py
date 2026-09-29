"""
fairness_analysis.py
--------------------
Provides comprehensive subgroup fairness auditing using Microsoft Fairlearn.

Evaluates AI model predictions across demographic & socioeconomic subgroups:
- Gender ('Female', 'Male')
- School Type ('Government', 'Private Aided', 'Private Unaided')
- Location ('Rural', 'Semi-Urban', 'Urban')

Calculates:
1. Selection Rate: P(Y_hat=1 | Group)
2. Statistical Parity Difference: max(SR) - min(SR)
3. Disparate Impact Ratio: min(SR) / max(SR) (Four-Fifths / 80% Rule)
4. False Positive Rate (FPR): P(Y_hat=1 | Y=0, Group)
5. False Negative Rate (FNR): P(Y_hat=0 | Y=1, Group)
6. Group-specific Accuracy, Precision, Recall

RESPONSIBLE AI PRINCIPLE:
"Different selection rates do not automatically prove discrimination.
They indicate a statistical disparity that requires deeper contextual and institutional investigation."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

def compute_fairness_audit(bundle, sensitive_col='gender'):
    """
    Computes rigorous fairness metrics for a chosen sensitive feature.
    """
    df_audit = bundle['df_test_audit'].copy()
    y_true = bundle['y_test']
    y_pred = df_audit['y_pred']
    sensitive_series = df_audit[sensitive_col]

    # Metrics dictionary for Fairlearn MetricFrame
    metrics_spec = {
        'Selection Rate': selection_rate,
        'Accuracy': accuracy_score,
        'False Positive Rate': false_positive_rate,
        'False Negative Rate': false_negative_rate,
        'Precision': lambda yt, yp: precision_score(yt, yp, zero_division=0),
        'Recall': lambda yt, yp: recall_score(yt, yp, zero_division=0)
    }

    mf = MetricFrame(
        metrics=metrics_spec,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_series
    )

    by_group_df = mf.by_group.copy()
    
    # Calculate group sample sizes and positive ground truth counts
    counts = sensitive_series.value_counts()
    pos_gt = df_audit.groupby(sensitive_col)[bundle['y_test'].name].sum()
    by_group_df['Sample Size'] = counts
    by_group_df['Actual Eligible'] = pos_gt
    
    # Overall summary metrics
    overall_sr = float(selection_rate(y_true, y_pred))
    min_sr = float(by_group_df['Selection Rate'].min())
    max_sr = float(by_group_df['Selection Rate'].max())
    
    stat_parity_diff = max_sr - min_sr
    disparate_impact_ratio = (min_sr / max_sr) if max_sr > 0 else 1.0
    
    min_fpr = float(by_group_df['False Positive Rate'].min())
    max_fpr = float(by_group_df['False Positive Rate'].max())
    fpr_diff = max_fpr - min_fpr
    
    min_fnr = float(by_group_df['False Negative Rate'].min())
    max_fnr = float(by_group_df['False Negative Rate'].max())
    fnr_diff = max_fnr - min_fnr
    
    # Four-fifths (80%) rule compliance check
    is_four_fifths_compliant = disparate_impact_ratio >= 0.80

    audit_summary = {
        'sensitive_col': sensitive_col,
        'overall_selection_rate': overall_sr,
        'group_metrics_df': by_group_df,
        'statistical_parity_difference': stat_parity_diff,
        'disparate_impact_ratio': disparate_impact_ratio,
        'is_four_fifths_compliant': is_four_fifths_compliant,
        'fpr_difference': fpr_diff,
        'fnr_difference': fnr_diff,
        'min_group': by_group_df['Selection Rate'].idxmin(),
        'max_group': by_group_df['Selection Rate'].idxmax()
    }

    return audit_summary

def plot_selection_rates(audit_summary):
    """
    Renders a bar chart comparing selection rates across groups against the overall rate.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    df = audit_summary['group_metrics_df'].reset_index()
    sens_col = audit_summary['sensitive_col']
    
    palette = sns.color_palette("mako", len(df))
    bars = ax.bar(df[sens_col], df['Selection Rate'] * 100, color=palette, edgecolor='black', alpha=0.88, width=0.5)
    
    overall_pct = audit_summary['overall_selection_rate'] * 100
    ax.axhline(overall_pct, color='#d9534f', linestyle='--', linewidth=1.8, label=f'Overall Rate ({overall_pct:.1f}%)')
    
    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    ax.set_title(f"Scholarship Selection Rate by {sens_col.replace('_', ' ').title()}", fontsize=12, pad=12, fontweight='bold')
    ax.set_ylabel("Selection Rate (%)", fontsize=10)
    ax.set_xlabel(sens_col.replace('_', ' ').title(), fontsize=10)
    ax.set_ylim(0, max(60, (df['Selection Rate'].max() * 100) + 15))
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_error_rates(audit_summary):
    """
    Plots False Positive Rate and False Negative Rate side-by-side.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    df = audit_summary['group_metrics_df'].reset_index()
    sens_col = audit_summary['sensitive_col']
    
    x = np.arange(len(df))
    width = 0.32
    
    bars1 = ax.bar(x - width/2, df['False Positive Rate'] * 100, width, label='False Positive Rate (FPR)',
                   color='#f0ad4e', edgecolor='black', alpha=0.9)
    bars2 = ax.bar(x + width/2, df['False Negative Rate'] * 100, width, label='False Negative Rate (FNR)',
                   color='#d9534f', edgecolor='black', alpha=0.9)
    
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
    ax.set_title(f"Error Rate Disparities (FPR vs FNR) across {sens_col.replace('_', ' ').title()}", fontsize=12, pad=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df[sens_col], fontsize=10)
    ax.set_ylabel("Rate (%)", fontsize=10)
    ax.set_ylim(0, max(50, max(df['False Positive Rate'].max(), df['False Negative Rate'].max()) * 100 + 15))
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    return fig

def get_fairness_interpretation(audit_summary):
    """
    Generates plain-language Responsible AI interpretation for the audit findings.
    """
    sens_col = audit_summary['sensitive_col'].replace('_', ' ').title()
    di = audit_summary['disparate_impact_ratio']
    spd = audit_summary['statistical_parity_difference'] * 100
    min_g = audit_summary['min_group']
    max_g = audit_summary['max_group']
    fpr_diff = audit_summary['fpr_difference'] * 100
    fnr_diff = audit_summary['fnr_difference'] * 100

    rule_status = "satisfies" if audit_summary['is_four_fifths_compliant'] else "violates"
    
    text = f"""
###  Responsible AI Audit Interpretation for {sens_col}

1. **Statistical Parity Assessment**:
   - The overall selection rate is **{audit_summary['overall_selection_rate']*100:.1f}%**.
   - The highest selection rate is in **{max_g}**, while the lowest is in **{min_g}**.
   - **Statistical Parity Difference**: **{spd:.1f}%**.
   - **Disparate Impact Ratio**: **{di:.2f}** (Threshold: $\\ge 0.80$). This outcome **{rule_status}** the classic EEOC Four-Fifths Rule.

2. **Error Rate Parity (Equalized Odds)**:
   - **False Positive Rate Disparity**: The difference in FPR between groups is **{fpr_diff:.1f}%**. False positives mean ineligible students are recommended.
   - **False Negative Rate Disparity**: The difference in FNR between groups is **{fnr_diff:.1f}%**. False negatives mean deserving students from disadvantaged backgrounds are inadvertently omitted from scholarship consideration.

3. **Core Responsible AI Takeaway**:
   > **"Different selection rates do not automatically prove intentional discrimination. They indicate an empirical disparity that requires institutional investigation."**
   
   - If demographic groups differ in baseline financial need or institutional schooling access, parity metrics must be analyzed alongside equity goals.
   - In scholarship allocation, higher recommendation rates for students from economically disadvantaged school backgrounds (e.g. Government schools) may reflect intended affirmative merit-cum-means policy rather than algorithm defect.
   - Human review is strictly required to verify whether differences stem from legitimate policy criteria or unintended proxy biases.
"""
    return text

if __name__ == "__main__":
    import joblib
    bundle = joblib.load("models/scholarship_model.pkl")
    audit = compute_fairness_audit(bundle, 'gender')
    print("Fairness audit completed for gender:")
    print(audit['group_metrics_df'])
