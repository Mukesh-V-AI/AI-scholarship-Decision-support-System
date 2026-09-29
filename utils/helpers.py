"""
utils/helpers.py
----------------
Utility functions for data loading, human review logging, student appeals,
and dashboard styling components.
"""

import os
import joblib
import pandas as pd
from datetime import datetime

REVIEW_CSV_PATH = "data/review_decisions.csv"
APPEALS_CSV_PATH = "data/student_appeals.csv"

def load_data(filepath="data/scholarship_students.csv"):
    if not os.path.exists(filepath):
        from generate_dataset import generate_scholarship_dataset
        return generate_scholarship_dataset(output_path=filepath)
    return pd.read_csv(filepath)

def load_model_bundle(model_path="models/scholarship_model.pkl"):
    if not os.path.exists(model_path):
        from train_model import train_scholarship_model
        return train_scholarship_model(model_dir=os.path.dirname(model_path))
    return joblib.load(model_path)

def format_currency_inr(amount, symbol="₹"):
    """
    Formats integers into standard Indian Rupee notation (e.g. ₹1,50,000 or Rs. 1,50,000).
    """
    try:
        amount = int(amount)
        orig = str(amount)
        if len(orig) <= 3:
            return f"{symbol}{orig}"
        last_three = orig[-3:]
        remaining = orig[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        return f"{symbol}{','.join(groups)},{last_three}"
    except Exception:
        return f"{symbol}{amount}"

# =========================================================================
# Human Review Audit Persistence
# =========================================================================

def init_review_db():
    if not os.path.exists(REVIEW_CSV_PATH):
        df = pd.DataFrame(columns=[
            'timestamp', 'student_id', 'student_name', 'ai_recommendation',
            'ai_confidence', 'human_decision', 'reviewer_name',
            'reviewer_role', 'reviewer_comments'
        ])
        df.to_csv(REVIEW_CSV_PATH, index=False)

def load_review_decisions():
    init_review_db()
    try:
        return pd.read_csv(REVIEW_CSV_PATH)
    except Exception:
        return pd.DataFrame(columns=[
            'timestamp', 'student_id', 'student_name', 'ai_recommendation',
            'ai_confidence', 'human_decision', 'reviewer_name',
            'reviewer_role', 'reviewer_comments'
        ])

def save_review_decision(student_id, student_name, ai_rec, ai_conf,
                         human_dec, reviewer_name, reviewer_role, comments):
    init_review_db()
    df = load_review_decisions()
    
    new_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'student_id': student_id,
        'student_name': student_name,
        'ai_recommendation': ai_rec,
        'ai_confidence': f"{ai_conf:.1f}%",
        'human_decision': human_dec,
        'reviewer_name': reviewer_name,
        'reviewer_role': reviewer_role,
        'reviewer_comments': comments
    }
    
    df = pd.concat([pd.DataFrame([new_entry]), df], ignore_index=True)
    df.to_csv(REVIEW_CSV_PATH, index=False)
    return df

# =========================================================================
# Student Appeal & Grievance Redressal Persistence
# =========================================================================

def init_appeals_db():
    if not os.path.exists(APPEALS_CSV_PATH):
        df = pd.DataFrame(columns=[
            'appeal_id', 'timestamp', 'student_id', 'student_name',
            'ground_of_appeal', 'additional_evidence', 'status', 'committee_notes'
        ])
        # Add 2 realistic sample appeals for immediate viva demonstration
        samples = [
            {
                'appeal_id': 'APP-2024-001',
                'timestamp': '2024-10-12 14:30:00',
                'student_id': 'SCH-2024042',
                'student_name': 'Kavya Raman',
                'ground_of_appeal': 'Extenuating family medical crisis impacting semester attendance',
                'additional_evidence': 'Hospital admission records and medical certificate submitted',
                'status': 'Under Review',
                'committee_notes': 'Medical documentation verified with registrar. Assigned to subcommittee.'
            },
            {
                'appeal_id': 'APP-2024-002',
                'timestamp': '2024-10-15 11:15:00',
                'student_id': 'SCH-2024108',
                'student_name': 'Deepak Subramanian',
                'ground_of_appeal': 'Correction in family income documentation (father retired)',
                'additional_evidence': 'Updated income certificate and pension statement',
                'status': 'Approved on Appeal',
                'committee_notes': 'Income recalculation confirmed student qualifies for Category A grant.'
            }
        ]
        df = pd.DataFrame(samples)
        df.to_csv(APPEALS_CSV_PATH, index=False)

def load_student_appeals():
    init_appeals_db()
    try:
        return pd.read_csv(APPEALS_CSV_PATH)
    except Exception:
        return pd.DataFrame(columns=[
            'appeal_id', 'timestamp', 'student_id', 'student_name',
            'ground_of_appeal', 'additional_evidence', 'status', 'committee_notes'
        ])

def submit_student_appeal(student_id, student_name, ground, evidence):
    init_appeals_db()
    df = load_student_appeals()
    new_id = f"APP-2024-{len(df) + 1:03d}"
    new_entry = {
        'appeal_id': new_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'student_id': student_id,
        'student_name': student_name,
        'ground_of_appeal': ground,
        'additional_evidence': evidence,
        'status': 'Pending Verification',
        'committee_notes': 'Appeal lodged. Queued for human scholarship ombudsman review.'
    }
    df = pd.concat([pd.DataFrame([new_entry]), df], ignore_index=True)
    df.to_csv(APPEALS_CSV_PATH, index=False)
    return new_id

# =========================================================================
# Custom UI Styles & Component Templates
# =========================================================================

def get_custom_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Academic Banner */
    .academic-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    
    .academic-banner h1 {
        color: white !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin: 0 0 8px 0 !important;
    }
    
    .academic-banner p {
        color: #e0f2fe !important;
        font-size: 14px !important;
        margin: 0 !important;
    }
    
    /* Disclaimer Card */
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        color: #92400e;
        font-size: 13.5px;
    }
    
    /* Recommendation Badges */
    .badge-recommended {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    
    .badge-not-recommended {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        display: inline-block;
    }
    
    .badge-human-review {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-card-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin: 4px 0;
    }
    
    .metric-card-label {
        font-size: 12.5px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Decision Support Notice Box */
    .dss-alert {
        background: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 12px 18px;
        border-radius: 6px;
        margin: 14px 0;
        font-size: 13.5px;
        color: #166534;
    }
    </style>
    """
