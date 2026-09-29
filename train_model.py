"""
train_model.py
--------------
Trains the Machine Learning Scholarship Recommendation Model.

RESPONSIBLE AI CORE DESIGN:
- Model is a Decision-Support System (DSS) that outputs recommendation probabilities.
- Feature set includes legitimate scholarship criteria:
  academic_score, attendance, family_income, financial_need,
  extracurricular_score, first_generation_student, school_type, location.
- PROTECTED ATTRIBUTE EXCLUSION:
  'gender' is strictly EXCLUDED from the training feature set X to avoid direct disparate treatment.
  It is preserved alongside other demographic variables in the audit metadata for Fairlearn analysis.
- Performance metrics evaluated: Accuracy, Precision, Recall, F1, Confusion Matrix.
- Explicit reminder: High model accuracy is NOT proof of fairness.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

def train_scholarship_model(data_path="data/scholarship_students.csv", model_dir="models"):
    if not os.path.exists(data_path):
        from generate_dataset import generate_scholarship_dataset
        print(f"Data file {data_path} not found. Generating synthetic dataset...")
        generate_scholarship_dataset(output_path=data_path)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    # Feature definitions
    # Legitimate scholarship criteria
    numerical_features = [
        'academic_score', 'attendance', 'family_income',
        'financial_need', 'extracurricular_score', 'first_generation_student'
    ]
    categorical_features = ['school_type', 'location']
    
    # Target
    target = 'eligible'
    
    # Audit attributes (retained for fairness auditing, not trained on)
    audit_features = ['gender', 'school_type', 'location', 'student_id', 'name']

    X = df[numerical_features + categorical_features]
    y = df[target]

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Keep corresponding audit slice for test set
    df_test_audit = df.loc[X_test.index, audit_features + [target]].copy()
    df_train_audit = df.loc[X_train.index, audit_features + [target]].copy()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # Fit preprocessor
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numerical_features + cat_feature_names

    X_train_df = pd.DataFrame(X_train_trans, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=all_feature_names, index=X_test.index)

    # Primary Model: Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_df, y_train)

    # Baseline Model: Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_df, y_train)

    # Evaluation on Test Set (Random Forest)
    y_pred = rf_model.predict(X_test_df)
    y_prob = rf_model.predict_proba(X_test_df)[:, 1]

    # Evaluation on Test Set (Logistic Regression)
    y_pred_lr = lr_model.predict(X_test_df)
    y_prob_lr = lr_model.predict_proba(X_test_df)[:, 1]

    # Store predictions in test audit dataframe
    df_test_audit['y_pred'] = y_pred
    df_test_audit['y_prob'] = y_prob
    df_test_audit['y_pred_lr'] = y_pred_lr
    df_test_audit['y_prob_lr'] = y_prob_lr

    # Metrics
    rf_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_prob)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    lr_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'precision': float(precision_score(y_test, y_pred_lr)),
        'recall': float(recall_score(y_test, y_pred_lr)),
        'f1': float(f1_score(y_test, y_pred_lr)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_lr)),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr).tolist()
    }

    print("\n=======================================================")
    print("      MODEL TRAINING & EVALUATION RESULTS")
    print("=======================================================")
    print(f"Random Forest Accuracy : {rf_metrics['accuracy']:.4f}")
    print(f"Random Forest Precision: {rf_metrics['precision']:.4f}")
    print(f"Random Forest Recall   : {rf_metrics['recall']:.4f}")
    print(f"Random Forest F1-Score : {rf_metrics['f1']:.4f}")
    print(f"Random Forest ROC-AUC  : {rf_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix:\n{np.array(rf_metrics['confusion_matrix'])}")
    print("\nIMPORTANT RESPONSIBLE AI NOTICE:")
    print("A high accuracy score (e.g. >90%) DOES NOT prove that the system is fair.")
    print("Demographic Parity and Equalized Odds must be formally assessed.")
    print("=======================================================\n")

    # Package everything for application and explainability
    bundle = {
        'rf_model': rf_model,
        'lr_model': lr_model,
        'preprocessor': preprocessor,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'all_feature_names': all_feature_names,
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_df': X_train_df,
        'X_test_df': X_test_df,
        'y_train': y_train,
        'y_test': y_test,
        'df_test_audit': df_test_audit,
        'df_train_audit': df_train_audit,
        'rf_metrics': rf_metrics,
        'lr_metrics': lr_metrics
    }

    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "scholarship_model.pkl")
    joblib.dump(bundle, model_save_path)
    print(f"Model bundle successfully saved to: {model_save_path}")

    return bundle

if __name__ == "__main__":
    train_scholarship_model()
