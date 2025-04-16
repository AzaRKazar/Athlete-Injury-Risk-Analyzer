# Model without threshold-derived features
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import joblib

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
figures_dir = os.path.join(script_dir, '../figures')
tables_dir = os.path.join(script_dir, '../tables')
model_dir = os.path.join(script_dir, '../trained-model')

# Create directories if they don't exist
for dir_path in [figures_dir, tables_dir, model_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Load model data
model_data_path = os.path.join(data_dir, 'model_data.csv')
df = pd.read_csv(model_data_path)
print(f"Loaded model data: {df.shape}")

# Define force_asymmetry_threshold (usually 0.10 or 10%)
force_asymmetry_threshold = 0.10

# Create binary target variable for high injury risk based on MaxForceSymmetry
# We do this to demonstrate the issue, but we'll then exclude this feature from modeling
df['injury_risk_high'] = ((df['MaxForceSymmetry'] < (1 - force_asymmetry_threshold)) | 
                        (df['MaxForceSymmetry'] > (1 + force_asymmetry_threshold))).astype(int)

print(f"Created injury_risk_high target with {df['injury_risk_high'].sum()} high risk samples ({df['injury_risk_high'].mean():.1%})")

# Approach 1: Remove direct symmetry metrics from features
# This approach removes MaxForceSymmetry and TorqueSymmetry which are directly related 
# to the target definition, while keeping raw measurements like leftMaxForce, rightMaxForce
indirect_features = [
    'leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque',
    'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque',
    'ForceSymmetry', 'ImpulseSymmetry',  # We can keep these as they're not directly used for target definition
    'days_since_first_test'
]

# Option: If you have sport data, include it
if 'sport' in df.columns:
    # One-hot encode the sport column
    sport_dummies = pd.get_dummies(df['sport'], prefix='sport')
    df = pd.concat([df, sport_dummies], axis=1)
    # Add sport dummy columns to features
    sport_features = list(sport_dummies.columns)
    indirect_features.extend(sport_features)

# Prepare data for modeling without direct features
X = df[indirect_features]
y = df['injury_risk_high']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nModeling with {len(indirect_features)} features, excluding MaxForceSymmetry and TorqueSymmetry")
print(f"Features: {indirect_features}")

# Create and train four different models
models = {
    "RF_No_Balancing": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF_with_SMOTE": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF_with_SMOTEENN": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF_with_Class_Weights": RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Apply sampling technique if applicable
    if name == "RF_with_SMOTE":
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Applied SMOTE: training samples {len(X_train)} → {len(X_train_resampled)}")
        model.fit(X_train_resampled, y_train_resampled)
    elif name == "RF_with_SMOTEENN":
        smoteenn = SMOTEENN(random_state=42)
        X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)
        print(f"Applied SMOTEENN: training samples {len(X_train)} → {len(X_train_resampled)}")
        model.fit(X_train_resampled, y_train_resampled)
    else:
        # No balancing or class weights (handled in model definition)
        model.fit(X_train, y_train)
    
    # Save the model
    model_path = os.path.join(model_dir, f"{name}_without_direct_features.pkl")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Cross-validation for robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    print(f"  Performance: Accuracy={accuracy:.3f}, F1={f1:.3f}, ROC AUC={roc_auc:.3f}")
    print(f"  Cross-validation ROC AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Store results
    results[name] = {
        "Accuracy": accuracy,
        "F1": f1,
        "ROC AUC": roc_auc,
        "CV ROC AUC Mean": cv_scores.mean(),
        "CV ROC AUC Std": cv_scores.std(),
        "Model": model
    }
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Low Risk', 'High Risk'],
        yticklabels=['Low Risk', 'High Risk']
    )
    plt.title(f'Confusion Matrix - {name} (Without Direct Features)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{name}_without_direct_features.png'), dpi=300)
    plt.close()

# Find the best model based on F1 score
best_model_name = max(results, key=lambda k: results[k]["F1"])
best_model = results[best_model_name]
print(f"\nBest model: {best_model_name} (F1={best_model['F1']:.3f})")

# Get feature importance for the best model
best_model_obj = best_model["Model"]
feature_importance = pd.DataFrame({
    'Feature': indirect_features,
    'Importance': best_model_obj.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title(f'Feature Importance - {best_model_name} (Without Direct Features)')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, f'feature_importance_{best_model_name}_without_direct_features.png'), dpi=300)
plt.close()

# Save results to CSV
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['Accuracy'] for m in results],
    'F1': [results[m]['F1'] for m in results],
    'ROC AUC': [results[m]['ROC AUC'] for m in results],
    'CV ROC AUC': [results[m]['CV ROC AUC Mean'] for m in results],
    'CV ROC AUC Std': [results[m]['CV ROC AUC Std'] for m in results]
})

results_df.to_csv(os.path.join(tables_dir, 'model_results_without_direct_features.csv'), index=False)
print(f"Saved results to {os.path.join(tables_dir, 'model_results_without_direct_features.csv')}")

# Compare to original results
print("\n--- Comparison with Original Models ---")
print("Without direct features that leak information:")
print(results_df[['Model', 'Accuracy', 'F1', 'ROC AUC']])
print("\nNote: Original models would have nearly perfect metrics (close to 1.0) due to data leakage.")