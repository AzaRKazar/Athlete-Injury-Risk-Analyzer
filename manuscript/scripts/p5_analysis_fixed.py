import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import pickle
import warnings
# Additional imports for new features
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shap
from sklearn.inspection import permutation_importance
import matplotlib.gridspec as gridspec
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# Setting paths - Updated to work from manuscript/scripts directory
main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_dir = os.path.join(main_dir, 'data')
figures_dir = os.path.join(main_dir, 'manuscript', 'figures')
tables_dir = os.path.join(main_dir, 'manuscript', 'tables')
model_dir = os.path.join(main_dir, 'trained-model')

# Create directories if they don't exist
for dir_path in [figures_dir, tables_dir, model_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Load the model data
model_data_path = os.path.join(data_dir, 'model_data.csv')
df = pd.read_csv(model_data_path)
print(f"Loaded model data: {df.shape}")

# Display the column names to verify
print("Available columns:", df.columns.tolist())

# Define the list of numeric features based on actual columns
numeric_features = ['leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque',
              'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque',
              'ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']

# Define features based on actual columns in the dataset
original_features = ['leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque',
              'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque',
              'ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']

# Define improved features set (removing direct symmetry metrics to prevent data leakage)
improved_features = ['leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque',
              'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque',
              'ForceSymmetry', 'ImpulseSymmetry'] # Removed MaxForceSymmetry and TorqueSymmetry

# Create a synthetic target for demonstration (you'll need to adjust this with your actual target)
# For demonstration, let's create a high-risk label for athletes with high force asymmetry
if 'injury_risk_high' not in df.columns:
    print("Creating synthetic target 'injury_risk_high' based on force asymmetry")
    # Consider high risk if force asymmetry is significantly different from 1.0 (perfect symmetry)
    df['injury_risk_high'] = ((df['MaxForceSymmetry'] > 1.1) | (df['MaxForceSymmetry'] < 0.9)).astype(int)
    print(f"Created synthetic target with {df['injury_risk_high'].sum()} high-risk samples")

# Create an improved longitudinal target if possible
if 'sbuid' in df.columns and 'testDateUtc' in df.columns:
    print("Creating improved longitudinal performance decline target")
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['testDateUtc']):
        df['testDateUtc'] = pd.to_datetime(df['testDateUtc'])
    
    # Find athletes with multiple tests
    athlete_test_counts = df['sbuid'].value_counts()
    multi_test_athletes = athlete_test_counts[athlete_test_counts > 1].index
    
    if len(multi_test_athletes) > 0:
        # Initialize the new target column
        df['performance_decline'] = 0
        
        for athlete in multi_test_athletes:
            athlete_data = df[df['sbuid'] == athlete].sort_values('testDateUtc')
            
            if len(athlete_data) >= 2:
                # Get first and last test
                first_test = athlete_data.iloc[0]
                last_test = athlete_data.iloc[-1]
                
                # Define performance decline as worsening symmetry over time
                max_force_decline = abs(1 - last_test['MaxForceSymmetry']) > abs(1 - first_test['MaxForceSymmetry'])
                torque_decline = abs(1 - last_test['TorqueSymmetry']) > abs(1 - first_test['TorqueSymmetry'])
                
                # Mark as declined if either metric worsened
                if max_force_decline or torque_decline:
                    df.loc[athlete_data.index, 'performance_decline'] = 1
        
        print(f"Created longitudinal target with {df['performance_decline'].sum()} decline cases")
        
        # Use this as an alternative target
        longitudinal_target = 'performance_decline'
    else:
        print("Not enough multi-test athletes for longitudinal analysis")
        longitudinal_target = None
else:
    longitudinal_target = None

# Check for target label
target = 'injury_risk_high'
if target not in df.columns:
    print(f"Error: Target column '{target}' not found in the dataset")
    exit(1)

# Add sbuid as athlete ID if needed for grouping
if 'sbuid' in df.columns:
    print(f"Found {df['sbuid'].nunique()} unique athletes in the dataset")

# Check for date column to perform temporal analysis
if 'testDateUtc' in df.columns:
    # Convert to datetime
    df['testDateUtc'] = pd.to_datetime(df['testDateUtc'])
    print(f"Date range: {df['testDateUtc'].min()} to {df['testDateUtc'].max()}")

# Select data for modeling
# Original approach - using potentially problematic features
X_original = df[original_features]
y = df[target]

# Improved approach - removing direct symmetry metrics to prevent data leakage
X_improved = df[improved_features]

# Print feature sets
print(f"Original features: {len(original_features)}")
print(f"Improved features (removed direct symmetry metrics): {len(improved_features)}")

# Create models for both feature sets to demonstrate improvement
X = X_improved  # Use improved features as default

# Perform exploratory data analysis
print("\n--- Exploratory Data Analysis ---")

# Summary statistics
summary = X.describe().T
summary['missing'] = X.isnull().sum()
summary['missing_percentage'] = (X.isnull().sum() / len(X)) * 100
print("Feature Summary Statistics:")
print(summary)

# Save summary to CSV
summary.to_csv(os.path.join(tables_dir, 'feature_summary_statistics.csv'))

# Data distribution visualization
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features[:9]):  # Limit to first 9 features for readability
    plt.subplot(3, 3, i+1)
    sns.histplot(X[feature], kde=True)
    plt.title(feature)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'feature_distributions.png'), dpi=300)

# Risk distribution
plt.figure(figsize=(10, 6))
risk_counts = df[target].value_counts()
sns.barplot(x=risk_counts.index.map({0: 'Low Risk', 1: 'High Risk'}), y=risk_counts.values)
plt.title('Risk Category Distribution')
plt.xlabel('Risk Category')
plt.ylabel('Count')
plt.savefig(os.path.join(figures_dir, 'risk_category_distribution.png'), dpi=300)

# Calculate effect sizes for each feature by risk category
print("\n--- Effect Size Analysis ---")
effect_sizes = []

for feature in numeric_features:
    high_risk_values = df[df[target] == 1][feature].dropna()
    low_risk_values = df[df[target] == 0][feature].dropna()
    
    if len(high_risk_values) > 0 and len(low_risk_values) > 0:
        # Calculate Cohen's d effect size
        mean_diff = high_risk_values.mean() - low_risk_values.mean()
        pooled_std = np.sqrt(((high_risk_values.std() ** 2) + (low_risk_values.std() ** 2)) / 2)
        
        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
        else:
            cohens_d = 0
            
        # Calculate p-value from t-test
        t_stat, p_value = stats.ttest_ind(high_risk_values, low_risk_values, equal_var=False, nan_policy='omit')
        
        effect_sizes.append({
            'feature': feature,
            'high_risk_mean': high_risk_values.mean(),
            'low_risk_mean': low_risk_values.mean(),
            'mean_difference': mean_diff,
            'cohens_d': cohens_d,
            'p_value': p_value
        })

effect_size_df = pd.DataFrame(effect_sizes)
effect_size_df = effect_size_df.sort_values('cohens_d', ascending=False)
print("Top features by effect size:")
print(effect_size_df.head())

# Save effect sizes to CSV
effect_size_df.to_csv(os.path.join(tables_dir, 'effect_sizes.csv'), index=False)

# Plot effect size heatmap
plt.figure(figsize=(10, 12))
effect_size_pivot = effect_size_df.pivot_table(
    index='feature', 
    values=['cohens_d', 'p_value'], 
    aggfunc='first'
)
effect_size_pivot = effect_size_pivot.sort_values('cohens_d', ascending=False)

# Create a separate column indicating statistical significance
effect_size_pivot['significant'] = effect_size_pivot['p_value'] < 0.05

# Plot heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(
    effect_size_pivot[['cohens_d']],
    annot=True,
    cmap='RdBu_r',
    center=0,
    fmt='.2f',
    linewidths=0.5,
    cbar_kws={'label': "Cohen's d Effect Size"}
)
plt.title("Effect Size Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'effect_size_heatmap.png'), dpi=300)

# Correlation analysis
plt.figure(figsize=(12, 10))
corr_matrix = X.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, 
            linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=300)

# Athlete Progress Analysis
if 'sbuid' in df.columns and 'testDateUtc' in df.columns:
    print("\n--- Athlete Progress Analysis ---")
    
    # Get athletes with multiple tests
    athlete_test_counts = df['sbuid'].value_counts()
    multi_test_athletes = athlete_test_counts[athlete_test_counts > 1].index.tolist()
    
    if multi_test_athletes:
        print(f"Found {len(multi_test_athletes)} athletes with multiple tests")
        
        # Plot progress for a sample athlete
        sample_athlete = multi_test_athletes[0]
        athlete_data = df[df['sbuid'] == sample_athlete].sort_values('testDateUtc')
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(athlete_data['testDateUtc'], athlete_data['ForceSymmetry'], 'o-')
        plt.title(f'Force Symmetry Over Time (Athlete {sample_athlete})')
        plt.xticks(rotation=45)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # Perfect symmetry line
        
        plt.subplot(2, 2, 2)
        plt.plot(athlete_data['testDateUtc'], athlete_data['MaxForceSymmetry'], 'o-')
        plt.title('Max Force Symmetry Over Time')
        plt.xticks(rotation=45)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        
        plt.subplot(2, 2, 3)
        plt.plot(athlete_data['testDateUtc'], athlete_data['leftMaxForce'], 'b-o', label='Left')
        plt.plot(athlete_data['testDateUtc'], athlete_data['rightMaxForce'], 'r-o', label='Right')
        plt.title('Max Force Comparison')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(athlete_data['testDateUtc'], athlete_data['leftAvgForce'], 'b-o', label='Left')
        plt.plot(athlete_data['testDateUtc'], athlete_data['rightAvgForce'], 'r-o', label='Right')
        plt.title('Avg Force Comparison')
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'athlete_progress_example.png'), dpi=300)
        
        # Improvement analysis
        if len(multi_test_athletes) >= 10:
            improvement_data = []
            
            for athlete_id in multi_test_athletes[:20]:  # Analyze up to 20 athletes
                athlete_data = df[df['sbuid'] == athlete_id].sort_values('testDateUtc')
                
                if len(athlete_data) >= 2:
                    first_test = athlete_data.iloc[0]
                    last_test = athlete_data.iloc[-1]
                    
                    # Calculate improvements
                    force_sym_improvement = abs(1 - last_test['ForceSymmetry']) < abs(1 - first_test['ForceSymmetry'])
                    max_force_sym_improvement = abs(1 - last_test['MaxForceSymmetry']) < abs(1 - first_test['MaxForceSymmetry'])
                    left_force_change = (last_test['leftMaxForce'] - first_test['leftMaxForce']) / first_test['leftMaxForce']
                    right_force_change = (last_test['rightMaxForce'] - first_test['rightMaxForce']) / first_test['rightMaxForce']
                    
                    improvement_data.append({
                        'athlete_id': athlete_id,
                        'tests_count': len(athlete_data),
                        'days_between_tests': (last_test['testDateUtc'] - first_test['testDateUtc']).days,
                        'force_sym_improvement': force_sym_improvement,
                        'max_force_sym_improvement': max_force_sym_improvement,
                        'left_force_change': left_force_change,
                        'right_force_change': right_force_change
                    })
            
            if improvement_data:
                improvement_df = pd.DataFrame(improvement_data)
                
                # Summary statistics
                print("Improvement Summary:")
                print(f"Force symmetry improved in {improvement_df['force_sym_improvement'].mean():.1%} of athletes")
                print(f"Max force symmetry improved in {improvement_df['max_force_sym_improvement'].mean():.1%} of athletes")
                print(f"Average left force change: {improvement_df['left_force_change'].mean():.1%}")
                print(f"Average right force change: {improvement_df['right_force_change'].mean():.1%}")
                
                # Save to CSV
                improvement_df.to_csv(os.path.join(tables_dir, 'athlete_improvements.csv'), index=False)

# Initialize models dictionary 
models = {}

# Before we start the performance decline prediction, initialize the results list
results = []

# Performance Decline Prediction Analysis
print("\n--- Performance Decline Prediction Analysis ---")

# Check if we have the necessary data for longitudinal analysis
if 'sbuid' in df.columns and 'testDateUtc' in df.columns:
    # Find athletes with multiple tests
    athlete_test_counts = df['sbuid'].value_counts()
    multi_test_athletes = athlete_test_counts[athlete_test_counts > 1].index.tolist()
    
    if len(multi_test_athletes) > 0:
        print(f"Creating longitudinal dataset from {len(multi_test_athletes)} athletes with multiple tests")
        
        # Create a longitudinal dataset with paired observations
        longitudinal_data = []
        
        for athlete_id in multi_test_athletes:
            athlete_tests = df[df['sbuid'] == athlete_id].sort_values('testDateUtc')
            
            if len(athlete_tests) >= 2:
                # Get tests in chronological order
                tests = athlete_tests.to_dict('records')
                
                # Create pairs of consecutive tests
                for i in range(len(tests) - 1):
                    initial_test = tests[i]
                    follow_up_test = tests[i + 1]
                    
                    # Calculate days between tests
                    days_between = (follow_up_test['testDateUtc'] - initial_test['testDateUtc']).days
                    
                    # Calculate symmetry changes
                    force_sym_worse = abs(1 - follow_up_test['ForceSymmetry']) > abs(1 - initial_test['ForceSymmetry'])
                    impulse_sym_worse = abs(1 - follow_up_test['ImpulseSymmetry']) > abs(1 - initial_test['ImpulseSymmetry'])
                    max_force_sym_worse = abs(1 - follow_up_test['MaxForceSymmetry']) > abs(1 - initial_test['MaxForceSymmetry'])
                    torque_sym_worse = abs(1 - follow_up_test['TorqueSymmetry']) > abs(1 - initial_test['TorqueSymmetry'])
                    
                    # Calculate significant symmetry changes (>10% deterioration)
                    force_sym_sig_worse = (abs(1 - follow_up_test['ForceSymmetry']) - abs(1 - initial_test['ForceSymmetry'])) > 0.1
                    impulse_sym_sig_worse = (abs(1 - follow_up_test['ImpulseSymmetry']) - abs(1 - initial_test['ImpulseSymmetry'])) > 0.1
                    max_force_sym_sig_worse = (abs(1 - follow_up_test['MaxForceSymmetry']) - abs(1 - initial_test['MaxForceSymmetry'])) > 0.1
                    torque_sym_sig_worse = (abs(1 - follow_up_test['TorqueSymmetry']) - abs(1 - initial_test['TorqueSymmetry'])) > 0.1
                    
                    # Count worsening metrics
                    num_worse = sum([force_sym_worse, impulse_sym_worse, max_force_sym_worse, torque_sym_worse])
                    num_sig_worse = sum([force_sym_sig_worse, impulse_sym_sig_worse, max_force_sym_sig_worse, torque_sym_sig_worse])
                    
                    # Create target variables
                    any_symmetry_worse = num_worse > 0
                    significant_symmetry_worse = num_sig_worse > 0
                    
                    # Categorize decline severity
                    if num_worse == 0:
                        decline_severity = "None"
                    elif num_worse == 1:
                        decline_severity = "Minimal"
                    elif num_worse == 2:
                        decline_severity = "Moderate"
                    else:
                        decline_severity = "Severe"
                    
                    # Extract features from initial test only (for fair prediction)
                    initial_features = {feature: initial_test[feature] for feature in improved_features}
                    
                    # Create entry with initial features and decline targets
                    entry = {
                        'athlete_id': athlete_id,
                        'initial_test_date': initial_test['testDateUtc'],
                        'follow_up_test_date': follow_up_test['testDateUtc'],
                        'days_between_tests': days_between,
                        'any_symmetry_worse': any_symmetry_worse,
                        'significant_symmetry_worse': significant_symmetry_worse,
                        'num_worse_metrics': num_worse,
                        'num_sig_worse_metrics': num_sig_worse,
                        'decline_severity': decline_severity
                    }
                    
                    # Add initial test features
                    entry.update(initial_features)
                    
                    # Add sport if available
                    if 'sport' in initial_test:
                        entry['sport'] = initial_test['sport']
                    
                    longitudinal_data.append(entry)
        
        # Create DataFrame from longitudinal data
        if longitudinal_data:
            longitudinal_df = pd.DataFrame(longitudinal_data)
            
            # Save longitudinal dataset
            longitudinal_df.to_csv(os.path.join(data_dir, 'longitudinal_model_data.csv'), index=False)
            
            print(f"Created longitudinal dataset with {len(longitudinal_df)} paired observations")
            
            # Analysis of decline targets
            any_worse_pct = longitudinal_df['any_symmetry_worse'].mean() * 100
            sig_worse_pct = longitudinal_df['significant_symmetry_worse'].mean() * 100
            
            print(f"Percentage with any symmetry worsening: {any_worse_pct:.1f}%")
            print(f"Percentage with significant symmetry worsening: {sig_worse_pct:.1f}%")
            
            # Prepare data for modeling
            X_long = longitudinal_df[improved_features]
            if 'sport' in longitudinal_df.columns:
                # One-hot encode sport
                X_sport = pd.get_dummies(longitudinal_df[['sport']], prefix='sport')
                X_long = pd.concat([X_long, X_sport], axis=1)
            
            # Add days between tests as a feature
            X_long['days_between_tests'] = longitudinal_df['days_between_tests']
            
            # Define targets for performance decline
            y_any_worse = longitudinal_df['any_symmetry_worse']
            y_sig_worse = longitudinal_df['significant_symmetry_worse']
            
            # Train models for both targets
            # First, for any symmetry worsening
            print("\nModeling any symmetry worsening:")
            X_train_any, X_test_any, y_train_any, y_test_any = train_test_split(
                X_long, y_any_worse, test_size=0.2, random_state=42, stratify=y_any_worse
            )
            
            # Apply SMOTE for balance
            smote = SMOTE(random_state=42)
            X_train_any_smote, y_train_any_smote = smote.fit_resample(X_train_any, y_train_any)
            
            # Train RF model
            rf_any = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_any.fit(X_train_any_smote, y_train_any_smote)
            
            # Evaluate
            y_pred_any = rf_any.predict(X_test_any)
            y_pred_proba_any = rf_any.predict_proba(X_test_any)[:, 1]
            
            # Print metrics
            accuracy_any = accuracy_score(y_test_any, y_pred_any)
            precision_any = precision_score(y_test_any, y_pred_any, zero_division=0)
            recall_any = recall_score(y_test_any, y_pred_any)
            f1_any = f1_score(y_test_any, y_pred_any)
            roc_auc_any = roc_auc_score(y_test_any, y_pred_proba_any)
            
            print(f"  Accuracy: {accuracy_any:.3f}")
            print(f"  Precision: {precision_any:.3f}")
            print(f"  Recall: {recall_any:.3f}")
            print(f"  F1 Score: {f1_any:.3f}")
            print(f"  ROC AUC: {roc_auc_any:.3f}")
            
            # Save model metrics
            any_worse_metrics = {
                'Model': 'RF_Any_Symmetry_Worse',
                'Accuracy': accuracy_any,
                'Precision': precision_any,
                'Recall': recall_any,
                'F1': f1_any,
                'ROC AUC': roc_auc_any
            }
            
            # Feature importance
            any_worse_importance = pd.DataFrame({
                'Feature': X_long.columns,
                'Importance': rf_any.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Save feature importance
            any_worse_importance.to_csv(os.path.join(tables_dir, 'any_worse_feature_importance.csv'), index=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=any_worse_importance.head(10))
            plt.title('Top 10 Feature Importances - Any Symmetry Worsening')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'any_worse_feature_importance.png'), dpi=300)
            
            # Now, for significant symmetry worsening
            print("\nModeling significant symmetry worsening:")
            X_train_sig, X_test_sig, y_train_sig, y_test_sig = train_test_split(
                X_long, y_sig_worse, test_size=0.2, random_state=42, stratify=y_sig_worse
            )
            
            # Apply SMOTEENN for balance
            smoteenn = SMOTEENN(random_state=42)
            X_train_sig_smoteenn, y_train_sig_smoteenn = smoteenn.fit_resample(X_train_sig, y_train_sig)
            
            # Train RF model
            rf_sig = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_sig.fit(X_train_sig_smoteenn, y_train_sig_smoteenn)
            
            # Evaluate
            y_pred_sig = rf_sig.predict(X_test_sig)
            y_pred_proba_sig = rf_sig.predict_proba(X_test_sig)[:, 1]
            
            # Print metrics
            accuracy_sig = accuracy_score(y_test_sig, y_pred_sig)
            precision_sig = precision_score(y_test_sig, y_pred_sig, zero_division=0)
            recall_sig = recall_score(y_test_sig, y_pred_sig)
            f1_sig = f1_score(y_test_sig, y_pred_sig)
            roc_auc_sig = roc_auc_score(y_test_sig, y_pred_proba_sig)
            
            print(f"  Accuracy: {accuracy_sig:.3f}")
            print(f"  Precision: {precision_sig:.3f}")
            print(f"  Recall: {recall_sig:.3f}")
            print(f"  F1 Score: {f1_sig:.3f}")
            print(f"  ROC AUC: {roc_auc_sig:.3f}")
            
            # Save model metrics
            sig_worse_metrics = {
                'Model': 'RF_Significant_Symmetry_Worse',
                'Accuracy': accuracy_sig,
                'Precision': precision_sig,
                'Recall': recall_sig,
                'F1': f1_sig,
                'ROC AUC': roc_auc_sig
            }
            
            # Feature importance
            sig_worse_importance = pd.DataFrame({
                'Feature': X_long.columns,
                'Importance': rf_sig.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Save feature importance
            sig_worse_importance.to_csv(os.path.join(tables_dir, 'sig_worse_feature_importance.csv'), index=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=sig_worse_importance.head(10))
            plt.title('Top 10 Feature Importances - Significant Symmetry Worsening')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'sig_worse_feature_importance.png'), dpi=300)
            
            # Plot decline severity distribution
            plt.figure(figsize=(10, 6))
            severity_counts = longitudinal_df['decline_severity'].value_counts().sort_index()
            sns.barplot(x=severity_counts.index, y=severity_counts.values)
            plt.title('Distribution of Symmetry Decline Severity')
            plt.xlabel('Decline Severity')
            plt.ylabel('Count')
            plt.savefig(os.path.join(figures_dir, 'decline_severity_distribution.png'), dpi=300)
            
            # Analyze effect of time between tests
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='decline_severity', y='days_between_tests', data=longitudinal_df)
            plt.title('Days Between Tests vs. Decline Severity')
            plt.xlabel('Decline Severity')
            plt.ylabel('Days Between Tests')
            plt.savefig(os.path.join(figures_dir, 'time_vs_decline_severity.png'), dpi=300)
            
            # Save longitudinal models
            with open(os.path.join(model_dir, 'rf_any_worse.pkl'), 'wb') as f:
                pickle.dump(rf_any, f)
            
            with open(os.path.join(model_dir, 'rf_sig_worse.pkl'), 'wb') as f:
                pickle.dump(rf_sig, f)
                
            # Compare with original models
            all_models_metrics = pd.concat([
                pd.DataFrame([result for result in results]),
                pd.DataFrame([any_worse_metrics, sig_worse_metrics])
            ])
            
            # Save comparison
            all_models_metrics.to_csv(os.path.join(tables_dir, 'all_models_comparison.csv'), index=False)
            
            # Plot comparison
            plt.figure(figsize=(12, 8))
            sns.barplot(x='F1', y='Model', data=all_models_metrics.sort_values('F1'))
            plt.title('Model Performance Comparison (F1 Score)')
            plt.xlabel('F1 Score')
            plt.xlim(0, 1.1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'all_models_f1_comparison.png'), dpi=300)
            
            # ROC curve comparison
            plt.figure(figsize=(10, 8))
            
            # Original model ROC
            if "RF_No_Balancing" in models:
                y_pred_proba_orig = models["RF_No_Balancing"].predict_proba(X_test)[:, 1]
                fpr_orig, tpr_orig, _ = roc_curve(y_test, y_pred_proba_orig)
                plt.plot(fpr_orig, tpr_orig, label=f'Original Model (AUC = {roc_auc_score(y_test, y_pred_proba_orig):.3f})')
            
            # Any worse ROC
            fpr_any, tpr_any, _ = roc_curve(y_test_any, y_pred_proba_any)
            plt.plot(fpr_any, tpr_any, label=f'Any Worse Model (AUC = {roc_auc_any:.3f})')
            
            # Significant worse ROC
            fpr_sig, tpr_sig, _ = roc_curve(y_test_sig, y_pred_proba_sig)
            plt.plot(fpr_sig, tpr_sig, label=f'Significant Worse Model (AUC = {roc_auc_sig:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(figures_dir, 'roc_comparison.png'), dpi=300)
            
        else:
            print("No longitudinal data pairs could be created.")
    else:
        print("No athletes with multiple tests found.")
else:
    print("Required columns for longitudinal analysis not found.")

# Clustering Analysis
print("\n--- Cluster Analysis ---")

# Standardize features for clustering
scaler = StandardScaler()
X_cluster = scaler.fit_transform(X)

# Determine optimal number of clusters using silhouette score
silhouette_scores = []
k_range = range(2, min(10, len(df) // 5 + 1))  # Ensure we don't try to create more clusters than practical

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    silhouette_avg = silhouette_score(X_cluster, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), silhouette_scores, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True)
plt.savefig(os.path.join(figures_dir, 'cluster_silhouette_scores.png'), dpi=300)

# Choose optimal number of clusters
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")

# Perform K-means clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)

# Add cluster labels to the dataframe
df['cluster'] = cluster_labels

# Calculate injury risk by cluster
cluster_risk = df.groupby('cluster')[target].agg(['mean', 'count']).reset_index()
cluster_risk = cluster_risk.rename(columns={'mean': 'injury_risk_rate'})
cluster_risk['injury_risk_rate'] = cluster_risk['injury_risk_rate'] * 100  # Convert to percentage
cluster_risk = cluster_risk.sort_values('injury_risk_rate', ascending=False)

# Save cluster risk to CSV
cluster_risk.to_csv(os.path.join(tables_dir, 'cluster_injury_risk.csv'), index=False)

# Calculate cluster centers and characteristics
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=X.columns  # Use X.columns instead of numeric_features to match dimensions
)

# Add cluster and risk rate to the centers dataframe
cluster_centers['cluster'] = range(optimal_k)
cluster_centers = cluster_centers.merge(cluster_risk[['cluster', 'injury_risk_rate']], on='cluster')

# Save cluster centers to CSV
cluster_centers.to_csv(os.path.join(tables_dir, 'cluster_centers.csv'), index=False)

# Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('PCA Visualization of Athlete Clusters')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(figures_dir, 'cluster_pca_visualization.png'), dpi=300)

# Machine Learning Models
print("\n--- Machine Learning Models ---")

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create models dictionary
models = {
    "RF_No_Balancing": RandomForestClassifier(n_estimators=100, random_state=42),
    "RF_with_SMOTE": None,  # Will be defined after SMOTE
    "RF_with_SMOTEENN": None  # Will be defined after SMOTEENN
}

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
models["RF_with_SMOTE"] = RandomForestClassifier(n_estimators=100, random_state=42)

# Apply SMOTEENN for combination of over and undersampling
smoteenn = SMOTEENN(random_state=42)
X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
models["RF_with_SMOTEENN"] = RandomForestClassifier(n_estimators=100, random_state=42)

# Train and evaluate models
results = []
feature_importances = {}
saved_models = {}

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining and evaluating {name}...")
    
    # Select appropriate training data
    if name == "RF_No_Balancing":
        X_train_current, y_train_current = X_train, y_train
    elif name == "RF_with_SMOTE":
        X_train_current, y_train_current = X_train_smote, y_train_smote
    elif name == "RF_with_SMOTEENN":
        X_train_current, y_train_current = X_train_smoteenn, y_train_smoteenn
    
    # Train the model
    model.fit(X_train_current, y_train_current)
    
    # Save model
    model_filename = f"{name.lower().replace(' ', '_')}.pkl"
    with open(os.path.join(model_dir, model_filename), 'wb') as f:
        pickle.dump(model, f)
    saved_models[name] = model_filename
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC AUC': roc_auc,
        'CV Mean ROC AUC': np.mean(cv_scores),
        'CV Std ROC AUC': np.std(cv_scores)
    })
    
    # Store feature importances
    feature_importances[name] = pd.DataFrame({
        'Feature': list(X.columns),  # Convert to list to ensure correct length matching
        'Importance': list(model.feature_importances_)  # Convert to list to ensure correct length matching
    }).sort_values('Importance', ascending=False)
    
    # Save feature importances to CSV
    feature_importances[name].to_csv(os.path.join(tables_dir, f'feature_importance_{name}.csv'), index=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances[name].head(10))
    plt.title(f'Top 10 Feature Importances - {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'feature_importance_{name}.png'), dpi=300)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{name}.png'), dpi=300)
    
    # SHAP Values for Model Interpretability
    if name == "RF_No_Balancing":  # Just do SHAP for one model to save time
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=numeric_features, show=False)
            plt.title(f'SHAP Summary Plot - {name}')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f'shap_summary_{name}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not generate SHAP plots due to error: {e}")

# Create results dataframe and save
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(tables_dir, 'model_comparison.csv'), index=False)

# Compare feature importances across models
all_importances = []
for name, imp_df in feature_importances.items():
    model_importances = imp_df.copy()
    model_importances['Model'] = name
    all_importances.append(model_importances)

if all_importances:
    combined_importances = pd.concat(all_importances)
    
    # Create pivot table for comparison
    importance_pivot = combined_importances.pivot_table(
        index='Feature',
        columns='Model',
        values='Importance'
    ).fillna(0)
    
    # Save to CSV
    importance_pivot.to_csv(os.path.join(tables_dir, 'feature_importance_comparison.csv'))
    
    # Plot comparison
    plt.figure(figsize=(12, 10))
    sns.heatmap(importance_pivot, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title('Feature Importance Comparison Across Models')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance_comparison.png'), dpi=300)

print("\nAnalysis complete. Results saved to tables and figures directories.")

# Create a summary report
with open(os.path.join(tables_dir, 'analysis_summary.md'), 'w') as f:
    f.write("# Athlete Injury Risk Analysis Summary\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
    
    f.write("## Dataset Overview\n\n")
    f.write(f"- Total samples: {len(df)}\n")
    f.write(f"- High-risk samples: {df[target].sum()} ({df[target].mean():.1%})\n")
    
    if 'sbuid' in df.columns:
        f.write(f"- Unique athletes: {df['sbuid'].nunique()}\n")
    
    if 'testDateUtc' in df.columns:
        f.write(f"- Date range: {df['testDateUtc'].min().strftime('%Y-%m-%d')} to {df['testDateUtc'].max().strftime('%Y-%m-%d')}\n")
    
    f.write("\n## Key Findings\n\n")
    
    # Top risk factors
    if len(effect_size_df) > 0:
        f.write("### Top Risk Factors\n\n")
        significant_factors = effect_size_df[effect_size_df['p_value'] < 0.05].head(5)
        if len(significant_factors) > 0:
            f.write(significant_factors[['feature', 'cohens_d', 'p_value']].to_markdown(index=False))
        else:
            f.write("No statistically significant risk factors identified.\n")
    
    # Cluster insights
    f.write("\n### Athlete Clusters\n\n")
    f.write(f"- Optimal number of clusters: {optimal_k}\n")

# Cross-Sport Analysis
print("\n--- Cross-Sport Analysis ---")

if 'sport' in df.columns:
    # Get unique sports
    sports = df['sport'].unique().tolist()
    print(f"Found {len(sports)} unique sports in the dataset")
    
    # Create a directory for sport-specific figures if it doesn't exist
    sports_figures_dir = os.path.join(figures_dir, 'sports')
    os.makedirs(sports_figures_dir, exist_ok=True)
    
    # Get sport sample sizes
    sport_counts = df['sport'].value_counts()
    print("Sample sizes by sport:")
    for sport, count in sport_counts.items():
        print(f"  {sport}: {count}")
    
    # Plot sport sample sizes
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sport_counts.index, y=sport_counts.values)
    plt.title('Sample Size by Sport')
    plt.xlabel('Sport')
    plt.ylabel('Number of Athletes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(sports_figures_dir, 'sport_sample_sizes.png'), dpi=300)
    
    # Compare injury rates across sports using our target variable 'injury_risk_high' instead of 'injured'
    injury_by_sport = df.groupby('sport')[target].mean().sort_values(ascending=False)
    
    # Plot injury rates by sport
    plt.figure(figsize=(12, 6))
    sns.barplot(x=injury_by_sport.index, y=injury_by_sport.values)
    plt.title('Injury Risk Rate by Sport')
    plt.xlabel('Sport')
    plt.ylabel('Injury Risk Rate')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(sports_figures_dir, 'injury_rate_by_sport.png'), dpi=300)
    
    # Create a table with injury rates by sport
    injury_rate_table = pd.DataFrame({
        'Sport': injury_by_sport.index,
        'Injury_Rate': injury_by_sport.values,
        'Sample_Size': [sport_counts[sport] for sport in injury_by_sport.index]
    })
    injury_rate_table['Injury_Rate'] = injury_rate_table['Injury_Rate'] * 100  # Convert to percentage
    injury_rate_table = injury_rate_table.round({'Injury_Rate': 1})
    injury_rate_table.columns = ['Sport', 'Injury Risk Rate (%)', 'Sample Size']
    
    # Save injury rate table
    injury_rate_table.to_csv(os.path.join(tables_dir, 'injury_rate_by_sport.csv'), index=False)
    
    # Compare symmetry metrics across sports
    symmetry_metrics = ['ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']
    
    for metric in symmetry_metrics:
        if metric in df.columns:
            # Calculate absolute asymmetry (distance from 1.0)
            df[f'{metric}_Asymmetry'] = abs(df[metric] - 1.0)
            
            # Compare asymmetry across sports
            asymmetry_by_sport = df.groupby('sport')[f'{metric}_Asymmetry'].mean().sort_values(ascending=False)
            
            # Plot asymmetry by sport
            plt.figure(figsize=(12, 6))
            sns.barplot(x=asymmetry_by_sport.index, y=asymmetry_by_sport.values)
            plt.title(f'{metric} Asymmetry by Sport')
            plt.xlabel('Sport')
            plt.ylabel('Mean Absolute Asymmetry')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(sports_figures_dir, f'{metric.lower()}_asymmetry_by_sport.png'), dpi=300)
    
    # Compare asymmetry between high-risk and low-risk athletes across sports
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(symmetry_metrics):
        if metric in df.columns:
            plt.subplot(2, 2, i+1)
            
            # Box plot comparing asymmetry between high-risk and low-risk athletes across sports
            sns.boxplot(x='sport', y=f'{metric}_Asymmetry', hue=target, data=df)
            plt.title(f'{metric} Asymmetry by Sport and Risk Status')
            plt.xlabel('Sport')
            plt.ylabel('Absolute Asymmetry')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='High Risk')
    
    plt.tight_layout()
    plt.savefig(os.path.join(sports_figures_dir, 'asymmetry_by_sport_and_injury.png'), dpi=300)
    
    # Train sport-specific models
    min_samples = 50  # Minimum sample size to train a sport-specific model
    sport_models = {}
    sport_results = []
    
    for sport in sports:
        sport_df = df[df['sport'] == sport]
        
        if len(sport_df) >= min_samples:
            print(f"\nTraining model for {sport} (n={len(sport_df)})")
            
            # Prepare data
            X_sport = sport_df[improved_features]
            y_sport = sport_df[target]  # Use target variable instead of 'injured'
            
            # Create train/test split
            X_train_sport, X_test_sport, y_train_sport, y_test_sport = train_test_split(
                X_sport, y_sport, test_size=0.2, random_state=42, stratify=y_sport
            )
            
            # Apply SMOTE for balance
            try:
                smote = SMOTE(random_state=42)
                X_train_sport_smote, y_train_sport_smote = smote.fit_resample(X_train_sport, y_train_sport)
                
                # Train RF model
                rf_sport = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_sport.fit(X_train_sport_smote, y_train_sport_smote)
                
                # Evaluate
                y_pred_sport = rf_sport.predict(X_test_sport)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_sport, y_pred_sport)
                precision = precision_score(y_test_sport, y_pred_sport, zero_division=0)
                recall = recall_score(y_test_sport, y_pred_sport)
                f1 = f1_score(y_test_sport, y_pred_sport)
                
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1 Score: {f1:.3f}")
                
                # Save results
                sport_result = {
                    'Sport': sport,
                    'Model': f'RF_{sport.replace(" ", "_")}',
                    'Sample_Size': len(sport_df),
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1
                }
                sport_results.append(sport_result)
                
                # Save model
                sport_models[sport] = rf_sport
                
                # Feature importance
                sport_importance = pd.DataFrame({
                    'Feature': improved_features,
                    'Importance': rf_sport.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                plt.figure(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=sport_importance.head(10))
                plt.title(f'Top 10 Feature Importances - {sport}')
                plt.tight_layout()
                plt.savefig(os.path.join(sports_figures_dir, f'{sport.replace(" ", "_")}_feature_importance.png'), dpi=300)
                
                # Save feature importance
                sport_importance.to_csv(os.path.join(tables_dir, f'{sport.replace(" ", "_")}_feature_importance.csv'), index=False)
                
            except Exception as e:
                print(f"  Error training model for {sport}: {str(e)}")
        else:
            print(f"Insufficient samples for {sport} (n={len(sport_df)}), skipping model training")
    
    # Create table of sport-specific results
    if sport_results:
        sport_results_df = pd.DataFrame(sport_results)
        sport_results_df = sport_results_df.sort_values('F1', ascending=False)
        
        # Save sport results
        sport_results_df.to_csv(os.path.join(tables_dir, 'sport_specific_model_results.csv'), index=False)
        
        # Plot sport model comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(x='F1', y='Sport', data=sport_results_df)
        plt.title('Sport-Specific Model Performance (F1 Score)')
        plt.xlabel('F1 Score')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(sports_figures_dir, 'sport_model_comparison.png'), dpi=300)
        
        # Compare sport-specific vs. general model
        # Extract top features from each sport model to see commonalities and differences
        top_features_by_sport = {}
        
        for sport in sport_models:
            importance = pd.DataFrame({
                'Feature': improved_features,
                'Importance': sport_models[sport].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features_by_sport[sport] = importance.head(5)['Feature'].tolist()
        
        # Create heatmap of top feature overlap
        if len(top_features_by_sport) > 1:
            # Get all unique top features across sports
            all_top_features = set()
            for sport_features in top_features_by_sport.values():
                all_top_features.update(sport_features)
            
            # Create matrix of feature importance ranks
            feature_rank_matrix = pd.DataFrame(0, index=list(top_features_by_sport.keys()), columns=list(all_top_features))
            
            for sport, features in top_features_by_sport.items():
                for i, feature in enumerate(features):
                    feature_rank_matrix.loc[sport, feature] = 5 - i  # Higher rank for more important features
            
            # Plot heatmap
            plt.figure(figsize=(max(12, len(all_top_features) * 0.8), max(8, len(top_features_by_sport) * 0.8)))
            sns.heatmap(feature_rank_matrix, cmap='YlOrRd', linewidths=0.5, annot=True, fmt='.0f')
            plt.title('Top Feature Importance Ranking Across Sports')
            plt.tight_layout()
            plt.savefig(os.path.join(sports_figures_dir, 'cross_sport_feature_importance.png'), dpi=300)
    
else:
    print("No 'sport' column found in the dataset. Skipping cross-sport analysis.")