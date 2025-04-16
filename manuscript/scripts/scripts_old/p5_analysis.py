# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys
import joblib  # Added missing import

# Add proper path handling for data files
# Get the absolute path of the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# Set the data path
data_dir = os.path.join(root_dir, 'data')
# Set paths for saving figures and tables
figures_dir = os.path.join(os.path.dirname(__file__), '../figures')
tables_dir = os.path.join(os.path.dirname(__file__), '../tables')
# Create directories if they don't exist
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
# Change the working directory to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Function to save figures with standardized formatting
def save_figure(fig, filename, dpi=300):
    fig.savefig(os.path.join(figures_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
# Function to save tables as CSV files
def save_table(df, filename):
    df.to_csv(os.path.join(tables_dir, filename), index=True)
    
# %%
vald_data = pd.read_csv(os.path.join(data_dir, 'model_data.csv'))

# %%
# Display basic information and first few rows for review
vald_data_info = vald_data.info()
vald_data_head = vald_data.head()

vald_data_info, vald_data_head

# %%
# Define the provided thresholds for symmetry metrics
symmetry_thresholds = {
    'ForceSymmetry': (250 / 310, 300 / 260),  # Based on AvgForce thresholds
    'ImpulseSymmetry': (150 / 260, 250 / 160),  # Based on Impulse thresholds
    'MaxForceSymmetry': (270 / 330, 320 / 280),  # Based on MaxForce thresholds
    'TorqueSymmetry': (250 / 310, 300 / 260)  # Based on Torque thresholds
}

# %%
def calculate_symmetry_risk(data, thresholds, metric_stds=None, buffer_factor=1.0):
    for metric, (min_val, max_val) in thresholds.items():
        # Use dynamic buffer based on metric standard deviation
        if metric_stds:
            buffer = metric_stds[metric] * buffer_factor
        else:
            buffer = (max_val - min_val) * 0.1  # Default 10% buffer

        data[f'{metric}Risk'] = data[metric].apply(
            lambda x: 'Low Risk' if min_val <= x <= max_val
            else 'Medium Risk' if (min_val - buffer <= x < min_val) or (max_val < x <= max_val + buffer)
            else 'High Risk'
        )
    return data


# %%

# Calculate standard deviations for symmetry metrics
symmetry_metrics = ['ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']
metric_stds = vald_data[symmetry_metrics].std().to_dict()

# Apply the updated logic with dynamic buffers based on standard deviations
def calculate_symmetry_risk_dynamic(data, thresholds, metric_stds=None, buffer_factor=1.0):
    for metric, (min_val, max_val) in thresholds.items():
        # Use dynamic buffer based on metric standard deviation
        buffer = metric_stds[metric] * buffer_factor if metric_stds else (max_val - min_val) * 0.1
        data[f'{metric}Risk'] = data[metric].apply(
            lambda x: 'Low Risk' if min_val <= x <= max_val
            else 'Medium Risk' if (min_val - buffer <= x < min_val) or (max_val < x <= max_val + buffer)
            else 'High Risk'
        )
    return data

# Reapply the logic
vald_data = calculate_symmetry_risk_dynamic(vald_data, symmetry_thresholds, metric_stds, buffer_factor=1.0)

# Validate the distribution of athletes across risk categories for each symmetry metric
risk_distributions = {}
for metric in symmetry_metrics:
    risk_distributions[metric] = vald_data[f'{metric}Risk'].value_counts()

# Convert distributions into a DataFrame for better readability
risk_distribution_df = pd.DataFrame(risk_distributions).transpose()

# Save risk distribution table
save_table(risk_distribution_df, 'risk_distribution.csv')

# Visualize risk distribution for each symmetry metric
plt.figure(figsize=(20, 12))
for i, metric in enumerate(symmetry_metrics):
    plt.subplot(2, 2, i + 1)
    sns.countplot(data=vald_data, x=f'{metric}Risk', order=['Low Risk', 'Medium Risk', 'High Risk'])
    plt.title(f'Risk Distribution for {metric}')
    plt.xlabel('Risk Category')
    plt.ylabel('Count')

plt.tight_layout()
# Save the figure
save_figure(plt.gcf(), 'risk_category_distribution.png')
plt.close()

# Save risk distribution table
save_table(risk_distribution_df, 'risk_distribution.csv')

# %% [markdown]
# Step 1: Validate Distribution
# What It Does:
# 
# Analyzes the proportions of athletes in each risk category (Low Risk, Medium Risk, High Risk) for each symmetry metric.
# How It Helps:
# 
# Ensures that the thresholds and buffer logic produce meaningful distributions.
# Identifies potential issues, such as:
# Overrepresentation of a specific category (e.g., too many in Low Risk).
# Underrepresentation of Medium Risk or High Risk, which could indicate overly strict thresholds or inappropriate buffer sizes.
# Allows refinement of thresholds and buffers to align with domain expectations.
# Outcome:
# 
# A clear understanding of the distribution of athletes across risk levels for each metric.
# Validation or refinement of the risk categorization logic.

# %% [markdown]
# Step 2: Statistical Validation (Hypothesis Testing)
# What It Does:
# 
# Tests whether symmetry metrics differ significantly between risk categories (Low Risk, Medium Risk, High Risk).
# How It Helps:
# 
# Adds statistical rigor to support your thresholds:
# For example, does ForceSymmetry significantly differ across risk levels? If not, the thresholds may not align with meaningful separation.
# Validates that the risk categories represent distinct groups based on the symmetry metrics.
# Identifies metrics that are most relevant for risk categorization (e.g., metrics with significant differences across groups).
# Outcome:
# 
# Evidence-backed conclusions about whether the symmetry metrics justify the defined risk categories.
# Insights into which metrics are most impactful for risk prediction.

# %% [markdown]
# Step 3: Refine Thresholds (if needed)
# What It Does:
# 
# Adjusts thresholds and buffer logic based on insights from distribution analysis and hypothesis testing.
# How It Helps:
# 
# Ensures thresholds align with the data distribution and statistical significance.
# Allows domain knowledge to guide adjustments (e.g., symmetry ratios closer to 1.0 may require stricter thresholds).
# Outcome:
# 
# Improved thresholds that create meaningful and balanced risk distributions.
# Better alignment between risk categories and the underlying data.

# %%
# Validate the distribution of athletes across risk categories for each symmetry metric
risk_distributions = {}
for metric in symmetry_metrics:
    risk_distributions[metric] = vald_data[f'{metric}Risk'].value_counts()

# Convert distributions into a DataFrame for better readability
risk_distribution_df = pd.DataFrame(risk_distributions).transpose()

# Display the distribution 

risk_distribution_df


# %%
# Visualize the distribution of risk categories for each symmetry metric
plt.figure(figsize=(16, 8))
for i, metric in enumerate(symmetry_metrics):
    plt.subplot(2, 2, i + 1)
    sns.barplot(
        x=risk_distribution_df.columns,
        y=risk_distribution_df.loc[metric],
        palette="viridis"
    )
    plt.title(f'Risk Distribution for {metric}')
    plt.xlabel('Risk Category')
    plt.ylabel('Count')

plt.tight_layout()
# Save the figure
save_figure(plt.gcf(), 'risk_distribution_barplot.png')
plt.close()

# %%
# correlation between symmetry metrics and risk categories
plt.figure(figsize=(10, 8))
vald_data_corr = vald_data[symmetry_metrics].corr()
sns.heatmap(vald_data_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Symmetry Metrics')
plt.tight_layout()
# Save the correlation heatmap
save_figure(plt.gcf(), 'correlation_heatmap.png')
plt.close()

# Save table of correlation values
save_table(vald_data_corr, 'symmetry_correlations.csv')

# %%
# Calculate descriptive statistics for Medium and High Risk athletes for each symmetry metric
medium_high_risk_stats = {}

for metric in ['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']:
    medium_risk = vald_data[vald_data[f'{metric}Risk'] == 'Medium Risk'][metric]
    high_risk = vald_data[vald_data[f'{metric}Risk'] == 'High Risk'][metric]
    
    medium_high_risk_stats[metric] = {
        'Medium Risk Mean': medium_risk.mean(),
        'Medium Risk Std Dev': medium_risk.std(),
        'High Risk Mean': high_risk.mean(),
        'High Risk Std Dev': high_risk.std(),
        'Medium Risk Count': len(medium_risk),
        'High Risk Count': len(high_risk)
    }

# Convert the statistics to a DataFrame for better readability
medium_high_risk_stats_df = pd.DataFrame(medium_high_risk_stats).transpose()

# Display the statistics
medium_high_risk_stats_df

# Save the statistics DataFrame as a table
save_table(medium_high_risk_stats_df, 'risk_statistics.csv')

# %%
# Visualize the separation between risk categories for each symmetry metric
plt.figure(figsize=(18, 6))
for i, metric in enumerate(['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']):
    plt.subplot(1, 3, i + 1)
    # Plot all three risk categories
    sns.kdeplot(vald_data[vald_data[f'{metric}Risk'] == 'Low Risk'][metric], 
               label='Low Risk', color='green', fill=True, alpha=0.3)
    sns.kdeplot(vald_data[vald_data[f'{metric}Risk'] == 'Medium Risk'][metric], 
               label='Medium Risk', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(vald_data[vald_data[f'{metric}Risk'] == 'High Risk'][metric], 
               label='High Risk', color='red', fill=True, alpha=0.3)
    plt.title(f'{metric} Distribution by Risk Category')
    plt.xlabel(metric)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
# Save the density distributions figure
save_figure(plt.gcf(), 'risk_density_distributions.png')
plt.close()

# %%
# Create RiskCategory based on the Highest Risk Approach
def assign_highest_risk(row):
    risks = [row['ForceSymmetryRisk'], row['MaxForceSymmetryRisk'], row['TorqueSymmetryRisk']]
    if 'High Risk' in risks:
        return 'High Risk'
    elif 'Medium Risk' in risks:
        return 'Medium Risk'
    else:
        return 'Low Risk'

vald_data['RiskCategory'] = vald_data.apply(assign_highest_risk, axis=1)

# Check the distribution of the newly created RiskCategory
risk_category_distribution = vald_data['RiskCategory'].value_counts()

# Display the distribution
risk_category_distribution

# Save risk category distribution as a table
risk_category_df = pd.DataFrame(risk_category_distribution).reset_index()
risk_category_df.columns = ['Risk Category', 'Count']
risk_category_df['Percentage'] = (risk_category_df['Count'] / risk_category_df['Count'].sum()) * 100
save_table(risk_category_df, 'overall_risk_distribution.csv')

# Visualize overall risk distribution
plt.figure(figsize=(10, 6))
plt.pie(risk_category_df['Count'], labels=risk_category_df['Risk Category'], 
        autopct='%1.1f%%', colors=['green', 'orange', 'red'])
plt.title('Overall Risk Category Distribution')
save_figure(plt.gcf(), 'overall_risk_pie.png')
plt.close()

# %%
vald_data

# %%
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare results storage
models_results = {}

# Step 1: Prepare the data (X and y from vald_data)
risk_mapping = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
X = vald_data[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']]
y = vald_data['RiskCategory'].map(risk_mapping)

# MODEL 1: SMOTE Oversampling
smote = SMOTE(random_state=42)
X_resampled_1, y_resampled_1 = smote.fit_resample(X, y)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_resampled_1, y_resampled_1, test_size=0.2, stratify=y_resampled_1, random_state=42)
rf_model_1 = RandomForestClassifier(random_state=42)
rf_model_1.fit(X_train_1, y_train_1)
y_pred_1 = rf_model_1.predict(X_test_1)
models_results['Model 1 (SMOTE)'] = {
    'accuracy': accuracy_score(y_test_1, y_pred_1),
    'classification_report': classification_report(y_test_1, y_pred_1, target_names=risk_mapping.keys()),
    'confusion_matrix': confusion_matrix(y_test_1, y_pred_1),
}

# MODEL 2: No Balancing
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rf_model_2 = RandomForestClassifier(random_state=42)
rf_model_2.fit(X_train_2, y_train_2)
y_pred_2 = rf_model_2.predict(X_test_2)
models_results['Model 2 (No Balancing)'] = {
    'accuracy': accuracy_score(y_test_2, y_pred_2),
    'classification_report': classification_report(y_test_2, y_pred_2, target_names=risk_mapping.keys()),
    'confusion_matrix': confusion_matrix(y_test_2, y_pred_2),
}

# MODEL 3: SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_resampled_3, y_resampled_3 = smoteenn.fit_resample(X, y)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_resampled_3, y_resampled_3, test_size=0.2, stratify=y_resampled_3, random_state=42)
rf_model_3 = RandomForestClassifier(random_state=42)
rf_model_3.fit(X_train_3, y_train_3)
y_pred_3 = rf_model_3.predict(X_test_3)
models_results['Model 3 (SMOTEENN)'] = {
    'accuracy': accuracy_score(y_test_3, y_pred_3),
    'classification_report': classification_report(y_test_3, y_pred_3, target_names=risk_mapping.keys()),
    'confusion_matrix': confusion_matrix(y_test_3, y_pred_3),
}

# MODEL 4: Class Weights
rf_model_4 = RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 2, 2: 3})
rf_model_4.fit(X_train_2, y_train_2)  # Use unbalanced data
y_pred_4 = rf_model_4.predict(X_test_2)
models_results['Model 4 (Class Weights)'] = {
    'accuracy': accuracy_score(y_test_2, y_pred_4),
    'classification_report': classification_report(y_test_2, y_pred_4, target_names=risk_mapping.keys()),
    'confusion_matrix': confusion_matrix(y_test_2, y_pred_4),
}


# %%
# Print model comparisons
for model_name, results in models_results.items():
    print(f"\n{model_name}")
    print(f"Accuracy: {results['accuracy']}")
    print("Classification Report:")
    print(results['classification_report'])
    print("Confusion Matrix:")
    print(results['confusion_matrix'])


# %%
# Fix the path for saving models
joblib.dump(rf_model_1, os.path.join(root_dir, "rf_model_smote.pkl"))
joblib.dump(rf_model_2, os.path.join(root_dir, "rf_model_no_balancing.pkl"))
joblib.dump(rf_model_3, os.path.join(root_dir, "rf_model_smoteenn.pkl"))
joblib.dump(rf_model_4, os.path.join(root_dir, "rf_model_class_weights.pkl"))

print("Models saved successfully!")


# %% [markdown]
# TESTING ON UNSEEN DATA

# %%
# Fix the path for loading unseen data
unseen_data = pd.read_csv(os.path.join(data_dir, 'unseen_data.csv'))

# Calculate symmetry metrics
unseen_data['ForceSymmetry'] = unseen_data['leftAvgForce'] / unseen_data['rightAvgForce']
unseen_data['ImpulseSymmetry'] = unseen_data['leftImpulse'] / unseen_data['rightImpulse']
unseen_data['MaxForceSymmetry'] = unseen_data['leftMaxForce'] / unseen_data['rightMaxForce']
unseen_data['TorqueSymmetry'] = unseen_data['leftTorque'] / unseen_data['rightTorque']

# Prepare features
X_unseen = unseen_data[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']]

# Predict with all models
unseen_predictions = {}
for model_name, model in zip(
    ['Model 1 (SMOTE)', 'Model 2 (No Balancing)', 'Model 3 (SMOTEENN)', 'Model 4 (Class Weights)'],
    [rf_model_1, rf_model_2, rf_model_3, rf_model_4]
):
    unseen_predictions[model_name] = model.predict(X_unseen)

# Add predictions to unseen data
for model_name in unseen_predictions:
    unseen_data[model_name] = unseen_predictions[model_name]

# Map predictions back to RiskCategory labels
risk_mapping_inverse = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
for model_name in unseen_predictions:
    unseen_data[model_name] = unseen_data[model_name].map(risk_mapping_inverse)

# Display the unseen data with predictions
print(unseen_data)

# %% [markdown]
# ## Statistical Validation of Risk Categories
# Performing formal statistical tests to validate that our risk categories represent significantly different groups

# %%
# Import necessary statistical libraries
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# %% [markdown]
# ### ANOVA Analysis for Risk Categories
# Testing whether the symmetry metrics differ significantly across risk categories

# %%
# Function to perform ANOVA and post-hoc tests for a given metric
def perform_anova_analysis(data, metric, risk_category):
    # Create a DataFrame for ANOVA
    anova_df = data[[metric, f'{metric}Risk']].copy()
    
    # Perform one-way ANOVA
    formula = f'{metric} ~ C({metric}Risk)'
    model = ols(formula, data=anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Perform Tukey's HSD post-hoc test
    comp = sm.stats.multicomp.MultiComparison(anova_df[metric], anova_df[f'{metric}Risk'])
    post_hoc = comp.tukeyhsd()
    
    return {
        'anova_table': anova_table,
        'post_hoc': post_hoc
    }

# Perform ANOVA for each symmetry metric
anova_results = {}
for metric in ['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']:
    anova_results[metric] = perform_anova_analysis(vald_data, metric, f'{metric}Risk')
    
    print(f"\nANOVA Results for {metric}:")
    print(anova_results[metric]['anova_table'])
    print("\nTukey's HSD Post-hoc Results:")
    print(anova_results[metric]['post_hoc'])

# %% [markdown]
# ### Kruskal-Wallis Test for ImpulseSymmetry
# Using non-parametric test for ImpulseSymmetry in case of non-normal distribution

# %%
# Perform Kruskal-Wallis test for ImpulseSymmetry
# Checking the groups first before running the test
impulse_groups = [
    vald_data[vald_data['ImpulseSymmetryRisk'] == 'Low Risk']['ImpulseSymmetry'],
    vald_data[vald_data['ImpulseSymmetryRisk'] == 'Medium Risk']['ImpulseSymmetry'],
    vald_data[vald_data['ImpulseSymmetryRisk'] == 'High Risk']['ImpulseSymmetry']
]

# Print the sizes of each group
print("\nGroup sizes for ImpulseSymmetry:")
for i, group_name in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
    print(f"{group_name}: {len(impulse_groups[i])}")

# Filter out empty groups
non_empty_groups = [group for group in impulse_groups if len(group) > 0]

# Check if we have at least two groups to perform the test
if len(non_empty_groups) >= 2:
    kw_stat, kw_p = stats.kruskal(*non_empty_groups)
    print(f"Kruskal-Wallis test for ImpulseSymmetry: statistic={kw_stat}, p-value={kw_p}")
else:
    print(f"Cannot perform Kruskal-Wallis test: only {len(non_empty_groups)} non-empty groups available (need at least 2)")

# %% [markdown]
# ### Effect Size Analysis for Risk Categories
# Calculating effect sizes to quantify the magnitude of differences between risk categories

# %%
from scipy.stats import f_oneway

# Calculate effect sizes using Cohen's d
def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return None
    
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(), group2.var()
    
    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = abs(mean1 - mean2) / pooled_sd
    return d

# Calculate effect sizes for each metric between risk categories
effect_sizes = {}
for metric in ['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']:
    low_risk = vald_data[vald_data[f'{metric}Risk'] == 'Low Risk'][metric]
    medium_risk = vald_data[vald_data[f'{metric}Risk'] == 'Medium Risk'][metric]
    high_risk = vald_data[vald_data[f'{metric}Risk'] == 'High Risk'][metric]
    
    effect_sizes[metric] = {
        'Low-Medium': cohen_d(low_risk, medium_risk),
        'Medium-High': cohen_d(medium_risk, high_risk),
        'Low-High': cohen_d(low_risk, high_risk)
    }

# Display effect sizes
effect_sizes_df = pd.DataFrame(effect_sizes)
print("Effect Sizes (Cohen's d) between Risk Categories:")
print(effect_sizes_df)

# %%
# Visualize effect sizes
plt.figure(figsize=(12, 6))
effect_sizes_df_melted = pd.melt(effect_sizes_df.reset_index(), 
                                 id_vars=['index'], 
                                 var_name='Metric', 
                                 value_name='Effect Size')

# Create a heatmap of effect sizes
plt.figure(figsize=(10, 6))
heatmap_data = effect_sizes_df.copy()
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Effect Sizes (Cohen\'s d) Between Risk Categories')
plt.xlabel('Symmetry Metrics')
plt.ylabel('Risk Category Comparisons')
plt.tight_layout()
# Save the effect size heatmap
save_figure(plt.gcf(), 'effect_size_heatmap.png')
plt.close()

# Save effect sizes table
save_table(effect_sizes_df, 'effect_sizes.csv')

# %% [markdown]
# ## Feature Importance Analysis
# Analyzing the relative importance of different symmetry metrics in risk prediction

# %%
# Permutation feature importance
from sklearn.inspection import permutation_importance

# Function to calculate permutation importance for each model
def calculate_feature_importance(model, X_test, y_test, n_repeats=10):
    r = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats, 
        random_state=42
    )
    
    # Create a DataFrame with feature importances
    feature_importance = pd.DataFrame(
        {'Feature': X_test.columns,
         'Importance': r.importances_mean,
         'Std': r.importances_std}
    ).sort_values('Importance', ascending=False)
    
    return feature_importance

# Calculate feature importance for all models
feature_importances = {}
for model_name, model, X_test, y_test in [
    ('Model 1 (SMOTE)', rf_model_1, X_test_1, y_test_1),
    ('Model 2 (No Balancing)', rf_model_2, X_test_2, y_test_2),
    ('Model 3 (SMOTEENN)', rf_model_3, X_test_3, y_test_3),
    ('Model 4 (Class Weights)', rf_model_4, X_test_2, y_test_2)
]:
    feature_importances[model_name] = calculate_feature_importance(model, X_test, y_test)

# Display feature importances for each model
for model_name, importance_df in feature_importances.items():
    print(f"\nFeature Importance for {model_name}:")
    print(importance_df)

# %% [markdown]
# ### Visualizing Feature Importance Across Models

# %%
# Create a combined visualization of feature importance across models
plt.figure(figsize=(14, 8))

# Set up positions for grouped bar chart
bar_width = 0.2
positions = np.arange(len(X.columns))

# Plot feature importances for each model
for i, (model_name, importance_df) in enumerate(feature_importances.items()):
    plt.bar(
        positions + i * bar_width, 
        importance_df['Importance'],
        width=bar_width,
        label=model_name,
        yerr=importance_df['Std'],
        capsize=5
    )

plt.xlabel('Features')
plt.ylabel('Permutation Importance')
plt.title('Feature Importance Comparison Across Models')
plt.xticks(positions + bar_width * 1.5, X.columns)
plt.legend()
plt.tight_layout()
# Save the feature importance comparison
save_figure(plt.gcf(), 'feature_importance_comparison.png')
plt.close()

# Save combined feature importance table
combined_importance = pd.DataFrame()
for model_name, importance_df in feature_importances.items():
    for feature in importance_df['Feature']:
        row = importance_df[importance_df['Feature'] == feature]
        combined_importance.loc[feature, f"{model_name}_Importance"] = row['Importance'].values[0]
        combined_importance.loc[feature, f"{model_name}_Std"] = row['Std'].values[0]
save_table(combined_importance, 'feature_importance_comparison.csv')

# %% [markdown]
# ### Direct Feature Importance from Random Forest

# %%
# Extract feature importance directly from Random Forest models
for model_name, model in [
    ('Model 1 (SMOTE)', rf_model_1),
    ('Model 2 (No Balancing)', rf_model_2),
    ('Model 3 (SMOTEENN)', rf_model_3),
    ('Model 4 (Class Weights)', rf_model_4)
]:
    # Get feature importance from model
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    # Create DataFrame
    forest_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances,
        'Std': std
    }).sort_values('Importance', ascending=False)
    
    print(f"\nRandom Forest Feature Importance for {model_name}:")
    print(forest_importances)
    
    # Save the feature importance table
    save_table(forest_importances, f'feature_importance_{model_name.split()[0]}{model_name.split()[1]}.csv')
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(forest_importances['Feature'], forest_importances['Importance'], yerr=forest_importances['Std'], capsize=5)
    plt.xlabel('Features')
    plt.ylabel('Mean Decrease in Impurity (MDI)')
    plt.title(f'Feature Importance for {model_name}')
    plt.tight_layout()
    
    # Save the figure - special handling for Model 3 (our best model)
    if model_name == 'Model 3 (SMOTEENN)':
        save_figure(plt.gcf(), 'model3_feature_importance.png')
    else:
        save_figure(plt.gcf(), f'feature_importance_{model_name.split()[0]}{model_name.split()[1]}.png')
    plt.close()

# %% [markdown]
# ## Cross-Validation Performance Analysis
# Implementing k-fold cross-validation to assess model stability and generalization

# %%
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to perform cross-validation and display results
def cross_validate_model(model_name, model, X, y, n_splits=5):
    # Set up stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores for multiple metrics
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    precision_macro = cross_val_score(model, X, y, cv=cv, scoring='precision_macro')
    recall_macro = cross_val_score(model, X, y, cv=cv, scoring='recall_macro')
    f1_macro = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    
    # Combine results
    cv_results = {
        'Accuracy': accuracy,
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1 Score': f1_macro
    }
    
    # Display results
    print(f"\nCross-Validation Results for {model_name}:")
    for metric, scores in cv_results.items():
        print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return cv_results

# Perform cross-validation for all models
cv_results = {}
for model_name, model, X_data, y_data in [
    ('Model 1 (SMOTE)', rf_model_1, X_resampled_1, y_resampled_1),
    ('Model 2 (No Balancing)', rf_model_2, X, y),
    ('Model 3 (SMOTEENN)', rf_model_3, X_resampled_3, y_resampled_3),
    ('Model 4 (Class Weights)', rf_model_4, X, y)
]:
    cv_results[model_name] = cross_validate_model(model_name, model, X_data, y_data)

# %%
# Create a comparative visualization of cross-validation results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
model_names = list(cv_results.keys())

plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    
    # Extract mean and std for the current metric across all models
    means = [cv_results[model][metric].mean() for model in model_names]
    stds = [cv_results[model][metric].std() for model in model_names]
    
    # Create bar chart
    plt.bar(range(len(model_names)), means, yerr=stds, capsize=10)
    plt.title(f'Cross-Validation {metric}')
    plt.ylabel(metric)
    plt.xticks(range(len(model_names)), [m.split(' ')[0] + ' ' + m.split(' ')[1] for m in model_names], rotation=45)
    plt.ylim([0, 1.1])
    
    # Add value labels on top of each bar
    for j, v in enumerate(means):
        plt.text(j, v + stds[j] + 0.02, f"{v:.3f}", ha='center')

plt.tight_layout()
# Save the cross-validation performance figure
save_figure(plt.gcf(), 'cross_validation_performance.png')
plt.close()

# Save cross-validation results as a table
cv_summary = pd.DataFrame()
for model_name in model_names:
    for metric in metrics:
        cv_summary.loc[model_name, f"{metric}_Mean"] = cv_results[model_name][metric].mean()
        cv_summary.loc[model_name, f"{metric}_Std"] = cv_results[model_name][metric].std()
save_table(cv_summary, 'cross_validation_results.csv')

# %% [markdown]
# ### Visualizing Cross-Validation Performance

# %%
# Create a comparative visualization of cross-validation results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
model_names = list(cv_results.keys())

plt.figure(figsize=(14, 10))
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    
    # Extract mean and std for the current metric across all models
    means = [cv_results[model][metric].mean() for model in model_names]
    stds = [cv_results[model][metric].std() for model in model_names]
    
    # Create bar chart
    plt.bar(range(len(model_names)), means, yerr=stds, capsize=10)
    plt.title(f'Cross-Validation {metric}')
    plt.ylabel(metric)
    plt.xticks(range(len(model_names)), [m.split(' ')[0] + ' ' + m.split(' ')[1] for m in model_names], rotation=45)
    plt.ylim([0, 1.1])
    
    # Add value labels on top of each bar
    for j, v in enumerate(means):
        plt.text(j, v + stds[j] + 0.02, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comprehensive Model Comparison and ROC Analysis
# Comparing models using Receiver Operating Characteristic (ROC) curves and AUC scores

# %%
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Function to calculate ROC curve and AUC for multiclass classification
def calculate_roc_auc_multiclass(model, X_test, y_test):
    # Binarize the labels for multiclass ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # Predict probabilities
    y_score = model.predict_proba(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return fpr, tpr, roc_auc

# Calculate ROC curves for all models
roc_results = {}
for model_name, model, X_test, y_test in [
    ('Model 1 (SMOTE)', rf_model_1, X_test_1, y_test_1),
    ('Model 2 (No Balancing)', rf_model_2, X_test_2, y_test_2),
    ('Model 3 (SMOTEENN)', rf_model_3, X_test_3, y_test_3),
    ('Model 4 (Class Weights)', rf_model_4, X_test_2, y_test_2)
]:
    roc_results[model_name] = calculate_roc_auc_multiclass(model, X_test, y_test)

# %% [markdown]
# ### Visualizing ROC Curves for All Models

# %%
# Plot ROC curves for all models
plt.figure(figsize=(15, 10))

# Plot micro-average ROC curve for each model
for model_name, (fpr, tpr, roc_auc) in roc_results.items():
    plt.plot(
        fpr["micro"], tpr["micro"],
        label=f'{model_name} (area = {roc_auc["micro"]:.3f})',
        linewidth=2
    )

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC Curves for All Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Model Predictions Consistency Analysis
# Analyzing the consistency of predictions across different models

# %%
# Create a function to compare model predictions on the unseen data
def analyze_prediction_consistency(predictions_dict):
    # Create a DataFrame with all model predictions
    consistency_df = pd.DataFrame(predictions_dict)
    
    # Check agreement between models
    agreement_count = consistency_df.apply(lambda row: row.value_counts().max(), axis=1)
    full_agreement = (agreement_count == 4).sum()  # All 4 models agree
    majority_agreement = ((agreement_count >= 3) & (agreement_count < 4)).sum()  # 3 models agree
    split_agreement = (agreement_count == 2).sum()  # 2-2 split
    
    # Create summary
    agreement_summary = {
        'Full Agreement (4/4)': full_agreement,
        'Majority Agreement (3/4)': majority_agreement,
        'Split Agreement (2/2)': split_agreement,
        'Total Samples': len(consistency_df)
    }
    
    # Calculate percentages
    for key in list(agreement_summary.keys())[:-1]:
        agreement_summary[f'{key} (%)'] = (agreement_summary[key] / agreement_summary['Total Samples']) * 100
    
    # Identify samples with disagreement
    disagreement_samples = consistency_df[agreement_count < 4]
    
    return {
        'summary': agreement_summary,
        'disagreement_samples': disagreement_samples
    }

# Analyze consistency of model predictions on unseen data
consistency_analysis = analyze_prediction_consistency(unseen_predictions)

# Display consistency summary
print("\nModel Prediction Consistency Analysis:")
for key, value in consistency_analysis['summary'].items():
    if '(%)' in key:
        print(f"{key}: {value:.2f}%")
    else:
        print(f"{key}: {value}")

# Display samples with disagreement if there are any
if len(consistency_analysis['disagreement_samples']) > 0:
    print("\nSamples with Model Disagreement:")
    print(consistency_analysis['disagreement_samples'])

# %% [markdown]
# ## Temporal Performance Analysis
# Analyzing model performance stability over time periods to assess temporal generalization

# %%
# Assuming we have temporal information in the dataset
# If not available in the current dataset, this code would need to be adapted
# Let's simulate by creating a temporal split

# Create a function to evaluate model performance over different time periods
def evaluate_temporal_performance(model, X, y, n_splits=4):
    # Create equal-sized chunks to simulate temporal splits
    chunk_size = len(X) // n_splits
    temporal_performance = []
    
    for i in range(n_splits):
        # Create temporal test split
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_splits - 1 else len(X)
        
        # Split data
        X_temp_train = pd.concat([X.iloc[:start_idx], X.iloc[end_idx:]])
        y_temp_train = pd.concat([y.iloc[:start_idx], y.iloc[end_idx:]])
        X_temp_test = X.iloc[start_idx:end_idx]
        y_temp_test = y.iloc[start_idx:end_idx]
        
        # Train model on this temporal fold
        temp_model = RandomForestClassifier(random_state=42)
        temp_model.fit(X_temp_train, y_temp_train)
        
        # Evaluate on test set
        y_temp_pred = temp_model.predict(X_temp_test)
        accuracy = accuracy_score(y_temp_test, y_temp_pred)
        
        # Store performance
        temporal_performance.append({
            'Period': f'Split {i+1}',
            'Accuracy': accuracy,
            'Samples': len(X_temp_test)
        })
    
    return pd.DataFrame(temporal_performance)

# Evaluate temporal performance for the SMOTEENN model (our best performer)
temporal_results = evaluate_temporal_performance(rf_model_3, pd.DataFrame(X), pd.Series(y))

# Display temporal performance
print("\nTemporal Performance Analysis:")
print(temporal_results)

# Visualize temporal performance
plt.figure(figsize=(10, 6))
plt.bar(temporal_results['Period'], temporal_results['Accuracy'], color='teal')
plt.axhline(y=temporal_results['Accuracy'].mean(), color='red', linestyle='--',
            label=f'Mean Accuracy: {temporal_results["Accuracy"].mean():.3f}')
plt.xlabel('Time Period')
plt.ylabel('Accuracy')
plt.title('Model Performance Across Different Time Periods')
plt.legend()
plt.ylim([0, 1.1])
plt.tight_layout()
plt.show()



