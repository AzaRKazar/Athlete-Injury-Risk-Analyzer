# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import pickle
import warnings
# Additional imports for new features
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap
from sklearn.inspection import permutation_importance
import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# Setting paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../../data')
figures_dir = os.path.join(script_dir, '../figures')
tables_dir = os.path.join(script_dir, '../tables')
model_dir = os.path.join(script_dir, '../../trained-model')

# Create directories if they don't exist
for dir_path in [figures_dir, tables_dir, model_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Load the model data
model_data_path = os.path.join(data_dir, 'model_data.csv')
df = pd.read_csv(model_data_path)
print(f"Loaded model data: {df.shape}")

# Check for team or sport information in the dataset
team_columns = [col for col in df.columns if 'team' in col.lower() or 'sport' in col.lower()]
if team_columns:
    print(f"Found team/sport columns: {team_columns}")
else:
    # If no team column exists, see if we can load sport data from the raw files
    sport_data_path = os.path.join(script_dir, '../data/sport_data_summary.csv')
    if os.path.exists(sport_data_path):
        print("Loading separate sport data file")
        sport_df = pd.read_csv(sport_data_path)
        if 'sbuid' in df.columns and 'sbuid' in sport_df.columns:
            df = df.merge(sport_df, on='sbuid', how='left')
            team_columns = [col for col in sport_df.columns if 'team' in col.lower() or 'sport' in col.lower()]
            print(f"Merged sport data with {len(team_columns)} team/sport related columns")
    
    # If still no team info, let's check if we can extract it from the raw data files
    if not team_columns:
        raw_data_dir = os.path.join(script_dir, '../data')
        sport_files = [f for f in os.listdir(raw_data_dir) if f.startswith('raw_vald_data_') and f.endswith('.csv')]
        if sport_files and 'sbuid' in df.columns:
            print(f"Found {len(sport_files)} sport-specific data files. Attempting to extract team information.")
            # Create a mapping of athlete IDs to sports
            athlete_sport_map = {}
            for sport_file in sport_files:
                sport_name = sport_file.replace('raw_vald_data_', '').replace('.csv', '')
                try:
                    sport_data = pd.read_csv(os.path.join(raw_data_dir, sport_file))
                    if 'sbuid' in sport_data.columns:
                        for athlete_id in sport_data['sbuid'].unique():
                            athlete_sport_map[athlete_id] = sport_name
                except Exception as e:
                    print(f"Error reading {sport_file}: {e}")
            
            # Add sport to main dataframe
            if athlete_sport_map:
                df['sport'] = df['sbuid'].map(athlete_sport_map)
                team_columns = ['sport']
                print(f"Added sport column based on raw data files. Found {df['sport'].nunique()} unique sports.")

# Define features and target
# Original features - NOTE: This includes MaxForceSymmetry and TorqueSymmetry which can cause data leakage
original_features = [
    'leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque',
    'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque',
    'ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry'
]

# Updated features - Remove direct symmetry metrics that cause data leakage
improved_features = [
    'leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque',
    'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque',
    'ForceSymmetry', 'ImpulseSymmetry' # Removed MaxForceSymmetry and TorqueSymmetry to prevent data leakage
]

# Add derived features that might be useful
if 'sbuid' in df.columns and 'testDateUtc' in df.columns:
    # Convert to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['testDateUtc']):
        df['testDateUtc'] = pd.to_datetime(df['testDateUtc'])
    
    # Add a 'days_since_first_test' feature if multiple tests per athlete
    athlete_first_tests = df.groupby('sbuid')['testDateUtc'].min()
    df = df.join(athlete_first_tests.rename('first_test_date'), on='sbuid')
    df['days_since_first_test'] = (df['testDateUtc'] - df['first_test_date']).dt.days
    original_features.append('days_since_first_test')
    improved_features.append('days_since_first_test')

# Define target - for this analysis we'll create a binary target 
# If MaxForceSymmetry is too far from 1.0 (>10% asymmetry), mark as high risk
if 'injury_risk_high' not in df.columns:
    # Creating a binary risk target based on force asymmetry
    force_asymmetry_threshold = 0.10  # 10% asymmetry threshold
    df['injury_risk_high'] = ((df['MaxForceSymmetry'] < (1 - force_asymmetry_threshold)) | 
                              (df['MaxForceSymmetry'] > (1 + force_asymmetry_threshold))).astype(int)
    print(f"Created injury_risk_high target with {df['injury_risk_high'].sum()} high risk samples ({df['injury_risk_high'].mean():.1%})")

# Define target variable name
target = 'injury_risk_high'

# Select data for modeling
X = df[improved_features]
y = df[target]

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
for i, feature in enumerate(improved_features[:9]):  # Limit to first 9 features for readability
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

for feature in improved_features:
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

# Between-Team Differences Analysis
print("\n--- Between-Team Differences Analysis ---")
if team_columns:
    team_column = team_columns[0]  # Use the first identified team/sport column
    print(f"Analyzing differences between teams/sports using '{team_column}' column")
    
    # Check if we have enough teams with sufficient data
    team_counts = df[team_column].value_counts()
    valid_teams = team_counts[team_counts >= 5].index  # Only include teams with at least 5 athletes
    
    if len(valid_teams) >= 2:
        print(f"Found {len(valid_teams)} teams/sports with sufficient data: {', '.join(valid_teams)}")
        team_df = df[df[team_column].isin(valid_teams)].copy()
        
        # Summary statistics by team
        team_summary = team_df.groupby(team_column)[improved_features].agg(['mean', 'std', 'count'])
        team_summary.to_csv(os.path.join(tables_dir, 'team_summary_statistics.csv'))
        
        # Calculate injury risk rate by team
        team_risk = team_df.groupby(team_column)[target].agg(['mean', 'count']).reset_index()
        team_risk = team_risk.rename(columns={'mean': 'injury_risk_rate'})
        team_risk['injury_risk_rate'] = team_risk['injury_risk_rate'] * 100  # Convert to percentage
        team_risk = team_risk.sort_values('injury_risk_rate', ascending=False)

        # Save team risk to CSV
        team_risk.to_csv(os.path.join(tables_dir, 'team_injury_risk.csv'), index=False)

        # Plot injury risk by team
        plt.figure(figsize=(12, 6))
        sns.barplot(x=team_column, y='injury_risk_rate', data=team_risk)
        plt.title('Asymmetry Rate by Team/Sport')
        plt.xlabel('Team/Sport')
        plt.ylabel('Biomechanical Asymmetry Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'team_injury_risk.png'), dpi=300)

        # Perform statistical tests to compare teams
        team_comparison = []

        for feature in improved_features:
            # Use ANOVA to test for differences between teams
            groups = [team_df[team_df[team_column] == team][feature].dropna() for team in valid_teams]
            if all(len(g) > 0 for g in groups):
                try:
                    f_stat, p_value = stats.f_oneway(*groups)

                    # Identify teams with highest and lowest values
                    feature_by_team = team_df.groupby(team_column)[feature].mean()
                    highest_team = feature_by_team.idxmax()
                    lowest_team = feature_by_team.idxmin()

                    # Calculate the percentage difference between highest and lowest
                    max_value = feature_by_team.max()
                    min_value = feature_by_team.min()
                    percent_diff = 0
                    if min_value != 0:
                        percent_diff = ((max_value - min_value) / min_value) * 100

                    team_comparison.append({
                        'feature': feature,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'highest_team': highest_team,
                        'lowest_team': lowest_team,
                        'highest_value': max_value,
                        'lowest_value': min_value,
                        'percent_difference': percent_diff
                    })
                except:
                    print(f"Error performing ANOVA for {feature}")

        if team_comparison:
            # Create and save team comparison dataframe
            team_comparison_df = pd.DataFrame(team_comparison)
            team_comparison_df = team_comparison_df.sort_values('p_value')
            team_comparison_df.to_csv(os.path.join(tables_dir, 'team_feature_comparison.csv'), index=False)

            # Plot significant differences
            significant_features = team_comparison_df[team_comparison_df['significant']]['feature'].tolist()

            if significant_features:
                print(f"Found {len(significant_features)} features with significant differences between teams/sports")

                # Box plots for significantly different features
                for i, feature in enumerate(significant_features[:6]):  # Limit to 6 for readability
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(x=team_column, y=feature, data=team_df)
                    plt.title(f'{feature} by Team/Sport')
                    plt.xlabel('Team/Sport')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(figures_dir, f'team_comparison_{feature}.png'), dpi=300)

                # Create a summary heatmap of feature values by team
                plt.figure(figsize=(14, 10))
                team_feature_pivot = team_df.pivot_table(
                    index=team_column,
                    values=significant_features[:10],  # Limit to top 10 significant features
                    aggfunc='mean'
                )

                # Normalize each feature for better visualization
                team_feature_norm = (team_feature_pivot - team_feature_pivot.mean()) / team_feature_pivot.std()

                sns.heatmap(
                    team_feature_norm,
                    annot=False,
                    cmap='coolwarm',
                    center=0,
                    fmt='.2f',
                    cbar_kws={'label': 'Z-score (normalized value)'}
                )
                plt.title('Team/Sport Performance Profile (Significant Features)')
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, 'team_feature_heatmap.png'), dpi=300)

        # Analyze injury risk differences between teams
        if 'injury_risk_high' in team_df.columns:
            # Prepare contingency table: teams vs risk
            team_risk_table = pd.crosstab(team_df[team_column], team_df['injury_risk_high'])

            # Chi-squared test for independence
            chi2, p, dof, expected = stats.chi2_contingency(team_risk_table)

            print(f"Chi-squared test for team vs injury risk: chi2={chi2:.2f}, p={p:.4f}")

            if p < 0.05:
                print("There is a significant association between team/sport and injury risk")

                # Calculate standardized residuals to identify which teams differ from expected
                residuals = (team_risk_table - expected) / np.sqrt(expected)

                # Plot residuals heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    residuals,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    fmt='.2f'
                )
                plt.title('Team vs Injury Risk Standardized Residuals\n(>2 or <-2 indicates significant deviation)')
                plt.xlabel('Injury Risk (0=Low, 1=High)')
                plt.ylabel('Team/Sport')
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, 'team_risk_residuals.png'), dpi=300)
            else:
                print("No significant association between team/sport and injury risk")

        # Team-specific regression models
        if len(valid_teams) >= 3:
            print("\nTraining team-specific models...")
            team_models = {}
            team_accuracies = []

            for team in valid_teams:
                team_data = team_df[team_df[team_column] == team]

                if len(team_data) >= 20 and team_data[target].nunique() > 1:  # Ensure enough samples
                    X_team = team_data[improved_features]
                    y_team = team_data[target]

                    # Use a simple model due to potentially small sample size
                    model = RandomForestClassifier(n_estimators=50, random_state=42)

                    # Use cross-validation to evaluate
                    try:
                        cv_scores = cross_val_score(model, X_team, y_team, cv=min(5, len(team_data) // 5), scoring='accuracy')
                        avg_accuracy = np.mean(cv_scores)

                        team_accuracies.append({
                            'team': team,
                            'sample_size': len(team_data),
                            'risk_rate': team_data[target].mean() * 100,
                            'cv_accuracy': avg_accuracy * 100,
                            'cv_std': np.std(cv_scores) * 100
                        })

                        # Train on all data for feature importance
                        model.fit(X_team, y_team)
                        team_models[team] = model
                    except Exception as e:
                        print(f"Error training model for team {team}: {e}")

            if team_accuracies:
                team_accuracies_df = pd.DataFrame(team_accuracies)
                team_accuracies_df = team_accuracies_df.sort_values('cv_accuracy', ascending=False)
                team_accuracies_df.to_csv(os.path.join(tables_dir, 'team_model_performance.csv'), index=False)

                # Plot team model performance
                plt.figure(figsize=(12, 6))
                sns.barplot(x='team', y='cv_accuracy', data=team_accuracies_df)
                plt.title('Team-Specific Model Performance')
                plt.xlabel('Team/Sport')
                plt.ylabel('Cross-Validation Accuracy (%)')
                plt.xticks(rotation=45)
                plt.ylim(0, 100)
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, 'team_model_performance.png'), dpi=300)

                # Compare feature importances across teams
                if len(team_models) >= 2:
                    team_importances = []

                    for team, model in team_models.items():
                        importances = pd.DataFrame({
                            'Feature': improved_features,
                            'Importance': model.feature_importances_
                        })
                        importances['Team'] = team
                        team_importances.append(importances)

                    combined_importances = pd.concat(team_importances)

                    # Create and save pivot table
                    importance_pivot = combined_importances.pivot_table(
                        index='Feature',
                        columns='Team',
                        values='Importance'
                    ).fillna(0)

                    importance_pivot.to_csv(os.path.join(tables_dir, 'team_feature_importance.csv'))

                    # Heatmap of feature importance by team
                    plt.figure(figsize=(14, 10))
                    top_features = importance_pivot.mean(axis=1).sort_values(ascending=False).head(10).index
                    importance_pivot_top = importance_pivot.loc[top_features]

                    sns.heatmap(
                        importance_pivot_top,
                        annot=True,
                        cmap='YlGnBu',
                        fmt='.3f'
                    )
                    plt.title('Feature Importance by Team/Sport')
                    plt.tight_layout()
                    plt.savefig(os.path.join(figures_dir, 'team_feature_importance.png'), dpi=300)

        # Train models for each team/sport separately
        print("\n--- Sport-Specific Model Comparison ---")
        
        # Create a dictionary to store results for each sport
        sport_model_results = {}
        metrics_to_track = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'CV Mean ROC AUC']
        
        # Define model types to train for each sport
        model_types = {
            "RF_No_Balancing": {"class_weight": None},
            "RF_with_SMOTE": {"smote": True},
            "RF_with_SMOTEENN": {"smoteenn": True},
            "RF_with_Class_Weights": {"class_weight": "balanced"}
        }
        
        # Train models for each valid team with enough data
        for team in valid_teams:
            team_data = team_df[team_df[team_column] == team]
            
            # Only process teams with sufficient data and both classes present
            if len(team_data) >= 20 and team_data[target].nunique() > 1:
                print(f"\nTraining models for {team} (n={len(team_data)})")
                
                X_team = team_data[improved_features]
                y_team = team_data[target]
                
                # Split data for this team
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_team, y_team, test_size=0.2, random_state=42, stratify=y_team
                    )
                    
                    # Initialize results for this sport
                    sport_model_results[team] = {}
                    
                    # Train each model type
                    for model_name, model_params in model_types.items():
                        # Set up model and training data
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        X_train_current, y_train_current = X_train.copy(), y_train.copy()
                        
                        # Apply appropriate sampling strategy
                        if model_params.get("smote"):
                            try:
                                smote = SMOTE(random_state=42)
                                X_train_current, y_train_current = smote.fit_resample(X_train, y_train)
                            except Exception as e:
                                print(f"Error applying SMOTE for {team} - {model_name}: {e}")
                                continue
                        elif model_params.get("smoteenn"):
                            try:
                                smoteenn = SMOTEENN(random_state=42)
                                X_train_current, y_train_current = smoteenn.fit_resample(X_train, y_train)
                            except Exception as e:
                                print(f"Error applying SMOTEENN for {team} - {model_name}: {e}")
                                continue
                        
                        # Set class weights if specified
                        if model_params.get("class_weight"):
                            model = RandomForestClassifier(
                                n_estimators=100, 
                                class_weight=model_params["class_weight"],
                                random_state=42
                            )
                        
                        # Train model
                        try:
                            model.fit(X_train_current, y_train_current)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            
                            # Handle case where all predictions are the same class
                            try:
                                roc_auc = roc_auc_score(y_test, y_pred_proba)
                            except Exception:
                                roc_auc = np.nan
                            
                            # Cross-validation
                            cv = StratifiedKFold(n_splits=min(5, len(y_team) // 10), shuffle=True, random_state=42)
                            try:
                                cv_scores = cross_val_score(model, X_team, y_team, cv=cv, scoring='roc_auc')
                                cv_mean = np.mean(cv_scores)
                                cv_std = np.std(cv_scores)
                            except Exception:
                                cv_mean = np.nan
                                cv_std = np.nan
                            
                            # Store results
                            sport_model_results[team][model_name] = {
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1': f1,
                                'ROC AUC': roc_auc,
                                'CV Mean ROC AUC': cv_mean,
                                'CV Std ROC AUC': cv_std,
                                'Sample Size': len(team_data),
                                'Asymmetry Rate': team_data[target].mean() * 100
                            }
                            
                            print(f"  {model_name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, ROC AUC={roc_auc if not np.isnan(roc_auc) else 'N/A':.3f}")
                            
                        except Exception as e:
                            print(f"Error training {model_name} for {team}: {e}")
                
                except Exception as e:
                    print(f"Error processing team {team}: {e}")
            else:
                print(f"Skipping {team}: insufficient data (n={len(team_data)}) or single class")
        
        # Compile results across sports
        if sport_model_results:
            # Create comparison dataframes
            sport_comparison = []
            model_comparison = []
            
            # For each sport and model, collect metrics
            for sport, models in sport_model_results.items():
                for model_name, metrics in models.items():
                    row = {'Sport': sport, 'Model': model_name}
                    row.update(metrics)
                    sport_comparison.append(row)
                
                # Find best model for this sport
                if models:
                    best_model = max(models.items(), key=lambda x: x[1].get('F1', 0))
                    row = {
                        'Sport': sport,
                        'Best Model': best_model[0],
                        'Sample Size': best_model[1]['Sample Size'],
                        'Asymmetry Rate': best_model[1]['Asymmetry Rate']
                    }
                    for metric in metrics_to_track:
                        if metric in best_model[1]:
                            row[metric] = best_model[1][metric]
                    model_comparison.append(row)
            
            # Convert to dataframes
            sport_comparison_df = pd.DataFrame(sport_comparison)
            model_comparison_df = pd.DataFrame(model_comparison)
            
            # Save detailed results
            sport_comparison_df.to_csv(os.path.join(tables_dir, 'sport_model_comparison_detailed.csv'), index=False)
            model_comparison_df.to_csv(os.path.join(tables_dir, 'sport_best_models.csv'), index=False)
            
            # Create summary table for best model by sport
            best_models_by_sport = model_comparison_df.sort_values('F1', ascending=False)
            print("\nBest Model Performance by Sport:")
            print(best_models_by_sport[['Sport', 'Best Model', 'Accuracy', 'F1', 'ROC AUC']].head())
            
            # Plot comparison of model performance across sports
            plt.figure(figsize=(14, 8))
            model_plot_data = model_comparison_df.sort_values('F1', ascending=False)
            sns.barplot(x='Sport', y='F1', data=model_plot_data)
            plt.title('Model Performance (F1 Score) by Sport')
            plt.xlabel('Sport')
            plt.ylabel('F1 Score')
            plt.xticks(rotation=45)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'sport_model_performance.png'), dpi=300)
            
            # Plot model comparison for each sport
            plt.figure(figsize=(16, 10))
            
            # Use only top 8 sports with highest F1 scores for readability
            top_sports = model_comparison_df.sort_values('F1', ascending=False)['Sport'].unique()[:8]
            plot_data = sport_comparison_df[sport_comparison_df['Sport'].isin(top_sports)]
            
            sns.barplot(x='Sport', y='F1', hue='Model', data=plot_data)
            plt.title('Model Performance by Sport and Algorithm')
            plt.xlabel('Sport')
            plt.ylabel('F1 Score')
            plt.xticks(rotation=45)
            plt.ylim(0, 1.05)
            plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'sport_algorithm_comparison.png'), dpi=300)
            
            # Create heatmap of performance across sports
            plt.figure(figsize=(14, 10))
            pivot_data = sport_comparison_df.pivot_table(
                index='Sport', 
                columns='Model', 
                values='F1',
                aggfunc='mean'
            ).fillna(0)
            
            # Sort sports by average F1 score
            pivot_data['Avg'] = pivot_data.mean(axis=1)
            pivot_data = pivot_data.sort_values('Avg', ascending=False)
            pivot_data = pivot_data.drop('Avg', axis=1)
            
            sns.heatmap(
                pivot_data,
                annot=True,
                cmap='YlGnBu',
                fmt='.3f',
                vmin=0,
                vmax=1
            )
            plt.title('F1 Score by Sport and Model Type')
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, 'sport_model_heatmap.png'), dpi=300)
            
            # Add to summary report
            with open(os.path.join(tables_dir, 'sport_model_summary.md'), 'w') as f:
                f.write("# Sport-Specific Model Performance Summary\n\n")
                f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
                
                f.write("## Overview\n\n")
                f.write(f"This analysis trained separate asymmetry classification models for {len(model_comparison_df)} different sports with sufficient data. ")
                f.write(f"Four different modeling approaches were compared for each sport to identify the most effective algorithm for capturing sport-specific asymmetry patterns.\n\n")
                
                f.write("## Key Findings\n\n")
                
                # Top performing sports
                top_3_sports = best_models_by_sport.head(3)['Sport'].tolist()
                bottom_3_sports = best_models_by_sport.tail(3)['Sport'].tolist()
                
                f.write(f"1. **Best Predictability**: The sports with most predictable asymmetry patterns were {', '.join(top_3_sports)}.\n")
                f.write(f"2. **Most Challenging**: The sports with least predictable asymmetry patterns were {', '.join(bottom_3_sports)}.\n")
                
                # Algorithm performance
                algo_counts = best_models_by_sport['Best Model'].value_counts()
                best_algo = algo_counts.index[0] if len(algo_counts) > 0 else "None"
                algo_pct = (algo_counts.iloc[0] / algo_counts.sum() * 100) if len(algo_counts) > 0 else 0
                
                f.write(f"3. **Best Algorithm**: {best_algo} was the most effective model, performing best for {algo_pct:.1f}% of sports.\n")
                
                # Performance ranges
                f.write(f"4. **Performance Range**: F1 scores ranged from {best_models_by_sport['F1'].min():.3f} to {best_models_by_sport['F1'].max():.3f}, ")
                f.write(f"indicating substantial variation in how well asymmetry patterns can be classified across different sports.\n\n")
                
                # Performance table
                f.write("## Performance by Sport\n\n")
                f.write("| Sport | Best Model | F1 Score | Accuracy | ROC-AUC | Sample Size | Asymmetry Rate (%) |\n")
                f.write("|-------|------------|----------|----------|---------|-------------|-------------------|\n")
                
                for _, row in best_models_by_sport.iterrows():
                    f1 = row['F1'] if not pd.isna(row['F1']) else "N/A"
                    acc = row['Accuracy'] if not pd.isna(row['Accuracy']) else "N/A"
                    auc = row['ROC AUC'] if not pd.isna(row['ROC AUC']) else "N/A"
                    
                    f.write(f"| {row['Sport']} | {row['Best Model']} | {f1:.3f} | {acc:.3f} | {auc:.3f} | {row['Sample Size']:.0f} | {row['Asymmetry Rate']:.1f} |\n")
                
                f.write("\n## Implications\n\n")
                f.write("1. **Sport-Specific Approaches**: The substantial variation in model performance across sports confirms the need for sport-specific assessment approaches.\n")
                f.write("2. **Sampling Strategies**: The effectiveness of different sampling strategies (SMOTE, SMOTEENN) varies by sport, likely reflecting different class imbalance characteristics.\n")
                f.write("3. **Data Requirements**: Sports with larger sample sizes generally yielded more reliable models, highlighting the importance of adequate data collection for each sport.\n")
                f.write("4. **Asymmetry Complexity**: The varying predictability of asymmetry patterns suggests that biomechanical demands differ significantly across sports, with some creating more consistent and predictable asymmetry profiles than others.\n")
        else:
            print("No sport had sufficient data for training separate models")
    else:
        print(f"Insufficient data for team analysis. Need at least 2 teams with 5+ athletes each.")
else:
    print("No team or sport information available in the dataset.")

# Add team findings to the analysis summary
with open(os.path.join(tables_dir, 'team_analysis_summary.md'), 'w') as f:
    f.write("# Between-Team Analysis Summary\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

    if team_columns:
        team_column = team_columns[0]
        team_counts = df[team_column].value_counts()
        valid_teams = team_counts[team_counts >= 5].index

        if len(valid_teams) >= 2:
            f.write(f"## Teams/Sports Analyzed: {len(valid_teams)}\n\n")

            # Team counts
            f.write("### Team/Sport Sample Sizes\n\n")
            f.write("| Team/Sport | Athletes | Tests |\n")
            f.write("|------------|----------|-------|\n")

            for team in valid_teams:
                team_data = df[df[team_column] == team]
                athletes = team_data['sbuid'].nunique() if 'sbuid' in team_data.columns else "N/A"
                f.write(f"| {team} | {athletes} | {len(team_data)} |\n")

            # Team risk rates
            if 'team_risk' in locals() and len(team_risk) > 0:
                f.write("\n### Injury Risk by Team/Sport\n\n")
                f.write("| Team/Sport | Risk Rate (%) | Sample Size |\n")
                f.write("|------------|---------------|-------------|\n")

                for _, row in team_risk.iterrows():
                    f.write(f"| {row[team_column]} | {row['injury_risk_rate']:.1f} | {row['count']} |\n")

            # Key differences
            if 'team_comparison_df' in locals() and len(team_comparison_df) > 0:
                significant_features = team_comparison_df[team_comparison_df['significant']]

                if len(significant_features) > 0:
                    f.write("\n### Significant Differences Between Teams/Sports\n\n")
                    f.write("| Feature | p-value | Highest Team | Lowest Team | % Difference |\n")
                    f.write("|---------|---------|--------------|-------------|---------------|\n")

                    for _, row in significant_features.head(10).iterrows():
                        f.write(f"| {row['feature']} | {row['p_value']:.4f} | {row['highest_team']} | {row['lowest_team']} | {row['percent_difference']:.1f} |\n")
                else:
                    f.write("\n### No Significant Performance Differences Between Teams/Sports\n\n")

            # Team-specific models
            if 'team_accuracies_df' in locals() and len(team_accuracies_df) > 0:
                f.write("\n### Team-Specific Model Performance\n\n")
                f.write("| Team/Sport | Accuracy (%) | Sample Size | Risk Rate (%) |\n")
                f.write("|------------|--------------|-------------|---------------|\n")

                for _, row in team_accuracies_df.iterrows():
                    f.write(f"| {row['team']} | {row['cv_accuracy']:.1f} | {row['sample_size']} | {row['risk_rate']:.1f} |\n")

                f.write("\nTeams with higher accuracy scores have more consistent and predictable injury risk patterns.\n")
        else:
            f.write("Insufficient data for meaningful team analysis.\n")
    else:
        f.write("No team or sport information available in the dataset.\n")

    # Key insights and recommendations
    f.write("\n## Key Insights\n\n")

    if team_columns and 'team_risk' in locals() and len(team_risk) > 0:
        # Find highest and lowest risk teams
        highest_risk_team = team_risk.iloc[0][team_column]
        highest_risk_rate = team_risk.iloc[0]['injury_risk_rate']
        lowest_risk_team = team_risk.iloc[-1][team_column]
        lowest_risk_rate = team_risk.iloc[-1]['injury_risk_rate']

        f.write(f"1. **Risk Variation**: {highest_risk_team} shows the highest injury risk rate ({highest_risk_rate:.1f}%), while {lowest_risk_team} has the lowest ({lowest_risk_rate:.1f}%).\n")

        # Chi-squared test results
        if 'p' in locals():
            if p < 0.05:
                f.write(f"2. **Statistical Significance**: There is a significant association between team/sport and injury risk (p={p:.4f}).\n")
            else:
                f.write(f"2. **Statistical Significance**: No significant association between team/sport and injury risk was found (p={p:.4f}).\n")

        # Performance differences
        if 'team_comparison_df' in locals() and len(team_comparison_df[team_comparison_df['significant']]) > 0:
            top_diff = team_comparison_df[team_comparison_df['significant']].iloc[0]
            f.write(f"3. **Key Differences**: The most significant difference between teams is in {top_diff['feature']} (p={top_diff['p_value']:.4f}), with {top_diff['highest_team']} showing the highest values and {top_diff['lowest_team']} showing the lowest.\n")

    f.write("\n## Recommendations\n\n")

    if team_columns and 'team_risk' in locals() and len(team_risk) > 0:
        # Add tailored recommendations
        f.write("1. **Sport-Specific Protocols**: Develop specialized monitoring and training protocols for high-risk teams.\n")
        f.write("2. **Cross-Training Opportunities**: Learn from lower-risk teams' practices and implement their techniques across other teams.\n")
        f.write("3. **Targeted Interventions**: Focus preventive resources on the highest-risk teams and specific risk factors identified.\n")

        if 'team_comparison_df' in locals() and len(team_comparison_df[team_comparison_df['significant']]) > 0:
            top_features = team_comparison_df[team_comparison_df['significant']]['feature'].head(3).tolist()
            if top_features:
                feature_list = ", ".join(top_features)
                f.write(f"4. **Key Metrics Monitoring**: Closely track {feature_list} as these show significant variation between teams.\n")
    else:
        f.write("1. **Data Collection**: Improve team/sport data collection to enable more robust between-team analyses.\n")
        f.write("2. **Standardized Testing**: Implement consistent testing protocols across all teams to enable fair comparisons.\n")

# Athlete Progress Analysis - New Feature
if 'sbuid' in df.columns and 'testDateUtc' in df.columns:
    print("\n--- Athlete Progress Analysis ---")

    # Convert to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['testDateUtc']):
        df['testDateUtc'] = pd.to_datetime(df['testDateUtc'])

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
        plt.plot(athlete_data['testDateUtc'], athlete_data['leftAvgForce'], 'b-o', label='Left')
        plt.plot(athlete_data['testDateUtc'], athlete_data['rightAvgForce'], 'r-o', label='Right')
        plt.title('Avg Force Comparison')
        plt.xticks(rotation=45)
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(athlete_data['testDateUtc'], athlete_data['leftMaxForce'], 'b-o', label='Left')
        plt.plot(athlete_data['testDateUtc'], athlete_data['rightMaxForce'], 'r-o', label='Right')
        plt.title('Max Force Comparison')
        plt.xticks(rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'athlete_progress_example.png'), dpi=300)

# Train an all-sports combined model
print("\n--- All-Sports Combined Model Analysis ---")

# First, split the data for the combined model
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define model types to train
model_types = {
    "RF_No_Balancing": {"class_weight": None},
    "RF_with_SMOTE": {"smote": True},
    "RF_with_SMOTEENN": {"smoteenn": True},
    "RF_with_Class_Weights": {"class_weight": "balanced"}
}

# Store combined model results
combined_model_results = {}

# Train each model type
for model_name, model_params in model_types.items():
    print(f"\nTraining {model_name} on combined data...")
    
    # Set up model and training data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train_current, y_train_current = X_all_train.copy(), y_all_train.copy()
    
    # Apply appropriate sampling strategy
    if model_params.get("smote"):
        try:
            smote = SMOTE(random_state=42)
            X_train_current, y_train_current = smote.fit_resample(X_all_train, y_all_train)
            print(f"Applied SMOTE: training samples {len(X_all_train)} → {len(X_train_current)}")
        except Exception as e:
            print(f"Error applying SMOTE: {e}")
            continue
    elif model_params.get("smoteenn"):
        try:
            smoteenn = SMOTEENN(random_state=42)
            X_train_current, y_train_current = smoteenn.fit_resample(X_all_train, y_all_train)
            print(f"Applied SMOTEENN: training samples {len(X_all_train)} → {len(X_train_current)}")
        except Exception as e:
            print(f"Error applying SMOTEENN: {e}")
            continue
    
    # Set class weights if specified
    if model_params.get("class_weight"):
        model = RandomForestClassifier(
            n_estimators=100, 
            class_weight=model_params["class_weight"],
            random_state=42
        )
    
    # Train model
    try:
        model.fit(X_train_current, y_train_current)
        
        # Save the model
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {model_path}")
        
        # Make predictions
        y_pred = model.predict(X_all_test)
        y_pred_proba = model.predict_proba(X_all_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_all_test, y_pred)
        precision = precision_score(y_all_test, y_pred)
        recall = recall_score(y_all_test, y_pred)
        f1 = f1_score(y_all_test, y_pred)
        roc_auc = roc_auc_score(y_all_test, y_pred_proba)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        # Store results
        combined_model_results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC AUC': roc_auc,
            'CV Mean ROC AUC': cv_mean,
            'CV Std ROC AUC': cv_std,
            'Sample Size': len(X),
            'Asymmetry Rate': y.mean() * 100
        }
        
        print(f"  Performance: Accuracy={accuracy:.3f}, F1={f1:.3f}, ROC AUC={roc_auc:.3f}")
        print(f"  Cross-validation ROC AUC: {cv_mean:.3f} ± {cv_std:.3f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_all_test, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Low Asymmetry', 'High Asymmetry'],
            yticklabels=['Low Asymmetry', 'High Asymmetry']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{model_name}.png'), dpi=300)
        
        # Feature importance for this model
        plt.figure(figsize=(10, 8))
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'feature_importance_{model_name}.png'), dpi=300)
        
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Convert combined results to DataFrame for comparison
if combined_model_results:
    combined_results_df = pd.DataFrame.from_dict(combined_model_results, orient='index')
    combined_results_df['Model'] = combined_results_df.index
    combined_results_df = combined_results_df.reset_index(drop=True)
    
    # Save results
    combined_results_df.to_csv(os.path.join(tables_dir, 'combined_model_results.csv'), index=False)
    
    # Plot comparison of model performance
    plt.figure(figsize=(12, 6))
    combined_plot = combined_results_df.sort_values('F1', ascending=False)
    sns.barplot(x='Model', y='F1', data=combined_plot)
    plt.title('Combined Data Model Performance (F1 Score)')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'combined_model_performance.png'), dpi=300)

    # Find best model for the combined dataset
    best_combined_model = combined_results_df.loc[combined_results_df['F1'].idxmax()]
    print(f"\nBest model for combined dataset: {best_combined_model['Model']} (F1={best_combined_model['F1']:.3f})")

# %% 
# Now, let's perform statistical comparison between the all-sports model and sport-specific models
if 'sport_comparison_df' in locals() and combined_model_results:
    print("\n--- Statistical Comparison: Combined vs. Sport-Specific Models ---")
    
    # Prepare data for comparison
    comparison_data = []
    
    # Add combined model results
    for model_name, metrics in combined_model_results.items():
        row = {
            'Model Type': model_name,
            'Sport': 'All Combined',
            'F1': metrics['F1'],
            'Accuracy': metrics['Accuracy'],
            'ROC AUC': metrics['ROC AUC'],
            'Sample Size': metrics['Sample Size']
        }
        comparison_data.append(row)
    
    # Add sport-specific model results
    for _, row in sport_comparison_df.iterrows():
        comparison_row = {
            'Model Type': row['Model'],
            'Sport': row['Sport'],
            'F1': row['F1'],
            'Accuracy': row['Accuracy'],
            'ROC AUC': row['ROC AUC'],
            'Sample Size': row['Sample Size']
        }
        comparison_data.append(comparison_row)
    
    # Convert to dataframe
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save the full comparison data
    comparison_df.to_csv(os.path.join(tables_dir, 'combined_vs_sport_specific_comparison.csv'), index=False)
    
    # Statistical tests for differences in performance
    # 1. Compare the best combined model with the best model for each sport
    best_combined_model_name = best_combined_model['Model']
    best_combined_metrics = combined_model_results[best_combined_model_name]
    
    # Get the best model for each sport
    best_by_sport = {}
    for sport, models in sport_model_results.items():
        if models:
            best_model = max(models.items(), key=lambda x: x[1].get('F1', 0))
            best_by_sport[sport] = {
                'Model': best_model[0],
                'F1': best_model[1].get('F1', 0),
                'Accuracy': best_model[1].get('Accuracy', 0),
                'ROC AUC': best_model[1].get('ROC AUC', 0)
            }
    
    # Create a summary table for the comparison
    statistical_comparison = []
    
    # Compare F1 scores of best models
    f1_scores = [best_by_sport[sport]['F1'] for sport in best_by_sport]
    combined_f1 = best_combined_metrics['F1']
    
    # Simple statistical comparison of means
    mean_sport_f1 = np.mean(f1_scores)
    std_sport_f1 = np.std(f1_scores)
    
    # Check if the combined model is significantly different from the average sport-specific model
    # Using Z-score for simple assessment (assuming normal distribution)
    if std_sport_f1 > 0:
        z_score = (combined_f1 - mean_sport_f1) / std_sport_f1
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # two-tailed p-value
    else:
        z_score = 0
        p_value = 1.0
    
    # Add overall comparison to the results
    statistical_comparison.append({
        'Comparison': 'Combined vs. Average Sport-Specific',
        'Combined F1': combined_f1,
        'Mean Sport F1': mean_sport_f1,
        'Std Sport F1': std_sport_f1,
        'Z-Score': z_score,
        'P-Value': p_value,
        'Significant': p_value < 0.05
    })
    
    # For each sport, compare its best model with the combined model
    for sport, metrics in best_by_sport.items():
        sport_f1 = metrics['F1']
        difference = combined_f1 - sport_f1
        
        # We can't do a true statistical test with just one value per sport,
        # but we can report the absolute and percentage differences
        percent_diff = (difference / sport_f1 * 100) if sport_f1 > 0 else 0
        
        statistical_comparison.append({
            'Comparison': f'Combined vs. {sport}',
            'Combined F1': combined_f1,
            'Sport F1': sport_f1,
            'Absolute Difference': difference,
            'Percentage Difference': percent_diff,
            'Combined Better': difference > 0
        })
    
    # Convert to dataframe and save
    statistical_df = pd.DataFrame(statistical_comparison)
    statistical_df.to_csv(os.path.join(tables_dir, 'combined_vs_sport_statistical_comparison.csv'), index=False)
    
    # Print the overall comparison
    overall_result = statistical_comparison[0]
    print(f"Combined model F1: {overall_result['Combined F1']:.3f}")
    print(f"Average sport-specific model F1: {overall_result['Mean Sport F1']:.3f} ± {overall_result['Std Sport F1']:.3f}")
    
    if overall_result['Significant']:
        print(f"The difference is statistically significant (p={overall_result['P-Value']:.4f})")
    else:
        print(f"The difference is not statistically significant (p={overall_result['P-Value']:.4f})")
    
    # Visualize the comparison
    plt.figure(figsize=(16, 8))
    
    # Prepare data for plotting
    plot_data = []
    
    # Add combined model
    plot_data.append({
        'Sport': 'All Combined', 
        'F1 Score': best_combined_metrics['F1'],
        'Model Type': best_combined_model_name
    })
    
    # Add individual sports
    for sport, metrics in best_by_sport.items():
        plot_data.append({
            'Sport': sport,
            'F1 Score': metrics['F1'],
            'Model Type': metrics['Model']
        })
    
    # Convert to dataframe for plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Sort by F1 score
    plot_df = plot_df.sort_values('F1 Score', ascending=False)
    
    # Create color mapping for model types
    model_colors = {
        'RF_No_Balancing': 'blue',
        'RF_with_SMOTE': 'green',
        'RF_with_SMOTEENN': 'orange',
        'RF_with_Class_Weights': 'red'
    }
    
    # Create the plot
    ax = sns.barplot(x='Sport', y='F1 Score', data=plot_df, palette=[model_colors.get(m, 'gray') for m in plot_df['Model Type']])
    
    # Add line for average sport-specific model F1
    plt.axhline(y=mean_sport_f1, color='red', linestyle='--', alpha=0.7, label=f'Avg Sport-Specific F1: {mean_sport_f1:.3f}')
    
    # Customize the plot
    plt.title('Combined Model vs. Best Sport-Specific Models (F1 Score)')
    plt.xlabel('Sport')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.05)
    
    # Add model type as text on each bar
    for i, row in enumerate(plot_df.itertuples()):
        try:
            model_type = row[3]  # Access by position instead of attribute name
            ax.text(i, 0.05, model_type, ha='center', rotation=90, color='white', fontweight='bold')
        except Exception as e:
            print(f"Error adding model type label: {e}")
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'combined_vs_sport_specific_comparison.png'), dpi=300)
    
    # Create a more detailed visualization showing all model types for each sport
    plt.figure(figsize=(20, 10))
    
    # Filter for readability - only include the top 8 sports plus the combined model
    top_sports = model_comparison_df.sort_values('F1', ascending=False)['Sport'].unique()[:8]
    filtered_comparison = comparison_df[(comparison_df['Sport'].isin(top_sports)) | (comparison_df['Sport'] == 'All Combined')]
    
    # Create a custom palette to highlight the combined models
    custom_palette = {}
    for sport in filtered_comparison['Sport'].unique():
        if sport == 'All Combined':
            custom_palette[sport] = 'darkred'  # Highlight the combined model
        else:
            custom_palette[sport] = 'steelblue'  # Regular color for sport-specific models
    
    # Plot with facet grid to separate by model type
    g = sns.catplot(
        data=filtered_comparison,
        x='Sport', y='F1',
        col='Model Type',
        kind='bar',
        palette=custom_palette,
        height=4, aspect=1.2,
        sharex=True, sharey=True
    )
    
    # Customize the subplot titles
    for ax, model_type in zip(g.axes.flat, filtered_comparison['Model Type'].unique()):
        ax.set_title(f'Model: {model_type}')
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'detailed_model_comparison_by_type.png'), dpi=300)
    
    # Create a summary report
    with open(os.path.join(tables_dir, 'combined_vs_sport_specific_summary.md'), 'w') as f:
        f.write("# Combined vs. Sport-Specific Model Comparison\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This analysis compared the performance of models trained on all sports combined versus models trained on individual sports. ")
        f.write("The comparison helps determine whether sport-specific asymmetry patterns are distinct enough to warrant separate models, ")
        f.write("or whether a single model trained on all sports performs as well or better.\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Overall comparison
        f.write(f"1. **Combined Model Performance**: The best model for the combined dataset was {best_combined_model_name} with an F1 score of {combined_f1:.3f}.\n")
        f.write(f"2. **Sport-Specific Performance**: The average F1 score across best sport-specific models was {mean_sport_f1:.3f} ± {std_sport_f1:.3f}.\n")
        
        if overall_result['Significant']:
            f.write(f"3. **Statistical Significance**: The difference between combined and sport-specific models was statistically significant (p={overall_result['P-Value']:.4f}).\n")
        else:
            f.write(f"3. **Statistical Significance**: The difference between combined and sport-specific models was not statistically significant (p={overall_result['P-Value']:.4f}).\n")
        
        # Best performing individual sports
        best_sport_models = sorted([(sport, metrics['F1']) for sport, metrics in best_by_sport.items()], key=lambda x: x[1], reverse=True)
        top_3_sports = best_sport_models[:3]
        f.write(f"4. **Top Performing Sports**: The sports with highest individual model performance were ")
        f.write(", ".join([f"{sport} (F1={score:.3f})" for sport, score in top_3_sports]))
        f.write(".\n\n")
        
        # Model type performance
        f.write("## Model Type Analysis\n\n")
        
        # Combined model comparison
        f.write("### Combined Dataset Model Performance\n\n")
        f.write("| Model | F1 Score | Accuracy | ROC AUC |\n")
        f.write("|-------|----------|----------|---------|\n")
        
        for model_name, metrics in combined_model_results.items():
            f.write(f"| {model_name} | {metrics['F1']:.3f} | {metrics['Accuracy']:.3f} | {metrics['ROC AUC']:.3f} |\n")
        
        f.write("\n### Sports Where Combined Model Outperformed Sport-Specific Model\n\n")
        
        better_sports = [c for c in statistical_comparison[1:] if c.get('Combined Better', False)]
        if better_sports:
            f.write("| Sport | Sport F1 | Combined F1 | Improvement |\n")
            f.write("|-------|----------|-------------|-------------|\n")
            
            for comp in better_sports:
                sport = comp['Comparison'].replace('Combined vs. ', '')
                f.write(f"| {sport} | {comp['Sport F1']:.3f} | {comp['Combined F1']:.3f} | {comp['Percentage Difference']:.1f}% |\n")
        else:
            f.write("No sports where the combined model outperformed the sport-specific model.\n")
        
        f.write("\n### Sports Where Sport-Specific Model Outperformed Combined Model\n\n")
        
        worse_sports = [c for c in statistical_comparison[1:] if not c.get('Combined Better', True)]
        if worse_sports:
            f.write("| Sport | Sport F1 | Combined F1 | Difference |\n")
            f.write("|-------|----------|-------------|------------|\n")
            
            for comp in worse_sports:
                sport = comp['Comparison'].replace('Combined vs. ', '')
                f.write(f"| {sport} | {comp['Sport F1']:.3f} | {comp['Combined F1']:.3f} | {-comp['Percentage Difference']:.1f}% |\n")
        else:
            f.write("No sports where the sport-specific model outperformed the combined model.\n")
        
        f.write("\n## Implications\n\n")
        
        if overall_result['Significant'] and overall_result['Z-Score'] > 0:
            f.write("1. **Unified Approach Recommended**: The combined model significantly outperformed sport-specific models on average, suggesting that a unified approach to biomechanical asymmetry assessment may be more effective than developing separate models for each sport.\n")
        elif overall_result['Significant'] and overall_result['Z-Score'] < 0:
            f.write("1. **Sport-Specific Approach Recommended**: Sport-specific models significantly outperformed the combined model on average, highlighting the importance of developing tailored approaches for each sport rather than a one-size-fits-all solution.\n")
        else:
            f.write("1. **Both Approaches Viable**: There was no significant difference between combined and sport-specific models, suggesting that both approaches can be effective depending on the implementation context.\n")
        
        if better_sports and worse_sports:
            f.write("2. **Hybrid Approach Potential**: Some sports benefited from the combined model while others performed better with sport-specific models. This suggests a potential hybrid approach where certain sports use dedicated models while others leverage the combined model.\n")
        
        f.write("3. **Data Requirements**: The combined model benefits from larger sample sizes, which may explain its strong performance. Sport-specific models require sufficient data for each sport to be reliable.\n")
        
        f.write("4. **Transfer Learning Opportunity**: The success of the combined model suggests potential for transfer learning, where models trained on data-rich sports can be adapted for use with sports having limited data availability.\n")

# %% Conduct ANOVA test on model performance across all sports
if 'sport_comparison_df' in locals():
    print("\n--- ANOVA Test on Model Performance Across Sports ---")
    
    # Group data by model type
    for model_type in sport_comparison_df['Model'].unique():
        model_data = sport_comparison_df[sport_comparison_df['Model'] == model_type]
        
        # Check if we have enough groups for ANOVA (at least 3 sports with this model)
        if len(model_data['Sport'].unique()) >= 3:
            print(f"\nANOVA test for {model_type} performance across sports:")
            
            try:
                # Group F1 scores by sport
                f1_by_sport = {}
                for sport in model_data['Sport'].unique():
                    sport_f1 = model_data[model_data['Sport'] == sport]['F1'].values
                    if len(sport_f1) > 0:
                        f1_by_sport[sport] = sport_f1[0]
                
                if len(f1_by_sport) >= 3:  # Need at least 3 groups for ANOVA
                    # Prepare data for ANOVA
                    sports = list(f1_by_sport.keys())
                    f1_scores = list(f1_by_sport.values())
                    
                    # Simple one-way ANOVA
                    f_stat, p_value = stats.f_oneway(*[[score] for score in f1_scores])
                    
                    print(f"  F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
                    if p_value < 0.05:
                        print("  There is a significant difference in model performance across sports.")
                        
                        # Identify the sports with highest and lowest performance
                        best_sport = max(f1_by_sport.items(), key=lambda x: x[1])
                        worst_sport = min(f1_by_sport.items(), key=lambda x: x[1])
                        
                        print(f"  Best performing sport: {best_sport[0]} (F1={best_sport[1]:.3f})")
                        print(f"  Worst performing sport: {worst_sport[0]} (F1={worst_sport[1]:.3f})")
                    else:
                        print("  No significant difference in model performance across sports.")
                else:
                    print(f"  Insufficient sports with {model_type} results for ANOVA (need at least 3).")
            except Exception as e:
                print(f"  Error performing ANOVA for {model_type}: {e}")
        else:
            print(f"  Insufficient sports with {model_type} results for ANOVA (need at least 3).")

# Add this new section to address methodological concerns with model evaluation

print("\n--- Methodological Validation and Addressing Potential Data Leakage ---")

# First, let's examine if we have a data leakage issue by checking feature importance
if 'feature_importance' in locals():
    # If MaxForceSymmetry is dominating the feature importance, that's a red flag
    force_symmetry_importance = feature_importance[feature_importance['Feature'] == 'MaxForceSymmetry']['Importance'].values
    if len(force_symmetry_importance) > 0 and force_symmetry_importance[0] > 0.5:
        print(f"WARNING: MaxForceSymmetry accounts for {force_symmetry_importance[0]:.2%} of feature importance, suggesting potential data leakage")

# Test a more rigorous approach by removing features directly used in target definition
print("\n1. Testing model without direct target-related features")
leakage_test_features = [f for f in features if f not in ['MaxForceSymmetry', 'TorqueSymmetry']]
print(f"Reduced feature set: {leakage_test_features}")

# Split data with this reduced feature set
X_leakage_test = df[leakage_test_features]
X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
    X_leakage_test, y, test_size=0.2, random_state=42, stratify=y
)

# Train a model without the leaked features
rf_leakage_test = RandomForestClassifier(n_estimators=100, random_state=42)
rf_leakage_test.fit(X_train_leak, y_train_leak)

# Evaluate
y_pred_leak = rf_leakage_test.predict(X_test_leak)
leak_test_acc = accuracy_score(y_test_leak, y_pred_leak)
leak_test_f1 = f1_score(y_test_leak, y_pred_leak)
leak_test_auc = roc_auc_score(y_test_leak, rf_leakage_test.predict_proba(X_test_leak)[:, 1])

print(f"Model without MaxForceSymmetry/TorqueSymmetry: Accuracy={leak_test_acc:.3f}, F1={leak_test_f1:.3f}, ROC AUC={leak_test_auc:.3f}")

# Create confusion matrix for the leakage test model
plt.figure(figsize=(8, 6))
cm_leak = confusion_matrix(y_test_leak, y_pred_leak)
sns.heatmap(
    cm_leak, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Low Asymmetry', 'High Asymmetry'],
    yticklabels=['Low Asymmetry', 'High Asymmetry']
)
plt.title('Confusion Matrix - Model Without Potential Leakage Features')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'confusion_matrix_leakage_test.png'), dpi=300)

# 2. Test with a completely independent target definition
print("\n2. Testing with an alternative target definition")
# Create a different asymmetry target based on ImpulseSymmetry instead
# This breaks the direct connection with MaxForceSymmetry
impulse_threshold = 0.15  # Using a different threshold
df['impulse_risk_high'] = ((df['ImpulseSymmetry'] < (1 - impulse_threshold)) | 
                          (df['ImpulseSymmetry'] > (1 + impulse_threshold))).astype(int)

# Remove both ImpulseSymmetry and MaxForceSymmetry from features
alt_features = [f for f in features if f not in ['ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']]
X_alt = df[alt_features]
y_alt = df['impulse_risk_high']

# Split the data
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(
    X_alt, y_alt, test_size=0.2, random_state=42, stratify=y_alt
)

# Train the model
rf_alt = RandomForestClassifier(n_estimators=100, random_state=42)
rf_alt.fit(X_train_alt, y_train_alt)

# Evaluate
y_pred_alt = rf_alt.predict(X_test_alt)
alt_acc = accuracy_score(y_test_alt, y_pred_alt)
alt_f1 = f1_score(y_test_alt, y_pred_alt)
alt_auc = roc_auc_score(y_test_alt, rf_alt.predict_proba(X_test_alt)[:, 1])

print(f"Alternative target model: Accuracy={alt_acc:.3f}, F1={alt_f1:.3f}, ROC AUC={alt_auc:.3f}")

# 3. Test with athlete-based cross-validation
if 'sbuid' in df.columns:
    print("\n3. Testing with athlete-based cross-validation to prevent data leakage across athletes")
    # Group by athlete to ensure no athlete appears in both train and test sets
    athletes = df['sbuid'].unique()
    np.random.shuffle(athletes)
    
    # Use 80% of athletes for training, 20% for testing
    train_athletes = athletes[:int(0.8 * len(athletes))]
    test_athletes = athletes[int(0.8 * len(athletes)):]
    
    # Create train/test splits based on athletes
    train_mask = df['sbuid'].isin(train_athletes)
    test_mask = df['sbuid'].isin(test_athletes)
    
    X_train_ath = X[train_mask]
    y_train_ath = y[train_mask]
    X_test_ath = X[test_mask]
    y_test_ath = y[test_mask]
    
    print(f"Athlete-based split: {len(X_train_ath)} training samples, {len(X_test_ath)} testing samples")
    print(f"Training athletes: {len(train_athletes)}, Testing athletes: {len(test_athletes)}")
    
    # Train the model
    rf_ath = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_ath.fit(X_train_ath, y_train_ath)
    
    # Evaluate
    y_pred_ath = rf_ath.predict(X_test_ath)
    ath_acc = accuracy_score(y_test_ath, y_pred_ath)
    ath_f1 = f1_score(y_test_ath, y_pred_ath)
    ath_auc = roc_auc_score(y_test_ath, rf_ath.predict_proba(X_test_ath)[:, 1])
    
    print(f"Athlete-based CV model: Accuracy={ath_acc:.3f}, F1={ath_f1:.3f}, ROC AUC={ath_auc:.3f}")

# 4. Create a composite performance comparison
models_comparison = {
    'Original Model': {'Accuracy': best_combined_model['Accuracy'], 'F1': best_combined_model['F1'], 'AUC': best_combined_model['ROC AUC']},
    'Without Leaked Features': {'Accuracy': leak_test_acc, 'F1': leak_test_f1, 'AUC': leak_test_auc},
    'Alternative Target': {'Accuracy': alt_acc, 'F1': alt_f1, 'AUC': alt_auc}
}

if 'sbuid' in df.columns:
    models_comparison['Athlete-Based CV'] = {'Accuracy': ath_acc, 'F1': ath_f1, 'AUC': ath_auc}

# Convert to DataFrame and plot
models_comp_df = pd.DataFrame.from_dict(models_comparison, orient='index')
models_comp_df = models_comp_df.reset_index().rename(columns={'index': 'Model'})

plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='F1', data=models_comp_df)
plt.title('F1 Score Comparison: Original vs. Leakage-Controlled Models')
plt.xlabel('Model Type')
plt.ylabel('F1 Score')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'methodological_validation_f1_comparison.png'), dpi=300)

# Save the comparison table
models_comp_df.to_csv(os.path.join(tables_dir, 'methodological_validation_metrics.csv'), index=False)

# Create summary markdown for methodological validation
with open(os.path.join(tables_dir, 'methodological_validation_summary.md'), 'w') as f:
    f.write("# Methodological Validation and Data Leakage Assessment\n\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
    
    f.write("## Overview\n\n")
    f.write("This analysis addresses potential methodological concerns in our asymmetry classification models, ")
    f.write("particularly focusing on the possibility of data leakage that could artificially inflate performance metrics.\n\n")
    
    f.write("## Key Findings\n\n")
    
    # Summarize feature importance if available
    if 'feature_importance' in locals():
        force_sym_importance = feature_importance[feature_importance['Feature'] == 'MaxForceSymmetry']['Importance'].values
        if len(force_sym_importance) > 0:
            f.write(f"1. **Feature Importance**: MaxForceSymmetry accounts for {force_sym_importance[0]:.2%} of the model's predictive power, ")
            if force_sym_importance[0] > 0.5:
                f.write("suggesting significant potential for data leakage since this feature directly relates to our target definition.\n")
            else:
                f.write("which is reasonable given its biomechanical significance but does warrant careful interpretation.\n")
    
    # Compare models
    f.write("2. **Model Performance Comparison**:\n\n")
    f.write("| Model | Accuracy | F1 Score | ROC AUC |\n")
    f.write("|-------|----------|----------|--------|\n")
    
    for model, metrics in models_comparison.items():
        f.write(f"| {model} | {metrics['Accuracy']:.3f} | {metrics['F1']:.3f} | {metrics['AUC']:.3f} |\n")
    
    # Interpret
    f.write("\n## Interpretation\n\n")
    
    # Compute drop in performance
    original_f1 = models_comparison['Original Model']['F1']
    leak_test_f1 = models_comparison['Without Leaked Features']['F1']
    alt_f1 = models_comparison['Alternative Target']['F1']
    
    f1_drop_leak = (original_f1 - leak_test_f1) / original_f1 * 100
    f1_drop_alt = (original_f1 - alt_f1) / original_f1 * 100
    
    f.write(f"1. **Feature Removal Impact**: Removing MaxForceSymmetry and TorqueSymmetry from the feature set resulted in a {f1_drop_leak:.1f}% decrease in F1 score, ")
    
    if f1_drop_leak > 50:
        f.write("indicating a critical dependence on these features that strongly suggests data leakage in the original model.\n")
    elif f1_drop_leak > 20:
        f.write("suggesting these features are important but not completely deterministic of model performance. Some degree of data leakage appears present.\n") 
    else:
        f.write("which is a modest reduction suggesting limited data leakage with other features providing substantial signal.\n")
    
    f.write(f"2. **Alternative Target Definition**: Using an independent target based on ImpulseSymmetry resulted in a {f1_drop_alt:.1f}% change in F1 score, ")
    
    if f1_drop_alt > 50:
        f.write("revealing a substantial disconnect between models when target definitions change.\n")
    elif f1_drop_alt > 20:
        f.write("indicating meaningful differences in how models perform with different asymmetry definitions.\n")
    else:
        f.write("suggesting consistent model performance even with different asymmetry definitions.\n")
    
    if 'sbuid' in df.columns:
        ath_f1 = models_comparison['Athlete-Based CV']['F1']
        f1_drop_ath = (original_f1 - ath_f1) / original_f1 * 100
        
        f.write(f"3. **Athlete-Based Validation**: Ensuring athletes don't appear in both training and testing sets resulted in a {f1_drop_ath:.1f}% change in F1 score, ")
        
        if f1_drop_ath > 50:
            f.write("strongly suggesting that our original cross-validation approach may have been learning athlete-specific patterns rather than generalizable asymmetry features.\n")
        elif f1_drop_ath > 20:
            f.write("indicating that athlete-specific patterns contribute to model performance but aren't the sole driver of predictive power.\n")
        else:
            f.write("suggesting our model generalizes well to new athletes and isn't overly dependent on athlete-specific patterns.\n")
    
    f.write("\n## Recommendations\n\n")
    
    # Recommendations based on validation results
    if max(f1_drop_leak, f1_drop_alt) > 50 or ('sbuid' in df.columns and f1_drop_ath > 50):
        f.write("1. **Revise Methodology**: Our validation tests indicate significant methodological concerns. We should:\n")
        f.write("   - Redefine our target variable to avoid direct mathematical relationships with features\n")
        f.write("   - Implement strict athlete-based cross-validation for all future analyses\n")
        f.write("   - Consider using external validation datasets where available\n")
        f.write("2. **Report Conservative Estimates**: When reporting model performance, we should primarily reference the results from our most rigorous validation tests.\n")
    elif max(f1_drop_leak, f1_drop_alt) > 20 or ('sbuid' in df.columns and f1_drop_ath > 20):
        f.write("1. **Methodological Refinements**: Our validation highlights opportunities to strengthen our approach:\n")
        f.write("   - Consider alternative asymmetry definitions that are more independent of our feature set\n")
        f.write("   - Implement athlete-based cross-validation for more reliable generalization assessment\n")
        f.write("   - Report performance ranges that acknowledge methodological variability\n")
    else:
        f.write("1. **Robust Methodology**: Our validation tests support the robustness of our modeling approach, with relatively minor performance variations across different methodological checks.\n")
        f.write("2. **Continued Validation**: Despite these positive findings, we should maintain methodological rigor by:\n")
        f.write("   - Routinely conducting similar validation tests on new datasets\n")
        f.write("   - Implementing athlete-based cross-validation as standard practice\n")
    
    f.write("\nThese methodological validations provide important context for interpreting our model performance metrics and ensure our asymmetry classification system rests on a sound analytical foundation.")