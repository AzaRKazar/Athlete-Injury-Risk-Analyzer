# Define a new performance decline target
import pandas as pd
import numpy as np
import os
import glob

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))  # Go up two levels to get to project root
data_dir = os.path.join(project_root, 'data')
figures_dir = os.path.join(project_root, 'manuscript', 'figures')
tables_dir = os.path.join(project_root, 'manuscript', 'tables')
model_dir = os.path.join(project_root, 'trained-model')  # Add path to trained models directory

print(f"Project root: {project_root}")
print(f"Data directory: {data_dir}")
print(f"Models directory: {model_dir}")

# Create directories if they don't exist
for dir_path in [figures_dir, tables_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Find the model data files
model_data_files = glob.glob(os.path.join(data_dir, 'model_data*.csv'))
print(f"Found model data files: {model_data_files}")

if not model_data_files:
    print(f"No model data files found in {data_dir}")
    exit(1)

# Use the combined file or the first file found
model_data_path = next((f for f in model_data_files if 'combined' in f.lower()), model_data_files[0])
print(f"Using model data file: {model_data_path}")

# Load model data
df = pd.read_csv(model_data_path)
print(f"Loaded model data: {df.shape}")

# Check for required columns
required_columns = ['sbuid', 'testDateUtc', 'ForceSymmetry', 'MaxForceSymmetry', 'ImpulseSymmetry', 'TorqueSymmetry']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Missing required columns: {missing_columns}")
    # Try to adapt to available columns
    if 'sbuid' not in df.columns and 'athlete_id' in df.columns:
        print("Using 'athlete_id' instead of 'sbuid'")
        df['sbuid'] = df['athlete_id']
    
    if 'testDateUtc' not in df.columns and 'test_date' in df.columns:
        print("Using 'test_date' instead of 'testDateUtc'")
        df['testDateUtc'] = df['test_date']
    
    # Recheck missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Still missing required columns: {missing_columns}")
        print("Available columns:", df.columns.tolist())
        exit(1)

# Ensure date is in datetime format
if 'testDateUtc' in df.columns:
    try:
        df['testDateUtc'] = pd.to_datetime(df['testDateUtc'])
    except:
        print("Warning: Could not convert testDateUtc to datetime format")
        # Try with different formats
        try:
            df['testDateUtc'] = pd.to_datetime(df['testDateUtc'], format='%Y-%m-%d')
        except:
            try:
                df['testDateUtc'] = pd.to_datetime(df['testDateUtc'], format='%m/%d/%Y')
            except:
                print("Failed to parse dates. Using numeric days_since_first_test if available.")
                if 'days_since_first_test' in df.columns:
                    # Sort by this column instead
                    df['testDateUtc'] = df['days_since_first_test']

# Step 1: Sort data by athlete ID and test date
if 'sbuid' in df.columns and 'testDateUtc' in df.columns:
    print(f"Sorting data by athlete ID and test date")
    df = df.sort_values(['sbuid', 'testDateUtc'])
    
    # Step 2: Group by athlete and calculate changes between consecutive tests
    athlete_groups = []
    athlete_count = 0
    multi_test_athlete_count = 0
    
    for athlete_id, athlete_data in df.groupby('sbuid'):
        athlete_count += 1
        if len(athlete_data) >= 2:  # Athlete must have at least 2 tests
            multi_test_athlete_count += 1
            # Sort by date
            athlete_data = athlete_data.sort_values('testDateUtc')
            
            # Calculate changes in important metrics
            athlete_data['next_ForceSymmetry'] = athlete_data['ForceSymmetry'].shift(-1)
            athlete_data['next_MaxForceSymmetry'] = athlete_data['MaxForceSymmetry'].shift(-1)
            athlete_data['next_ImpulseSymmetry'] = athlete_data['ImpulseSymmetry'].shift(-1)
            athlete_data['next_TorqueSymmetry'] = athlete_data['TorqueSymmetry'].shift(-1)
            
            # Calculate absolute changes
            athlete_data['change_ForceSymmetry'] = abs(athlete_data['next_ForceSymmetry'] - athlete_data['ForceSymmetry'])
            athlete_data['change_MaxForceSymmetry'] = abs(athlete_data['next_MaxForceSymmetry'] - athlete_data['MaxForceSymmetry'])
            athlete_data['change_ImpulseSymmetry'] = abs(athlete_data['next_ImpulseSymmetry'] - athlete_data['ImpulseSymmetry'])
            athlete_data['change_TorqueSymmetry'] = abs(athlete_data['next_TorqueSymmetry'] - athlete_data['TorqueSymmetry'])
            
            # Calculate worsening flags (further from perfect symmetry of 1.0)
            athlete_data['worse_ForceSymmetry'] = np.where(
                abs(athlete_data['next_ForceSymmetry'] - 1.0) > abs(athlete_data['ForceSymmetry'] - 1.0),
                1, 0
            )
            athlete_data['worse_MaxForceSymmetry'] = np.where(
                abs(athlete_data['next_MaxForceSymmetry'] - 1.0) > abs(athlete_data['MaxForceSymmetry'] - 1.0),
                1, 0
            )
            athlete_data['worse_ImpulseSymmetry'] = np.where(
                abs(athlete_data['next_ImpulseSymmetry'] - 1.0) > abs(athlete_data['ImpulseSymmetry'] - 1.0),
                1, 0
            )
            athlete_data['worse_TorqueSymmetry'] = np.where(
                abs(athlete_data['next_TorqueSymmetry'] - 1.0) > abs(athlete_data['TorqueSymmetry'] - 1.0),
                1, 0
            )
            
            # Calculate time between tests if possible
            if isinstance(athlete_data['testDateUtc'].iloc[0], pd.Timestamp):
                athlete_data['days_to_next_test'] = (athlete_data['testDateUtc'].shift(-1) - athlete_data['testDateUtc']).dt.days
            else:
                # Numerical date representation 
                athlete_data['days_to_next_test'] = athlete_data['testDateUtc'].shift(-1) - athlete_data['testDateUtc']
            
            # Drop the last row as it doesn't have a next test
            athlete_data = athlete_data.iloc[:-1]
            
            athlete_groups.append(athlete_data)
    
    print(f"Found {athlete_count} total athletes")
    print(f"Found {multi_test_athlete_count} athletes with multiple tests")
    
    # Combine all athlete data
    if athlete_groups:
        longitudinal_df = pd.concat(athlete_groups)
        print(f"Created longitudinal dataset with {len(longitudinal_df)} records from {len(athlete_groups)} athletes")
        
        # Create target variable for performance decline
        # Option 1: Binary target - any worsening in symmetry
        longitudinal_df['any_symmetry_worse'] = (
            (longitudinal_df['worse_ForceSymmetry'] == 1) | 
            (longitudinal_df['worse_MaxForceSymmetry'] == 1) | 
            (longitudinal_df['worse_ImpulseSymmetry'] == 1) | 
            (longitudinal_df['worse_TorqueSymmetry'] == 1)
        ).astype(int)
        
        # Option 2: Binary target - significant worsening in symmetry (e.g., >10% change)
        significant_change_threshold = 0.10
        longitudinal_df['significant_symmetry_worse'] = (
            ((longitudinal_df['change_ForceSymmetry'] > significant_change_threshold) & (longitudinal_df['worse_ForceSymmetry'] == 1)) |
            ((longitudinal_df['change_MaxForceSymmetry'] > significant_change_threshold) & (longitudinal_df['worse_MaxForceSymmetry'] == 1)) |
            ((longitudinal_df['change_ImpulseSymmetry'] > significant_change_threshold) & (longitudinal_df['worse_ImpulseSymmetry'] == 1)) |
            ((longitudinal_df['change_TorqueSymmetry'] > significant_change_threshold) & (longitudinal_df['worse_TorqueSymmetry'] == 1))
        ).astype(int)
        
        # Option 3: Multi-class target based on number of metrics that worsened
        longitudinal_df['num_metrics_worse'] = (
            longitudinal_df['worse_ForceSymmetry'] + 
            longitudinal_df['worse_MaxForceSymmetry'] + 
            longitudinal_df['worse_ImpulseSymmetry'] + 
            longitudinal_df['worse_TorqueSymmetry']
        )
        
        # Create categorical version (0=none, 1=minimal, 2=moderate, 3-4=severe decline)
        longitudinal_df['decline_severity'] = pd.cut(
            longitudinal_df['num_metrics_worse'],
            bins=[-1, 0, 1, 2, 4],
            labels=['None', 'Minimal', 'Moderate', 'Severe']
        )
        
        # Save the longitudinal dataset with performance decline targets
        longitudinal_file = os.path.join(data_dir, 'longitudinal_model_data.csv')
        longitudinal_df.to_csv(longitudinal_file, index=False)
        print(f"Saved longitudinal dataset to {longitudinal_file}")
        
        # Print summary statistics
        print("\nPerformance Decline Target Distribution:")
        print(f"Any symmetry worse: {longitudinal_df['any_symmetry_worse'].mean():.1%} of athletes")
        print(f"Significant symmetry worse: {longitudinal_df['significant_symmetry_worse'].mean():.1%} of athletes")
        print("\nNumber of metrics worsened distribution:")
        print(longitudinal_df['num_metrics_worse'].value_counts().sort_index())
        print("\nDecline severity distribution:")
        print(longitudinal_df['decline_severity'].value_counts())
        
    else:
        print("No athletes with multiple tests found for longitudinal analysis")
else:
    print("Required columns 'sbuid' or 'testDateUtc' not found in the dataset")
    print("Available columns:", df.columns.tolist())