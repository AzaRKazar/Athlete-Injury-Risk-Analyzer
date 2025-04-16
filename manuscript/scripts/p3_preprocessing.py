# %%
import pandas as pd
import numpy as np
import os
import glob

# Setting paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../../data')
output_dir = data_dir
os.makedirs(output_dir, exist_ok=True)

# List all pivot vald data files for different sports
pivot_data_files = glob.glob(os.path.join(data_dir, 'pivot_vald_data_*.csv'))
combined_pivot_file = os.path.join(data_dir, 'pivot_vald_data.csv')

if os.path.exists(combined_pivot_file):
    pivot_data_files.append(combined_pivot_file)

print(f"Found {len(pivot_data_files)} pivot data files to process")

# Define preprocessing function
def preprocess_data(data):
    # Drop unnecessary columns if they exist
    columns_to_drop = ['leftCalibration', 'rightCalibration', 'leftRepetitions', 'rightRepetitions']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)
    
    # Adding new columns for each symmetry ratio
    data['ForceSymmetry'] = data['leftAvgForce'] / data['rightAvgForce']
    data['ImpulseSymmetry'] = data['leftImpulse'] / data['rightImpulse']
    data['MaxForceSymmetry'] = data['leftMaxForce'] / data['rightMaxForce']
    data['TorqueSymmetry'] = data['leftTorque'] / data['rightTorque']
    
    # Replace infinity values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Remove duplicate rows
    data.drop_duplicates(inplace=True)
    
    # Remove rows where all relevant metrics are zero
    metric_columns = ['leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque', 
                      'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque']
    data = data[~(data[metric_columns].eq(0).any(axis=1))]
    
    # Define symmetry columns for outlier treatment
    symmetry_columns = ['ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']
    
    # Outlier Treatment using IQR method
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    # Apply the function to symmetry columns
    for col in symmetry_columns:
        data[col] = cap_outliers(data[col])
    
    # Convert date column to datetime format
    data["testDateUtc"] = pd.to_datetime(data["testDateUtc"]).dt.date
    
    return data

# Process each sport pivot file individually
sport_dfs = []
for file_path in pivot_data_files:
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")
    
    # Read the pivot data
    vald_data = pd.read_csv(file_path)
    
    # Apply preprocessing
    processed_data = preprocess_data(vald_data)
    
    # Save individual sport processed data if it's not the combined file
    if file_name != 'pivot_vald_data.csv':
        sport_name = file_name.replace('pivot_vald_data_', '').replace('.csv', '')
        sport_output_path = os.path.join(output_dir, f'preprocessed_vald_data_{sport_name}.csv')
        processed_data.to_csv(sport_output_path, index=False)
        print(f"Saved preprocessed data for {sport_name} to {sport_output_path}")
        
        # Add to list for combined dataset statistics
        sport_dfs.append(processed_data)
    else:
        # Save combined preprocessed data
        combined_output_path = os.path.join(output_dir, 'preprocessed_vald_data.csv')
        processed_data.to_csv(combined_output_path, index=False)
        print(f"Saved combined preprocessed data to {combined_output_path}")

# Display summary statistics for each sport
if sport_dfs:
    print("\nData Summary by Sport:")
    for df in sport_dfs:
        sport = df['sport'].iloc[0] if 'sport' in df.columns else "Unknown"
        print(f"\n{sport} - {len(df)} records")
        print(df[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']].describe())

# %%
# Display combined data summary if it exists
if os.path.exists(os.path.join(output_dir, 'preprocessed_vald_data.csv')):
    combined_data = pd.read_csv(os.path.join(output_dir, 'preprocessed_vald_data.csv'))
    print("\nCombined Dataset Summary:")
    if 'sport' in combined_data.columns:
        sport_counts = combined_data.groupby('sport').size().reset_index(name='count')
        print(sport_counts)
    print(combined_data[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']].describe())
    