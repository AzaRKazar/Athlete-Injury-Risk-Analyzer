# %%
import pandas as pd
import numpy as np
import os
import glob

# Setting paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
output_dir = os.path.join(script_dir, '../../data')
os.makedirs(output_dir, exist_ok=True)

# List all raw vald data files for different sports
raw_data_files = glob.glob(os.path.join(data_dir, 'raw_vald_data_*.csv'))
print(f"Found {len(raw_data_files)} raw data files: {[os.path.basename(f) for f in raw_data_files]}")

# Process each sport individually
sport_dfs = []
for file_path in raw_data_files:
    sport_name = os.path.basename(file_path).replace('raw_vald_data_', '').replace('.csv', '')
    print(f"Processing {sport_name}...")
    
    # Read the raw data
    vald_data = pd.read_csv(file_path)
    
    # Create pivot table
    pivot_data = vald_data.pivot_table(
        index=['sbuid', 'testDateUtc'],
        columns='metric', 
        values='value'
    ).reset_index()
    
    # Add sport column
    pivot_data['sport'] = sport_name
    
    # Save individual sport pivot data
    sport_output_path = os.path.join(output_dir, f'pivot_vald_data_{sport_name}.csv')
    pivot_data.to_csv(sport_output_path, index=False)
    print(f"Saved {sport_name} pivot data to {sport_output_path}")
    
    # Add to list for combined dataset
    sport_dfs.append(pivot_data)

# Create combined dataset with all sports
combined_pivot_data = pd.concat(sport_dfs, ignore_index=True)
combined_output_path = os.path.join(output_dir, 'pivot_vald_data.csv')
combined_pivot_data.to_csv(combined_output_path, index=False)
print(f"Saved combined pivot data to {combined_output_path}")

# Display data summary
print("\nData Summary by Sport:")
sport_summary = combined_pivot_data.groupby('sport').size().reset_index(name='count')
print(sport_summary)

# %%
combined_pivot_data.head()