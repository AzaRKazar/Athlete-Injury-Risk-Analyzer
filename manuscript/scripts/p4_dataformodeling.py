# %%
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# Setting paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../../data')
output_dir = data_dir
os.makedirs(output_dir, exist_ok=True)

# List all preprocessed vald data files for different sports
preprocessed_data_files = glob.glob(os.path.join(data_dir, 'preprocessed_vald_data_*.csv'))
combined_preprocessed_file = os.path.join(data_dir, 'preprocessed_vald_data.csv')

if os.path.exists(combined_preprocessed_file):
    preprocessed_data_files.append(combined_preprocessed_file)

print(f"Found {len(preprocessed_data_files)} preprocessed data files to split")

# Set the cutoff date for train/test split (using January 1, 2025 as the cutoff)
cutoff_date = '2025-01-01'

# Process each sport's preprocessed data file individually
for file_path in preprocessed_data_files:
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")
    
    # Read the preprocessed data
    vald_data = pd.read_csv(file_path)
    
    # Convert date column to datetime for comparison
    vald_data['testDateUtc'] = pd.to_datetime(vald_data['testDateUtc'])
    
    # Split data into model data (before cutoff) and unseen data (after cutoff)
    model_data = vald_data[vald_data['testDateUtc'] < cutoff_date]
    unseen_data = vald_data[vald_data['testDateUtc'] >= cutoff_date]
    
    # Convert dates back to string format if needed
    model_data['testDateUtc'] = model_data['testDateUtc'].dt.strftime('%Y-%m-%d')
    unseen_data['testDateUtc'] = unseen_data['testDateUtc'].dt.strftime('%Y-%m-%d')
    
    # Save split datasets
    if file_name != 'preprocessed_vald_data.csv':
        sport_name = file_name.replace('preprocessed_vald_data_', '').replace('.csv', '')
        
        # Save sport-specific model data
        model_output_path = os.path.join(output_dir, f'model_data_{sport_name}.csv')
        model_data.to_csv(model_output_path, index=False)
        print(f"Saved model data for {sport_name} to {model_output_path}")
        
        # Save sport-specific unseen data
        unseen_output_path = os.path.join(output_dir, f'unseen_data_{sport_name}.csv')
        unseen_data.to_csv(unseen_output_path, index=False)
        print(f"Saved unseen data for {sport_name} to {unseen_output_path}")
    else:
        # Save combined model data
        combined_model_path = os.path.join(output_dir, 'model_data.csv')
        model_data.to_csv(combined_model_path, index=False)
        print(f"Saved combined model data to {combined_model_path}")
        
        # Save combined unseen data
        combined_unseen_path = os.path.join(output_dir, 'unseen_data.csv')
        unseen_data.to_csv(combined_unseen_path, index=False)
        print(f"Saved combined unseen data to {combined_unseen_path}")

# Create summary of the data split
print("\nData Split Summary:")
total_model_records = 0
total_unseen_records = 0

# For combined data
if os.path.exists(os.path.join(output_dir, 'model_data.csv')):
    combined_model_data = pd.read_csv(os.path.join(output_dir, 'model_data.csv'))
    combined_unseen_data = pd.read_csv(os.path.join(output_dir, 'unseen_data.csv'))
    
    total_model_records = len(combined_model_data)
    total_unseen_records = len(combined_unseen_data)
    
    print(f"Combined Data - Model: {total_model_records} records, Unseen: {total_unseen_records} records")
    
    if 'sport' in combined_model_data.columns:
        sport_counts = combined_model_data.groupby('sport').size().reset_index(name='model_count')
        sport_counts_unseen = combined_unseen_data.groupby('sport').size().reset_index(name='unseen_count')
        
        # Merge the counts
        sport_summary = pd.merge(sport_counts, sport_counts_unseen, on='sport', how='outer').fillna(0)
        sport_summary[['model_count', 'unseen_count']] = sport_summary[['model_count', 'unseen_count']].astype(int)
        sport_summary['total'] = sport_summary['model_count'] + sport_summary['unseen_count']
        sport_summary['model_pct'] = (sport_summary['model_count'] / sport_summary['total'] * 100).round(1)
        
        # Sort by total count descending
        sport_summary = sport_summary.sort_values('total', ascending=False)
        
        print("\nSport-wise Split:")
        print(sport_summary[['sport', 'model_count', 'unseen_count', 'total', 'model_pct']])
        
        # Calculate total athletes by sport
        if 'sbuid' in combined_model_data.columns:
            print("\nUnique Athletes by Sport:")
            athlete_counts = combined_model_data.groupby('sport')['sbuid'].nunique().reset_index(name='athlete_count')
            print(athlete_counts.sort_values('athlete_count', ascending=False))

# Individual sport files summary
print("\nIndividual Sport Files Summary:")
sport_files = []
for sport_file in glob.glob(os.path.join(output_dir, 'model_data_*.csv')):
    sport_name = os.path.basename(sport_file).replace('model_data_', '').replace('.csv', '')
    model_data = pd.read_csv(sport_file)
    
    # Check for corresponding unseen data
    unseen_file = os.path.join(output_dir, f'unseen_data_{sport_name}.csv')
    unseen_count = 0
    if os.path.exists(unseen_file):
        unseen_data = pd.read_csv(unseen_file)
        unseen_count = len(unseen_data)
    
    # Count unique athletes if possible
    athlete_count = "N/A"
    if 'sbuid' in model_data.columns:
        athlete_count = model_data['sbuid'].nunique()
    
    sport_files.append({
        'sport': sport_name,
        'model_records': len(model_data),
        'unseen_records': unseen_count,
        'total_records': len(model_data) + unseen_count,
        'unique_athletes': athlete_count
    })

if sport_files:
    sport_summary_df = pd.DataFrame(sport_files)
    print(sport_summary_df.sort_values('total_records', ascending=False))

# Create a consolidated summary file
summary_path = os.path.join(output_dir, 'data_split_summary.csv')
if sport_files:
    pd.DataFrame(sport_files).to_csv(summary_path, index=False)
    print(f"\nSaved detailed data split summary to {summary_path}")