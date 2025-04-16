# %%
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from sqlalchemy import create_engine
import os.path

# %%
# Retrieve credentials
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

print("Credentials loaded from .env!")
print(f"MYSQL_USER: {MYSQL_USER}")
print(f"MYSQL_PASSWORD: {MYSQL_PASSWORD}")
print(f"MYSQL_HOST: {MYSQL_HOST}")
print(f"MYSQL_DATABASE: {MYSQL_DATABASE}")

# %%
# Create the connection URL
connection_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"

# %%
# Fetch data
def fetch_data(query):
    try:
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            data = pd.read_sql(query, connection)
        print("Data fetched successfully!")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Query the database
query = "SELECT * FROM sbu_athletics.vald;"  # Replace with your table name
data = fetch_data(query)

if data is not None:
    print(data.head())

# %%
# Create a 'manuscript/data' directory if it doesn't exist
data_dir = 'manuscript/data'
os.makedirs(data_dir, exist_ok=True)

# %%
# Instead of filtering for just Football, include multiple sports
# Option 1: Get all sports available
all_sports = data['sbu_sport'].unique()
print(f"Available sports: {all_sports}")
print(f"Length of available sports: {len(all_sports)}")
print(f"Length of data: {len(data)}")
## print group-by and count rows by sport
sport_counts = data.groupby('sbu_sport').size().reset_index(name='record_count')
print(sport_counts)

# Option 2: Select specific sports (modify as needed)
selected_sports = all_sports
# Filter to include only sports that exist in the data
selected_sports = [sport for sport in selected_sports if sport in all_sports]
print(f"Selected sports for analysis: {selected_sports}")

# %%
# Extract data for each sport and save separately
for sport in selected_sports:
    sport_data = data.loc[data['sbu_sport'] == sport, ['sbuid', 'testDateUtc', 'metric', 'value', 'sbu_sport']]
    
    # Save sport-specific file
    try:
        sport_filename = f"{data_dir}/raw_vald_data_{sport.lower()}.csv"
        sport_data.to_csv(sport_filename, index=False)
        print(f"Saved {sport} data with {len(sport_data)} records to {sport_filename}")
        
        # Display sample
        print(f"\nSample data for {sport}:")
        print(sport_data.head())
    except Exception as e:
        print(f"Error saving {sport} data: {e}")

# %%
# Also create a combined dataset with all selected sports
combined_data = data.loc[data['sbu_sport'].isin(selected_sports), 
                         ['sbuid', 'testDateUtc', 'metric', 'value', 'sbu_sport']]

# Save combined dataset
combined_filename = f"{data_dir}/raw_vald_data_combined.csv"
combined_data.to_csv(combined_filename, index=False)
print(f"\nSaved combined data with {len(combined_data)} records to {combined_filename}")

# %%
# Create a summary of records per sport
sport_summary = combined_data.groupby('sbu_sport').size().reset_index(name='record_count')
sport_summary_filename = f"{data_dir}/sport_data_summary.csv"
sport_summary.to_csv(sport_summary_filename, index=False)
print(f"\nSport data summary:")
print(sport_summary)

# %%
# For backward compatibility, also save the Football-only dataset in the original location
football_data = data.loc[data['sbu_sport'] == 'Football', ['sbuid', 'testDateUtc', 'metric', 'value']]
football_data.to_csv('data/raw_vald_data.csv', index=False)
print(f"\nSaved football-only data to 'data/raw_vald_data.csv' for backward compatibility")