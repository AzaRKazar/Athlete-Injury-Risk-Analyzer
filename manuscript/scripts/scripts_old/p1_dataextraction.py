# %%
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from sqlalchemy import create_engine


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
# Filter the relevant columns and rows
filtered_data = data.loc[data['sbu_sport'] == 'Football', ['sbuid', 'testDateUtc', 'metric', 'value']]

# Display the filtered data
print("Filtered Data:")
print(filtered_data.head())


# %%
filtered_data.to_csv('data/raw_vald_data.csv', index=False)


