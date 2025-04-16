# %%
import pandas as pd
import numpy as np


# %%
vald_data = pd.read_csv('data/raw_vald_data.csv')

# %%
pivot_data=vald_data.pivot_table(
    index=['sbuid', 'testDateUtc'],
    columns='metric', 
    values='value').reset_index()
pivot_data

# %%
pivot_data.to_csv('data/pivot_vald_data.csv', index=False)


