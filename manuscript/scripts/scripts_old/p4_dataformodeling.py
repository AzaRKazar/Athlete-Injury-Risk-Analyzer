# %%
import pandas as pd
import numpy as np  


# %%
vald_data = pd.read_csv('data/preprocessed_vald_data.csv')

# %%
vald_data.head()

# %%
model_data=vald_data[vald_data['testDateUtc']<'2025-01-01']
unseen_data=vald_data[vald_data['testDateUtc']>='2025-01-01']

# %%
model_data.to_csv('data/model_data.csv',index=False)
unseen_data.to_csv('data/unseen_data.csv',index=False)


