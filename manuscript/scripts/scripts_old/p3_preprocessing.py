# %%
import pandas as pd
import numpy as np

# %%
vald_data = pd.read_csv('data/pivot_vald_data.csv')

# %%
vald_data

# %%
vald_data.drop(columns=['leftCalibration','rightCalibration','leftRepetitions','rightRepetitions'], inplace=True)

# %%
vald_data

# %%
data=vald_data
# Adding new columns for  each symmetry ratio
data['ForceSymmetry'] = data['leftAvgForce'] / data['rightAvgForce']
data['ImpulseSymmetry'] = data['leftImpulse'] / data['rightImpulse']
data['MaxForceSymmetry'] = data['leftMaxForce'] / data['rightMaxForce']
data['TorqueSymmetry'] = data['leftTorque'] / data['rightTorque']

# %%
data[["ForceSymmetry","ImpulseSymmetry","MaxForceSymmetry","TorqueSymmetry"]].describe()

# %%
# Replace infinity values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# %%

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# %%

# Remove rows where all relevant metrics are zero
metric_columns = ['leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque', 
                  'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque']
data = data[~(data[metric_columns].eq(0).any(axis=1))]

# %%
# data_dupliate=data.copy()
data.describe()

# %%
symmetry_columns = ['ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']
# Step 4: Outlier Treatment
# Define a function to cap outliers based on the IQR method
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

# %%
data.describe()

# %%
data[symmetry_columns].describe()

# %%
data["testDateUtc"] = pd.to_datetime(data["testDateUtc"]).dt.date

# %%
data.to_csv('data/preprocessed_vald_data.csv', index=False)


