import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc

# Load the best model from MLflow
model_uri = 'runs:/7429131a01464de68dc4aa191c18bcd3/RF_SMOTEENN'
best_model = mlflow.pyfunc.load_model(model_uri)

# Risk Mapping
risk_mapping_inverse = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

# Streamlit UI
st.title("Athlete Injury Risk Analyzer")

# Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.write(data.head())

    # Validate required columns
    required_columns = ['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']
    if all(col in data.columns for col in required_columns):

        # Prepare features
        X_unseen = data[required_columns]

        # Make predictions
        predictions = best_model.predict(X_unseen)

        # Convert NumPy array to Pandas Series before mapping
        data['Predicted Risk'] = pd.Series(predictions).map(risk_mapping_inverse)

        st.write("### Predictions:")
        st.write(data)

        # Allow users to download results
        csv = data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="predicted_risks.csv", mime="text/csv")

    else:
        st.error(f"Uploaded file must contain the following columns: {', '.join(required_columns)}")
else:
    st.info("Upload a CSV file to get predictions.")

st.write("Developed by [AzaR Kazar](https://github.com/AzaRKazar/Athlete-Injury-Risk-Analyzer)")
