import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained models (update paths with your saved model files)
rf_model_1 = joblib.load("rf_model_smote.pkl")
rf_model_2 = joblib.load("rf_model_no_balancing.pkl")
rf_model_3 = joblib.load("rf_model_smoteenn.pkl")
rf_model_4 = joblib.load("rf_model_class_weights.pkl")

# Define risk category mapping
risk_mapping_inverse = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

# App title
st.title("Athlete Injury Risk Analyzer")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data:")
    st.write(data.head())

    # Check required columns
    required_columns = [
        'leftAvgForce', 'leftImpulse', 'leftMaxForce', 'leftTorque',
        'rightAvgForce', 'rightImpulse', 'rightMaxForce', 'rightTorque'
    ]
    if all(col in data.columns for col in required_columns):
        # Calculate symmetry metrics
        data['ForceSymmetry'] = data['leftAvgForce'] / data['rightAvgForce']
        data['ImpulseSymmetry'] = data['leftImpulse'] / data['rightImpulse']
        data['MaxForceSymmetry'] = data['leftMaxForce'] / data['rightMaxForce']
        data['TorqueSymmetry'] = data['leftTorque'] / data['rightTorque']

        # Handle infinite or missing values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        st.write("### Processed Data with Symmetry Metrics:")
        st.write(data[['ForceSymmetry', 'ImpulseSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']].head())

        # Prepare features for prediction
        X_unseen = data[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']]

        # Predict risk categories using all models
        unseen_predictions = {}
        for model_name, model in zip(
            ['Model 1 (SMOTE)', 'Model 2 (No Balancing)', 'Model 3 (SMOTEENN)', 'Model 4 (Class Weights)'],
            [rf_model_1, rf_model_2, rf_model_3, rf_model_4]
        ):
            unseen_predictions[model_name] = model.predict(X_unseen)

        # Add predictions to unseen data
        for model_name in unseen_predictions:
            # Convert predictions (NumPy array) to pandas Series and map labels
            data[model_name] = pd.Series(unseen_predictions[model_name]).map(risk_mapping_inverse)

        st.write("### Predictions:")
        st.write(data)

        # Download option for predictions
        csv = data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    else:
        st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.write("Developed by [AzaR Kazar](https://github.com/AzaRKazar/Athlete-Injury-Risk-Analyzer)")
