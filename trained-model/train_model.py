import mlflow.pyfunc
import pandas as pd
import os
# Define the model URI from MLflow
model_uri = os.getenv('MODELURI')  # Replace with your actual MODELURI

# Load the trained model
best_model = mlflow.pyfunc.load_model(model_uri)

# Example: Loading unseen data
unseen_data = pd.read_csv("unseen_athlete_data.csv")  # Replace with actual unseen dataset

# Ensure the unseen data contains the required symmetry metrics
X_unseen = unseen_data[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']]

# Make predictions
predictions = best_model.predict(X_unseen)

# Mapping predictions back to Risk Categories
risk_mapping_inverse = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
unseen_data['Predicted Risk'] = predictions.map(risk_mapping_inverse)

# Save the predictions to a new CSV file
unseen_data.to_csv("predicted_risks.csv", index=False)

print("Predictions saved to predicted_risks.csv")
