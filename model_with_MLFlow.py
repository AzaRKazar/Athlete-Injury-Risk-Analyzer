import os
import mlflow.pyfunc
import pandas as pd


# Get model URI from environment variable
model_uri = os.getenv("model_uri","models:/RF_SMOTEENN/Version 1")

if model_uri is None:
    raise ValueError("MLFLOW_MODEL_URI is not set! Please pass it as an argument or set it in GitHub Secrets.")

# Load the trained MLflow model
best_model = mlflow.pyfunc.load_model(model_uri)
print(f"Using registered model from: {model_uri}")

# Load unseen data and predict
unseen_data = pd.read_csv("data/unseen_athlete.csv")
X_unseen = unseen_data[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']]
predictions = best_model.predict(X_unseen)

# Save predictions
unseen_data["Predicted Risk"] = predictions
unseen_data.to_csv("predicted_risks.csv", index=False)
print("Predictions saved to predicted_risks.csv")
