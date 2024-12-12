from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import json
import mlflow
from mlflow.tracking import MlflowClient
import xgboost as xgb

app = FastAPI()

# -----------------------------------
# Pydantic Model for Input Validation
# -----------------------------------
# Expects JSON input data with a 'data' field containing a list of records.
class InputData(BaseModel):
    data: list

# -----------------------------------
# Load Artifacts on Startup
# -----------------------------------
# On startup, we attempt to load:
# 1. The latest version of the XGBoost model from MLflow Model Registry.
# 2. The corresponding scaler used for preprocessing.
# 3. The expected columns for the model.
try:
    client = MlflowClient()
    model_name = "xgboost_model"
    
    # Retrieve the latest versions of the given model
    latest_version_info = client.get_latest_versions(name=model_name)
    if not latest_version_info:
        raise RuntimeError(f"No versions found for model '{model_name}'.")

    # Get the latest model version from the returned list
    latest_version = latest_version_info[-1].version

    # Construct the URI for the latest model version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.xgboost.load_model(model_uri)

    # Download the scaler artifact using the run_id associated with the latest model version
    run_id = latest_version_info[-1].run_id
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Download the expected columns artifact
    columns_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="expected_columns.json")
    with open(columns_path, "r") as f:
        expected_columns = json.load(f)

except Exception as e:
    raise RuntimeError(f"Error loading model or artifacts from MLflow: {e}")

# -----------------------------------
# API Endpoints
# -----------------------------------

@app.get("/")
async def read_root():
    """Root endpoint to verify the API is running."""
    return {"message": "Welcome to the Predict API. Visit /docs for API documentation."}

@app.get("/expected_columns")
async def get_expected_columns():
    """Retrieve the expected input columns for the model."""
    return {"expected_columns": expected_columns}

@app.post("/scale")
async def scale(input_data: InputData):
    """
    Scale input data according to the previously saved scaler.
    Does not perform any predictions, just returns the scaled values.
    """
    try:
        # Convert input data to a DataFrame and ensure columns match expected model input columns
        df = pd.DataFrame(input_data.data).reindex(columns=expected_columns, fill_value=0)

        # Scale the data using the loaded scaler
        df_scaled = scaler.transform(df.values)

        # Convert scaled data back to a list of records for the JSON response
        scaled_data = pd.DataFrame(df_scaled, columns=expected_columns).to_dict(orient="records")

        return {"scaled_data": scaled_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaling error: {e}")

@app.post("/scale_and_predict")
async def scale_and_predict(input_data: InputData):
    """
    Scale the input data and perform predictions using the loaded XGBoost model.
    Returns both predicted classes and the predicted probabilities.
    """
    try:
        # Convert input data to a DataFrame and ensure columns match expected model input columns
        df = pd.DataFrame(input_data.data).reindex(columns=expected_columns, fill_value=0)

        # Scale the data
        df_scaled = scaler.transform(df.values)

        # Convert scaled data into XGBoost DMatrix format
        dmat = xgb.DMatrix(df_scaled)

        # Predict probabilities for each class
        predicted_probs = model.predict(dmat)

        # Identify the predicted class (highest probability)
        predicted_classes = predicted_probs.argmax(axis=1)

        # Prepare the results, including probabilities
        results = [
            {
                "predicted_class": int(pred_class),
                "probabilities": prob.tolist()
            }
            for pred_class, prob in zip(predicted_classes, predicted_probs)
        ]

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
