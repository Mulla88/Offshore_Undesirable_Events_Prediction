from prefect import task, flow, get_run_logger
import subprocess
import requests
import time
import os

# -------------------------------
# Helper Function: Start MLflow UI
# -------------------------------
def start_mlflow_ui():
    mlflow_ui_url = "http://127.0.0.1:5000"
    try:
        # Check if MLflow UI is already running
        response = requests.get(mlflow_ui_url)
        if response.status_code == 200:
            print("MLflow UI is already running.")
            return
    except requests.ConnectionError:
        print("MLflow UI is not running. Starting it now...")

    # Start MLflow UI
    mlflow_process = subprocess.Popen(["mlflow", "ui", "--host", "127.0.0.1", "--port", "5000"])

    # Wait until MLflow UI is up
    while True:
        try:
            response = requests.get(mlflow_ui_url)
            if response.status_code == 200:
                print("MLflow UI is up and running.")
                break
        except requests.ConnectionError:
            print("Waiting for MLflow UI to be available...")
            time.sleep(3)

    return mlflow_process

# -------------------------------
# Helper Function: Start Prefect Server
# -------------------------------
def start_prefect_server():
    # Start the Prefect server in the background on the specified host and port
    prefect_server_process = subprocess.Popen(["prefect", "server", "start", "--host", "127.0.0.1", "--port", "4200"])

    # Allow some time for the Prefect server to initialize
    time.sleep(10)

    # Set the Prefect server URL environment variable
    server_url = "http://127.0.0.1:4200/api"
    os.environ["PREFECT_API_URL"] = server_url

    # Continuously check if the server is reachable before proceeding
    while True:
        try:
            response = requests.get(server_url.replace("/api", ""))
            if response.status_code == 200:
                print("Prefect server is up and running.")
                break
        except requests.ConnectionError:
            print("Waiting for Prefect server to be available...")
            time.sleep(3)

    # Persist the server URL in Prefect configuration for CLI usage
    subprocess.run(["prefect", "config", "set", f"PREFECT_API_URL={server_url}"])
    return prefect_server_process

# -------------------------------
# Task: Run EDA with Streamlit
# -------------------------------
@task
def run_eda():
    logger = get_run_logger()
    logger.info("Starting Streamlit EDA app...")
    try:
        # Start the EDA Streamlit app
        subprocess.Popen(["streamlit", "run", "eda.py"])
        # Allow some time for Streamlit app to initialize
        time.sleep(5)
    except Exception as e:
        logger.error(f"Failed to start Streamlit EDA: {e}")

# -------------------------------
# Task: Train the Model
# -------------------------------
@task
def train_model():
    logger = get_run_logger()
    logger.info("Training the model with train.py...")
    try:
        # Run the training script
        subprocess.run(["python", "trainXGboost.py"], check=True)
        logger.info("Model training completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")

# -------------------------------
# Task: Deploy the Model
# -------------------------------
@task
def deploy_model():
    logger = get_run_logger()
    try:
        # Check if the FastAPI server is already running
        response = requests.get("http://0.0.0.0:8000")
        if response.status_code == 200:
            logger.info("FastAPI server is already running.")
            return
    except requests.ConnectionError:
        logger.info("Starting FastAPI server...")
        try:
            # Start the FastAPI server
            subprocess.Popen(["python", "predict_api.py"])
            logger.info("FastAPI server started successfully.")
        except Exception as e:
            logger.error(f"Failed to start FastAPI server: {e}")

# -------------------------------
# Flow: Model Training and Deployment
# -------------------------------
@flow(name="Model_Training_and_Deployment")
def main_flow():
    # Run EDA and model training in parallel
    eda_task = run_eda.submit()
    train_task = train_model.submit()

    # Wait for the training to complete before deploying the model
    train_task.result()
    deploy_model()

# -------------------------------
# Main Entry Point
# -------------------------------
if __name__ == "__main__":
    # Start MLflow UI
    mlflow_process = start_mlflow_ui()

    # Start the local Prefect server
    prefect_server_process = start_prefect_server()
    try:
        # Execute the main flow
        main_flow()
        # Note: No termination here to allow Prefect server, FastAPI, and MLflow UI to continue running
    except KeyboardInterrupt:
        print("Process interrupted. Exiting gracefully.")
        # Optionally stop the Prefect server if needed
        prefect_server_process.terminate()
        print("Prefect server stopped.")
        # Optionally stop MLflow UI if needed
        if mlflow_process:
            mlflow_process.terminate()
            print("MLflow UI stopped.")
