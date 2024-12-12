import pandas as pd
import requests
import time
import numpy as np
from collections import deque

# -------------------------------
# Constants and API URLs
# -------------------------------
SCALE_URL = "http://localhost:8000/scale"
PREDICTION_URL = "http://localhost:8000/scale_and_predict"
COLUMN_URL = "http://localhost:8000/expected_columns"
REQUEST_DELAY = 0.1  # Delay between requests in seconds
WINDOW_SIZE = 90

# -------------------------------
# Helper Function: Trigger Alarm
# -------------------------------
def trigger_alarm(event_name):
    print(f"ALARM: Prediction indicates a {event_name} above threshold!\n")

# -------------------------------
# Fetch Expected Columns from the API
# -------------------------------
try:
    column_response = requests.get(COLUMN_URL)
    column_response.raise_for_status()
    expected_columns = column_response.json().get("expected_columns", [])
except requests.exceptions.RequestException as e:
    raise SystemExit(f"Error fetching expected columns: {e}")

# -------------------------------
# Load and Preprocess a Sample Data 
# -------------------------------
df = pd.read_parquet('SIMULATED_00007.parquet').dropna(axis=1, how='all')

# Rename columns for clarity (same as training)
df.rename(columns={
    'ABER-CKGL': 'GLCK_Open_Percent',
    'ABER-CKP': 'PCK_Open_Percent',
    'ESTADO-DHSV': 'Downhole_SV_State',
    'ESTADO-M1': 'Production_MV_State',
    'ESTADO-M2': 'Annulus_MV_State',
    'ESTADO-PXO': 'PigCrossoverV_State',
    'ESTADO-SDV-GL': 'Gas_Lift_SDV_State',
    'ESTADO-SDV-P': 'Production_SDV_State',
    'ESTADO-W1': 'Production_Wing_Valve_State',
    'ESTADO-W2': 'Annulus_Wing_Valve_State',
    'ESTADO-XO': 'CrossoverV_State',
    'P-ANULAR': 'Annulus_Pressure_Pa',
    'P-JUS-CKGL': 'GLCK_Downstream_Pressure_Pa',
    'P-JUS-CKP': 'PCK_Downstream_Pressure_Pa',
    'P-MON-CKP': 'PCK_Upstream_Pressure_Pa',
    'P-PDG': 'PDG_Pressure_Pa',
    'P-TPT': 'TPT_Pressure_Pa',
    'QGL': 'Gas_Lift_Flow_Rate_m3_per_s',
    'T-JUS-CKP': 'PCK_Downstream_Temperature_C',
    'T-MON-CKP': 'PCK_Upstream_Temperature_C',
    'T-PDG': 'PDG_Temperature_C',
    'T-TPT': 'TPT_Temperature_C'
}, inplace=True)

# Ensure expected columns exist, fill missing
df = df.reindex(columns=expected_columns, fill_value=0)

# Forward/backward fill missing values globally if any exist
if df.isnull().values.any():
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)

# Identify numeric columns (in training, these were all non-'class' numeric columns)
# Here we just take all numeric columns since we don't have 'class' now.
numeric_columns = [col for col in df.select_dtypes(include='number').columns]

# Identify window columns: those that are numeric and not ending with '_State', and not already rolling features
rolling_feature_suffixes = ['_moving_avg', '_moving_diff']
window_columns = [
    col for col in numeric_columns
    if not any(col.endswith(sfx) for sfx in rolling_feature_suffixes) and not col.endswith('_State')
]

# Initialize rolling buffers for each window column
rolling_buffers = {col: deque(maxlen=WINDOW_SIZE) for col in window_columns}

def apply_rolling_features(row):
    # For each window column, add the current value to the buffer
    for col in window_columns:
        rolling_buffers[col].append(row[col])

    # Compute the rolling features
    for col in window_columns:
        # moving_avg
        if len(rolling_buffers[col]) > 0:
            row[f'{col}_moving_avg'] = np.mean(rolling_buffers[col])
        else:
            row[f'{col}_moving_avg'] = row[col]

        # moving_diff
        if len(rolling_buffers[col]) == WINDOW_SIZE:
            row[f'{col}_moving_diff'] = rolling_buffers[col][-1] - rolling_buffers[col][0]
        else:
            row[f'{col}_moving_diff'] = 0

    return row

previous_class = None

# -------------------------------
# Real-Time Simulation
# -------------------------------
for index, original_row in df.iterrows():
    # Apply rolling features
    row = apply_rolling_features(original_row.copy())

    # Ensure all expected columns exist (including rolling ones). Fill missing if any.
    for col in expected_columns:
        if col not in row:
            row[col] = 0

    row_dict = row.to_dict()

    # Scale the input data using the `/scale` endpoint
    try:
        scale_response = requests.post(SCALE_URL, json={"data": [row_dict]})
        scale_response.raise_for_status()
        scaled_data = scale_response.json().get("scaled_data", [{}])[0]
        print(f"\nRow {index} - Scaled Values: {scaled_data}")
    except requests.exceptions.RequestException as e:
        print(f"Error with scaling request at index {index}: {e}")
        continue

    # Predict class probabilities using `/scale_and_predict` endpoint
    try:
        response = requests.post(PREDICTION_URL, json={"data": [row_dict]})
        response.raise_for_status()
        prediction_info = response.json()[0]

        class_index = prediction_info.get("predicted_class", -1)
        probabilities = prediction_info.get("probabilities", [])
        print(f"Predicted Class: {class_index}, Probabilities: {probabilities}")

        # If the class changes compared to the previous one, alert
        if previous_class is not None and class_index != previous_class:
            print(f"ALARM: Class change detected! Transition from Class {previous_class} to Class {class_index}")

        # Trigger alarm if predicted class is not 0 (assuming 0 is 'Normal')
        if class_index != 0:
            trigger_alarm(f"Class {class_index}")

        previous_class = class_index
    except requests.exceptions.RequestException as e:
        print(f"Error with prediction request at index {index}: {e}")

    # Simulate a delay to mimic real-time data streaming
    time.sleep(REQUEST_DELAY)
