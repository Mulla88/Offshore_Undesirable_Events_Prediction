# %%
# -------------------------------
# Imports and Setup
# -------------------------------
# The following libraries are used for data manipulation, machine learning, model training
# and experiment tracking. Make sure they are installed in your environment before running this code.

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb
import mlflow
from collections import defaultdict


# %%
# -------------------------------
# Configuration
# -------------------------------
# This section defines constants and configurations for the data processing and training steps.

num_files_per_event = 500 # Number of files to process per event

# A human-readable mapping of class labels to event names
event_names = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line"
}

# Mapping of original class labels in the input files to the desired normalized class labels.
# Each file will map original classes to a unified set of classes defined above.
event_files = {
    "combined_data_event1.parquet": {0: 0, 101: 1, 1: 99},
    "combined_data_event2.parquet": {0: 0, 102: 2, 2: 99},
    "combined_data_event5.parquet": {0: 0, 105: 3, 5: 99},
    "combined_data_event6.parquet": {0: 0, 106: 4, 6: 99},
    "combined_data_event7.parquet": {0: 0, 107: 5, 7: 99},
    "combined_data_event8.parquet": {0: 0, 108: 6, 8: 99},
    "combined_data_event9.parquet": {0: 0, 109: 7, 9: 99}
}


# %%
# -------------------------------
# Data Loading and Processing
# -------------------------------
# This section reads event files, applies class mappings, samples a subset of files per event (if desired),
# and concatenates them into a single DataFrame for further processing.

data_frames = []

for file, class_map in event_files.items():
    # Load the event file from parquet format
    df = pd.read_parquet(file)


    # Print null percentage for each column
    print(f"Null Percentage for {file}:")
    print(df.isnull().mean() * 100) 

    # Forward/Backward fill class labels for each filename group and then apply mapping
    df['class'] = df.groupby('filename')['class'].ffill().bfill()
    df['class'] = df['class'].replace(class_map)

    # Remove intermediate classes (labeled 99) that are not desired in final output
    df = df[df['class'] != 99]

    # Add a unique prefix to filename based on event, ensuring distinct naming across events
    event_prefix = file.split("_")[-1].split(".")[0]
    df['filename'] = event_prefix + "_" + df['filename']

    # Only keep the data starting with event2

    #df = df[df['filename'].str.startswith('event6_')]

    print(df.head())
    
    # Randomly sample a fixed number of files per event if desired
    if num_files_per_event is not None:
        unique_filenames = df['filename'].unique()
        max_files_to_sample = min(num_files_per_event, len(unique_filenames))
        sampled_filenames = np.random.choice(unique_filenames, size=max_files_to_sample, replace=False)
        df = df[df['filename'].isin(sampled_filenames)]

    # Append the processed DataFrame
    data_frames.append(df)

# Concatenate all DataFrames into a single large DataFrame
# and drop columns that are entirely NaN
df_gru = pd.concat(data_frames, ignore_index=True).dropna(axis=1, how='all')


# %%
# -------------------------------
# Rename Columns for Clarity
# -------------------------------
# Rename columns to more meaningful names for clarity.

df_gru.rename(columns={
    # Valve positions and states
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

    # Pressure readings
    'P-ANULAR': 'Annulus_Pressure_Pa',
    'P-JUS-CKGL': 'GLCK_Downstream_Pressure_Pa',
    'P-JUS-CKP': 'PCK_Downstream_Pressure_Pa',
    'P-MON-CKP': 'PCK_Upstream_Pressure_Pa',
    'P-PDG': 'PDG_Pressure_Pa',
    'P-TPT': 'TPT_Pressure_Pa',

    # Flow rate and temperature readings
    'QGL': 'Gas_Lift_Flow_Rate_m3_per_s',
    'T-JUS-CKP': 'PCK_Downstream_Temperature_C',
    'T-MON-CKP': 'PCK_Upstream_Temperature_C',
    'T-PDG': 'PDG_Temperature_C',
    'T-TPT': 'TPT_Temperature_C'
}, inplace=True)


# %%
# -------------------------------
# Handle Missing Values
# -------------------------------
# Identify numeric columns, excluding 'class'. For each filename group, forward/backward fill
# missing values. If a column is entirely NaN for a file, fill it with 0.

all_numeric_columns = [col for col in df_gru.select_dtypes(include='number').columns if col != 'class']

for filename, group in df_gru.groupby('filename'):
    for column in all_numeric_columns:
        if group[column].isnull().all():
            # If a column is completely missing, fill with 0
            df_gru.loc[group.index, column] = 0
        elif group[column].isnull().any():
            # Otherwise, forward fill and backward fill missing values
            df_gru.loc[group.index, column] = group[column].ffill().bfill()

# At this point, no missing values should remain in these numeric columns.
assert not df_gru[all_numeric_columns].isnull().values.any(), "Data contains missing values after processing."


# %%
# -------------------------------
# Apply Window Functions
# -------------------------------
# For continuous variables, we apply moving averages and differences over a specified window.
# This can help the model capture trends and changes over time.

window_columns = [col for col in all_numeric_columns if not col.endswith('_State')]
window_size = 60

for column in window_columns:
    # Compute a moving average over 'window_size' time steps
    df_gru[f'{column}_moving_avg'] = df_gru.groupby('filename')[column].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    # Compute a difference over 'window_size' time steps (lagged difference)
    df_gru[f'{column}_moving_diff'] = df_gru.groupby('filename')[column].transform(
        lambda x: x.diff(periods=window_size).fillna(0)
    )

# Update the list of numeric columns after adding new features
# Exclude 'state' from numeric_columns
numeric_columns = [col for col in df_gru.select_dtypes(include='number').columns if col != 'class' and col != 'state']

# Number of classes in the dataset
num_classes = df_gru['class'].nunique()


# %%
# -------------------------------
# Data Splitting by Event Groups
# -------------------------------
# Split data into training and testing sets based on event groups (filename prefixes).
# This ensures that events in the test set are truly unseen.

# Extract event prefixes (e.g., "event1", "event2", etc.)
filename_groups = df_gru['filename'].str.extract(r'^(event\d+)_')[0].unique()

train_filenames = []
test_filenames = []

for event_prefix in filename_groups:
    # Filter filenames for the current event prefix
    event_filenames = df_gru[df_gru['filename'].str.startswith(event_prefix)]['filename'].unique()
    
    # Group filenames by category (WELL, DRAWN, SIMULATED)
    category_groups = defaultdict(list)
    for filename in event_filenames:
        if 'WELL' in filename:
            category_groups['WELL'].append(filename)
        elif 'DRAWN' in filename:
            category_groups['DRAWN'].append(filename)
        elif 'SIMULATED' in filename:
            category_groups['SIMULATED'].append(filename)
    
    # Ensure each category is split such that at least one instance exists in both train and test
    train_split = []
    test_split = []

    for category, filenames in category_groups.items():
        if len(filenames) == 1:
            # If there's only one file, randomly assign it to train or test
            if np.random.rand() > 0.5:
                train_split.append(filenames[0])
            else:
                test_split.append(filenames[0])
        else:
            # Split the files ensuring at least one in each set
            split_train, split_test = train_test_split(filenames, test_size=0.2)
            train_split.extend(split_train)
            test_split.extend(split_test)

    train_filenames.extend(train_split)
    test_filenames.extend(test_split)

# Create separate DataFrames for training and testing
df_train = df_gru[df_gru['filename'].isin(train_filenames)]
df_test = df_gru[df_gru['filename'].isin(test_filenames)]

# print unique filenames exisit in df
print(f"Unique filenames in train: {df_train['filename'].unique()}")
print(f"Unique filenames in test: {df_test['filename'].unique()}")

# Convert to numeric arrays
X_train = df_train[numeric_columns].values.astype(np.float64)
y_train = df_train['class'].values
X_test = df_test[numeric_columns].values.astype(np.float64)
y_test = df_test['class'].values

# Scale features using RobustScaler (fit on train, apply on test)
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %%
# -------------------------------
# MLflow Experiment Setup and Model Training
# -------------------------------
# We use MLflow to track experiments, parameters, and metrics. We train an XGBoost model
# on the training set and log the model and performance metrics to MLflow.

if mlflow.active_run():
    mlflow.end_run()

mlflow.set_experiment("xgboost_gpu_multievent")

with mlflow.start_run() as run:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': len(event_names),
        'tree_method': 'gpu_hist',
        'eval_metric': 'mlogloss',
        'max_depth': 8,
        'learning_rate': 0.1
    }
    
    # Train the XGBoost model
    model = xgb.train(params, dtrain, num_boost_round=150)
    
    # Log the model to MLflow along with an input example
    mlflow.xgboost.log_model(model, "xgboost_model", input_example=X_train[:5])
    model_uri = f"runs:/{run.info.run_id}/xgboost_model"
    mlflow.register_model(model_uri=model_uri, name="xgboost_model")

    # Log the expected columns (features) for future reference
    with open("expected_columns.json", "w") as f:
        json.dump(numeric_columns, f)
    mlflow.log_artifact("expected_columns.json")

    # Save the scaler for future transformations and log it
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("scaler.pkl")

    # -------------------------------
    # Quick Feature Importance Logging
    # -------------------------------

    # Create a mapping of default XGBoost feature names to original feature names
    feature_mapping = {f"f{i}": col for i, col in enumerate(numeric_columns)}

    # Function to replace feature names in a DataFrame
    def replace_feature_names(df, mapping):
        df['Feature'] = df['Feature'].map(mapping)
        return df

    # Compute and log feature importance for different types: weight, gain, and cover
    importance_types = ['weight', 'gain', 'cover']

    for importance_type in importance_types:
        # Get feature importance from the model
        feature_importance = model.get_score(importance_type=importance_type)
        
        # Convert to DataFrame for easy visualization
        df_importance = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values(by='Importance', ascending=False)
        
        # Replace generic feature names with original names
        df_importance = replace_feature_names(df_importance, feature_mapping)

        # Plot the feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(df_importance['Feature'], df_importance['Importance'])
        plt.xlabel(importance_type.capitalize())
        plt.ylabel('Features')
        plt.title(f'XGBoost Feature Importance ({importance_type.capitalize()})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Log the plot to MLflow
        mlflow.log_figure(plt.gcf(), f"feature_importance_{importance_type}.png")
        plt.close()

        # Log the feature importance data as CSV directly to MLflow
        from io import StringIO
        csv_buffer = StringIO()
        df_importance.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), f"feature_importance_{importance_type}.csv")

    # -------------------------------
    # Evaluation on Test Filenames
    # -------------------------------

    # Initialize metrics
    validation_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'time_lag': []}

    for filename in test_filenames:
        # Filter data for the current test file
        mask = df_test['filename'] == filename
        X_test_filename = X_test[mask]
        y_true_filename = y_test[mask]

        # Generate predictions
        y_pred_probs = model.predict(xgb.DMatrix(X_test_filename))
        y_pred_filename = y_pred_probs.argmax(axis=1)

        # Compute metrics
        acc = accuracy_score(y_true_filename, y_pred_filename)
        f1 = f1_score(y_true_filename, y_pred_filename, average='weighted', zero_division=0)
        precision = precision_score(y_true_filename, y_pred_filename, average='weighted', zero_division=0)

        validation_metrics['accuracy'].append(acc)
        validation_metrics['f1'].append(f1)
        validation_metrics['precision'].append(precision)

        # -------------------------------
        # Compute Average Time Lag
        # -------------------------------
        # Calculate the average time difference where predictions and true labels differ
        mismatches = np.where(y_true_filename != y_pred_filename)[0]
        avg_time_lag = np.mean(mismatches) if len(mismatches) > 0 else 0
        validation_metrics['time_lag'].append(avg_time_lag)

        # Plot and log the True vs. Predicted classes for visualization
        plt.figure(figsize=(10, 4))
        plt.plot(y_true_filename, label='True Class')
        plt.plot(y_pred_filename, label='Predicted Class', linestyle='--')
        plt.xlabel('Time Steps')
        plt.ylabel('Class Label')
        plt.yticks(ticks=np.arange(len(event_names)), labels=list(event_names.values()), fontsize=8)
        plt.title(f'True vs Predicted Classes for {filename}')
        plt.legend()
        mlflow.log_figure(plt.gcf(), f"{filename}_plot.png")
        plt.close()

        # Compute and log confusion matrix if there are multiple classes
        if len(np.unique(y_true_filename)) > 1 and len(np.unique(y_pred_filename)) > 1:
            all_classes = sorted(np.unique(np.concatenate([y_true_filename, y_pred_filename])))
            unique_labels = [event_names[label] for label in all_classes]

            cm = confusion_matrix(y_true_filename, y_pred_filename, labels=all_classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
            plt.figure(figsize=(10, 8))
            disp.plot(cmap='Blues', xticks_rotation=45)
            plt.title(f'Confusion Matrix for {filename}')
            mlflow.log_figure(plt.gcf(), f"{filename}_confusion_matrix.png")
            plt.close('all')

    # -------------------------------
    # Log Average Metrics Across All Test Files
    # -------------------------------
    avg_accuracy = np.mean(validation_metrics['accuracy'])
    avg_f1 = np.mean(validation_metrics['f1'])
    avg_precision = np.mean(validation_metrics['precision'])
    avg_time_lag = np.mean(validation_metrics['time_lag'])

    mlflow.log_metric("test_avg_accuracy", avg_accuracy)
    mlflow.log_metric("test_avg_f1", avg_f1)
    mlflow.log_metric("test_avg_precision", avg_precision)
    mlflow.log_metric("test_avg_time_lag", avg_time_lag)

    # Print final evaluation metrics
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Time Lag: {avg_time_lag:.4f}")

print("XGBoost model training and testing completed with MLflow logging.")
