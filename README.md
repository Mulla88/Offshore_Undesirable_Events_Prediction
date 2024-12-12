# Petrobras Anomaly Detection Project

This project is designed to detect and classify undesirable events in offshore oil wells using the **Petrobras 3W dataset**. The solution leverages Python scripts for data exploration, model training, real-time prediction, and workflow automation. The core model used is **XGBoost**, enhanced with **rolling window methods** for preserving temporal integrity.

## **Project Overview**

The Petrobras 3W dataset contains time-series data representing different types of undesirable events such as:

- **Abrupt Increase of Basic Sediment and Water (BSW)**
- **Spurious Closure of Downhole Safety Valve**
- **Severe Slugging**
- **Flow Instability**
- **Rapid Productivity Loss**
- **Quick Restriction in Production Choke**
- **Scaling in Production Choke**
- **Hydrate Formation in Production Lines**
- **Hydrate Formation in Service Lines**

The project workflow is streamlined through the following Python scripts:

1. **`eda.py`**:  
   Streamlit-based Exploratory Data Analysis (EDA) tool to visualize the dataset and identify trends.

2. **`train.py`**:  
   Preprocesses data using rolling window methods and trains the XGBoost model.

3. **`predict_api.py`**:  
   FastAPI-based RESTful service for real-time anomaly detection.

4. **`workflow.py`**:  
   Prefect automation script to run EDA, train the model, and deploy the prediction API.

5. **`real_time_prediction.py`**:  
   Simulates real-time data streaming and anomaly detection by calling the prediction API.

## **Setup Instructions**

Follow these steps to get the project up and running on your local machine:

### **1. Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/petrobras_anomaly_detection.git
cd petrobras_anomaly_detection
```

### **2. Install Dependencies**

Ensure you have Python installed, then install the required libraries:

```bash
pip install -r requirements.txt
```

### **3: Run the Workflow**

Execute the workflow.py script to automate EDA, model training, and API deployment:

```bash
python workflow.py
```
This script performs the following tasks:

Launches the EDA tool.
Preprocesses the data.
Trains the XGBoost model.
Deploys the prediction API at http://localhost:8000/docs.

### **4: Simulate Real-Time Data**

To simulate real-time data streaming and anomaly detection, run:

```bash
python real_time_prediction.py
```

This script sends data to the prediction API in real-time, mimicking the flow of sensor data from offshore wells.