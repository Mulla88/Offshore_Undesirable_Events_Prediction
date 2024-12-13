# 🚢 Petrobras Anomaly Detection Project

This project is designed to detect and classify undesirable events in offshore oil wells using the **Petrobras 3W dataset**. The solution leverages Python scripts for **data exploration, model training, real-time prediction**, and **workflow automation**. The core model used is **XGBoost**, enhanced with **rolling window methods** to preserve temporal integrity.

## 📊 **Project Overview**

### **Dataset**

The **Petrobras 3W dataset** contains time-series data representing various types of undesirable events, such as:

1. **Abrupt Increase of Basic Sediment and Water (BSW)**
2. **Spurious Closure of Downhole Safety Valve (DHSV)**
3. **Severe Slugging**
4. **Flow Instability**
5. **Rapid Productivity Loss**
6. **Quick Restriction in Production Choke**
7. **Scaling in Production Choke**
8. **Hydrate Formation in Production Lines**
9. **Hydrate Formation in Service Lines**

The **Dataset** can be downloaded from the following link: 

https://drive.google.com/drive/folders/1wUSA42A2zOiZOyyXi2tacnLeRm7Y_Zok?usp=drive_link


### **Objective**

The goal of this project is to develop a **machine learning model** that accurately detects **offshore well failures** at their **earliest stages**. By leveraging the **Petrobras 3W dataset**, the model aims to identify the **type** and **timing** of potential issues **before they escalate**, enabling **prompt interventions**. This approach minimizes **downtime**, reduces **unplanned maintenance costs**, enhances **operational efficiency**, supports **safer and more sustainable operations**, and facilitates **data-driven decision-making** in the oil and gas industry.

### **Core Components**

The project workflow is managed through several Python scripts:

| **Script**                  | **Description**                                                                                         |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| **`eda.py`**                | Streamlit-based Exploratory Data Analysis (EDA) tool to visualize and analyze the dataset.             |
| **`train.py`**              | Preprocesses data with rolling window methods and trains the XGBoost model.                            |
| **`predict_api.py`**        | FastAPI-based RESTful service for real-time anomaly detection.                                          |
| **`workflow.py`**           | Prefect automation script to orchestrate EDA, model training, and API deployment.                      |
| **`real_time_prediction.py`** | Simulates real-time data streaming and anomaly detection by interacting with the prediction API.        |

## 🛠️ **Setup Instructions**

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

## 📈 **Model Performance for Event 2**

### **Metrics**

| **Metric**               | **Score**  |
|---------------------------|------------|
| **Accuracy**             | 92.10%     |
| **F1 Score**             | 92.29%     |
| **Precision**            | 93.46%     |
| **Early Warning Time**   | 20 sec     |

### **Confusion Matrix**

|                             | **Predicted Normal** | **Predicted Fault** |
|-----------------------------|----------------------|---------------------|
| **Actual Normal**           | 5,758                | 1                   |
| **Actual Fault** (DHSV)     | 0                    | 1,513               |

**Key Insights:**

- **True Negatives:** The model accurately identified **5,758 cases** of normal operation.
- **True Positives:** The model correctly detected **1,513 cases** of DHSV (Downhole Safety Valve) closure events.
- **False Negatives:** **0 false negatives** – no failure events were missed.
- **False Positives:** Only **1 false positive**, indicating a very low false alarm rate.

This high performance ensures reliable detection of faults while minimizing unnecessary interventions.

### **Early Warning Time (EWT)**

- **Average Early Warning Time:** 20 seconds  
  The model detects events **20 seconds** in advance, with a **90-second rolling window validation** to confirm predictions and reduce false alarms.

---

## 🚀 **Technologies Used**

- **Python** (pandas, numpy, XGBoost, FastAPI, Streamlit)
- **XGBoost** (gradient-boosted decision trees for anomaly classification)
- **Prefect** (workflow automation)
- **MLflow** (model tracking and management)
- **FastAPI** (RESTful API for real-time predictions)
- **Streamlit** (interactive EDA and dashboard)

---

## 🌱 **Future Improvements**

1. **Cloud Integration:** Deploy the system on **AWS** or **Azure** for scalability.
2. **Real-Time Dashboards:** Enhance **Streamlit** dashboards with real-time anomaly visualization.
3. **Model Retraining Pipelines:** Automate model retraining with new data to maintain accuracy.
4. **Additional Fault Types:** Extend the model to detect more categories of undesirable events.
5. **Edge Deployment:** Optimize the model for edge devices to enable real-time processing on offshore rigs.

---

## 📚 **References**

- [Petrobras 3W Dataset Repository](https://github.com/petrobras/3W/tree/main/dataset)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [Prefect Documentation](https://docs.prefect.io/)

---

## 👥 **Contributing**

Contributions are welcome! Follow these steps to contribute:

1. **Fork** the repository.
2. **Create a new branch:** `git checkout -b feature-branch`
3. **Commit changes:** `git commit -m "Add new feature"`
4. **Push the branch:** `git push origin feature-branch`
5. **Submit a Pull Request**

---

