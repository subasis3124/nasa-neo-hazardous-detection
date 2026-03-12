# 🚀 NASA NEO Hazard Classification  

<p align="center">

  <!-- GitHub Actions -->
  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/data_pusher.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/data_pusher.yml?branch=main&label=Data%20Pusher&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/predict.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/predict.yml?branch=main&label=Predict%20Fresh%20Data&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/data_drift_check.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/data_drift_check.yml?branch=main&label=Data%20Drift%20Check&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/continous_integration.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/continous_integration.yml?branch=main&label=Continuous%20Integration&logo=github" />
  </a>

  <a href="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/actions/workflows/auto_risk_analysis.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Subrat1920/Nasa-Near-Earth-Hazardous-Detection/auto_risk_analysis.yml?branch=main&label=Auto%20Risk%20Analysis&logo=github" />
  </a>

  <!-- MLflow -->
  <a href="https://dagshub.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection.mlflow/#/experiments/8">
    <img src="https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow" />
  </a>

  <!-- PostgreSQL -->
  <img src="https://img.shields.io/badge/Database-PostgreSQL-blue?logo=postgresql" />

  <!-- Grafana -->
  <a href="https://subrat1920.grafana.net/public-dashboards/888774ac60ce422fb3f0abb526512867">
    <img src="https://img.shields.io/badge/Monitoring-Grafana-orange?logo=grafana" />
  </a>

  <!-- Python -->
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?logo=python" />

  <!-- License -->
  <img src="https://img.shields.io/badge/License-NonCommercial-orange" />

</p>

 

This project builds a **production-grade ML pipeline** to classify **Near-Earth Objects (NEOs)** as *hazardous* or *non-hazardous*.  
It automates **data ingestion, preprocessing, model training, monitoring, and drift detection** using **GitHub Actions, PostgreSQL, MLflow, DAGsHub, and Grafana**.  

---

## 📑 Project Overview  

- **Goal**: Predict whether an asteroid/NEO poses a potential hazard to Earth.  
- **Data Source**: NASA Near Earth Object (NEO) dataset (orbital data, diameters, velocities, etc.).  
- **Core Features**:  
  - Automated data ingestion (weekly updates)  
  - Preprocessing & handling class imbalance (SMOTE)  
  - Model training with **MLflow tracking on DAGsHub**  
  - Logging models & preprocessors into **PostgreSQL**  
  - **Grafana dashboards** for monitoring & visualization  
  - Weekly **data drift detection** with notifications  

---

## ⚙️ Tech Stack  

- **Orchestration**: GitHub Actions (scheduled workflows)  
- **Database**: PostgreSQL  
- **Machine Learning**: scikit-learn, imbalanced-learn (SMOTE)  
- **Experiment Tracking**: MLflow (integrated with DAGsHub)  
- **Monitoring & Visualization**: Grafana   

---

## 🔄 Workflow Pipeline  

### 1. Data Ingestion  
- GitHub Actions scheduled weekly workflow.  
- Fetches latest NASA NEO dataset.  
- Loads data into **PostgreSQL**.  

### 2. Data Transformation  
- Data cleaning, feature scaling, and encoding.  
- **SMOTE** applied to balance hazardous vs non-hazardous classes.  

### 3. Model Training  
- Multiple ML models trained (Logistic Regression, Random Forest, XGBoost, etc.).  
- Tracked with **MLflow + DAGsHub authentication**.  
- Best model automatically selected.  

### 4. Model Logging  
- Best model + preprocessing pipeline stored in **PostgreSQL**.  
- Enables consistent inference and reproducibility.  

### 5. Monitoring with Grafana  
- Grafana dashboard connected to PostgreSQL.  
- Visualizes:  
  - Ingestion stats  
  - Model performance (accuracy, precision, recall, F1)  
  - Drift metrics  

### 6. Data Drift Detection  
- Weekly scheduled GitHub Action.  
- Drift metrics computed & logged in PostgreSQL.  
- Results visualized in Grafana.  
  

---

## 🌳 Repository Structure  

```
├── 📁 .github/
│   └── 📁 workflows/
│       ├── ⚙️ continous_integration.yml
│       ├── ⚙️ data_drift_check.yml
│       ├── ⚙️ data_pusher.yml
│       └── ⚙️ predict.yml
├── 📁 Data/
│   └── 📄 neo_data.csv
├── 📁 Notebook/
│   └── 📓 border_line_smote.ipynb
├── 📁 Research_Paper/
│   ├── 📁 Images/
│   │   └── 🖼️ System Architecture.png
│   ├── 📄 Automated End.docx
│   ├── 📕 Automated End.pdf
│   └── 📊 Presentation.pptx
├── 📁 pusher/
│   ├── 🐍 __init__.py
│   └── 🐍 weekly_data_pusher.py
├── 📁 src/
│   ├── 📁 constants/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config_entity.py
│   │   ├── 🐍 entity.py
│   │   └── 🐍 params.py
│   ├── 📁 custom/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 data_ingestion.py
│   │   ├── 🐍 data_transformation.py
│   │   ├── 🐍 data_validation.py
│   │   └── 🐍 model_trainer.py
│   ├── 📁 pipeline/
│   │   ├── 📁 prediction/
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 predict.py
│   │   └── 🐍 __init__.py
│   ├── 📁 utils/
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 utils.py
│   ├── 🐍 __init__.py
│   ├── 🐍 exception.py
│   └── 🐍 logging.py
├── 📁 tests/
│   ├── 🐍 __init__.py
│   ├── 🐍 test_data_ingestion.py
│   └── 🐍 test_utils.py
├── 📜 LICENSE
├── 📖 README.md
├── 🐍 app.py
├── 🐍 main.py
├── 🐍 predict_example.py
├── 📄 requirements.txt
└── 🐍 setup.py
```
---


---

## 📊 Grafana Dashboards  

- **Data Ingestion**: Track weekly data updates.  
- **Model Performance**: Compare metrics over time.  
- **Drift Monitoring**: Detect and visualize changes in data distribution.  

---

## 🔔 Alerts  

- **Email Notification** → Sent when data drift detected.  
- **Grafana Alerts** → Triggered for abnormal performance drops.  

---

## 🚀 Future Improvements  

- Deploy inference API for real-time predictions.  
- Add Docker + Kubernetes for scalable deployment.  
- Integrate SHAP for explainability of hazardous predictions.  

---

## 👨‍💻 Author  

**Subasis Mishra**  

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue?style=flat&logo=internet-explorer)](https://mishra-subrat.framer.website)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/subrat1920/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Subrat1920)
[![Medium](https://img.shields.io/badge/Medium-Read-black?style=flat&logo=medium)](https://medium.com/@subrat1920/1671404ef449)

---





