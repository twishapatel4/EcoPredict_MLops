# 🌍 EcoPredict MLOps: Global Energy & Carbon Forecasting System

### *An End-to-End Automated Intelligence Factory for Planetary Sustainability*

---

## 🎯 The Ultimate Goal
The ultimate goal of **EcoPredict** is to move beyond "Notebook-based" research into **"Systemic AI."** We are building a production-grade, automated system that provides **actionable, long-term forecasts** of a country’s carbon footprint. By analyzing the relationship between **GDP (Economic Growth)**, **Energy Consumption**, and **Population**, this system identifies environmental trends and predicts global CO2 emissions 5 years into the future.

---

## 🚀 Technical Goals & Achievements

### 1. Data Engineering & Orchestration (The Plumbing)
*   **Goal:** Eliminate manual data cleaning and handle unreliable external sources.
*   **Achievement:** A fully **Automated ETL Pipeline** using **Apache Airflow**. We automate the fetching of live data from the **World Bank API**, handle rate-limiting/retries, and store structured data in a **PostgreSQL** "Source of Truth."

### 2. MLOps & Reproducibility (The Factory)
*   **Goal:** Solve the "it works on my machine" problem and track model history.
*   **Achievement:** 
    *   **Environment Parity:** Every component is **Dockerized** to run identically on any server.
    *   **Data Versioning:** Using **DVC**, we track the exact version of the dataset used for every model.
    *   **Experiment Tracking:** Using **MLflow**, we record every training run, hyperparameter, and accuracy metric for complete auditability.

### 3. ML Science & Explainability (The Brains)
*   **Goal:** Build a model that is both accurate and transparent.
*   **Achievement:**
    *   **Predictive Power:** Using **Scikit-Learn** and **Optuna** for automated hyperparameter tuning.
    *   **Transparent AI:** Integrating **SHAP** to explain *why* the AI predicts an increase or decrease in emissions (e.g., identifying GDP vs. Energy as the primary driver).

### 4. Full-Stack Serving (The Interface)
*   **Goal:** Make AI accessible to non-technical users.
*   **Achievement:** A high-performance **FastAPI** backend serving predictions to a beautiful, interactive **Streamlit** dashboard.

---

## 🛠 The Full-Stack Toolset

| Category | Tool | Specific Purpose in this Project |
| :--- | :--- | :--- |
| **Orchestrator** | **Apache Airflow** | Automates the schedule for data ingestion and retraining. |
| **Data Logic** | **DVC & dbt** | Versions the raw data and transforms it via modular SQL. |
| **Database** | **PostgreSQL** | Stores historical global data and generated forecasts. |
| **Tracking** | **MLflow** | The central hub for logging model experiments and artifacts. |
| **API Layer** | **FastAPI** | The bridge connecting the ML model to the web. |
| **UI Layer** | **Streamlit** | Interactive maps and time-series charts for the user. |
| **Infrastructure**| **Docker** | Ensures the entire stack is portable and scalable. |
| **Automation** | **GitHub Actions**| Automatically tests code and builds images on every "Push." |

---

## 🏗 Modular Project Architecture

```text
EcoPredict_MLOps/
├── .github/workflows/      # CI/CD (GitHub Actions)
├── dags/                   # Airflow Schedules & Orchestration
├── data/                   # Global Datasets (Tracked by DVC)
├── logs/                   # Continuous Activity Tracking
├── artifacts/              # Saved Models & Processors (.pkl)
├── src/                    # THE BRAIN (Modular Python Package)
│   ├── components/         # Ingestion, Transformation, Trainer
│   ├── pipeline/           # Training and Prediction Pipelines
│   ├── logger.py           # Professional logging system
│   └── exception.py        # Custom Error/Exception Handling
├── app.py                  # Backend API (FastAPI)
├── streamlit_app.py        # Frontend Dashboard
├── setup.py                # Package Installation logic
└── Dockerfile              # Container Blueprint
```

---

## 📡 Operational Logic (The Pipeline)

1.  **INGEST:** Airflow triggers a request to the World Bank API $\rightarrow$ Data is saved to Postgres.
2.  **TRANSFORM:** We handle missing data via **Interpolation**, scale the numbers, and encode regions.
3.  **TRAIN:** Optuna finds the best settings $\rightarrow$ Scikit-Learn trains the model $\rightarrow$ Results logged in MLflow.
4.  **SERVE:** FastAPI loads the "Champion" model to handle incoming prediction requests.
5.  **MONITOR:** We check for "Data Drift" to ensure the model remains accurate as the world changes.

---

### **Summary of Full-Stack AI Competencies**
By building this project, I am demonstrating mastery in **Data Systems**, **Predictive Modeling**, **Cloud Infrastructure**, and **Web Development**. This is a complete "End-to-End" MLOps system designed for the 2026 AI landscape.