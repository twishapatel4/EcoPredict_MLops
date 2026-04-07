# 🌍 MLOps Project Blueprint: EcoPredict

### 1. Project Vision
**Name:** EcoPredict MLOps: Global Energy & Carbon Forecasting System.  
**Objective:** Build a professional, end-to-end automated system to forecast global CO2 emissions (per capita) using World Bank economic (GDP) and energy consumption indicators. This is a **Zero-Budget** project designed to showcase **Full-Stack AI Engineering** skills (Data Engineering, ML Science, and DevOps).

### 2. The "Zero-Budget" Tech Stack
To maintain a ₹0 cost, the system uses the following professional alternatives:
*   **Orchestration:** Apache Airflow via Astro CLI (Running on Local Docker).
*   **Data Versioning:** DVC (using Google Drive as free remote storage).
*   **Experiment Tracking:** MLflow (using DagsHub as a free hosted tracking server).
*   **Database:** PostgreSQL (Containerized via Docker).
*   **ML Stack:** XGBoost (Model), Optuna (Hyperparameter Tuning), SHAP (Explainability).
*   **Infrastructure:** Docker, GitHub Actions (CI/CD).
*   **Serving:** FastAPI (Backend API), Streamlit (Frontend Dashboard).

### 3. Modular System Architecture
The project follows the **Krish Naik Industrial Standard** for modular coding:
*   `setup.py`: Makes the `src` folder an installable local package (`-e .`).
*   `src/logger.py` & `src/exception.py`: Production-grade logging and custom error handling (captures file name and line number).
*   `src/components/`: Modular logic for `data_ingestion.py`, `data_transformation.py`, and `model_trainer.py`.
*   `src/pipeline/`: Connectors for `train_pipeline.py` and `predict_pipeline.py`.
*   `artifacts/`: Local storage for data versions and serialized models (`.pkl`).

### 4. Core Logic & Scientific Requirements
*   **Data Source:** World Bank API (Indicators: `NY.GDP.PCAP.CD`, `EN.ATM.CO2E.PC`, `EG.USE.PCAP.KG.OE`).
*   **Merging:** Inner join on Country, ISO3 Code, and Year.
*   **Validation Strategy:** **Time-Series Split** (Crucial: Not random). Training on years 2000–2018; Testing on 2019–2023 to simulate future forecasting.
*   **Missing Data:** Must handle global data gaps using **Interpolation/Imputation** (not simple row deletion).

### 5. Current Progress (Project State)
*   [x] **Environment:** Virtual environment (Python 3.12) configured and tested.
*   [x] **Scaffolding:** Full folder structure and `__init__.py` files established.
*   [x] **Foundation:** `logger.py` and `exception.py` implemented.
*   [x] **Packaging:** `setup.py` successfully links `src` as a library.
*   [x] **Ingestion:** `data_ingestion.py` written to fetch, merge, and split World Bank data chronologically.
*   [ ] **Next Step:** **Phase 2: Data Transformation.** Implementing a preprocessing pipeline (imputing, scaling, and encoding) to prepare the "Feature Store."

---

## 💡 Guidelines for the AI assistant:
1.  **Code Style:** Always provide modular, class-based Python code within the `src/` structure.
2.  **ML Focus:** Emphasize "Signal over Noise." Focus on Feature Engineering (Lag features, rolling averages).
3.  **Ops Focus:** Ensure all components are compatible with Docker and Airflow.
4.  **Budget:** Do not suggest paid AWS/Azure services; stick to the free local/community alternatives listed above.

---

### **Next Action for you:**

Now that the blueprint is ready, keep this saved. 

**Let's move to Phase 2: Data Transformation.**
In this step, we will write **`src/components/data_transformation.py`**. 

**The Challenge:**
World data is messy. 
1. Some years are NaN (Missing).
2. GDP is in the trillions, while CO2 is in decimals.
3. We need to tell the model *which* country is *which* (Categorical Encoding).