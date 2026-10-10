# 🎓 Student Performance Prediction — End-to-End MLOps System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-0.7+-FF4F00?style=for-the-badge)](https://www.evidentlyai.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.9+-2753B8?style=for-the-badge)](https://optuna.org/)
[![Telegram Bot API](https://img.shields.io/badge/Telegram_Bot-Alerting-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)
[![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

An enterprise-ready, end-to-end Machine Learning Operations (**MLOps**) system for predicting student examination performance. This repository transitions an academic tabular machine learning problem from exploratory data analysis and Bayesian hyperparameter tuning into a robust, observable, self-monitoring production microservice with automated data drift detection, PostgreSQL persistence, incremental stream ingestion, SLA tracking, and real-time Telegram ChatOps alerting.

---

## 📑 Table of Contents

- [System Architecture](#-system-architecture)
- [MLOps Skills Demonstrated](#-mlops-skills-demonstrated)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Repository Structure](#-repository-structure)
- [End-to-End Workflow & Lifecycle](#-end-to-end-workflow--lifecycle)
- [Deep Dive: Core Subsystems](#-deep-dive-core-subsystems)
  - [1. Data Preprocessing & Modeling Pipeline](#1-data-preprocessing--modeling-pipeline)
  - [2. Production API & Serving Layer (FastAPI)](#2-production-api--serving-layer-fastapi)
  - [3. Transactional & Analytical Storage (PostgreSQL & JSONB)](#3-transactional--analytical-storage-postgresql--jsonb)
  - [4. Dual-Layer Logging Architecture](#4-dual-layer-logging-architecture)
  - [5. Incremental Stream Log Ingestion Engine](#5-incremental-stream-log-ingestion-engine)
  - [6. Continuous ML Monitoring & Drift Detection (Evidently AI)](#6-continuous-ml-monitoring--drift-detection-evidently-ai)
  - [7. Operational SLI/SLO Latency Tracking & Health Engine](#7-operational-slislo-latency-tracking--health-engine)
  - [8. Real-Time ChatOps Alerting & Auto-Recovery (Telegram)](#8-real-time-chatops-alerting--auto-recovery-telegram)
  - [9. Production Traffic Simulation & Load Testing](#9-production-traffic-simulation--load-testing)
  - [10. Scheduled Orchestration & Cron Automation](#10-scheduled-orchestration--cron-automation)
- [Installation & Local Setup](#-installation--local-setup)
- [Configuration & Environment Variables](#-configuration--environment-variables)
- [Running the System](#-running-the-system)
  - [Database Initialization](#database-initialization)
  - [Starting the FastAPI Microservice](#starting-the-fastapi-microservice)
  - [Simulating Production Inference Requests](#simulating-production-inference-requests)
  - [Running Incremental Ingestion](#running-incremental-ingestion)
  - [Running the Drift Monitoring Pipeline](#running-the-drift-monitoring-pipeline)
  - [Configuring Cron Automation](#configuring-cron-automation)
- [API Documentation & Schema](#-api-documentation--schema)
- [Monitoring Reports & Alerts Preview](#-monitoring-reports--alerts-preview)
- [Engineering Highlights & Best Practices](#-engineering-highlights--best-practices)

---

## 🏛️ System Architecture

The following diagram illustrates the complete dataflow, serving pipeline, persistence mechanisms, monitoring triggers, and alerting boundaries:

```mermaid
flowchart TD
    subgraph Data_Modeling["1. Pipeline Engineering & Training"]
        RawData["Raw Dataset (data/SPF.csv)"] --> EDA["EDA & Cleaning (notebooks/SPF.ipynb)"]
        EDA --> SklearnPipeline["ColumnTransformer & Preprocessing"]
        SklearnPipeline --> OptunaTuning["Optuna Bayesian Tuning"]
        OptunaTuning --> JoblibModel["Serialized Artifact (models/student_performance_pipeline.joblib)"]
    end

    subgraph Serving["2. Production Inference Service (FastAPI)"]
        Client(["HTTP Client / Students / Systems"]) -->|POST /api/v1/predict| FastAPIServer["FastAPI Server (backend/main.py)"]
        FastAPIServer --> PydanticValidation["Schema Validation (backend/schemas.py)"]
        JoblibModel -.->|Preloaded into memory| ModelInference["Model Inference Engine"]
        PydanticValidation --> ModelInference
        ModelInference --> LatencyTimer["High-Resolution Timer (time.perf_counter)"]
        ModelInference --> APIResponse(["Predicted Exam Score + Latency"])
        APIResponse --> Client
    end

    subgraph Observability_Persistence["3. Dual-Layer Logging & Persistence"]
        ModelInference -->|Async Structured JSON| FileLogger["logs/inference.log"]
        ModelInference -->|SQLAlchemy ORM (JSONB Features)| PostgresDB[("PostgreSQL: inference_logs")]
    end

    subgraph Ingestion_Stream["4. Incremental Log Processing"]
        FileLogger --> IngestScript["Incremental Ingest Engine (monitoring/ingest_logs.py)"]
        StateJSON["Byte Offset Checkpoint (monitoring/ingest_state.json)"] <--> IngestScript
        IngestScript --> ProdCSV["data/production_data.csv"]
    end

    subgraph Monitoring_Engine["5. Continuous Monitoring & SRE Alerting"]
        Scheduler[["System Cron / Orchestrator (run_monitoring.py)"]] --> DriftEngine["Drift Monitor (monitoring/monitor_drift.py)"]
        PostgresDB -->|Time-Windowed Query| DBLoader["Data Loader (monitoring/database_loader.py)"]
        RefData["Baseline Reference (data/reference_data.csv)"] --> DriftEngine
        RefPreds["Reference Predictions (data/reference_predictions.csv)"] --> DriftEngine
        DBLoader --> DriftEngine
        
        DriftEngine --> DataDrift["Input Feature Drift (Evidently AI DataDriftPreset)"]
        DriftEngine --> PredDrift["Prediction Drift (Evidently AI DataDriftPreset)"]
        DriftEngine --> LatencyMetrics["Latency SLA Metrics (Mean, P95)"]

        DataDrift --> HTMLReports["Interactive Reports (reports/*.html)"]
        PredDrift --> HTMLReports
        
        DriftEngine --> StatusEvaluator{"Health Status: NORMAL | WARNING | ALERT"}
        StatusEvaluator -->|Save Summary Metrics| MetricsTable[("PostgreSQL: monitoring_metrics")]
        StatusEvaluator --> AlertDeduplication{"State Changed?"}
        AlertDeduplication -->|Yes / Transition| TelegramAPI["Telegram Bot API (monitoring/alerting.py)"]
        TelegramAPI --> TelegramChat(["Telegram Channel / SRE On-Call"])
    end
```

---

## ⚡ MLOps Skills Demonstrated

This repository implements the full spectrum of production MLOps competencies:

| MLOps Competency | Engineering Implementation in Project | Core Files / Technologies |
| :--- | :--- | :--- |
| **1. Reproducible ML Pipelines** | Scikit-Learn `Pipeline` & `ColumnTransformer` with median/mode imputation and `OneHotEncoder(handle_unknown="ignore")`. Zero training-serving skew. | `notebooks/SPF.ipynb`, `sklearn` |
| **2. Hyperparameter Optimization** | Automated Bayesian optimization across continuous & discrete parameter spaces (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`). | `notebooks/SPF.ipynb`, `optuna` |
| **3. Model Artifact Versioning** | Atomic serialization of pipeline (preprocessing + estimator) saved as deployable artifact. | `models/student_performance_pipeline.joblib` |
| **4. High-Performance API Serving** | Asynchronous REST microservice with dependency-injected database sessions, lifespan model pre-loading, and CORS readiness. | `backend/main.py`, `backend/routes/prediction.py`, `fastapi`, `uvicorn` |
| **5. Strict Schema Validation** | Pydantic v2 schemas enforcing type contracts, bounded numeric fields (`ge=0`, `le=100`, `le=24`), and OpenAPI/Swagger auto-generation. | `backend/schemas.py`, `pydantic` |
| **6. Production Database Design** | Relational schema using SQLAlchemy 2.0 ORM with PostgreSQL native `JSONB` for input features, indexed fields, connection pooling (`pool_pre_ping`). | `backend/models.py`, `backend/database.py`, `backend/init_db.py`, `postgresql` |
| **7. Dual-Path Observability** | Parallel logging architecture: high-speed JSON file logging (`inference.log`) and ACID database transactional logging (`inference_logs`). | `backend/logging_config.py`, `backend/routes/prediction.py` |
| **8. Real-Time SLI/SLO Profiling** | Microsecond-accurate latency measurements using `time.perf_counter()`, computing request volume, mean latency, and 95th percentile (P95) latency. | `backend/routes/prediction.py`, `monitoring/monitor_drift.py` |
| **9. Stream & Incremental Ingestion** | Stateful byte-offset tracking (`tell()`, `seek()`) to incrementally read appended log streams without re-parsing historical records. Handles log rotation/truncation. | `monitoring/ingest_logs.py`, `monitoring/ingest_state.json` |
| **10. Feature & Data Drift Detection** | Evidently AI statistical comparison across 19 tabular features against validated reference distribution (`reference_data.csv`). | `monitoring/monitor_drift.py`, `evidently` |
| **11. Prediction & Concept Drift** | Output target distribution monitoring comparing production inferences against reference predictions (`reference_predictions.csv`). | `monitoring/monitor_drift.py`, `evidently` |
| **12. Historical Metrics Auditing** | Persisting operational run metrics, drift status, drifted feature count, and latency percentiles to `monitoring_metrics` table. | `backend/models.py`, `monitoring/monitor_drift.py` |
| **13. ChatOps & Automated SRE Alerting** | Automated webhook dispatch to Telegram Bot API with formatted diagnostics. Implements alert deduplication and auto-recovery notifications. | `monitoring/alerting.py`, `urllib.request` |
| **14. Automated Pipeline Orchestration** | Headless automated execution scripts designed for Linux `cron` hourly sliding-window batch jobs. | `monitoring/run_monitoring.py`, Linux Cron |
| **15. Traffic Simulation & Load Testing** | Configurable synthetic traffic generator with sampling with/without replacement, target stripping, delay pacing, and error handling. | `monitoring/simulate_requests.py` |

---

## 🎯 Key Features & Capabilities

- **Zero Training-Serving Skew:** All feature transformations (categorical one-hot encodings, missing value imputations, scalers) are encapsulated inside a single serialized `scikit-learn` pipeline object. Raw JSON payloads are fed directly to the model.
- **FastAPI Microservice with Type Safety:** Input parameters validated using strict Pydantic schemas. Unrecognized or out-of-bound variables return standardized HTTP 422 errors before reaching the model.
- **PostgreSQL JSONB Storage:** Inbound inference features are saved as native JSONB documents, allowing future schema flexibility, fast SQL JSON indexing, and historical auditing.
- **Microsecond Latency Tracking:** Every inference measures exact execution time and logs both successful transactions and failures with rollback safety.
- **Dual Drift Detection Engines:**
  - **Data Drift:** Evaluates Kolmogorov-Smirnov (KS) or Wasserstein distance/Chi-square tests per feature using Evidently AI.
  - **Prediction Drift:** Detects model output decay or demographic shifts before ground-truth labels are available.
- **Stateful Incremental Ingestion:** File pointer bookmarking ensures batch log processors ingest only net-new log lines without duplicate record creation.
- **De-duplicated ChatOps Incident Management:** Alert rules trigger on configurable thresholds (Mean Latency > 500ms, P95 > 500ms, Drift > 0). If an issue persists, spam is suppressed; when metrics normalize, an automated recovery notification is delivered.

---

## 📂 Repository Structure

```text
Linear_Regression/
├── backend/
│   ├── routes/
│   │   └── prediction.py       # Prediction endpoint (/api/v1/predict) with DB & log triggers
│   ├── config.py               # Central path configurations and settings
│   ├── database.py             # SQLAlchemy 2.0 engine, SessionLocal, and DB dependencies
│   ├── init_db.py              # Database table creation script
│   ├── logging_config.py       # Application-wide structured logging configuration
│   ├── main.py                 # FastAPI application instance, routes, and healthcheck
│   ├── model_loader.py         # Thread-safe pipeline artifact loader (joblib)
│   ├── models.py               # SQLAlchemy ORM models (InferenceLog, MonitoringMetric)
│   └── schemas.py              # Pydantic v2 validation models and response schemas
├── data/
│   ├── SPF.csv                 # Master Student Performance Factors dataset (raw)
│   ├── reference_data.csv      # Baseline reference feature distribution (6,600+ rows)
│   ├── reference_predictions.csv# Baseline prediction distribution for output drift
│   ├── production_data.csv     # Extracted production records from incremental ingestion
│   └── archive.zip             # Archived historical datasets
├── models/
│   ├── student_performance_pipeline.joblib # Optuna-tuned production pipeline artifact
│   └── model.joblib            # Baseline linear regression model artifact
├── monitoring/
│   ├── alerting.py             # Telegram Bot API client and payload message builders
│   ├── database_loader.py      # Time-windowed PostgreSQL query loader for production data
│   ├── ingest_logs.py          # Stateful incremental inference log parser with byte offsets
│   ├── ingest_state.json       # Checkpoint tracking file pointer position (tell/seek)
│   ├── monitor_drift.py        # Core Evidently AI monitoring pipeline, SLA evaluator & alert dispatcher
│   ├── run_monitoring.py       # Orchestration wrapper for cron jobs and scheduled runners
│   └── simulate_requests.py    # Production traffic simulator and API load testing script
├── notebooks/
│   ├── Data_exploration.ipynb  # Initial EDA and baseline Linear Regression benchmarking
│   ├── finding best model.ipynb# Data cleaning, outlier handling & regression comparisons
│   └── SPF.ipynb               # End-to-end preprocessing, Optuna tuning, and pipeline export
├── reports/
│   ├── data_drift_report.html  # Interactive Evidently visual report for input feature drift
│   └── production_drift_report.html # Interactive Evidently visual report for prediction drift
├── logs/
│   ├── inference.log           # Production API structured JSON access and prediction log
│   ├── ingest.log              # Log stream ingestion execution log
│   ├── monitoring.log          # Hourly monitoring run output and telemetry log
│   └── cron.log                # Scheduler automation run execution log
├── .env                        # Environment configurations (DB URL, Telegram secrets)
├── .gitignore                  # Git exclusion rules
└── requirements.txt            # Python dependencies (FastAPI, Evidently, SQLAlchemy, etc.)
```

---

## 🔄 End-to-End Workflow & Lifecycle

The lifecycle of an inference from model development to automated monitoring runs through four distinct stages:

```
[Phase 1: Model Engineering]
Raw Data (SPF.csv) ──> Preprocessing & Encoders ──> Optuna Tuning ──> Pipeline Serialization (.joblib)

[Phase 2: Real-Time Serving]
Client Request ──> Pydantic Validation ──> In-Memory Inference ──> Latency Profiling
                                                                        │
                   ┌────────────────────────────────────────────────────┴───────────────────────────────────┐
                   ▼                                                                                        ▼
       File Handler: logs/inference.log                                              PostgreSQL: inference_logs (JSONB)

[Phase 3: Stream Ingestion]
File Pointer Tracking (ingest_state.json) ──> Incremental Parsing ──> Append to data/production_data.csv

[Phase 4: ML Observability & SRE]
Hourly Window Query ──> Evidently AI Drift Engine ──> Metric Calculation ──> Threshold Check (NORMAL/WARN/ALERT)
                                                                                       │
                   ┌───────────────────────────────────────────────────────────────────┴────────────────────┐
                   ▼                                                                                        ▼
    HTML Report Generation (reports/*.html)                                           Telegram ChatOps Notification
```

---

## 🔬 Deep Dive: Core Subsystems

### 1. Data Preprocessing & Modeling Pipeline
- **Dataset:** Student Performance Factors (19 input features covering study habits, attendance, parental involvement, access to resources, sleep, tutoring, and family income).
- **Leakage Prevention & Modular Transformers:** Implemented via Scikit-Learn `ColumnTransformer`:
  ```python
  numeric_transformer = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="median"))
  ])
  categorical_transformer = Pipeline(steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("encoder", OneHotEncoder(handle_unknown="ignore"))
  ])
  preprocessor = ColumnTransformer(transformers=[
      ("num", numeric_transformer, numerical_cols),
      ("cat", categorical_transformer, categorical_cols)
  ])
  ```
- **Bayesian Optimization with Optuna:** 
  The hyperparameters for the `RandomForestRegressor` were optimized using Optuna to minimize Cross-Validation Root Mean Squared Error (RMSE).
  - Selected Optimal Parameters: `n_estimators=286`, `max_depth=17`, `min_samples_split=8`, `min_samples_leaf=4`.
- **Pipeline Packaging:** Saved as `student_performance_pipeline.joblib`. The pipeline self-contains imputers, encodings, and estimator weights, eliminating preprocessing synchronization errors.

---

### 2. Production API & Serving Layer (FastAPI)
The serving layer located in `backend/` exposes clean RESTful endpoints:
- **`GET /`**: Service health and identification endpoint.
- **`GET /health`**: Microservice liveness probe for load balancers.
- **`POST /api/v1/predict`**: Core inference route.
  - Takes `StudentInput` JSON payload.
  - Validates constraints (e.g. `Attendance: 0 to 100`, `Sleep_Hours: 0 to 24`).
  - Measures execution duration using high-precision `time.perf_counter()`.
  - Dispatches dual write: writes structured JSON to `logs/inference.log` and executes an ACID transaction to PostgreSQL.
  - Returns `PredictionResponse` (`status="success"`, `predicted_exam_score=...`).

---

### 3. Transactional & Analytical Storage (PostgreSQL & JSONB)
The project utilizes PostgreSQL with SQLAlchemy 2.0 ORM. The relational architecture balances write latency and analytical query efficiency:
- **`inference_logs` Table:**
  ```python
  class InferenceLog(Base):
      __tablename__ = "inference_logs"
      id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
      timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
      input_features: Mapped[dict] = mapped_column(JSONB, nullable=False)
      prediction: Mapped[float] = mapped_column(Float, nullable=False)
      latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
      status: Mapped[str] = mapped_column(String(20), default="success", nullable=False)
  ```
- **`monitoring_metrics` Table:**
  Tracks monitoring batch runs, request volumes, Mean and P95 latencies, drift detections, drifted feature counts, and overall system status (`NORMAL`, `WARNING`, `ALERT`).

---

### 4. Dual-Layer Logging Architecture
Rather than relying solely on file logging or solely on database transactions, the service employs **dual logging**:
1. **File Logging (`logs/inference.log`):** Low-latency, structured JSON logging handled by Python's logging facility.
2. **Database Persistence (`inference_logs`):** Stores structured records in PostgreSQL with connection pooling (`pool_pre_ping=True`), guaranteeing ACID consistency and enabling fast SQL time-window queries.

---

### 5. Incremental Stream Log Ingestion Engine
`monitoring/ingest_logs.py` implements an enterprise log-scraping pattern:
- **Byte-Offset Bookmarking:** Maintains the byte position of `inference.log` in `monitoring/ingest_state.json`.
- **Atomic Seek & Read:** On every invocation, calls `file.seek(last_position)` and reads only new lines appended since the last execution.
- **Log Rotation & Truncation Resiliency:** Checks whether `start_position > file_size`. If the file was rotated or cleared, the pointer resets to `0` automatically without crashing.
- **CSV Cold Storage Export:** Appends new parsed records to `data/production_data.csv`.

---

### 6. Continuous ML Monitoring & Drift Detection (Evidently AI)
`monitoring/monitor_drift.py` provides automated monitoring:
- **Hourly Sliding Window:** Calculates monitoring bounds for the preceding hour (`start_time` to `end_time`) and pulls production records from PostgreSQL.
- **Feature Data Drift:** Isolates the 19 input features and executes Evidently's `DataDriftPreset()` against `reference_data.csv`. Identifies drift using statistical two-sample tests.
- **Prediction Drift:** Extracts the `prediction` vector and compares it against baseline `reference_predictions.csv` to detect distribution shifts in model outputs.
- **Report Export:** Generates standalone interactive visual dashboards:
  - `reports/data_drift_report.html`
  - `reports/production_drift_report.html`

---

### 7. Operational SLI/SLO Latency Tracking & Health Engine
The monitoring engine calculates operational Service Level Indicators (SLIs) and classifies health status:
- **Request Volume:** Total inferences served within the window.
- **Average Latency:** Mean execution time in milliseconds.
- **P95 Latency:** 95th percentile latency (tail latency metric).

#### Classification Rules:
| Condition | System State | Action Taken |
| :--- | :--- | :--- |
| Avg Latency < 200ms & P95 < 300ms & No Drift | `NORMAL` | Logged, metrics stored in DB. If recovering, sends Recovery alert. |
| Avg Latency 200-500ms OR P95 300-500ms | `WARNING` | Logged, metrics stored in DB, Telegram warning dispatched. |
| Avg Latency > 500ms OR P95 > 500ms OR Data Drift Detected OR Prediction Drift Detected | `ALERT` | High-priority Telegram alert dispatched with root causes. |

---

### 8. Real-Time ChatOps Alerting & Auto-Recovery (Telegram)
`monitoring/alerting.py` integrates directly with the Telegram Bot API:
- **Anti-Spam State Deduplication:** Checks the latest record in `monitoring_metrics`. If the system was already in `ALERT` status during the last run, redundant notifications are suppressed.
- **Auto-Recovery Notifications:** If the system was previously in `WARNING` or `ALERT` status and returns to `NORMAL`, a green recovery message is automatically dispatched.
- **Payload Diagnostics:** Formatted Telegram notifications provide immediate visibility into request counts, latencies, drifted feature ratios, and explicit reasons for failure.

---

### 9. Production Traffic Simulation & Load Testing
`monitoring/simulate_requests.py` simulates realistic live production traffic against the FastAPI endpoint:
- **Sampling Capabilities:** Samples arbitrary batch sizes (`--requests N`), supporting replacement sampling if `N` exceeds dataset size.
- **Target Sanitization:** Automatically strips the ground-truth `Exam_Score` column from input vectors.
- **Pacing Control:** Configurable delay interval (`--delay S`, default 0.1s) to simulate varying traffic patterns.
- **Telemetry Output:** Reports per-request status codes, returned predictions, and summary statistics.

---

### 10. Scheduled Orchestration & Cron Automation
`monitoring/run_monitoring.py` acts as an automated entrypoint:
- Runs headless subprocesses to execute drift detection and alerting.
- Returns standard POSIX exit codes (`0` for success, non-zero for failure).
- Facilitates integration with standard Linux `cron` or cloud container schedulers (Kubernetes CronJobs, AWS EventBridge, etc.).

---

## 🛠️ Installation & Local Setup

### 1. Prerequisites
- **Python 3.11+** (Python 3.13 tested)
- **PostgreSQL 14+** running locally or remotely
- Active internet access (for Telegram Bot API alerts)

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/vardhan9296/Linear_Regression.git
cd Linear_Regression

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Configuration & Environment Variables

Create a `.env` file in the root directory:

```env
# PostgreSQL Database URL (format: postgresql+psycopg2://<user>:<password>@<host>:<port>/<dbname>)
DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/ml_monitoring

# Telegram Bot API Configuration for Automated Alerting
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

---

## 🚀 Running the System

### Database Initialization
Create the database tables (`inference_logs` and `monitoring_metrics`):
```bash
python backend/init_db.py
```

### Starting the FastAPI Microservice
Launch the production API server with Uvicorn:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Alternative ReDoc UI: `http://127.0.0.1:8000/redoc`
- Service Liveness Probe: `http://127.0.0.1:8000/health`

### Simulating Production Inference Requests
Send simulated student records to the running API:
```bash
# Send 100 requests with 0.05s delay
python monitoring/simulate_requests.py --requests 100 --delay 0.05
```

### Running Incremental Ingestion
Parse newly written records from `logs/inference.log` into `data/production_data.csv`:
```bash
python monitoring/ingest_logs.py
```

### Running the Drift Monitoring Pipeline
Trigger the drift evaluation, metrics persistence, and Telegram alerting:
```bash
python monitoring/monitor_drift.py
```
*(Or use the runner script: `python monitoring/run_monitoring.py`)*

### Configuring Cron Automation
Schedule hourly automated monitoring runs using system cron:
```bash
# Open crontab editor
crontab -e

# Run monitoring at minute 0 of every hour
0 * * * * cd /home/vardhan9296/Linear_Regression && /home/vardhan9296/Linear_Regression/venv/bin/python monitoring/run_monitoring.py >> logs/cron.log 2>&1
```

---

## 📡 API Documentation & Schema

### `POST /api/v1/predict`
Calculates predicted exam score for a student profile.

#### Request Headers
```http
Content-Type: application/json
```

#### Request Payload Example
```json
{
  "Hours_Studied": 25,
  "Attendance": 85,
  "Parental_Involvement": "Medium",
  "Access_to_Resources": "High",
  "Extracurricular_Activities": "Yes",
  "Sleep_Hours": 7,
  "Previous_Scores": 80,
  "Motivation_Level": "Medium",
  "Internet_Access": "Yes",
  "Tutoring_Sessions": 1,
  "Family_Income": "Medium",
  "Teacher_Quality": "Medium",
  "School_Type": "Public",
  "Peer_Influence": "Neutral",
  "Physical_Activity": 3,
  "Learning_Disabilities": "No",
  "Parental_Education_Level": "College",
  "Distance_from_Home": "Near",
  "Gender": "Female"
}
```

#### Response Payload Example (200 OK)
```json
{
  "status": "success",
  "predicted_exam_score": 70.32
}
```

---

## 📊 Monitoring Reports & Alerts Preview

### 1. Evidently AI Visual Reports
When `monitor_drift.py` runs, it generates interactive visual reports in `reports/`:
- **`data_drift_report.html`**: Statistical distribution comparison of all 19 input features (baseline vs. production window). Highlights drifted features, p-values, and quantile shifts.
- **`production_drift_report.html`**: Distribution analysis of model prediction scores across the current monitoring window compared to reference baseline predictions.

### 2. Telegram Alert Payload Samples

#### Incident Alert (`ALERT`)
```text
🚨 ML MONITORING ALERT

Status: ALERT

Metrics:
Requests: 195
Average latency: 84.12 ms
P95 latency: 122.40 ms
Data drift: True
Drifted features: 3/19
Prediction drift: True

Reasons:
• Data drift detected in 3 features
• Prediction drift detected
```

#### Incident Recovery (`NORMAL`)
```text
✅ ML MONITORING RECOVERY

The monitoring system has returned to NORMAL status.
```

---

## 🏆 Engineering Highlights & Best Practices

1. **Clean Code & Separation of Concerns:**
   - Routes, schemas, database models, configurations, and loaders are isolated in dedicated modules.
2. **Defensive Error Handling:**
   - All database operations are wrapped with try/except blocks and explicit `db.rollback()` on failures.
3. **High-Performance Database Practices:**
   - SQLAlchemy connection pooling with `pool_pre_ping=True` ensures stale connections are recycled gracefully.
4. **Stateful Ingestion with Fallbacks:**
   - Offset checkpoints guard against redundant log processing, with auto-reset fallback in case of log rotation.
5. **No Blind Alerts (Alert Hygiene):**
   - Alert state deduplication eliminates alert fatigue, ensuring on-call engineers are notified only on status transitions.

---

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this codebase for educational and commercial purposes.
