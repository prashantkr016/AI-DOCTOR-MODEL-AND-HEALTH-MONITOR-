# AI Dataset Doctor & Model Health Monitor
> Enterprise-Grade Pre-Training Data Quality Auditing & Post-Deployment MLOps Monitoring Platform

---

## 🌟 Overview

**AI Dataset Doctor & Model Health Monitor** is a full-stack, modular MLOps platform engineered for B.Tech Final-Year / Industry-Level AI/ML applications. It bridges pre-training dataset quality auditing with post-deployment model health & data drift monitoring.

---

## 🚀 Key Features

1. **Dataset Profiler**: Ingests CSV & Excel datasets, automatically detecting numerical, categorical, datetime, and target columns with full statistical summaries.
2. **Data Quality Doctor**: Automated diagnosis of missing values, duplicate rows, Z-score/IQR outliers, constant/near-constant features, high cardinality, class imbalance, and feature correlation matrices.
3. **Data Leakage Detector**: Identifies suspicious feature-target relationships, label name similarities, and single-predictor feature stumps.
4. **Bias & Fairness Analysis**: Calculates Demographic Parity Ratio (DPR) and statistical subgroup disparities across protected attributes.
5. **Automated ML Benchmark**: Trains and evaluates baseline models (Logistic Regression, Random Forest, XGBoost, SVM) across Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
6. **Explainable AI (SHAP)**: Computes global SHAP feature importances and individual sample local force explanations.
7. **Model Health & Drift Monitoring**: Tracks Kolmogorov-Smirnov (KS-test) continuous drift and Population Stability Index (PSI) categorical shifts post-deployment.
8. **Threshold Alert System**: Real-time warning triggers for low dataset health scores, target leakage, or production drift.
9. **Model Registry**: Semantic model versioning, training timestamps, hyperparameters, and deployment status.
10. **Modern React Dashboard**: Sleek dark-mode interface with interactive charts, gauge meters, and alert center.

---

## 🛠 Quick Start (Local Setup)

### Option A: Direct Python & Node.js Execution

#### 1. Backend Setup:
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

#### 2. Frontend Setup:
```bash
cd frontend
npm install
npm run dev
```
- Web Dashboard: [http://localhost:3000](http://localhost:3000)

---

### Option B: Docker Compose Setup

```bash
docker-compose up --build
```

---

## 🧪 Running Pytest Unit Suite

```bash
cd backend
pytest -v
```

---

## 📄 License
MIT License - Open Source MLOps Benchmark Suite.
