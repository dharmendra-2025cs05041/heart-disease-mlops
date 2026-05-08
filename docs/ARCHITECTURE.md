# System Architecture - Heart Disease Prediction MLOps

## Overview

This document describes the architecture of the Heart Disease Prediction MLOps system, including data flow, component interactions, and deployment strategy.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  UCI ML Repository → Download Script → Raw Data → Preprocessing │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  Data Processing → Feature Engineering → Model Training         │
│                    ↓                                             │
│              MLflow Tracking                                     │
│  (Parameters, Metrics, Artifacts, Models)                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Model Registry                              │
├─────────────────────────────────────────────────────────────────┤
│  Trained Models (pkl) + Preprocessor + Metadata                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Serving Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Application → Model Predictor → Predictions            │
│                    ↓                                             │
│         Prometheus Metrics + Logging                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Deployment Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  Docker Container → Kubernetes Cluster → Load Balancer          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Monitoring Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus (Metrics) → Grafana (Visualization) → Alerts        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Layer

**Components:**
- **Data Source:** UCI Machine Learning Repository
- **Download Script:** `scripts/download_data.py`
- **Storage:** Local file system (`data/`)

**Responsibilities:**
- Fetch raw data from UCI repository
- Store data in CSV format
- Maintain data versioning

### 2. Data Processing Pipeline

**Components:**
- **DataPreprocessor:** `src/data_processing/preprocessor.py`
- **Feature Engineering:** Standard scaling, imputation

**Flow:**
```
Raw Data → Missing Value Imputation → Feature Scaling → Train/Test Split
```

**Key Features:**
- Reproducible preprocessing with saved transformers
- Stratified splitting to maintain class balance
- Pipeline serialization for production use

### 3. Model Training Pipeline

**Components:**
- **Training Script:** `src/models/train.py`
- **MLflow Integration:** Experiment tracking
- **Model Algorithms:**
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - SVM

**Workflow:**
```
1. Load preprocessed data
2. For each model:
   a. Hyperparameter tuning (GridSearchCV)
   b. Train on full training set
   c. Evaluate on test set
   d. Log to MLflow
   e. Save model artifacts
3. Select best model based on ROC-AUC
```

### 4. Experiment Tracking

**Tool:** MLflow

**Tracked Information:**
- **Parameters:** Model hyperparameters
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Artifacts:** 
  - Trained models
  - Confusion matrices
  - ROC curves
  - Feature importance plots

**Storage:** Local MLflow tracking server (`mlruns/`)

### 5. Model Serving

**Framework:** FastAPI

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/predict` | POST | Heart disease prediction |
| `/info` | GET | Model metadata |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | API documentation |

**Request/Response Flow:**
```
Client Request → Input Validation → Preprocessing → Model Inference → Response
```

### 6. Containerization

**Technology:** Docker

**Multi-stage Build:**
```dockerfile
Stage 1 (Builder):
  - Install build dependencies
  - Install Python packages

Stage 2 (Runtime):
  - Copy from builder
  - Copy application code
  - Configure runtime environment
```

**Benefits:**
- Smaller image size (~450 MB)
- Faster deployment
- Security (minimal attack surface)

### 7. Orchestration

**Technology:** Kubernetes

**Resources:**

1. **Deployment:**
   - 3 replicas for high availability
   - Resource limits (CPU: 500m, Memory: 512Mi)
   - Health checks (liveness & readiness probes)

2. **Service:**
   - Type: LoadBalancer
   - Port: 80 → Container Port: 8000

3. **Horizontal Pod Autoscaler:**
   - Min: 2, Max: 10 replicas
   - CPU threshold: 70%
   - Memory threshold: 80%

**Architecture:**
```
Internet → Load Balancer → Service → Pods (3 replicas) → Containers
```

### 8. CI/CD Pipeline

**Tool:** GitHub Actions

**Pipeline Stages:**

```
1. Code Quality
   ├── Black (formatting)
   ├── Flake8 (linting)
   └── Pylint (static analysis)

2. Testing
   ├── Unit tests
   ├── Coverage report
   └── Integration tests

3. Model Training
   ├── Download data
   ├── Train models
   └── Store artifacts

4. Build & Package
   ├── Build Docker image
   ├── Test container
   └── Push to registry
```

**Triggers:**
- Push to main/develop
- Pull requests
- Manual dispatch

### 9. Monitoring & Observability

**Components:**

1. **Logging:**
   - Application logs (Python logging)
   - Request/response logging
   - Error tracking

2. **Metrics (Prometheus):**
   - Request count
   - Latency histogram
   - Prediction distribution
   - System metrics (CPU, memory)

3. **Visualization (Grafana):**
   - Request rate dashboard
   - Latency trends
   - Error rate monitoring
   - Resource utilization

## Data Flow

### Training Flow
```
1. Data Download
   UCI Repository → scripts/download_data.py → data/heart_disease.csv

2. Preprocessing
   CSV → DataPreprocessor → Scaled Features + Labels

3. Training
   Features → GridSearchCV → Trained Models

4. Tracking
   Models + Metrics → MLflow → mlruns/

5. Storage
   Best Model → models/random_forest.pkl
   Preprocessor → models/preprocessor.pkl
```

### Inference Flow
```
1. API Request
   Client → POST /predict → FastAPI

2. Validation
   FastAPI → Pydantic → Validated Input

3. Preprocessing
   Input → DataPreprocessor → Scaled Features

4. Prediction
   Features → Model → Probabilities

5. Response
   Probabilities → JSON → Client
```

## Security Considerations

1. **Input Validation:**
   - Pydantic models for strict type checking
   - Range validation for all features

2. **Container Security:**
   - Non-root user execution
   - Minimal base image
   - Regular security updates

3. **API Security (Future):**
   - Rate limiting
   - Authentication (OAuth2/JWT)
   - HTTPS encryption

## Scalability

**Horizontal Scaling:**
- Kubernetes HPA for automatic scaling
- Load balancing across multiple pods
- Stateless API design

**Performance Optimization:**
- Model caching
- Request batching
- Async prediction endpoints (future)

## Disaster Recovery

1. **Model Versioning:**
   - MLflow model registry
   - Git-tracked code
   - Docker image tags

2. **Rollback Strategy:**
   - Kubernetes deployment rollback
   - Previous model versions available
   - Blue-green deployment (future)

## Cost Optimization

- Auto-scaling to match demand
- Resource limits prevent over-provisioning
- Efficient Docker image (multi-stage build)
- Local development with Minikube

---

**Last Updated:** 2025-01-15  
**Version:** 1.0.0
