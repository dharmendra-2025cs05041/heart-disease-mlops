# Heart Disease Risk Prediction — MLOps Pipeline

[![CI/CD Pipeline](https://github.com/dharmendra-2025cs05041/heart-disease-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/dharmendra-2025cs05041/heart-disease-mlops/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository is my submission for **MLOps Assignment 1** of the
M.Tech course. It builds a binary classifier for heart-disease risk
on the UCI Heart Disease dataset and wraps it in a small but
realistic MLOps pipeline — experiment tracking, automated tests,
CI/CD, a containerised REST API, Kubernetes manifests, and a live
Prometheus + Grafana monitoring stack.

The project is intentionally modest in scope so that each piece of
the pipeline is easy to run on a laptop and easy to reason about.

## What is in here

```
heart-disease-mlops/
├── data/                     UCI dataset + cleaned CSV
├── notebooks/                01_eda, 02_training, 03_inference
├── src/
│   ├── data_processing/      Preprocessor (scaling + encoding)
│   ├── models/               Trainer for 4 classifiers, MLflow tracking
│   └── api/                  FastAPI service with /predict, /health, /metrics
├── tests/                    pytest suite (26 tests, ~72 % coverage)
├── deployment/
│   ├── docker/               Dockerfile + docker-compose (api + monitoring)
│   ├── kubernetes/           Deployment, Service, HPA, ConfigMap
│   └── monitoring/           prometheus.yml + Grafana provisioning
├── .github/workflows/        CI/CD — lint, test, build, security scan
├── scripts/                  setup_local_env.sh, download_data.py, test_api.sh
├── docs/                     ARCHITECTURE.md, SETUP_GUIDE.md
├── screenshots/              Architecture diagram + result captures
└── requirements.txt
```

## Dataset

The model is trained on the **UCI Heart Disease (Cleveland)** dataset
— 303 patients, 13 clinical features (age, sex, chest-pain type,
resting blood pressure, cholesterol, fasting blood sugar, resting
ECG, max heart rate, exercise-induced angina, ST depression, slope,
number of major vessels, thalassemia) and a binary target
(presence / absence of heart disease).
[Source on the UCI repository.](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

## Getting started

The fastest way to get running is the bootstrap script, which creates
the virtual environment, installs dependencies, downloads the data,
trains the four models, and runs the test suite in one go:

```bash
bash scripts/setup_local_env.sh
```

If you also want a local Docker engine and a local Kubernetes cluster
(via Colima and Minikube), pass `--all`:

```bash
bash scripts/setup_local_env.sh --all
```

Doing it manually is just as easy:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python src/models/train.py
pytest -q
```

Tested with **Python 3.10 – 3.12** on macOS and Linux.

## Running the pieces

**Train and track experiments**

```bash
python src/models/train.py            # trains 4 models, logs to ./mlruns
mlflow ui --backend-store-uri ./mlruns --port 5000
```

The trainer fits Logistic Regression, Random Forest, Gradient
Boosting and SVM, logs parameters, metrics and artefacts (confusion
matrix, ROC curve, model file) to MLflow, and writes the chosen
estimator to `models/`. Random Forest typically wins with ROC-AUC
around 0.96.

**Serve predictions**

```bash
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# Swagger UI:  http://localhost:8000/docs
# Health:      http://localhost:8000/health
# Metrics:     http://localhost:8000/metrics
```

A sample payload lives at `scripts/sample_payload.json`:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @scripts/sample_payload.json
```

**Bring up the full monitoring stack (API + Prometheus + Grafana)**

```bash
docker compose -f deployment/docker/docker-compose.yml up -d
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000   (admin / admin)
```

Grafana is provisioned at boot with the Prometheus datasource and a
dashboard called *Heart Disease API — Live Metrics* (request rate,
latency percentiles, predictions broken down by class).

**Deploy to Kubernetes**

```bash
kubectl apply -f deployment/kubernetes/
kubectl rollout status deployment/heart-disease-api
kubectl get pods,svc,hpa -l app=heart-disease-api
```

The manifests provision three replicas behind a `LoadBalancer`
service, with a `HorizontalPodAutoscaler` targeting 70 % CPU between
two and ten pods, plus liveness and readiness probes.

## Testing

```bash
pytest -q                                    # quick run
pytest -v --cov=src --cov-report=term        # with coverage
```

The same command runs in GitHub Actions on every push, alongside
flake8 / black checks, the Docker image build, and a Trivy security
scan of the resulting image.

## Further reading

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system diagram and
  per-component description.
- [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) — step-by-step setup
  on a fresh machine.
- The full assignment write-up is in
  `MLOps_Assignment1_Report_2025CS05041.docx`.

## Licence

Released under the MIT licence — see [`LICENSE`](LICENSE).

## Author

Dharmendra Parsaila (2025CS05041) — M.Tech, Semester 2, MLOps
Assignment 1.

## Repository

<https://github.com/dharmendra-2025cs05041/heart-disease-mlops>
