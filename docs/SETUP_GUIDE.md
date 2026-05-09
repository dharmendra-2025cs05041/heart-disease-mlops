# Setup Guide - Heart Disease Prediction MLOps Project

## Prerequisites

### Required Software
- Python 3.9 or higher
- Docker Desktop (for containerization)
- Kubernetes (Minikube or Docker Desktop Kubernetes)
- Git
- kubectl (Kubernetes CLI)

### Optional Tools
- MLflow (included in requirements.txt)
- Prometheus & Grafana (for monitoring)
- VSCode or PyCharm (IDE)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/dharmendra-2025cs05041/heart-disease-mlops.git
cd heart-disease-mlops
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset

```bash
python scripts/download_data.py
```

Expected output:
```
INFO - Downloading dataset from UCI ML Repository
INFO - Raw data saved to data/heart_disease_raw.data
INFO - Processed CSV saved to data/heart_disease.csv
INFO - Dataset shape: (303, 14)
INFO - ✓ Dataset download completed successfully!
```

### 5. Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Training Models

### Option 1: Using Python Script

```bash
python src/models/train.py
```

This will:
- Load and preprocess the data
- Train 4 different models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Perform hyperparameter tuning with GridSearchCV
- Log experiments to MLflow
- Save trained models to `models/` directory

### Option 2: Using Jupyter Notebook

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

## Viewing MLflow Experiments

```bash
mlflow ui --port 5000
```

Then open browser: http://localhost:5000

## Running the API Locally

### Option 1: Direct Python Execution

```bash
python src/api/app.py
```

### Option 2: Using Uvicorn

```bash
uvicorn src.api.app:app --reload --port 8000
```

API will be available at:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing the API

The full smoke-test (root, health, info, prediction × 2, validation
error, Prometheus metrics) lives in `scripts/test_api.sh`. It targets
`http://localhost:8000` by default and accepts a `BASE_URL` override:

```bash
bash scripts/test_api.sh                                              # local
BASE_URL=https://dharmendra-2025cs05041-heart-disease-api.hf.space \
  bash scripts/test_api.sh                                            # live
```

The same payloads are also available as standalone JSON files for
copy-paste curl or for the Swagger Examples dropdown on `POST /predict`:

| File                                      | Scenario                  | Expected response                          |
|-------------------------------------------|---------------------------|--------------------------------------------|
| `scripts/sample_payload_high_risk.json`   | 🔴 high-risk patient      | `prediction:1`, `risk_level:"High"`        |
| `scripts/sample_payload_low_risk.json`    | 🟢 low-risk patient       | `prediction:0`, `risk_level:"Low"`         |

#### Positive case — high-risk patient

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload_high_risk.json
```

Expected (against the live HF endpoint, Random Forest):
```json
{
  "prediction": 1,
  "probability_no_disease": 0.396,
  "probability_disease":   0.604,
  "confidence":            0.604,
  "risk_level": "Moderate"
}
```

#### Negative case — low-risk patient

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @scripts/sample_payload_low_risk.json
```

Expected:
```json
{
  "prediction": 0,
  "probability_no_disease": 0.970,
  "probability_disease":   0.030,
  "confidence":            0.970,
  "risk_level": "Low"
}
```

#### Validation-error case — out-of-range fields

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":250,"sex":5,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,
       "thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Expected (HTTP 422):
```json
{
  "detail": [
    {"loc":["body","age"],"msg":"Input should be less than or equal to 120","type":"less_than_equal"},
    {"loc":["body","sex"],"msg":"Input should be less than or equal to 1","type":"less_than_equal"}
  ]
}
```

### Testing the live (Hugging Face) deployment

The same image runs publicly on Hugging Face Spaces. Replace
`http://localhost:8000` in any of the commands above with:

```
https://dharmendra-2025cs05041-heart-disease-api.hf.space
```

| Resource     | URL                                                                 |
|--------------|---------------------------------------------------------------------|
| Live API     | <https://dharmendra-2025cs05041-heart-disease-api.hf.space>         |
| Swagger UI   | <https://dharmendra-2025cs05041-heart-disease-api.hf.space/docs>    |
| Health       | <https://dharmendra-2025cs05041-heart-disease-api.hf.space/health>  |
| Space page   | <https://huggingface.co/spaces/dharmendra-2025cs05041/heart-disease-api> |

The Space sleeps when idle (free tier), so the first request after
inactivity may take ~30 s while the container cold-starts.

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Run Specific Test File

```bash
pytest tests/test_data_processing.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v
```

## Docker Deployment

### One-shot demo stack (recommended)

The fastest way to bring up the full local demo (api + Prometheus +
Grafana with both Swagger Examples populated and Grafana panels
warmed up) is the orchestrator script. It rebuilds the api image
with `--no-cache`, brings the stack up, smoke-tests both labelled
payloads, fires a 30+30 warm-up burst, and prints every URL:

```bash
bash scripts/run_demo_stack.sh
```

Other modes:

```bash
bash scripts/run_demo_stack.sh --burst-only   # refresh metrics between demo takes
bash scripts/run_demo_stack.sh --status       # show stack + URLs, no traffic
bash scripts/run_demo_stack.sh --no-rebuild   # reuse cached image
bash scripts/run_demo_stack.sh --down         # stop stack (keep volumes)
bash scripts/run_demo_stack.sh --down --purge # stop stack + wipe prom/grafana data
```

After a successful run, open:

- API + Swagger : <http://localhost:8000/docs>
- Prometheus    : <http://localhost:9090/targets>
- Grafana       : <http://localhost:3000/d/heart-disease-api> (admin/admin, *Last 15 minutes*)

### Build Docker Image (manual)

```bash
docker build -t heart-disease-api:latest -f deployment/docker/Dockerfile .
```

### Run Docker Container (manual, single container)

```bash
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
```

### Test Docker Container

```bash
curl http://localhost:8000/health
```

### View Logs

```bash
docker logs heart-disease-api
```

### Stop Container

```bash
docker stop heart-disease-api
docker rm heart-disease-api
```

## Kubernetes Deployment

### Start Minikube (if using)

```bash
minikube start
```

### Deploy Application

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

### Check Deployment Status

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

### Access the Service

```bash
# Get service URL
minikube service heart-disease-api-service --url

# Or use port forwarding
kubectl port-forward service/heart-disease-api-service 8000:80
```

### Deploy Monitoring

```bash
kubectl apply -f deployment/kubernetes/monitoring.yaml
```

Access monitoring:
- Prometheus: http://localhost:30090
- Grafana: http://localhost:30300 (admin/admin)

## Troubleshooting

### Issue: Module not found
```bash
# Ensure you're in the project root and virtual environment is activated
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Issue: Docker build fails
```bash
# Clean Docker cache
docker system prune -a
```

### Issue: MLflow experiments not showing
```bash
# Check mlruns directory exists
ls -la mlruns/
```

## Next Steps

1. ✅ Explore the API documentation at `/docs`
2. ✅ View MLflow experiments
3. ✅ Review model performance in notebooks
4. ✅ Deploy to cloud (GCP/AWS/Azure)
5. ✅ Set up CI/CD pipeline with GitHub Actions
