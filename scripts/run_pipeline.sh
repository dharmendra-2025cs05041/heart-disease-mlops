#!/bin/bash

# Heart Disease MLOps - Complete Pipeline Execution Script
# This script runs the entire ML pipeline from data download to model training

set -e  # Exit on error

echo "======================================"
echo "Heart Disease MLOps Pipeline"
echo "======================================"
echo ""

# Step 1: Download Dataset
echo "Step 1: Downloading dataset..."
python scripts/download_data.py
echo "✓ Dataset downloaded successfully"
echo ""

# Step 2: Run EDA (optional - requires Jupyter)
echo "Step 2: EDA notebooks available at notebooks/01_eda.ipynb"
echo "Run: jupyter notebook notebooks/01_eda.ipynb"
echo ""

# Step 3: Train Models
echo "Step 3: Training models with MLflow tracking..."
python src/models/train.py
echo "✓ Models trained successfully"
echo ""

# Step 4: Run Tests
echo "Step 4: Running unit tests..."
pytest tests/ -v --cov=src --cov-report=term
echo "✓ Tests completed successfully"
echo ""

# Step 5: MLflow UI
echo "Step 5: Starting MLflow UI..."
echo "Access MLflow at: http://localhost:5000"
echo "Run manually: mlflow ui --port 5000"
echo ""

# Step 6: Start API
echo "Step 6: API can be started with:"
echo "  python src/api/app.py"
echo "  or"
echo "  uvicorn src.api.app:app --reload"
echo ""

echo "======================================"
echo "Pipeline Completed Successfully!"
echo "======================================"
echo ""
echo "Next Steps:"
echo "1. Review models in MLflow UI: mlflow ui --port 5000"
echo "2. Start API: python src/api/app.py"
echo "3. Test API: curl http://localhost:8000/health"
echo "4. Build Docker: docker build -t heart-disease-api -f deployment/docker/Dockerfile ."
echo "5. Deploy to K8s: kubectl apply -f deployment/kubernetes/deployment.yaml"
