#!/bin/bash

# API Testing Script
# Tests all endpoints of the Heart Disease Prediction API

API_URL="http://localhost:8000"

echo "======================================"
echo "Testing Heart Disease Prediction API"
echo "======================================"
echo ""

# Test 1: Root endpoint
echo "Test 1: Root Endpoint"
curl -s $API_URL/ | python -m json.tool
echo ""
echo ""

# Test 2: Health check
echo "Test 2: Health Check"
curl -s $API_URL/health | python -m json.tool
echo ""
echo ""

# Test 3: Model info
echo "Test 3: Model Info"
curl -s $API_URL/info | python -m json.tool
echo ""
echo ""

# Test 4: Prediction - High Risk Patient
echo "Test 4: Prediction - High Risk Patient"
curl -s -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }' | python -m json.tool
echo ""
echo ""

# Test 5: Prediction - Low Risk Patient
echo "Test 5: Prediction - Low Risk Patient"
curl -s -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 0,
    "cp": 1,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 0,
    "thalach": 170,
    "exang": 0,
    "oldpeak": 0.5,
    "slope": 1,
    "ca": 0,
    "thal": 2
  }' | python -m json.tool
echo ""
echo ""

# Test 6: Metrics endpoint
echo "Test 6: Prometheus Metrics"
curl -s $API_URL/metrics | head -20
echo ""
echo "..."
echo ""

echo "======================================"
echo "API Testing Completed"
echo "======================================"
