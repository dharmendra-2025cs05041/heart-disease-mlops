#!/bin/bash
# End-to-end smoke test for the Heart Disease Prediction API.
#
# By default targets a locally running server. Override with BASE_URL to test
# the live deployment, e.g.:
#   BASE_URL=https://dharmendra-2025cs05041-heart-disease-api.hf.space \
#     bash scripts/test_api.sh

set -u
BASE_URL="${BASE_URL:-http://localhost:8000}"

PYJSON='python3 -m json.tool'

echo "======================================"
echo "Testing Heart Disease Prediction API"
echo "  target: ${BASE_URL}"
echo "======================================"
echo ""

# Test 1: Root endpoint
echo "Test 1: Root Endpoint"
curl -s "${BASE_URL}/" | ${PYJSON}
echo ""

# Test 2: Health check
echo "Test 2: Health Check"
curl -s "${BASE_URL}/health" | ${PYJSON}
echo ""

# Test 3: Model info (model type and feature schema)
echo "Test 3: Model Info"
curl -sf "${BASE_URL}/info" | ${PYJSON} || echo "  /info not available on this deployment"
echo ""

# Test 4: POSITIVE / high-risk patient -- expects prediction: 1
echo "Test 4: Prediction - HIGH-RISK Patient (expect prediction=1)"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67, "sex": 1, "cp": 0, "trestbps": 160, "chol": 286,
    "fbs": 0, "restecg": 0, "thalach": 108, "exang": 1,
    "oldpeak": 1.5, "slope": 1, "ca": 3, "thal": 2
  }' | ${PYJSON}
echo ""

# Test 5: NEGATIVE / low-risk patient -- expects prediction: 0
echo "Test 5: Prediction - LOW-RISK Patient (expect prediction=0)"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 200,
    "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
  }' | ${PYJSON}
echo ""

# Test 6: VALIDATION error path -- expects HTTP 422 with field-level detail
echo "Test 6: Validation Error (expect HTTP 422 with details)"
curl -s -o /tmp/_pred_err.json -w "  http_status: %{http_code}\n" \
  -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 250, "sex": 5, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
${PYJSON} < /tmp/_pred_err.json
echo ""

# Test 7: Prometheus metrics (first 20 lines)
echo "Test 7: Prometheus Metrics"
curl -s "${BASE_URL}/metrics" | head -20
echo "..."
echo ""

echo "======================================"
echo "API Testing Completed"
echo "======================================"
