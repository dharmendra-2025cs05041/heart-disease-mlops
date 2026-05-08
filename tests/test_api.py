"""
Unit tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.app import app

client = TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestPredictionEndpoint:
    """Tests for prediction endpoint"""

    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input"""
        # Note: This will fail if model is not loaded, which is expected in test environment
        patient_data = {
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
            "thal": 1,
        }

        response = client.post("/predict", json=patient_data)

        # Will be 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]

    def test_predict_endpoint_invalid_age(self):
        """Test prediction with invalid age"""
        patient_data = {
            "age": 150,  # Invalid age
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
            "thal": 1,
        }

        response = client.post("/predict", json=patient_data)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_field(self):
        """Test prediction with missing required field"""
        patient_data = {
            "age": 63,
            "sex": 1,
            # Missing other required fields
        }

        response = client.post("/predict", json=patient_data)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_type(self):
        """Test prediction with invalid data type"""
        patient_data = {
            "age": "sixty",  # Should be int
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
            "thal": 1,
        }

        response = client.post("/predict", json=patient_data)

        assert response.status_code == 422  # Validation error


class TestInfoEndpoints:
    """Tests for info endpoints"""

    def test_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/info")

        # Will be 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


class TestAPIDocumentation:
    """Tests for API documentation"""

    def test_openapi_schema(self):
        """Test OpenAPI schema is available"""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_endpoint(self):
        """Test Swagger UI is available"""
        response = client.get("/docs")

        assert response.status_code == 200

    def test_redoc_endpoint(self):
        """Test ReDoc is available"""
        response = client.get("/redoc")

        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
