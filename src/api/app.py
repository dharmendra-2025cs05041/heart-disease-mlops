"""
FastAPI application for heart disease prediction
Production-ready API with logging and monitoring
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
import time
import logging
import sys
from pathlib import Path
from typing import Optional
import uvicorn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predictor import HeartDiseasePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds", "Prediction request latency"
)
PREDICTION_COUNTER = Counter(
    "predictions_by_class", "Predictions by class", ["predicted_class"]
)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered API for predicting heart disease risk",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Pydantic models for request/response validation
class PatientData(BaseModel):
    """Input features for heart disease prediction"""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(
        ..., ge=50, le=250, description="Resting blood pressure (mm Hg)"
    )
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)"
    )
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)"
    )
    oldpeak: float = Field(
        ..., ge=0, le=10, description="ST depression induced by exercise"
    )
    slope: int = Field(
        ..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)"
    )
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    prediction: int = Field(
        ..., description="Predicted class (0=No Disease, 1=Disease)"
    )
    probability_no_disease: float = Field(..., description="Probability of no disease")
    probability_disease: float = Field(..., description="Probability of disease")
    confidence: float = Field(..., description="Prediction confidence")
    risk_level: str = Field(..., description="Risk level (Low/Moderate/High)")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    version: str


# Global predictor instance
predictor: Optional[HeartDiseasePredictor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    try:
        # Prefer the promoted final model; fall back to random_forest.pkl
        # for backward compatibility with older training runs.
        candidates = ["models/final_model.pkl", "models/random_forest.pkl"]
        model_path = next((p for p in candidates if Path(p).exists()), candidates[-1])
        preprocessor_path = "models/preprocessor.pkl"

        predictor = HeartDiseasePredictor(model_path, preprocessor_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup, but log the error


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests"""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {duration:.4f}s"
    )

    return response


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient_data: PatientData):
    """
    Predict heart disease risk for a patient

    Args:
        patient_data: Patient health metrics

    Returns:
        Prediction result with probabilities and risk level
    """
    REQUEST_COUNT.inc()

    if predictor is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Start timing
        start_time = time.time()

        # Convert to dict
        features = patient_data.dict()

        # Make prediction
        result = predictor.predict_proba(features)

        # Update metrics
        PREDICTION_COUNTER.labels(predicted_class=str(result["prediction"])).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        # Log prediction
        logger.info(
            f"Prediction made - Class: {result['prediction']}, "
            f"Confidence: {result['confidence']:.4f}, "
            f"Risk: {result['risk_level']}"
        )

        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/info", tags=["Info"])
async def model_info():
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not available")

    return {
        "model_type": str(type(predictor.model).__name__),
        "features": predictor.feature_names,
        "n_features": len(predictor.feature_names),
        "description": "Heart disease prediction model trained on UCI dataset",
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
