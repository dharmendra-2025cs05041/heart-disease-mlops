"""
Model predictor for inference
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Union, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HeartDiseasePredictor:
    """
    Predictor class for heart disease prediction
    Handles model loading and inference
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initialize predictor with model and preprocessor

        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        logger.info(f"Loading preprocessor from {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        # Feature names expected by the model
        self.feature_names = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]

        logger.info("Predictor initialized successfully")

    def predict(self, features: Union[Dict, pd.DataFrame, np.ndarray]) -> int:
        """
        Make prediction for a single sample

        Args:
            features: Input features (dict, DataFrame, or array)

        Returns:
            Predicted class (0 or 1)
        """
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            features_df = pd.DataFrame([features], columns=self.feature_names)
        else:
            features_df = features

        # Ensure correct feature order
        features_df = features_df[self.feature_names]

        # Preprocess
        features_scaled = self.preprocessor.transform(features_df)

        # Predict
        prediction = self.model.predict(features_scaled)[0]

        return int(prediction)

    def predict_proba(
        self, features: Union[Dict, pd.DataFrame, np.ndarray]
    ) -> Dict[str, float]:
        """
        Make prediction with probability

        Args:
            features: Input features (dict, DataFrame, or array)

        Returns:
            Dictionary with prediction and probabilities
        """
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            features_df = pd.DataFrame([features], columns=self.feature_names)
        else:
            features_df = features

        # Ensure correct feature order
        features_df = features_df[self.feature_names]

        # Preprocess
        features_scaled = self.preprocessor.transform(features_df)

        # Predict
        prediction = self.model.predict(features_scaled)[0]

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features_scaled)[0]
            prob_no_disease = float(probabilities[0])
            prob_disease = float(probabilities[1])
        else:
            # For models without predict_proba (e.g., SVM with probability=False)
            prob_no_disease = 1.0 if prediction == 0 else 0.0
            prob_disease = 1.0 if prediction == 1 else 0.0

        return {
            "prediction": int(prediction),
            "probability_no_disease": prob_no_disease,
            "probability_disease": prob_disease,
            "confidence": max(prob_no_disease, prob_disease),
            "risk_level": (
                "High"
                if prob_disease > 0.7
                else "Moderate" if prob_disease > 0.4 else "Low"
            ),
        }

    def predict_batch(self, features_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple samples

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            result = self.predict_proba(features)
            results.append(result)

        return results


if __name__ == "__main__":
    """Example usage"""
    # Example prediction
    predictor = HeartDiseasePredictor(
        model_path="models/random_forest.pkl",
        preprocessor_path="models/preprocessor.pkl",
    )

    # Sample patient data
    sample_patient = {
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

    result = predictor.predict_proba(sample_patient)
    print(f"Prediction Result: {result}")
