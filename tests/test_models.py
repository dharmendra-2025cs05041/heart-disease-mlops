"""
Unit tests for models module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.models.predictor import HeartDiseasePredictor


class TestEvaluateModel:
    """Tests for evaluate_model function"""

    def test_evaluate_model_returns_metrics(self):
        """Test that evaluate_model returns expected metrics"""
        # Create dummy data
        X_train = np.random.rand(100, 10)
        X_test = np.random.rand(20, 10)
        y_train = np.random.randint(0, 2, 100)
        y_test = np.random.randint(0, 2, 20)

        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Check all expected metrics are present
        expected_metrics = [
            "train_accuracy",
            "test_accuracy",
            "train_precision",
            "test_precision",
            "train_recall",
            "test_recall",
            "train_f1",
            "test_f1",
            "train_roc_auc",
            "test_roc_auc",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0 <= metrics[metric] <= 1 or metric.endswith("_std")


class TestPlotFunctions:
    """Tests for plotting functions"""

    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])

        fig = plot_confusion_matrix(y_true, y_pred)

        assert fig is not None
        assert len(fig.axes) == 2  # Plot and colorbar

    def test_plot_roc_curve(self):
        """Test ROC curve plotting"""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        fig = plot_roc_curve(y_true, y_proba)

        assert fig is not None
        assert len(fig.axes) == 1


class TestHeartDiseasePredictor:
    """Tests for HeartDiseasePredictor class"""

    @pytest.fixture
    def mock_predictor(self, tmp_path):
        """Create a mock predictor for testing"""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from src.data_processing.preprocessor import DataPreprocessor

        # Create and save a dummy model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 13)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)

        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)

        # Create and save a dummy preprocessor
        preprocessor = DataPreprocessor()
        df_dummy = pd.DataFrame(
            X_dummy,
            columns=[
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
            ],
        )
        preprocessor.fit(df_dummy)

        preprocessor_path = tmp_path / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)

        return HeartDiseasePredictor(str(model_path), str(preprocessor_path))

    def test_predictor_initialization(self, mock_predictor):
        """Test predictor initialization"""
        assert mock_predictor is not None
        assert mock_predictor.model is not None
        assert mock_predictor.preprocessor is not None
        assert len(mock_predictor.feature_names) == 13

    def test_predict_with_dict(self, mock_predictor):
        """Test prediction with dictionary input"""
        features = {
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

        prediction = mock_predictor.predict(features)

        assert isinstance(prediction, int)
        assert prediction in [0, 1]

    def test_predict_proba_with_dict(self, mock_predictor):
        """Test prediction with probabilities"""
        features = {
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

        result = mock_predictor.predict_proba(features)

        assert "prediction" in result
        assert "probability_no_disease" in result
        assert "probability_disease" in result
        assert "confidence" in result
        assert "risk_level" in result

        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability_no_disease"] <= 1
        assert 0 <= result["probability_disease"] <= 1
        assert result["risk_level"] in ["Low", "Moderate", "High"]

    def test_predict_batch(self, mock_predictor):
        """Test batch prediction"""
        features_list = [
            {
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
            },
            {
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
                "thal": 2,
            },
        ]

        results = mock_predictor.predict_batch(features_list)

        assert len(results) == 2
        assert all("prediction" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
