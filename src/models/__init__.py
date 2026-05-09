"""Models module for heart disease prediction"""

from .predictor import HeartDiseasePredictor

# Training utilities pull in heavy deps (mlflow, matplotlib, seaborn) that
# are intentionally absent in serving-only deployments (e.g., HF Spaces).
# Import them lazily so `src.models` itself stays importable everywhere.
try:
    # Re-exported via __all__; flake8 can't see that, so silence F401.
    from .train import train_models, evaluate_model  # noqa: F401

    __all__ = ["train_models", "evaluate_model", "HeartDiseasePredictor"]
except ImportError:
    __all__ = ["HeartDiseasePredictor"]
