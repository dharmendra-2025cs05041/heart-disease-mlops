"""Models module for heart disease prediction"""

from .train import train_models, evaluate_model
from .predictor import HeartDiseasePredictor

__all__ = ['train_models', 'evaluate_model', 'HeartDiseasePredictor']
