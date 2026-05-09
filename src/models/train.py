"""
Model training module with MLflow experiment tracking
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from pathlib import Path
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Evaluate model performance and return metrics

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name of the model

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_train_proba = model.decision_function(X_train)
        y_test_proba = model.decision_function(X_test)

    # Calculate metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred, average="binary"),
        "test_precision": precision_score(y_test, y_test_pred, average="binary"),
        "train_recall": recall_score(y_train, y_train_pred, average="binary"),
        "test_recall": recall_score(y_test, y_test_pred, average="binary"),
        "train_f1": f1_score(y_train, y_train_pred, average="binary"),
        "test_f1": f1_score(y_test, y_test_pred, average="binary"),
        "train_roc_auc": roc_auc_score(y_train, y_train_proba),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba),
    }

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    metrics["cv_roc_auc_mean"] = cv_scores.mean()
    metrics["cv_roc_auc_std"] = cv_scores.std()

    # Log metrics
    logger.info(f"{model_name} Performance:")
    logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"  Test Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Test Recall: {metrics['test_recall']:.4f}")
    logger.info(f"  Test F1-Score: {metrics['test_f1']:.4f}")
    logger.info(f"  Test ROC-AUC: {metrics['test_roc_auc']:.4f}")
    logger.info(
        f"  CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']:.4f})"
    )

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix"
) -> plt.Figure:
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    return fig


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, title: str = "ROC Curve"
) -> plt.Figure:
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return fig


def train_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str = "heart-disease-prediction",
    models_dir: str = "models",
) -> Dict[str, Any]:
    """
    Train multiple models with MLflow tracking

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        experiment_name: MLflow experiment name
        models_dir: Directory to save models

    Returns:
        Dictionary of trained models and their metrics
    """
    # Create models directory
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Define models to train
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
        },
        "SVM": {
            "model": SVC(random_state=42, probability=True),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
    }

    results = {}
    best_model = None
    best_score = 0

    for model_name, config in models_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*60}")

        with mlflow.start_run(run_name=model_name):
            # Hyperparameter tuning with GridSearchCV
            logger.info("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                config["model"],
                config["params"],
                cv=5,
                scoring="roc_auc",
                # n_jobs=1 (serial) — parallel workers add no measurable speedup
                # on this 303-row dataset and avoid loky/ResourceTracker
                # shutdown warnings on macOS Python 3.12 (ChildProcessError 10).
                n_jobs=1,
                verbose=0,
            )

            grid_search.fit(X_train, y_train)

            # Best model
            model = grid_search.best_estimator_

            # Log parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("model_type", model_name)

            # Evaluate model
            metrics = evaluate_model(
                model, X_train, X_test, y_train, y_test, model_name
            )

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Generate and log plots
            # Confusion Matrix
            y_test_pred = model.predict(X_test)
            cm_fig = plot_confusion_matrix(
                y_test, y_test_pred, f"{model_name} - Confusion Matrix"
            )
            mlflow.log_figure(cm_fig, f"{model_name}_confusion_matrix.png")
            plt.close(cm_fig)

            # ROC Curve
            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = model.decision_function(X_test)

            roc_fig = plot_roc_curve(y_test, y_test_proba, f"{model_name} - ROC Curve")
            mlflow.log_figure(roc_fig, f"{model_name}_roc_curve.png")
            plt.close(roc_fig)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save model locally
            model_path = (
                Path(models_dir) / f"{model_name.replace(' ', '_').lower()}.pkl"
            )
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            # Store results
            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "best_params": grid_search.best_params_,
                "model_path": str(model_path),
            }

            # Track best model
            if metrics["test_roc_auc"] > best_score:
                best_score = metrics["test_roc_auc"]
                best_model = model_name

            logger.info(f"✓ {model_name} training completed")

    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_model} with ROC-AUC: {best_score:.4f}")
    logger.info(f"{'='*60}\n")

    # Save best model info
    results["best_model_name"] = best_model
    results["best_model_score"] = best_score

    # Promote the winning estimator to a stable filename so the API
    # does not need to know which algorithm won this run.
    if best_model and best_model in results:
        final_path = Path(models_dir) / "final_model.pkl"
        joblib.dump(results[best_model]["model"], final_path)
        logger.info(f"Saved final model to {final_path}")

    return results


if __name__ == "__main__":
    """Example usage"""
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent))

    from src.data_processing.preprocessor import load_data, preprocess_data

    # Load and preprocess data
    df = load_data("data/heart_disease.csv")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Save preprocessor
    preprocessor.save("models/preprocessor.pkl")

    logger.info("Training pipeline completed successfully!")
