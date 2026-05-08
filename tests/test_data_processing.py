"""
Unit tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.preprocessor import (
    DataPreprocessor,
    load_data,
    preprocess_data,
)


class TestDataPreprocessor:
    """Tests for DataPreprocessor class"""

    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized"""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert not preprocessor.fitted

    def test_preprocessor_fit(self):
        """Test preprocessor fit method"""
        # Create sample data
        X = pd.DataFrame(
            {"age": [50, 60, 70], "chol": [200, 250, 300], "trestbps": [120, 130, 140]}
        )

        preprocessor = DataPreprocessor()
        preprocessor.fit(X)

        assert preprocessor.fitted
        assert preprocessor.feature_names == ["age", "chol", "trestbps"]

    def test_preprocessor_transform(self):
        """Test preprocessor transform method"""
        # Create sample data
        X = pd.DataFrame(
            {"age": [50, 60, 70], "chol": [200, 250, 300], "trestbps": [120, 130, 140]}
        )

        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert isinstance(X_scaled, np.ndarray)
        # Check if scaled (mean should be close to 0)
        assert abs(X_scaled.mean()) < 1.0

    def test_preprocessor_transform_without_fit(self):
        """Test that transform fails without fit"""
        X = pd.DataFrame({"age": [50, 60, 70]})

        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError):
            preprocessor.transform(X)

    def test_preprocessor_handles_missing_values(self):
        """Test preprocessor handles missing values"""
        X = pd.DataFrame(
            {
                "age": [50, np.nan, 70],
                "chol": [200, 250, np.nan],
                "trestbps": [120, 130, 140],
            }
        )

        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X)

        # Check no NaN values after preprocessing
        assert not np.isnan(X_scaled).any()


class TestLoadData:
    """Tests for load_data function"""

    def test_load_data_returns_dataframe(self, tmp_path):
        """Test that load_data returns a DataFrame"""
        # Create a temporary CSV file
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({"age": [50, 60, 70], "target": [0, 1, 0]})
        df.to_csv(csv_file, index=False)

        # Load the data
        loaded_df = load_data(str(csv_file))

        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (3, 2)


class TestPreprocessData:
    """Tests for preprocess_data function"""

    def test_preprocess_data_splits_correctly(self):
        """Test that data is split correctly"""
        # Create sample data
        df = pd.DataFrame(
            {"age": range(100), "chol": range(100, 200), "target": [0, 1] * 50}
        )

        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            df, test_size=0.2, random_state=42
        )

        # Check shapes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Check preprocessor is fitted
        assert preprocessor.fitted

    def test_preprocess_data_stratifies(self):
        """Test that stratification maintains class balance"""
        # Create imbalanced data
        df = pd.DataFrame(
            {"age": range(100), "chol": range(100, 200), "target": [0] * 70 + [1] * 30}
        )

        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            df, test_size=0.2, random_state=42
        )

        # Check class proportions are similar
        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)

        assert abs(train_ratio - test_ratio) < 0.1  # Within 10%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
