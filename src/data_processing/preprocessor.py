"""
Data preprocessing module for heart disease prediction
Handles data loading, cleaning, and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
import joblib
from pathlib import Path
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for heart disease dataset
    Handles missing values, encoding, and scaling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.fitted = False
        
    def fit(self, X: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data
        
        Args:
            X: Feature dataframe
            
        Returns:
            self
        """
        logger.info("Fitting preprocessor...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Fit imputer and scaler
        X_imputed = self.imputer.fit_transform(X)
        self.scaler.fit(X_imputed)
        
        self.fitted = True
        logger.info("Preprocessor fitted successfully")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed feature array
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Impute missing values
        X_imputed = self.imputer.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            X: Feature dataframe
            
        Returns:
            Transformed feature array
        """
        return self.fit(X).transform(X)
    
    def save(self, filepath: str):
        """Save preprocessor to disk"""
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """Load preprocessor from disk"""
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load heart disease dataset
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath)
    
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    preprocessor: Optional[DataPreprocessor] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """
    Preprocess data and split into train/test sets
    
    Args:
        df: Input dataframe
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        preprocessor: Optional fitted preprocessor to use
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    logger.info("Starting data preprocessing...")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # Create or use existing preprocessor
    if preprocessor is None:
        preprocessor = DataPreprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)
    else:
        X_train_scaled = preprocessor.transform(X_train)
    
    X_test_scaled = preprocessor.transform(X_test)
    
    logger.info("Data preprocessing completed successfully")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, preprocessor
