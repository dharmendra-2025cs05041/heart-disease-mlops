#!/usr/bin/env python3
"""
Script to download Heart Disease dataset from UCI ML Repository
"""
import os
import sys
import requests
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset URL from UCI Repository
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names based on UCI documentation
COLUMN_NAMES = [
    'age',           # Age in years
    'sex',           # Sex (1 = male; 0 = female)
    'cp',            # Chest pain type (1-4)
    'trestbps',      # Resting blood pressure (mm Hg)
    'chol',          # Serum cholesterol (mg/dl)
    'fbs',           # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    'restecg',       # Resting electrocardiographic results (0-2)
    'thalach',       # Maximum heart rate achieved
    'exang',         # Exercise induced angina (1 = yes; 0 = no)
    'oldpeak',       # ST depression induced by exercise relative to rest
    'slope',         # Slope of the peak exercise ST segment (0-2)
    'ca',            # Number of major vessels colored by fluoroscopy (0-3)
    'thal',          # Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)
    'target'         # Diagnosis of heart disease (0 = no disease, 1-4 = disease)
]

def download_dataset(output_dir: str = "data") -> bool:
    """
    Download the Heart Disease dataset from UCI ML Repository
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Downloading dataset from {DATASET_URL}")
        
        # Download the data
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        
        # Save raw data
        raw_file = os.path.join(output_dir, "heart_disease_raw.data")
        with open(raw_file, 'w') as f:
            f.write(response.text)
        logger.info(f"Raw data saved to {raw_file}")
        
        # Parse and create CSV with proper column names
        from io import StringIO
        df = pd.read_csv(StringIO(response.text), names=COLUMN_NAMES, na_values='?')
        
        # Convert target to binary (0 = no disease, 1 = disease)
        df['target'] = (df['target'] > 0).astype(int)
        
        # Save as CSV
        csv_file = os.path.join(output_dir, "heart_disease.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Processed CSV saved to {csv_file}")
        
        # Display basic info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Target distribution:\n{df['target'].value_counts()}")
        
        # Save dataset info
        info_file = os.path.join(output_dir, "dataset_info.txt")
        with open(info_file, 'w') as f:
            f.write("Heart Disease UCI Dataset Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Source: {DATASET_URL}\n\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Rows: {df.shape[0]}\n")
            f.write(f"Columns: {df.shape[1]}\n\n")
            f.write("Column Names and Descriptions:\n")
            f.write("-" * 50 + "\n")
            for i, col in enumerate(COLUMN_NAMES):
                f.write(f"{i+1}. {col}\n")
            f.write("\n")
            f.write(f"Missing Values:\n{df.isnull().sum()}\n\n")
            f.write(f"Target Distribution:\n{df['target'].value_counts()}\n")
        
        logger.info(f"Dataset information saved to {info_file}")
        logger.info("✓ Dataset download completed successfully!")
        
        return True
        
    except requests.RequestException as e:
        logger.error(f"Error downloading dataset: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def main():
    """Main execution function"""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    
    # Download dataset
    success = download_dataset(str(data_dir))
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
