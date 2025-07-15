import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from .core import _run_model
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def full_dataset_model(csv_file_path, months_to_project=12):
    """
    Comprehensive model for Full dataset
    Uses 10 features with very high alpha for ultra-conservative projections
    """
    lifestage = Path(csv_file_path).parts[-3]
    print(f"🎯 Full Dataset Model for {lifestage}")
    print(f"   Tier: Comprehensive | Features: 11 | Alpha: 1.0")
    df_base = pd.read_csv("raw_data/fake_base_per_month.csv", encoding='latin-1', low_memory=False)
    # Load and prepare data
    df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
    df = df.merge(df_base, on='Month_Sequential', how='left')
    df = df[:-1]
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Calculate Churn Rate
    df['Churn_Rate'] = np.where(df['Starting Count'] > 0, 
                                df['Churn Count'] / df['Starting Count'], 0)
    df['Churn_Rate'] = df['Churn_Rate'].fillna(0)
    df['Churn_Rate'] = df['Churn_Rate'].replace([np.inf, -np.inf], 0)
    
    # Comprehensive feature set (10 features)
    X = df[["Month_Sequential", "Month_Number", "New MRR", "Churn",
            "Ending Count", "Ending PARPU", "Starting MRR", 
            "Starting PARPU", "ARPU", "Churn_Rate", "Base"]].values
    y = df["Ending MRR"].values
    
    # Clean data
    if not np.isfinite(X).all():
        print("Warning: Found non-finite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create model with very high alpha for ultra-conservative projections
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lasso', Lasso(alpha=1.0))
    ])
    
    return _run_model(df, X, y, poly_pipeline, lifestage, months_to_project, "Full Dataset")


def large_segment_model(csv_file_path, months_to_project=12):
    """
    High-capacity model for large segments (M2, M3, Y1)
    Uses 7 features with low alpha for complex pattern detection
    """
    lifestage = Path(csv_file_path).parts[-3]
    print(f"🎯 Large Segment Model for {lifestage}")
    print(f"   Tier: High-Capacity | Features: 7 | Alpha: 2.0")
    
    # Load and prepare data
    df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
    df = df[:-1]
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Calculate Churn Rate
    df['Churn_Rate'] = np.where(df['Starting Count'] > 0, 
                                df['Churn Count'] / df['Starting Count'], 0)
    df['Churn_Rate'] = df['Churn_Rate'].fillna(0)
    df['Churn_Rate'] = df['Churn_Rate'].replace([np.inf, -np.inf], 0)
    
    # High-capacity feature set
    X = df[["Month_Sequential", "Month_Number", "New MRR", "Churn", 
            "Ending Count", "Ending PARPU", "Churn_Rate"]].values
    y = df["Ending MRR"].values
    
    # Create model with low alpha
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lasso', Lasso(alpha=2.0))
    ])
    
    return _run_model(df, X, y, poly_pipeline, lifestage, months_to_project, "Large Segment")

def medium_segment_model(csv_file_path, months_to_project=12):
    """
    Balanced model for medium segments (F1, Y2, F2, M1)
    Uses 5 features with moderate alpha for balanced patterns
    """
    lifestage = Path(csv_file_path).parts[-3]
    print(f"🎯 Medium Segment Model for {lifestage}")
    print(f"   Tier: Balanced | Features: 5 | Alpha: 5.0")
    
    # Load and prepare data
    df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
    df = df[:-1]
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Balanced feature set
    X = df[["Month_Sequential", "New MRR", "Churn", 
            "Ending Count", "Ending PARPU"]].values
    y = df["Ending MRR"].values
    
    # Create model with moderate alpha
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lasso', Lasso(alpha=5.0))
    ])
    
    return _run_model(df, X, y, poly_pipeline, lifestage, months_to_project, "Medium Segment")

def small_segment_model(csv_file_path, months_to_project=12):
    """
    Simple-stable model for small segments (F3, M4, Y3)
    Uses 3 features with high alpha for stability
    """
    lifestage = Path(csv_file_path).parts[-3]
    print(f"🎯 Small Segment Model for {lifestage}")
    print(f"   Tier: Simple-Stable | Features: 3 | Alpha: 50.0")
    
    # Load and prepare data
    df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
    df = df[:-1]
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Simple feature set
    X = df[["Month_Sequential", "New MRR", "Churn"]].values
    y = df["Ending MRR"].values
    
    # Create model with high alpha
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lasso', Lasso(alpha=50.0))
    ])
    
    return _run_model(df, X, y, poly_pipeline, lifestage, months_to_project, "Small Segment")


