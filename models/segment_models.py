import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from .core import _run_model
from .group_segments import create_lifecycle_grouped_features
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def full_dataset_model(csv_file_path, months_to_project=12):
    """
    Enhanced comprehensive model for Full dataset with performance tier segmentation features
    """
    performance_tier = Path(csv_file_path).parts[-3]
    print(f"🎯 Enhanced Full Dataset Model for {performance_tier}")
    print(f"   Tier: Performance-Based | Features: 10 | Alpha: 15.0")

    # Load and prepare main data
    df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
    df = df[:-1]  # Remove last row
    df['Month'] = pd.to_datetime(df['Month'])

    # Add Month Number for seasonal effects
    df['Month_Number'] = df['Month'].dt.month

    # Create performance tier segmentation features
    segmentation_features = create_lifecycle_grouped_features()

    # Add segmentation features to main dataframe
    for feature_name, feature_data in segmentation_features.items():
        df[feature_name] = feature_data

    # Performance tier feature set (10 features)
    X = df[["Month_Sequential", "New MRR", "Churn", "Ending Count", "ARPU", "Month_Number", "Expansion",   
            "high_performers_net_new_mrr", "moderate_performers_net_new_mrr", "underperformers_net_new_mrr"
           ]].values
    y = df["Ending MRR"].values

    # Clean data
    if not np.isfinite(X).all():
        print("Warning: Found non-finite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Create model with linear features
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=1, include_bias=False)),
        ('lasso', Lasso(alpha=5.0))
    ])

    return _run_model(df, X, y, poly_pipeline, performance_tier, months_to_project, "Performance Tier Model")

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
    
  
    
    # High-capacity feature set
    X = df[["Month_Sequential", "Month_Number", "New MRR", "Churn", 
            "Ending Count", "Ending PARPU", "Starting MRR"]].values
    y = df["Ending MRR"].values
    
    # Create model with low alpha
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('lasso', Lasso(alpha=2.0))
    ])
    
    return _run_model(df, X, y, poly_pipeline, lifestage, months_to_project, "Large Segment")




