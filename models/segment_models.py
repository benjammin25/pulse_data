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

def ending_mrr_model(csv_file_path, months_to_project=12):
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
    segmentation_features = create_lifecycle_grouped_features(ratio_type="net_new_mrr")

    # Add segmentation features to main dataframe
    for feature_name, feature_data in segmentation_features.items():
        df[feature_name] = feature_data



    # Add this to your full_dataset_model function after loading segmentation features:

    # Create quality performers from the actual Net New MRR values
    df['qualityperformers_net_new_mrr'] = df['high_performers_net_new_mrr'] + df['moderate_performers_net_new_mrr']

    # Then calculate the ratio
    df['total_segment_mrr'] = df['qualityperformers_net_new_mrr'] + df['underperformers_net_new_mrr']
    df['qualityperformers_ratio'] = df['qualityperformers_net_new_mrr'] / df['total_segment_mrr']
    df['underperformers_ratio'] = df['underperformers_net_new_mrr'] / df['total_segment_mrr']

    # Use ratios in your model instead of absolute values
    X = df[["Month_Sequential", "Churn", "Ending Count", "ARPU", "Month_Number", "Expansion", "Base", "Site_Opens",
            "qualityperformers_ratio", "underperformers_ratio"
        ]].values  # 11 features

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

def net_new_mrr_model(csv_file_path, months_to_project=12):
    """
    Polynomial model for Full dataset with performance tier segmentation features
    Uses degree=1 polynomial features for capturing linear relationships
    """
    performance_tier = Path(csv_file_path).parts[-3]
    print(f"🎯 Polynomial Model for {performance_tier}")
    print(f"   Tier: Performance-Based | Features: 10 | Model: Polynomial (degree=1) | Alpha: 1.0")

    # Load and prepare main data
    df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
    df = df[:-1]  # Remove last row
    df['Month'] = pd.to_datetime(df['Month'])

    # Add Month Number for seasonal effects
    df['Month_Number'] = df['Month'].dt.month
    df["Quarter"] = df["Month"].dt.quarter
    # Create performance tier segmentation features
    segmentation_features = create_lifecycle_grouped_features(ratio_type="new_mrr")

    # Add segmentation features to main dataframe
    for feature_name, feature_data in segmentation_features.items():
        df[feature_name] = feature_data


    # Create quality performers from the actual Net New MRR values
    df['qualityperformers_new_mrr'] = df['high_performers_new_mrr'] + df['moderate_performers_new_mrr']

    # Then calculate the ratio
    df['total_segment_mrr'] = df['qualityperformers_new_mrr'] + df['underperformers_new_mrr']
    df['qualityperformers_ratio'] = df['qualityperformers_new_mrr'] / df['total_segment_mrr']
    df['underperformers_ratio'] = df['underperformers_new_mrr'] / df['total_segment_mrr']

    # High performers vs underperformers (biggest impact expected)
    df['high_to_under_direct'] = df['high_performers_new_mrr'] / (df['underperformers_new_mrr'].abs() + 100)
    # High performers vs moderate performers direct ratio
    df['high_to_moderate_direct'] = df['high_performers_new_mrr'] / (df['moderate_performers_new_mrr'] + 1000)

    df['H2_indicator'] = (df['Month_Number'] > 6).astype(int)  # 1 for Jul-Dec, 0 for Jan-Jun

    df['Month_Sequential_squared'] = df['Month_Sequential'] ** 2

    df['Quality_Season_interaction'] = df['Month_Number'].isin([3, 5, 6]).astype(int) * df['New Count']
    df['quality_weighted_performance'] = df['qualityperformers_ratio'] * df['total_segment_mrr']
    # Add correlation-based seasonal indicators
    df['Good_Months'] = df['Month_Number'].isin([3, 5, 6]).astype(int)    # Mar, May, Jun (positive correlations)
    df['Bad_Months'] = df['Month_Number'].isin([7, 8, 10, 12]).astype(int)  # Jul, Aug, Oct, Dec (negative correlations)

    # Feature selection 
    X = df[["Month_Sequential","Month_Sequential_squared", "Month_Number", "Good_Months", "Bad_Months",
        "H2_indicator", "qualityperformers_ratio", "quality_weighted_performance","underperformers_ratio",
         "high_to_under_direct"]].values  # 11 features
    
    y = df["Net New MRR"].values


    # Clean data
    if not np.isfinite(X).all():
        print("Warning: Found non-finite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Create polynomial model pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=1, include_bias=False)),
        ('lasso', Lasso(alpha=1.0))  # Higher alpha for polynomial to prevent overfitting
    ])

    return _run_model(df, X, y, poly_pipeline, performance_tier, months_to_project, "Linear Model", "Net New MRR")


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




