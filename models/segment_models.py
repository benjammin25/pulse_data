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



    df['high_performers_net_new_mrr'] = df['high_performers_net_new_mrr'].fillna(0)
    df['moderate_performers_net_new_mrr'] = df['moderate_performers_net_new_mrr'].fillna(0)
    df['underperformers_net_new_mrr'] = df['underperformers_net_new_mrr'].fillna(0)

    # Create quality performers from the actual Net New MRR values
    df['qualityperformers_net_new_mrr'] = df['high_performers_net_new_mrr'] + df['moderate_performers_net_new_mrr']

    # Then calculate the ratio
    df['total_segment_mrr'] = df['qualityperformers_net_new_mrr'] + df['underperformers_net_new_mrr']

    df['total_segment_mrr'] = np.where(df['total_segment_mrr'] == 0, 1, df['total_segment_mrr'])

    df['qualityperformers_ratio'] = df['qualityperformers_net_new_mrr'] / df['total_segment_mrr']
    df['underperformers_ratio'] = df['underperformers_net_new_mrr'] / df['total_segment_mrr']

    # High performers vs underperformers with robust NaN handling
    high_perf_clean = df['high_performers_net_new_mrr'].fillna(0)
    under_perf_clean = df['underperformers_net_new_mrr'].fillna(0)
    
    # Robust division with multiple protections
    denominator = under_perf_clean.abs() + 100
    denominator = np.where(denominator == 0, 100, denominator)  # Extra protection
    df['high_to_under_direct'] = high_perf_clean / denominator
    
    # Final cleanup for any remaining NaN/inf values
    df['high_to_under_direct'] = df['high_to_under_direct'].fillna(0)
    df['high_to_under_direct'] = np.where(np.isinf(df['high_to_under_direct']), 0, df['high_to_under_direct'])

    df['Month_Sequential'] = df['Month_Sequential'].fillna(df['Month_Sequential'].median())
    # Use ratios in your model instead of absolute values
    X = df[["Month_Sequential",  "Churn", "Ending Count", "ARPU", "Month_Number", "Expansion"]].values  # 11 features

    y = df["Ending MRR"].values

    # Clean data
    if not np.isfinite(X).all():
        print("Warning: Found non-finite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Create model with linear features
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=1, include_bias=False)),
        ('lasso', Lasso(alpha=15.0))
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

    # Add segmentation features to main dataframe with NaN handling
    for feature_name, feature_data in segmentation_features.items():
        df[feature_name] = pd.Series(feature_data).fillna(0)  # Fill NaN with 0

    # Create quality performers from the actual Net New MRR values with NaN handling
    df['high_performers_new_mrr'] = df['high_performers_new_mrr'].fillna(0)
    df['moderate_performers_new_mrr'] = df['moderate_performers_new_mrr'].fillna(0)
    df['underperformers_new_mrr'] = df['underperformers_new_mrr'].fillna(0)
    
    df['qualityperformers_new_mrr'] = df['high_performers_new_mrr'] + df['moderate_performers_new_mrr']

    # Calculate ratios with protection against division by zero and NaN
    df['total_segment_mrr'] = df['qualityperformers_new_mrr'] + df['underperformers_new_mrr']
    
    # Protect against division by zero
    df['total_segment_mrr'] = np.where(df['total_segment_mrr'] == 0, 1, df['total_segment_mrr'])
    
    df['qualityperformers_ratio'] = df['qualityperformers_new_mrr'] / df['total_segment_mrr']
    df['underperformers_ratio'] = df['underperformers_new_mrr'] / df['total_segment_mrr']
    
    # Fill any remaining NaN values in ratios
    df['qualityperformers_ratio'] = df['qualityperformers_ratio'].fillna(0)
    df['underperformers_ratio'] = df['underperformers_ratio'].fillna(0)

    # High performers vs underperformers with robust NaN handling
    high_perf_clean = df['high_performers_new_mrr'].fillna(0)
    under_perf_clean = df['underperformers_new_mrr'].fillna(0)
    
    # Robust division with multiple protections
    denominator = under_perf_clean.abs() + 100
    denominator = np.where(denominator == 0, 100, denominator)  # Extra protection
    df['high_to_under_direct'] = high_perf_clean / denominator
    
    # Final cleanup for any remaining NaN/inf values
    df['high_to_under_direct'] = df['high_to_under_direct'].fillna(0)
    df['high_to_under_direct'] = np.where(np.isinf(df['high_to_under_direct']), 0, df['high_to_under_direct'])

    # Other feature creation with NaN protection
    df['H2_indicator'] = (df['Month_Number'] > 6).astype(int)
    df['Month_Sequential_squared'] = df['Month_Sequential'] ** 2
    
    # Handle potential NaN in Month_Sequential
    df['Month_Sequential'] = df['Month_Sequential'].fillna(df['Month_Sequential'].median())
    df['Month_Sequential_squared'] = df['Month_Sequential_squared'].fillna(df['Month_Sequential_squared'].median())
    
    df['quality_weighted_performance'] = df['qualityperformers_ratio'] * df['total_segment_mrr']
    df['quality_weighted_performance'] = df['quality_weighted_performance'].fillna(0)
    
    # Add correlation-based seasonal indicators
    df['Good_Months'] = df['Month_Number'].isin([3, 5, 6]).astype(int)
    df['Bad_Months'] = df['Month_Number'].isin([7, 8, 10, 12]).astype(int)
    
    # Handle NaN in New Count
    df['New Count'] = df['New Count'].fillna(df['New Count'].median())

    # Feature selection with final NaN check
    feature_columns = ["Month_Sequential", "Month_Sequential_squared", "Month_Number", "Good_Months", "Bad_Months",
                      "H2_indicator", "qualityperformers_ratio", "quality_weighted_performance", "underperformers_ratio",
                      "high_to_under_direct", "New Count"]
    
    # Final NaN cleanup for all features
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = np.where(np.isinf(df[col]), 0, df[col])
    
    X = df[feature_columns].values
    y = df["Net New MRR"].fillna(0).values  # Handle target variable NaN too

    # Final data cleaning with more robust approach
    if not np.isfinite(X).all():
        print("Warning: Found non-finite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.isfinite(y).all():
        print("Warning: Found non-finite values in target. Cleaning...")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Verification
    print(f"✅ Data verification:")
    print(f"   Features shape: {X.shape}")
    print(f"   NaN in features: {np.isnan(X).sum()}")
    print(f"   Inf in features: {np.isinf(X).sum()}")
    print(f"   NaN in target: {np.isnan(y).sum()}")

    # Create polynomial model pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=1, include_bias=False)),
        ('lasso', Lasso(alpha=1.0))
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




