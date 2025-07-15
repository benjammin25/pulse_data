import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def _run_model(df, X, y, poly_pipeline, lifestage, months_to_project, model_type):
    """
    Common model execution logic shared by all three models
    """
    # Clean data
    if not np.isfinite(X).all():
        print("Warning: Found non-finite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Display data info
    print(f"📅 Data covers from {df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}")
    print(f"💰 MRR range: ${df['Ending MRR'].min():,.2f} to ${df['Ending MRR'].max():,.2f}")
    
    # Fit model
    poly_pipeline.fit(X, y)
    y_pred = poly_pipeline.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    cv_scores = cross_val_score(poly_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mean = -cv_scores.mean()
    
    print(f"\n🏆 {lifestage} {model_type} Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Mean Absolute Error: ${mae:,.2f}")
    print(f"   Cross-Validation MAE: ${cv_mean:,.2f}")
    
    # Future projections
    print(f"\n🔮 Projecting {months_to_project} months ahead...")
    
    # Create future dates and indices
    last_month_seq = df['Month_Sequential'].max()
    future_months_seq = np.arange(last_month_seq + 1, last_month_seq + 1 + months_to_project)
    last_date = df['Month'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_to_project, freq='M')
    future_month_nums = [future_date.month for future_date in future_dates]
    
    # Get recent values based on feature count
    feature_count = X.shape[1]
    
    if feature_count == 11:  # Full dataset
        future_X = np.column_stack([
            future_months_seq, future_month_nums,
            [df['New MRR'].iloc[-3:].mean()] * months_to_project,  # Use averages for Full
            [df['Churn'].iloc[-3:].mean()] * months_to_project,
            [df['Ending Count'].iloc[-1]] * months_to_project,
            [df['Ending PARPU'].iloc[-1]] * months_to_project,
            [df['Starting MRR'].iloc[-1]] * months_to_project,
            [df['Starting PARPU'].iloc[-1]] * months_to_project,
            [df['ARPU'].iloc[-1]] * months_to_project,
            [df['Churn_Rate'].iloc[-3:].mean()] * months_to_project,
            [0] * months_to_project  # Base = 0 for future (no more expansion)
        ])
        feature_names = ["Month_Sequential", "Month_Number", "New MRR", "Churn", 
                        "Ending Count", "Ending PARPU", "Starting MRR", 
                        "Starting PARPU", "ARPU", "Churn_Rate", "Base"]
    elif feature_count == 7:  # Large segment
        future_X = np.column_stack([
            future_months_seq, future_month_nums,
            [df['New MRR'].iloc[-1]] * months_to_project,
            [df['Churn'].iloc[-1]] * months_to_project,
            [df['Ending Count'].iloc[-1]] * months_to_project,
            [df['Ending PARPU'].iloc[-1]] * months_to_project,
            [df['Churn_Rate'].iloc[-1]] * months_to_project
        ])
        feature_names = ["Month_Sequential", "Month_Number", "New MRR", "Churn", 
                        "Ending Count", "Ending PARPU", "Churn_Rate"]
    elif feature_count == 5:  # Medium segment
        future_X = np.column_stack([
            future_months_seq, 
            [df['New MRR'].iloc[-1]] * months_to_project,
            [df['Churn'].iloc[-1]] * months_to_project,
            [df['Ending Count'].iloc[-1]] * months_to_project,
            [df['Ending PARPU'].iloc[-1]] * months_to_project
        ])
        feature_names = ["Month_Sequential", "New MRR", "Churn", "Ending Count", "Ending PARPU"]
    else:  # Small segment (3 features)
        future_X = np.column_stack([
            future_months_seq,
            [df['New MRR'].iloc[-1]] * months_to_project,
            [df['Churn'].iloc[-1]] * months_to_project
        ])
        feature_names = ["Month_Sequential", "New MRR", "Churn"]
    
    # Make predictions
    future_mrr = poly_pipeline.predict(future_X)
    future_mrr = np.maximum(future_mrr, df['Ending MRR'].iloc[-1] * 0.5)
    
    # Create projections DataFrame
    projections_df = pd.DataFrame({
        'Month': future_dates,
        'Month_Sequential': future_months_seq,
        'Projected_MRR': future_mrr,
        'Projected_ARR': future_mrr * 12,
        'Lifestage': lifestage
    })
    
    # Calculate growth metrics
    current_mrr = df['Ending MRR'].iloc[-1]
    final_projected_mrr = projections_df['Projected_MRR'].iloc[-1]
    total_growth = ((final_projected_mrr - current_mrr) / current_mrr) * 100
    monthly_growth_rate = (final_projected_mrr / current_mrr) ** (1/months_to_project) - 1
    
    print(f"📊 {lifestage} Projection Summary:")
    print(f"   Current MRR: ${current_mrr:,.2f}")
    print(f"   Projected MRR ({months_to_project}m): ${final_projected_mrr:,.2f}")
    print(f"   Total Growth: {total_growth:.1f}%")
    print(f"   Monthly Growth Rate: {monthly_growth_rate:.2%}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Main plot
    plt.subplot(2, 2, (1, 2))
    plt.plot(df['Month'], df['Ending MRR'], 'b-', linewidth=3, label='Historical MRR', marker='o', markersize=4)
    plt.plot(projections_df['Month'], projections_df['Projected_MRR'], 
             'r--', linewidth=3, label='Projected MRR', marker='s', markersize=4)
    
    # Add polynomial trend line for full period
    all_months_seq = np.concatenate([df['Month_Sequential'], future_months_seq])
    all_dates = pd.concat([df['Month'], pd.Series(future_dates)])
    
    # Build all_X based on feature count for trend line
    if feature_count == 11:  # Full dataset
        all_month_nums = [date.month for date in all_dates]
        all_X_trend = np.column_stack([
            all_months_seq, all_month_nums,
            np.concatenate([df['New MRR'], [df['New MRR'].iloc[-3:].mean()] * months_to_project]),
            np.concatenate([df['Churn'], [df['Churn'].iloc[-3:].mean()] * months_to_project]),
            np.concatenate([df['Ending Count'], [df['Ending Count'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Ending PARPU'], [df['Ending PARPU'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Starting MRR'], [df['Starting MRR'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Starting PARPU'], [df['Starting PARPU'].iloc[-1]] * months_to_project]),
            np.concatenate([df['ARPU'], [df['ARPU'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Churn_Rate'], [df['Churn_Rate'].iloc[-3:].mean()] * months_to_project]),
            np.concatenate([df['Base'], [0] * months_to_project])  # Base = 0 for future
    ])
    elif feature_count == 7:  # Large segment
        all_month_nums = [date.month for date in all_dates]
        all_X_trend = np.column_stack([
            all_months_seq, all_month_nums,
            np.concatenate([df['New MRR'], [df['New MRR'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Churn'], [df['Churn'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Ending Count'], [df['Ending Count'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Ending PARPU'], [df['Ending PARPU'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Churn_Rate'], [df['Churn_Rate'].iloc[-1]] * months_to_project])
        ])
    elif feature_count == 5:  # Medium segment
        all_X_trend = np.column_stack([
            all_months_seq,
            np.concatenate([df['New MRR'], [df['New MRR'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Churn'], [df['Churn'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Ending Count'], [df['Ending Count'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Ending PARPU'], [df['Ending PARPU'].iloc[-1]] * months_to_project])
        ])
    else:  # Small segment (3 features)
        all_X_trend = np.column_stack([
            all_months_seq,
            np.concatenate([df['New MRR'], [df['New MRR'].iloc[-1]] * months_to_project]),
            np.concatenate([df['Churn'], [df['Churn'].iloc[-1]] * months_to_project])
        ])
    
    all_predictions = poly_pipeline.predict(all_X_trend)
    plt.plot(all_dates, all_predictions, 'g:', alpha=0.8, linewidth=2, label='Polynomial Trend (Degree 2)')
    
    plt.title(f'MRR Projection - {lifestage} ({model_type})', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('MRR ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Residuals plot
    plt.subplot(2, 2, 3)
    residuals = y - y_pred
    plt.scatter(df['Month'], residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.title('Residuals Plot', fontsize=12)
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('Residuals ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # MRR drivers plot
    plt.subplot(2, 2, 4)
    plt.plot(df['Month'], df['New MRR'], 'green', linewidth=2, alpha=0.8, label='New MRR')
    plt.plot(df['Month'], -df['Churn'], 'red', linewidth=2, alpha=0.8, label='Churn (negative)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'{lifestage} - MRR Drivers', fontsize=12)
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('MRR Change ($)', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'lifestage': lifestage,
        'model_type': model_type,
        'model': poly_pipeline,
        'projections': projections_df,
        'performance': {'r2': r2, 'mae': mae, 'rmse': rmse, 'cv_mae': cv_mean},
        'monthly_growth_rate': monthly_growth_rate,
        'feature_names': feature_names,
        'current_metrics': {
            'current_mrr': current_mrr,
            'projected_mrr': final_projected_mrr,
            'total_growth': total_growth
        }
    }