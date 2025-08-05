import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def _run_model(df, X, y, poly_pipeline, performance_tier, months_to_project, model_type, target_variable="Ending MRR"):
    """
    Model execution logic for performance tier segmentation model
    """
    # Clean data
    if not np.isfinite(X).all():
        print("Warning: Found non-finite values in features. Cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.isnan(X).any():
        print("Warning: Still found NaN values after cleaning. Replacing with zeros...")
        X = np.where(np.isnan(X), 0.0, X)
    
    # Display data info
    print(f"📅 Data covers from {df['Month'].min().strftime('%Y-%m')} to {df['Month'].max().strftime('%Y-%m')}")
    print(f"🎯 Target Variable: {target_variable}")
    print(f"💰 Target range: ${y.min():,.2f} to ${y.max():,.2f}")
    
    # Fit model
    poly_pipeline.fit(X, y)
    y_pred = poly_pipeline.predict(X)

    # Display coefficients
    print("\n📊 Model Coefficients:")
    coefficients = poly_pipeline.named_steps['lasso'].coef_
    
    # Dynamic feature names based on target variable
    if target_variable == "Ending MRR":
        feature_names = ["Month_Sequential", "Churn", "Ending Count", "ARPU", "Month_Number", "Expansion"]
    elif target_variable == "Net New MRR":
        feature_names = ["Month_Sequential", "Month_Number", "Month_Sequential_squared", "Good_Months", "Bad_Months",
                        "H2_indicator",  "qualityperformers_ratio", "quality_weighted_performance", "underperformers_ratio",
                         "high_to_under_direct", "New Count"]
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    for name, coef in zip(feature_names, coefficients):
        if abs(coef) > 0.01:
            print(f"   {name}: {coef:.2f}")

    # Calculate performance metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    cv_scores = cross_val_score(poly_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mean = -cv_scores.mean()
    
    print(f"\n🏆 {performance_tier} {model_type} Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Mean Absolute Error: ${mae:,.2f}")
    print(f"   Cross-Validation MAE: ${cv_mean:,.2f}")
    
    # Future projections
    print(f"\n🔮 Projecting {months_to_project} months ahead...")

    # In your code, add these calculations:
    print(df['high_to_under_direct'].tail())
    print(df['underperformers_new_mrr'].tail())
    print(df['high_performers_new_mrr'].tail())
    
    # Create future dates
    last_month_seq = df['Month_Sequential'].max()
    last_date = df['Month'].max()
    future_months_seq = np.arange(last_month_seq + 1, last_month_seq + 1 + months_to_project)
    future_months_seq_squared = future_months_seq ** 2
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_to_project, freq='ME')
    future_month_nums = [future_date.month for future_date in future_dates]
    future_quarters = [future_date.quarter for future_date in future_dates]
    future_h2_indicators = [int(month_num > 6) for month_num in future_month_nums]

    # Build future feature matrix based on target variable
    if target_variable == "Ending MRR":
        # Ending MRR future projections - 11 features
        churn_improvement = np.linspace(df['Churn'].iloc[-1], df['Churn'].iloc[-1] * 1.00, months_to_project)
        customer_growth = np.linspace(df['Ending Count'].iloc[-1], df['Ending Count'].iloc[-1] * 1.00, months_to_project)
        arpu_growth = np.linspace(df['ARPU'].iloc[-1], df['ARPU'].iloc[-1] * 1.10, months_to_project)
        expansion_growth = np.linspace(df['Expansion'].iloc[-1], df['Expansion'].iloc[-1] * 1.00, months_to_project)
        base_growth = np.linspace(df['Base'].iloc[-1], df['Base'].iloc[-1] * 1.00, months_to_project)
        site_growth = np.linspace(df['Site_Opens'].iloc[-1], df['Site_Opens'].iloc[-1] * 1.00, months_to_project)
        high_to_under_growth = np.linspace(df['high_to_under_direct'].iloc[-1], df['high_to_under_direct'].iloc[-1] * 1.00, months_to_project)

       

        future_X = np.column_stack([
            future_months_seq,                    # Month_Sequential
            churn_improvement,                    # Churn
            customer_growth,                      # Ending Count
            arpu_growth,                         # ARPU
            future_month_nums,                   # Month_Number
            expansion_growth,                    # Expansion
            #base_growth,                         # Base_Added
            #site_growth,                         # Site_Opens
        ])
            
    elif target_variable == "Net New MRR":
        # Net New MRR future projections - Only grow the variables you can directly control
        
        # Variables you can directly influence through business actions:
        new_count_growth = np.linspace(df['New Count'].iloc[-1], df['New Count'].iloc[-1] * 1.00, months_to_project)
        
        high_to_under_growth = np.linspace(df['high_to_under_direct'].iloc[-1], df['high_to_under_direct'].iloc[-1] * 1.00,  months_to_project)
        
        # Keep other performance ratios STATIC (they'll change naturally as business improves)
        static_quality_ratio = [df['qualityperformers_ratio'].iloc[-1]] * months_to_project
        static_quality_weighted = [df['quality_weighted_performance'].iloc[-1]] * months_to_project
        static_underperformers_ratio = [df['underperformers_ratio'].iloc[-1]] * months_to_project
        
        # Calculate good/bad months for future
        future_good_months = [int(month in [3, 5, 6]) for month in future_month_nums]
        future_bad_months = [int(month in [7, 8, 10, 12]) for month in future_month_nums]

        future_X = np.column_stack([
            future_months_seq,                   # Month_Sequential
            future_months_seq_squared,           # Month_Sequential_squared
            future_month_nums,                   # Month_Number
            future_good_months,                  # Good_Months
            future_bad_months,                   # Bad_Months
            future_h2_indicators,                # H2_indicator
            static_quality_ratio,                # qualityperformers_ratio (STATIC)
            static_quality_weighted,             # quality_weighted_performance (STATIC)
            static_underperformers_ratio,        # underperformers_ratio (STATIC)
            high_to_under_growth,                # high_to_under_direct (GROWING - you can influence this)
            new_count_growth                     # New Count (GROWING - you can control this)
        ])
    
    # Clean future data
    if not np.isfinite(future_X).all():
        print("Warning: Found non-finite values in future features. Cleaning...")
        future_X = np.nan_to_num(future_X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.isnan(future_X).any():
        print("Warning: Still found NaN values in future features. Replacing with zeros...")
        future_X = np.where(np.isnan(future_X), 0.0, future_X)
    
    # Make predictions
    future_predictions = poly_pipeline.predict(future_X)
    
    # Apply target-specific bounds
    if target_variable == "Ending MRR":
        future_predictions = np.maximum(future_predictions, y[-1] * 0.5)  # Ending MRR shouldn't drop below 50% of current
    elif target_variable == "Net New MRR":
        # Net New MRR can be negative, so no lower bound needed
        pass
    
    # Create projections DataFrame
    projections_df = pd.DataFrame({
        'Month': future_dates,
        'Month_Sequential': future_months_seq,
        'Projected_Value': future_predictions,
        'Projected_ARR': future_predictions * 12 if target_variable == "Ending MRR" else None,
        'Lifestage': [performance_tier] * len(future_dates)
    })
    
    # Calculate growth metrics
    current_value = y[-1]
    final_projected_value = projections_df['Projected_Value'].iloc[-1]
    total_growth = ((final_projected_value - current_value) / abs(current_value)) * 100 if current_value != 0 else 0
    monthly_growth_rate = (final_projected_value / current_value) ** (1/months_to_project) - 1 if current_value > 0 else 0
    
    print(f"📊 {performance_tier} Projection Summary:")
    print(f"   Current {target_variable}: ${current_value:,.2f}")
    print(f"   Projected {target_variable} ({months_to_project}m): ${final_projected_value:,.2f}")
    print(f"   Total Growth: {total_growth:.1f}%")
    print(f"   Monthly Growth Rate: {monthly_growth_rate:.2%}")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Main plot - CONNECTED LINES
    plt.subplot(2, 2, (1, 2))
    
    # Plot historical data
    plt.plot(df['Month'], y, 'b-', linewidth=3, label=f'Historical {target_variable}', marker='o', markersize=4)
    
    # Plot separate projection line
    plt.plot(projections_df['Month'], projections_df['Projected_Value'], 
             'r--', linewidth=3, label=f'Projected {target_variable}', marker='s', markersize=4)
    
    # Add trend line for full period
    all_months_seq = np.concatenate([df['Month_Sequential'], future_months_seq])
    all_months_seq_squared = all_months_seq ** 2
    all_dates = pd.concat([df['Month'], pd.Series(future_dates)])
    
    # Build trend line data based on target variable
    if target_variable == "Ending MRR":
        # Ending MRR trend line
        historical_month_nums = df['Month'].dt.month.tolist()
        
        all_month_nums = historical_month_nums + future_month_nums
        
        all_X_trend = np.column_stack([
            all_months_seq,                                                         # Month_Sequential
            np.concatenate([df['Churn'], churn_improvement]),                              # Churn
            np.concatenate([df['Ending Count'], customer_growth]),                         # Ending Count
            np.concatenate([df['ARPU'], arpu_growth]),                                     # ARPU
            all_month_nums,                                                                # Month_Number
            np.concatenate([df['Expansion'], expansion_growth]),                           # Expansion
            #np.concatenate([df['Base'], base_growth]),                               # Base_Added
            #np.concatenate([df['Site_Opens'], site_growth]),                               # Site_Opens
        ])
        
    elif target_variable == "Net New MRR":
        # Net New MRR trend line
        historical_month_nums = df['Month'].dt.month.tolist()
        historical_quarters = df['Month'].dt.quarter.tolist()
        historical_h2_indicators = df['H2_indicator'].values
        all_month_nums = historical_month_nums + future_month_nums

        all_good_months = [int(month in [3, 5, 6]) for month in all_month_nums]
        all_bad_months = [int(month in [7, 8, 10, 12]) for month in all_month_nums]
        
        all_X_trend = np.column_stack([
            all_months_seq,                                                                # Month_Sequential
            all_months_seq_squared,                                                        # Month_Sequential_squared
            all_month_nums,                                                                # Month_Number
            all_good_months,                                                               # Good_Months
            all_bad_months,                                                                # Bad_Months
            np.concatenate([historical_h2_indicators, future_h2_indicators]),             # H2_indicator

            np.concatenate([df['qualityperformers_ratio'], static_quality_ratio]),
            np.concatenate([df['quality_weighted_performance'], static_quality_weighted]),
            np.concatenate([df['underperformers_ratio'], static_underperformers_ratio]),
            np.concatenate([df['high_to_under_direct'], high_to_under_growth]),
            np.concatenate([df['New Count'], new_count_growth]),                          # New Count
        ])
    
    # Clean trend data and generate trend line
    if not np.isfinite(all_X_trend).all():
        all_X_trend = np.nan_to_num(all_X_trend, nan=0.0, posinf=0.0, neginf=0.0)
    
    all_predictions = poly_pipeline.predict(all_X_trend)

    # ADD THIS BLOCK HERE:
    # Save predictions data for Excel graphing
    predictions_df = pd.DataFrame({
        'Month': all_dates,
        'Historical_Data': np.concatenate([y, [np.nan] * months_to_project]),
        'Projected_Data': np.concatenate([[np.nan] * len(y), future_predictions]),
        'Model_Trend_Line': all_predictions
    })

    # Save to CSV
    csv_filename = f"{target_variable.replace(' ', '_')}_{performance_tier}_predictions.csv"
    predictions_df.to_csv(csv_filename, index=False)
    print(f"📊 Predictions saved to: {csv_filename}")
    # END OF NEW BLOCK
    plt.plot(all_dates, all_predictions, 'g:', alpha=0.8, linewidth=2, label='Model Trend Line')
    
    plt.title(f'{target_variable} Projection - {performance_tier} ({model_type})', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(f'{target_variable} ($)', fontsize=12)
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
    plt.title(f'{performance_tier} - MRR Drivers', fontsize=12)
    plt.xlabel('Month', fontsize=10)
    plt.ylabel('MRR Change ($)', fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'performance_tier': performance_tier,
        'model_type': model_type,
        'model': poly_pipeline,
        'projections': projections_df,
        'performance': {'r2': r2, 'mae': mae, 'rmse': rmse, 'cv_mae': cv_mean},
        'monthly_growth_rate': monthly_growth_rate,
        'feature_names': feature_names,
        'current_metrics': {
            'current_value': current_value,
            'projected_value': final_projected_value,
            'total_growth': total_growth
        }
    }