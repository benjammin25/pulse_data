import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def basic_linear_model(csv_file_path, months_to_project=12):
    """
    Create a basic linear regression model to project revenue
    
    Parameters:
    csv_file_path: Path to your combined_mrr_arpu_results.csv file
    months_to_project: How many months ahead to project (default 12)
    """
    lifestage = Path(csv_file_path).parts[-3] # Extract lifestage from file path
    print(f" Modeling lifestage: {lifestage}")
   # Load the data
    df = pd.read_csv(csv_file_path, encoding='latin-1', low_memory=False)
    
    df = df[:-1]
    df['Month'] = pd.to_datetime(df['Month'])

    #Display date range and mrr range
    print(f"Data covers from {df['Month'].min().strftime("%Y-%m")} to {df['Month'].max().strftime("%Y-%m")}")
    print(f"Mrr range: {df['Ending MRR'].min():,.2f} to {df['Ending MRR'].max():,.2f}")

    # Prepare the data for linear regression
    df["MRR_Growth_1m"] = df["Ending MRR"].pct_change(1).fillna(0)
    df['MRR_Growth_3m'] = df['Ending MRR'].pct_change(3).fillna(0)
    df["MRR_Growth_6m"] = df["Ending MRR"].pct_change(6).fillna(0)

    X = df[["Month_Sequential", "Month_Number", "MRR_Growth_1m","MRR_Growth_3m", "MRR_Growth_6m", "Churn"]].values
    y = df["Ending MRR"].values

    #Fit the linear regression model
    model = LinearRegression()
    model.fit(X,y)

    # Make predictions on existing data (for training and evaluation)
    y_pred = model.predict(X)

    #calculate model performance metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print("📈 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Mean Absolute Error: ${mae:,.2f}")
    print(f"   Monthly Growth Rate: ${model.coef_[0]:,.2f}")
    print(f"   Starting Point: ${model.intercept_:,.2f}")
    
   

    # Project future months
    print(f"Projecting {months_to_project} months ahead...")

    #Create future month indices
    last_month_seq = df['Month_Sequential'].max()
    future_months_seq = np.arange(last_month_seq + 1, last_month_seq + 1 + months_to_project)

    #Create future dates
    last_date = df['Month'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_to_project, freq='M')

    # IF using both features, need both future values:
    if X.shape[1] == 6:  
        last_month_num = df['Month_Number'].iloc[-1]
    
        future_month_nums = [((last_month_num + i) % 12) + 1 for i in range(1, months_to_project + 1)]
        """
        # For New Count, use recent average
        recent_new_count = df['New Count'].iloc[-1]
        future_new_counts = [recent_new_count] * months_to_project
        
        # For Ending PARPU, use recent trend or average
        recent_parpu = df['Ending PARPU'].iloc[-1] # Or use trend
        future_parpu = [recent_parpu] * months_to_project
        """
        # For Churn, use recent average
        recent_churn = df['Churn'].iloc[-1]
        future_churn = [recent_churn] * months_to_project
        
        recent_growth_1m = df['MRR_Growth_1m'].iloc[-1]
        future_growth_1m = [recent_growth_1m] * months_to_project

        recent_growth_3m = df['MRR_Growth_3m'].iloc[-1]
        future_growth_3m = [recent_growth_3m] * months_to_project

        recent_growth_6m = df['MRR_Growth_6m'].iloc[-1]
        future_growth_6m = [recent_growth_6m] * months_to_project
        
        future_X = np.column_stack([future_months_seq, future_month_nums, future_growth_1m, future_growth_3m, future_growth_6m, future_churn])
    else:  # One feature
        future_X = future_months_seq.reshape(-1, 1)

    #Make future predictions
    future_mrr = model.predict(future_X)

    

    # Create DataFrame for projections
    projections_df = pd.DataFrame({
        'Month': future_dates,
        'Month_Sequential': future_months_seq,
        'Projected_MRR': future_mrr,
        'Projected_ARR': future_mrr * 12
    })

    # Calculate growth metrics
    current_mrr = df['Ending MRR'].iloc[-1]
    final_projected_mrr = projections_df['Projected_MRR'].iloc[-1]
    total_growth = ((final_projected_mrr - current_mrr) / current_mrr) * 100
    
    print("📊 Projection Summary:")
    print(f"   Current MRR: ${current_mrr:,.2f}")
    print(f"   Projected MRR (12m): ${final_projected_mrr:,.2f}")
    print(f"   Total Growth: {total_growth:.1f}%")
    print(f"   Current ARR: ${current_mrr * 12:,.2f}")
    print(f"   Projected ARR (12m): ${final_projected_mrr * 12:,.2f}")

    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Historical data
    plt.plot(df['Month'], df['Ending MRR'], 'b-', linewidth=2, label='Historical MRR', marker='o')
    
    # Projected data
    plt.plot(projections_df['Month'], projections_df['Projected_MRR'], 
             'r--', linewidth=2, label='Projected MRR', marker='s')
    
    # Add trend line for full period
    all_months_seq = np.concatenate([df['Month_Sequential'], future_months_seq])
    all_months_num = np.concatenate([df['Month_Number'], future_month_nums])
    all_growth_1m = np.concatenate([df['MRR_Growth_1m'], future_growth_1m])
    all_growth_3m = np.concatenate([df['MRR_Growth_3m'], future_growth_3m])
    all_growth_6m = np.concatenate([df['MRR_Growth_6m'], future_growth_6m])
    all_churn = np.concatenate([df['Churn'], future_churn])
    #all_new_counts = np.concatenate([df['New Count'], future_new_counts])
    #all_parpu = np.concatenate([df['Ending PARPU'], future_parpu])
    #all_churn = np.concatenate([df['Churn'], future_churn])
    all_X = np.column_stack([all_months_seq, all_months_num, all_growth_1m, all_growth_3m, all_growth_6m, all_churn])  
    all_predictions = model.predict(all_X)
    all_dates = pd.concat([df['Month'], pd.Series(future_dates)])
    plt.plot(all_dates, all_predictions, 'g:', alpha=0.7, label='Linear Trend')
    
    plt.title(f'MRR Projection - {lifestage} Lifestage', fontsize=16, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('MRR ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()
    
    # Combine historical and projected data
    historical_df = df[['Month', 'Month_Sequential', 'Ending MRR']].copy()
    historical_df['Type'] = 'Historical'
    historical_df['MRR'] = historical_df['Ending MRR']
    
    projected_df = projections_df[['Month', 'Month_Sequential', 'Projected_MRR']].copy()
    projected_df['Type'] = 'Projected'
    projected_df['MRR'] = projected_df['Projected_MRR']
    
    combined_df = pd.concat([
        historical_df[['Month', 'Month_Sequential', 'MRR', 'Type']], 
        projected_df[['Month', 'Month_Sequential', 'MRR', 'Type']]
    ], ignore_index=True)
    
    return {
        'model': model,
        'historical_data': df,
        'projections': projections_df,
        'combined_data': combined_df,
        'performance': {'r2': r2, 'mae': mae},
        'monthly_growth': model.coef_[0]
    }

if __name__ == "__main__":
    # Example usage
    result = basic_linear_model('output/Full/combined_results/baseline_combined_mrr_arpu_results.csv', months_to_project=12)
    
    # Access results
    print("\n📊 Projection Results:")
    print(result['projections'])
    
    print("\n📈 Combined Historical and Projected Data:")
    print(result['combined_data'].head())
    
    print("\n📊 Model Performance Metrics:")
    print(result['performance'])