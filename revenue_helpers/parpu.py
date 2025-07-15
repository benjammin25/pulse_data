import pandas as pd

# === ARPU Calculation Function ===
def calculate_arpu(df, time_key):
    return df.groupby(time_key).agg({
        'Price': 'sum',
        'Account Number': 'nunique'
    }).rename(columns={
        'Price': 'Total Period Activation',
        'Account Number': 'Unique Accounts'
    }).reset_index().assign(
        ARPU=lambda x: x['Total Period Activation'] / x['Unique Accounts']).round(2)
    
    

# === Currency Formatting Function ===
def format_currency_columns(df, columns):
    """
    Formats specified columns in a DataFrame as currency with $ sign, commas, and 2 decimal places.
    """
    df_formatted = df.copy()
    for col in columns:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
    return df_formatted


def calculate_cumulative_arpu(df):
    """
    Calculate cumulative ARPU - returns DataFrame with customer-level data
    
    Parameters:
    df: DataFrame with 'Account Number' and 'Price' columns
    
    Returns:
    DataFrame with columns: ['Account Number', 'Cumulative_ARPU']
    """
    customer_cumulative = df.groupby('Account Number')['Price'].sum().reset_index()
    customer_cumulative.rename(columns={'Price': 'Cumulative_ARPU'}, inplace=True)
    return customer_cumulative.sort_values('Cumulative_ARPU', ascending=False)

def cumulative_arpu_summary(cumulative_df):
    """
    Generate summary statistics from cumulative ARPU DataFrame
    
    Parameters:
    cumulative_df: Output from calculate_cumulative_arpu()
    
    Returns:
    Dictionary with summary metrics
    """
    return {
        'Total_Customers': len(cumulative_df),
        'Average_Cumulative_ARPU': cumulative_df['Cumulative_ARPU'].mean(),
        'Median_Cumulative_ARPU': cumulative_df['Cumulative_ARPU'].median(),
        'Total_Revenue': cumulative_df['Cumulative_ARPU'].sum(),
        'Max_Customer_Value': cumulative_df['Cumulative_ARPU'].max(),
        'Min_Customer_Value': cumulative_df['Cumulative_ARPU'].min(),
        'Top_10_Percent_ARPU': cumulative_df['Cumulative_ARPU'].quantile(0.9)
    }


def parpu_calculations(combined_df):
    # Define all PARPU calculations
    parpu_columns = {
        'Starting PARPU': ('Starting MRR', 'Starting Count'),
        'New PARPU': ('New MRR', 'New Count'),
        'Expansion PARPU': ('Expansion', 'Expansion Count'),
        'Contraction PARPU': ('Contraction', 'Contraction Count'),
        'Churn PARPU': ('Churn', 'Churn Count'),
        'Net PARPU': ('Net New MRR', 'Net Count'),
        'Ending PARPU': ('Ending MRR', 'Ending Count')
    }

    # Create all PARPU columns in one loop
    for new_col, (numerator, denominator) in parpu_columns.items():
        combined_df[new_col] = combined_df[numerator].div(combined_df[denominator]).fillna(0)

    combined_df = combined_df[['Month','Month_Sequential','Month_Number','Starting MRR', 'Starting PARPU', 'Starting Count', 'New MRR', 'New Count', 'New PARPU',
                            'Expansion', 'Expansion Count', 'Expansion PARPU',
                            'Contraction', 'Contraction Count', 'Contraction PARPU',
                            'Churn', 'Churn Count', 'Churn PARPU',
                            'Net New MRR', 'Net Count', 'Net PARPU', 
                            'Ending MRR', 'Ending Count', 'Ending PARPU',
                            'ARPU'
                ]]
    return combined_df
