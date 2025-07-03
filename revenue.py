import pandas as pd


def expansion_revenue(df_filtered):
    expansion_df = df_filtered[(df_filtered['WO Type Group'] == 'Change of Service') & 
                               (df_filtered['Reason Group'] == 'Upgrade')].copy()
    
    expansion_df = expansion_df.sort_values(['Account Number', 'Package Start Date'])
    expansion_df['Prev Price'] = expansion_df.groupby('Account Number')['Price'].shift(1)
    expansion_df['Delta'] = expansion_df['Price'] - expansion_df['Prev Price']
    
    # JUST REMOVE THE CAP - that's the main fix
    expansion_df['Expansion'] = expansion_df['Delta'].apply(lambda x: x if x > 0 else 0)
    
    monthly_expansion = expansion_df.groupby('Months_Key')['Expansion'].sum().reset_index(name='Expansion')
    return monthly_expansion


def contraction_revenue(df_filtered):
    # Filter downgrades only
    contraction_df = df_filtered[(df_filtered['WO Type Group'] == 'Change of Service') & 
                                 (df_filtered['Reason Group'] == 'Downgrade')].copy()
    
    # Sort by Account and date
    contraction_df = contraction_df.sort_values(['Account Number', 'Package Start Date'])
    
    # Get previous Price per Account to compare
    contraction_df['Prev Price'] = contraction_df.groupby('Account Number')['Price'].shift(1)
    
    # Calculate delta: Previous Price - New Price (for contraction)
    contraction_df['Delta'] = contraction_df['Prev Price'] - contraction_df['Price']
    
    # Only keep positive deltas (actual contraction)
    contraction_df['Contraction'] = contraction_df['Delta'].apply(lambda x: x if x > 0 else 0)
    
    # Sum monthly contraction
    monthly_contraction = contraction_df.groupby('Months_Key')['Contraction'].sum().reset_index(name='Contraction')
    
    return monthly_contraction


def churn_revenue(df_churned):
    # Only count revenue in the month they actually churned
    churned_df = df_churned[df_churned['Churn_Flag'] == 1].copy()
    churned_df['Churn_Month'] = pd.to_datetime(churned_df['Churn_Date']).dt.to_period('M').astype(str)
    
    # Group by churn month, not all months
    monthly_churn_revenue = churned_df.groupby('Churn_Month')['Price'].sum().reset_index()
    monthly_churn_revenue.columns = ['Month', 'Churn']
    return monthly_churn_revenue
