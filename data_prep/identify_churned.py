import pandas as pd


def identify_churned_accounts_with_date(df, account_col='Account Number', wo_col='WO Type Group', end_date_col='Package End Date'):
    df = df.copy()
    df[end_date_col] = pd.to_datetime(df[end_date_col], errors='coerce')

    # EDIT: Make this more conservative - only consider truly ended subscriptions
    active_activate_accounts = df[
        (df[wo_col] == 'Activate') & 
        ((df[end_date_col].dt.year >= 2999) | 
         (df[end_date_col].isnull()) |
         (df[end_date_col] > pd.Timestamp.today()))  # ADD: Future dates = active
    ][account_col].unique()

    cos_accounts = df[df[wo_col] == 'Change of Service'][account_col].unique()
    
    # EDIT: Add recent activity check - accounts with recent activity shouldn't be churned
    recent_cutoff = pd.Timestamp.today() - pd.DateOffset(months=0)  
    recent_activity_accounts = df[
        df['Package Start Date'] >= recent_cutoff
    ][account_col].unique()

    all_accounts = df[account_col].unique()
    churned_accounts = []
    churn_dates = {}

    for account in all_accounts:
        # EDIT: More conservative churn criteria
        if (account in active_activate_accounts or 
            account in cos_accounts or 
            account in recent_activity_accounts):  # ADD: Recent activity = active
            churn_dates[account] = pd.NaT
        else:
            # EDIT: Only churn if end date is clearly in the past
            end_dates = df.loc[df[account_col] == account, end_date_col].dropna()
            if not end_dates.empty:
                churn_date = end_dates.min()
                # ADD: Only consider churned if end date is more than 30 days ago
                if churn_date < (pd.Timestamp.today() - pd.DateOffset(days=30)):
                    churned_accounts.append(account)
                    churn_dates[account] = churn_date
                else:
                    churn_dates[account] = pd.NaT
            else:
                churn_dates[account] = pd.NaT

    # Rest of function stays the same...

    # Add churn flag
    df['Churn_Flag'] = df[account_col].apply(lambda x: 1 if x in churned_accounts else 0)

    # Map churn dates back to DataFrame
    df['Churn_Date'] = df[account_col].map(churn_dates)

    print(f"[Churn Detection] Total churned accounts identified: {len(churned_accounts)}")
    # After identify_churned_accounts_with_date returns df
    assert df.loc[df['Churn_Flag'] == 1, 'Churn_Date'].notnull().all(), "Some churned accounts missing churn date"
    return df