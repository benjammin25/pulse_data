import pandas as pd
    

"""
def get_active_accounts_by_month(df):
    df = df.copy()
    df['Start'] = pd.to_datetime(df['Package Start Date'], errors='coerce')
    df['End'] = pd.to_datetime(df['Package End Date'], errors='coerce')

    months = pd.date_range(start=df['Start'].min(), end=df['End'].max(), freq='MS').to_period('M')

    active_list = []

    for month in months:
        active_accounts = df[
            (df['Start'] <= month.end_time) &
            ((df['End'] >= month.start_time) | df['End'].isnull())
        ]['Account Number'].nunique()

        active_list.append({'Month': month, 'Active Accounts': active_accounts})

    return pd.DataFrame(active_list)
"""



def calculate_starting_and_new_mrr_prorated(df, start_col, end_col, price_col, account_col, churn_date_col=None):
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])

    max_analysis_date = pd.Timestamp.today().normalize()

    def fix_end_date(x):
        try:
            x_date = pd.to_datetime(x, errors='coerce')
            if pd.isnull(x_date):
                return max_analysis_date
            elif x_date.year >= 2999:  # Active subscription
                return max_analysis_date
            # EDIT: Don't cap recent future dates - they might be legitimate
            #elif x_date > max_analysis_date and x_date < (max_analysis_date + pd.DateOffset(years=2)):
                #return x_date  # Keep reasonable future dates
            #elif x_date > max_analysis_date:
                #return max_analysis_date
            else:
                return x_date
        except Exception:
            return max_analysis_date

    df[end_col] = df[end_col].apply(fix_end_date)

    # If churn_date_col provided, convert to datetime
    if churn_date_col:
        df[churn_date_col] = pd.to_datetime(df[churn_date_col], errors='coerce')

    first_month = df[start_col].min().replace(day=1)
    last_month = df[end_col].max().replace(day=1)
    months = pd.date_range(first_month, last_month, freq='MS')

    results = []

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(0)
        total_days = (month_end - month_start).days + 1

        # STARTING MRR: Filter subs active at month start
        active_mask = (
            (df[start_col] <= month_start) & 
            (df[end_col] >= month_start) 
        )

        df_active = df.loc[active_mask].copy()

        # FIXED: For starting MRR, use full monthly price unless they churn mid-month
        def prorate_start_fixed(row):
            # Determine when their service actually ends this month
            service_end = pd.Timestamp(row[end_col])
            
            # Apply churn date logic (but don't subtract extra day)
            #if churn_date_col and pd.notnull(row[churn_date_col]):
                #churn_date = pd.Timestamp(row[churn_date_col])
                #service_end = min(service_end, churn_date)
            # 🔧 ADD THIS TEST HERE:
            # If they're active the whole month, use full price
            if service_end >= month_end:
                return row[price_col]
            
            # If they end mid-month, prorate only from month start to end date
            elif service_end >= month_start:
                days_active = (service_end - month_start).days + 1
                return row[price_col] * days_active / total_days
            
            # If they ended before this month (shouldn't happen with our filter)
            else:
                return 0

        df_active["ProratedRevenue"] = df_active.apply(prorate_start_fixed, axis=1)
        starting_mrr = df_active["ProratedRevenue"].sum()
        starting_count = df_active[account_col].nunique()
        # NEW MRR: Filter subs starting within the month
        new_mask = (
            (df[start_col] >= month_start) & 
            (df[start_col] <= month_end) 
        )

        df_new = df.loc[new_mask].copy()

        # ADD THIS QUICK FIX HERE:
       # if not df_active.empty:
            #df_active = df_active.groupby([account_col, 'Package Group Condensed']).tail(1).copy()

        # FIXED: For new MRR, only prorate if they don't start on 1st or end early
        def prorate_new_fixed(row):
            start_date = pd.Timestamp(row[start_col])
            service_end = pd.Timestamp(row[end_col])
            
            # Apply churn date logic (but don't subtract extra day)
            #if churn_date_col and pd.notnull(row[churn_date_col]):
                #churn_date = pd.Timestamp(row[churn_date_col])
                #service_end = min(service_end, churn_date)
            
            # Determine the effective end date for this month
            effective_end = min(service_end, month_end)

                
            # If they started on the 1st and are active all month, use full price
            if start_date == month_start and effective_end >= month_end:
                return row[price_col]
            
            # Otherwise, prorate based on actual days active
            elif effective_end >= start_date:
                days_active = (effective_end - start_date).days + 1
                return row[price_col] * days_active / total_days
            
            # Service ended before it started (shouldn't happen)
            else:
                return 0

        df_new["ProratedRevenue"] = df_new.apply(prorate_new_fixed, axis=1)
        new_count = df_new[account_col].nunique()
        new_mrr = df_new["ProratedRevenue"].sum()
        """
        print(f"[{month_start.strftime('%Y-%m')}] Active Accounts: {df_active[account_col].nunique()} | New Accounts: {df_new[account_col].nunique()}")
        print(f"[{month_start.strftime('%Y-%m')}] Starting MRR: ${starting_mrr:.2f} | New MRR: ${new_mrr:.2f}")
        """
        results.append({
            'Month': month_start.strftime('%Y-%m'),
            'Starting MRR': starting_mrr,
            'New MRR': new_mrr,
            'New Count': new_count,
            'Starting Count': starting_count
        })

    return pd.DataFrame(results)




def mrr_waterfall(start_new_mrr_df, expansion_df, contraction_df, churn_df):
    """
    Builds an MRR Waterfall showing Starting MRR, New MRR, Expansion, Contraction, Churn, Net New MRR, and Ending MRR.
    Implements sequential logic where each month's Starting MRR = previous month's Ending MRR.

    Parameters:
        start_new_mrr_df (DataFrame): Columns = ['Month', 'Starting MRR', 'New MRR']
        expansion_df (DataFrame): Columns = ['Month', 'Expansion']
        contraction_df (DataFrame): Columns = ['Month', 'Contraction']
        churn_df (DataFrame): Columns = ['Month', 'Churn']

    Returns:
        DataFrame: MRR Waterfall with all monthly movements and Ending MRR
    """

    # Step 1: Start with Starting and New MRR Data
    final_df = start_new_mrr_df.copy()

    # Step 2: Merge Expansion
    if expansion_df is not None and not expansion_df.empty:
        final_df = final_df.merge(expansion_df, on='Month', how='left')
    else:
        final_df['Expansion'] = 0

    # Step 3: Merge Contraction
    if contraction_df is not None and not contraction_df.empty:
        final_df = final_df.merge(contraction_df, on='Month', how='left')
    else:
        final_df['Contraction'] = 0

    # Step 4: Merge Churn
    if churn_df is not None and not churn_df.empty:
        final_df = final_df.merge(churn_df, on='Month', how='left')
    else:
        final_df['Churn'] = 0

    # Step 5: Fill any NaNs (months with no activity for some components)
    final_df[['Expansion','Expansion Count', 'Contraction','Contraction Count', 'Churn','Churn Count']] = final_df[['Expansion', 'Expansion Count','Contraction', 'Contraction Count','Churn', 'Churn Count']].fillna(0)

    # Step 6: Sort by month to ensure proper sequential calculation
    final_df = final_df.sort_values('Month').reset_index(drop=True)

    # Step 7: Calculate Net New MRR and Ending MRR with sequential logic
    for i in range(len(final_df)):
        final_df.loc[i, 'Net New MRR'] = (
            final_df.loc[i, 'New MRR'] +
            final_df.loc[i, 'Expansion'] -
            final_df.loc[i, 'Contraction'] -
            final_df.loc[i, 'Churn']
        )
        
        # HYBRID: For months after the first, use the HIGHER of:
        # 1. Previous month's Ending MRR (sequential logic)
        # 2. Calculated Starting MRR (your calculation)
        final_df.loc[i, 'Ending MRR'] = final_df.loc[i, 'Starting MRR'] + final_df.loc[i, 'Net New MRR']

        # Calculate Net Count: New customers minus churned customers
        # (Expansion/Contraction don't change customer count - same customers, different revenue)
        final_df.loc[i, 'Net Count'] = final_df.loc[i, 'New Count'] - final_df.loc[i, 'Churn Count']
        
        # Calculate Ending Count: Starting + Net Change
        final_df.loc[i, 'Ending Count'] = final_df.loc[i, 'Starting Count'] + final_df.loc[i, 'Net Count']

    # Step 8: Optional - Add Average MRR for the month
    #final_df['Average MRR'] = (final_df['Starting MRR'] + final_df['Ending MRR']) / 2

    # Step 9: Clean column order
    final_df = final_df[['Month', 'Starting MRR','Starting Count', 'New MRR', 'New Count','Expansion', 'Expansion Count','Contraction', 'Contraction Count','Churn', 'Churn Count','Net New MRR', 'Net Count','Ending MRR', 'Ending Count']]

    return final_df







