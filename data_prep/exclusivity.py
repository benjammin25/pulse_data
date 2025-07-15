import pandas as pd

def deduplicate_mutual_exclusive(df):
    """
    Enhanced version that handles both revenue category conflicts 
    AND overlapping time periods for the same package
    """
    
    # PART 1: Your existing logic for revenue category conflicts
    df['Key'] = df['Account Number'] + '|' + df['Package Group Condensed'] + '|' + df['Months_Key']
    
    dupes = df.groupby('Key')['Revenue_Category'].nunique()
    conflict_keys = dupes[dupes > 1].index.tolist()
    
    conflicted = df[df['Key'].isin(conflict_keys)].copy()
    clean = df[~df['Key'].isin(conflict_keys)].copy()
    
    # Apply revenue category priority
    priority = {'New MRR': 1, 'Expansion': 2, 'Contraction': 3, 'Other': 4}
    conflicted['priority'] = conflicted['Revenue_Category'].map(priority)
    conflicted_sorted = conflicted.sort_values(by='priority')
    deduped_conflicted = conflicted_sorted.drop_duplicates(subset='Key', keep='first')
    deduped_conflicted = deduped_conflicted.drop(columns=['priority'])
    
    # Combine back
    df_step1 = pd.concat([clean, deduped_conflicted], ignore_index=True)
    df_step1.drop(columns=['Key'], inplace=True, errors='ignore')
    
    # PART 2: NEW - Handle overlapping time periods for SAME package
    df_step1['Overlap_Key'] = (df_step1['Account Number'] + '|' + 
                               df_step1['Package Group Condensed'] + '|' + 
                               df_step1['Package Start Date'].astype(str))
    
    # Find accounts with multiple contracts for same package on same start date
    overlap_dupes = df_step1.groupby('Overlap_Key').size()
    overlap_conflict_keys = overlap_dupes[overlap_dupes > 1].index.tolist()
    
    if overlap_conflict_keys:
        print(f"Found {len(overlap_conflict_keys)} overlapping time period conflicts")
        
        overlap_conflicted = df_step1[df_step1['Overlap_Key'].isin(overlap_conflict_keys)].copy()
        overlap_clean = df_step1[~df_step1['Overlap_Key'].isin(overlap_conflict_keys)].copy()
        
        # For overlapping time periods, keep the one with latest Package End Date
        overlap_conflicted_sorted = overlap_conflicted.sort_values(['Package End Date'], ascending=False)
        overlap_deduped = overlap_conflicted_sorted.drop_duplicates(subset='Overlap_Key', keep='first')
        
        df_final = pd.concat([overlap_clean, overlap_deduped], ignore_index=True)
    else:
        df_final = df_step1
    
    df_final.drop(columns=['Overlap_Key'], inplace=True, errors='ignore')
    
    return df_final


def tag_revenue_category(df):
    df = df.copy()

    # Default null
    df['Revenue_Category'] = None

    # New MRR (Activations)
    df.loc[df['WO Type Group'] == 'Activate', 'Revenue_Category'] = 'New MRR'

    # Expansion
    df.loc[
        (df['WO Type Group'] == 'Change of Service') &
        (df['Reason Group'] == 'Upgrade'),
        'Revenue_Category'
    ] = 'Expansion'

    # Contraction
    df.loc[
        (df['WO Type Group'] == 'Change of Service') &
        (df['Reason Group'] == 'Downgrade'),
        'Revenue_Category'
    ] = 'Contraction'

    # Churn
    df.loc[df['Churn_Flag'] == 1, 'Revenue_Category'] = 'Churn'

    # Other
    df.loc[df['Revenue_Category'].isnull(), 'Revenue_Category'] = 'Other'
    """
    # Report how many rows didn't get tagged
    untagged_count = df['Revenue_Category'].isnull().sum()
    if untagged_count > 0:
        print(f"Warning: {untagged_count} rows have no revenue category. Saving to 'Unmapped_Rows.csv' for review.")
        df[df['Revenue_Category'].isnull()].to_csv('Unmapped_Rows.csv', index=False)
    
    # Filter: Keep only rows with a valid category
    df = df[df['Revenue_Category'].notnull()]
    """
    df = df[df["Revenue_Category"] != "Other"]
    return df

def validate_revenue_category_exclusivity(df):
    # Count number of rows per Account/Start Date/Package that appear in multiple categories
    conflict_cols = ['Account Number', 'Package Start Date', 'Package Group Condensed']
    
    # Count how many revenue categories each (Account + Start Date + Package) appears in
    conflicts = (
        df.groupby(conflict_cols)['Revenue_Category']
        .nunique()
        .reset_index()
    )
    
    # Filter where count > 1 (meaning same account/package appears in multiple categories)
    conflicts = conflicts[conflicts['Revenue_Category'] > 1]

    if not conflicts.empty:
        print(f"\n⚠️ Mutual Exclusivity Warning: {len(conflicts)} account/package/date combos appear in multiple categories.")
        
        # Join back to full data for context
        conflicted_rows = df.merge(conflicts, on=conflict_cols, how='inner')
        conflicted_rows.to_csv('Mutual_Exclusivity_Errors.csv', index=False)
        print("Saved conflict rows to 'Mutual_Exclusivity_Errors.csv' for review.")

    else:
        print("\n✅ Revenue categories are mutually exclusive. No overlaps found.")