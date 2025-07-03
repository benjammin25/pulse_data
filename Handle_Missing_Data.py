import pandas as pd

# Missing data strategy
def handle_missing_data(df):
    # Analyze missing data patterns
    missing_analysis = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    }).sort_values('Missing_Percentage', ascending=False)

    print(missing_analysis)
        
    df_clean = df.copy()
    df_clean['Price'] = df_clean['Price'].str.replace('$', '').str.replace(',', '').str.strip()  # Clean Price column by removing $ and commas
    df_clean['Price'] = pd.to_numeric(df_clean['Price'], errors='coerce').fillna(0)  # Convert Price to numeric and fill NaN with 0

    #Drop columns with >50% missing (Package End Date, Prior Drop fields)
    high_missing = missing_analysis[missing_analysis['Missing_Percentage'] > 50]['Column'].tolist()
    # Correct filtering: Use list comprehension
    high_missing = [col for col in high_missing if col != 'Package End Date']

    # Now drop them
    df_clean = df_clean.drop(columns=high_missing, errors='ignore')



    # Fill 'Package Name', 'Package Group', and 'Package Group Condensed' with 'Standard Package'
    package_cols = ['Package Name', 'Package Group', 'Package Group Condensed']
    for col in package_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    

    # Fill all object (categorical) columns with 'Unknown'
    categorical_cols = df_clean.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        if col == "Package End Date":
            continue
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')

    # Fill numeric columns with median
    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    

    return df_clean


