import pandas as pd
import Time_Keys as tk
from Handle_Missing_Data import handle_missing_data
import revenue as rev
import EndingMRR as mrr
import ARPU_Results as arpu
import exclusivity as ex

def load_and_initial_clean(file_path):
    df = pd.read_csv(file_path)
    df.drop(df.columns[42:], axis=1, inplace=True)
    df.rename(columns={" Price ": "Price"}, inplace=True)
    return df


def print_missing_summary(df):
    summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    }).sort_values('Missing_Percentage', ascending=False)
    
    print(summary)


def analyze_and_handle_missing(df):
    print("\n--- Missing Data Before Handling ---")

    df_cleaned = handle_missing_data(df)
    df_cleaned.to_csv("CleanedData.csv",index= False)

    print("\n--- Missing Data After Handling ---")
    print_missing_summary(df_cleaned)
    
    return df_cleaned



def apply_business_filters(df):
    # Drop unneeded columns
    drop_cols = [
        'Account Name', 'Full Address', 'GIS LOCID', 'User Name', 'CXN Code', 
        'BH HP ID', 'CXN Name', 'CXN Code and Name', 'City', 'Zip', 'User'
    ]
    df.drop(columns=drop_cols, inplace=True)

    core_mrr_packages = [
       # Your existing core packages
       'Internet: 1 Gbps', 'Internet: 100 Mbps', 'Internet: 120 Mbps',
       'Internet: 250 Mbps', 'Internet: 350 Mbps', 'Internet: 500 Mbps',
       'Internet: 2 Gbps', 'Internet: 3 Gbps', 'Internet: 10 Gbps',
       'TV: Essentials', 'TV: Premier', 'TV: Favorites',
       'Voice: Local & Long Distance', 
       
       
       'TV: Premium Content',      
       'Internet: TiVo+',         
       'TV: Additional Feature',   
       'Voice: Additional Feature', 
       'TV: Broadcast Fee',       
       'Internet: Equipment',      
       'TV: Equipment',           
   ]



    filtered = df[df['Package Group Condensed'].isin(core_mrr_packages)]
    

    
    return filtered


def start_new_mrr_filters(df):
    df = df[
            (df['WO Type Group'] == 'Activate')  &
        (df['Test Flag'] == 0)
    ]
    return df

def add_time_keys(df):
    return tk.create_time_keys(df)


def user_arpu_selection(df):
    try:
        choice = int(input("Enter 1 for Weekly ARPU, 2 for Monthly ARPU, 3 for Quarterly ARPU, or 4 for Annual ARPU: "))
        key_map = {1: 'Weeks_Key', 2: 'Months_Key', 3: 'Quarters_Key', 4: 'Years_Key'}
        key = key_map.get(choice)

        if not key:
            raise ValueError("Invalid choice")

        arpu_df = arpu.calculate_arpu(df, key)
        

        return arpu_df

    except ValueError as e:
        print("Error:", e)
        return None







def main_pipeline():
    df_raw = load_and_initial_clean('NewAppended.csv')
    df_cleaned = analyze_and_handle_missing(df_raw)
    df_filtered = apply_business_filters(df_cleaned)
    df_filtered = add_time_keys(df_filtered)

    
    df_with_churn_flags = mrr.identify_churned_accounts_with_date(df_filtered)
    df_with_churn_flags.to_csv('FilteredData.csv', index=False)
    print("\nFiltered data saved to 'FilteredData.csv'.")
    df_tagged = ex.tag_revenue_category(df_with_churn_flags)
    df_tagged = ex.deduplicate_mutual_exclusive(df_tagged)
    ex.validate_revenue_category_exclusivity(df_tagged)
    
    expansion_df = rev.expansion_revenue(df_tagged).rename(columns={'Months_Key': 'Month'})
    contraction_df = rev.contraction_revenue(df_tagged).rename(columns={'Months_Key': 'Month'})
    churned_df = rev.churn_revenue(df_tagged[df_tagged['Churn_Flag'] == 1]).rename(columns={'Months_Key': 'Month'})

    start_new_mrr_filtered = start_new_mrr_filters(df_tagged)
    
    start_new_mrr_df = mrr.calculate_starting_and_new_mrr_prorated(start_new_mrr_filtered, "Package Start Date", "Package End Date", "Price", "Account Number","Churn_Date")
    
    mrr_waterfall_df = mrr.mrr_waterfall(start_new_mrr_df, expansion_df, contraction_df, churned_df)

    mrr_waterfall_df_formatted = arpu.format_currency_columns(mrr_waterfall_df, ["Starting MRR", "New MRR", "Expansion", "Contraction", "Churn", "Net New MRR", "Ending MRR", "Average MRR"])
    print(mrr_waterfall_df_formatted.iloc[:57])
    total_mrr = mrr_waterfall_df["Ending MRR"].iloc[:57].sum()
    
    print(f"\nTotal MRR for the first 57 months: ${total_mrr:,.2f}")


    # OPTION 1: Full dataset ARPU (recommended for comprehensive analysis)
    print("\n--- Full Customer Cumulative ARPU Analysis ---")
    cumulative_data_full = arpu.calculate_cumulative_arpu(df_tagged)
    summary_full = arpu.cumulative_arpu_summary(cumulative_data_full)
    print(f"Full Dataset - Average Customer Value: ${summary_full['Average_Cumulative_ARPU']:,.2f}")
    print(f"Full Dataset - Total Customers: {summary_full['Total_Customers']:,}")

    # OPTION 2: Activation-only ARPU (for MRR comparison)
    print("\n--- Activation-Only Cumulative ARPU Analysis ---")
    cumulative_data_activate = arpu.calculate_cumulative_arpu(start_new_mrr_filtered)
    summary_activate = arpu.cumulative_arpu_summary(cumulative_data_activate)
    print(f"Activate-Only - Average Customer Value: ${summary_activate['Average_Cumulative_ARPU']:,.2f}")
    print(f"Activate-Only - Total Customers: {summary_activate['Total_Customers']:,}")

    # Show top customers (using full dataset)
    print("\n--- Top 10 Customers (Full Dataset) ---")
    print(cumulative_data_full.head(10))

    # Period ARPU Analysis
    print("\n--- Period ARPU Analysis ---")
    arpu_df = user_arpu_selection(df_tagged).iloc[:57]
    arpu_df_formatted = arpu.format_currency_columns(arpu_df, ['Total Period Activation', 'ARPU'])

    print("\nPeriod ARPU Results:")
    print(arpu_df_formatted[['Total Period Activation', 'Unique Accounts', 'ARPU']])

    """
    # Save outputs
    recurring_df.to_csv('Recurring_Revenue.csv', index=False)
    monthly_revenue_df.to_csv('Monthly_Cumulative_Revenue.csv', index=False)
    df_filtered.to_csv('ARPU_Results.csv', index=False)
    """
    print("\nPipeline execution completed. Results saved.")


if __name__ == "__main__":
    main_pipeline()
