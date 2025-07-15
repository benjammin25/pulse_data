from . import clean_missing as cm
from . import exclusivity as ex
from .Time_Keys import add_time_keys
from .filters import apply_business_filters
from .identify_churned import identify_churned_accounts_with_date

def analyze_and_handle_missing(df, folders):
    #print("\n--- Missing Data Before Handling ---")
    #cm.print_missing_summary(df)

    df_cleaned = cm.handle_missing_data(df)
    # Save to organized folder
    df_cleaned.to_csv(folders['raw_cleaned'] / "cleaned_data.csv", index=False)
    print(f"💾 Saved cleaned data to: {folders['raw_cleaned'] / 'cleaned_data.csv'}")

    #print("\n--- Missing Data After Handling ---")
    #cm.print_missing_summary(df_cleaned)
    
    return df_cleaned


def filtering_tagging(filename, folders):
    df_raw = cm.load_and_initial_clean(filename)
    
    # STEP 1: Clean the data first
    df_cleaned = analyze_and_handle_missing(df_raw, folders)
    
    # STEP 2: Apply business filters BEFORE price changes
    df_filtered = apply_business_filters(df_cleaned)
    df_filtered = add_time_keys(df_filtered)
    
    # STEP 3: Add churn detection and revenue tagging
    df_with_churn_flags = identify_churned_accounts_with_date(df_filtered)
    df_tagged = ex.tag_revenue_category(df_with_churn_flags)
    
    # STEP 4: Enhanced deduplication
    print("\n🔧 APPLYING ENHANCED DEDUPLICATION")
    print("="*45)
    print(f"Before deduplication: {len(df_tagged)} contracts")
    df_deduped = ex.deduplicate_mutual_exclusive(df_tagged)
    print(f"After deduplication: {len(df_deduped)} contracts")
    print(f"Removed: {len(df_tagged) - len(df_deduped)} duplicate/overlapping contracts\n")

    return df_deduped