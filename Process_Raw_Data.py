import pandas as pd
from revenue_helpers import revenue as rev
from revenue_helpers import EndingMRR as mrr
from revenue_helpers import price_changes as pc
from revenue_helpers import parpu

from data_prep import filtering_tagging as ft
from data_prep import exclusivity as ex
from setup_output import setup_output_structure


"""
def base_added_per_month(df_filtered_customers):
    
    # Get sites where this segment's customers are located
    relevant_sites = df_filtered_customers['Site_ID'].unique()
    
    # Load clean site data
    sites_df = pd.read_csv('raw_data/sites_table.csv')  # Or wherever you store it
    segment_sites = sites_df[sites_df['Site_ID'].isin(relevant_sites)]
    
    # Clean aggregation - no duplicates to worry about!
    segment_sites["Site Release Month"] = pd.to_datetime(segment_sites["Site Release Date"]).dt.to_period("M")
    base_added = segment_sites.groupby("Site Release Month")["Base_Count"].sum().reset_index()
    base_added.columns = ["Month", "Base_Added"]
    
    return base_added


def site_opens_per_month(df):
    df["Site Release Month"] = pd.to_datetime(df["Site Release Date"]).dt.to_period("M")
    site_opens = df.groupby("Site Release Month")["Site_ID"].nunique()
    site_opens.columns = ["Month", "Site Opens"]

    return site_opens

def base_added_per_month(df):
    df["Site Release Month"] = pd.to_datetime(df["Site Release Date"]).dt.to_period("M")
    
    # Group by Site Release Month AND Site_ID to avoid double-counting
    base_added = df.groupby(["Site Release Month", "Site_ID"])["Base_Count"].first().reset_index()
    
    # Then sum by month to get total base expansion
    monthly_base = base_added.groupby("Site Release Month")["Base_Count"].sum().reset_index()
    monthly_base.columns = ["Month", "Base_Added"]
    
    return monthly_base

"""

def comprehensive_price_change_handling(df, price_changes):
    """Complete price change handling: new contracts + mid-contract splits"""
    
    print("🔧 APPLYING COMPREHENSIVE PRICE CHANGES")
    print("="*45)
    
    # Step 1: Update new contracts with correct pricing
    print("\n📅 Step 1: Updating new contracts...")
    df = pc.apply_price_changes_to_new_contracts(df, price_changes)
    
    # Step 2: Split contracts for mid-contract price changes  
    print("\n✂️  Step 2: Splitting mid-contract changes...")
    df = pc.split_contracts_for_price_changes(df, price_changes)
    
    return df



def waterfall_calculations(df):
    expansion_df = rev.expansion_revenue(df).rename(columns={'Months_Key': 'Month'})
    contraction_df = rev.contraction_revenue(df).rename(columns={'Months_Key': 'Month'})
    churned_df = rev.churn_revenue(df[df['Churn_Flag'] == 1]).rename(columns={'Months_Key': 'Month'})
    
    start_new_mrr_df = mrr.calculate_starting_and_new_mrr_prorated(df, "Package Start Date", "Package End Date", "Price", "Account Number","Churn_Date")
    
    mrr_waterfall_df = mrr.mrr_waterfall(start_new_mrr_df, expansion_df, contraction_df, churned_df)
    mrr_waterfall_df["Month_Number"] = pd.to_datetime(mrr_waterfall_df["Month"]).dt.month
    mrr_waterfall_df["Month_Sequential"] = range(1, len(mrr_waterfall_df) + 1)
    return mrr_waterfall_df

def combined_mrr_parpu_calculations(mrr_waterfall_df, df_for_arpu):
    monthly_arpu_df = parpu.calculate_arpu(df_for_arpu, 'Months_Key')
    combined_mrr_arpu_df = pd.concat([mrr_waterfall_df, monthly_arpu_df["ARPU"]], axis=1)
    combined_mrr_arpu_df.drop(columns=["Months_Key"], inplace=True, errors='ignore')
    combined_mrr_arpu_df = parpu.parpu_calculations(combined_mrr_arpu_df)
    

    """
    # Extract site metrics from the same dataframe
    site_opens = site_opens_per_month(df_for_arpu)
    base_added = base_added_per_month(df_for_arpu)
    
    # Direct merge since Month formats match
    combined_mrr_arpu_df = combined_mrr_arpu_df.merge(site_opens, on='Month', how='left')
    combined_mrr_arpu_df = combined_mrr_arpu_df.merge(base_added, on='Month', how='left')
    
    # Fill missing with 0
    combined_mrr_arpu_df['Site_Opens'] = combined_mrr_arpu_df['Site_Opens'].fillna(0)
    combined_mrr_arpu_df['Base_Added'] = combined_mrr_arpu_df['Base_Added'].fillna(0)
    """
    return combined_mrr_arpu_df





def main_pipeline(performance_tier='Full'):
    """Pipeline to process performance tiers instead of lifestage codes"""
    
    # Setup organized folder structure
    folders = setup_output_structure(performance_tier)
    
    # Process data with folder structure
    df_deduped = ft.filtering_tagging('raw_data/Added_Packages.csv', folders)

    # Filter by performance tier or keep full dataset
    if performance_tier != 'Full':
        df_filtered = ft.filter_on_performance_tier(df_deduped, performance_tier)
        print(f"▶️ Processing Performance Tier: {performance_tier}")
        print(f"Remaining rows after filtering: {len(df_filtered)}")
    else:
        df_filtered = df_deduped
        print("▶️ No filtering applied - processing Full dataset.")

    # Rest of pipeline remains the same...
    price_changes = pc.create_price_changes_from_user_input()

    if price_changes:
        df_with_prices = comprehensive_price_change_handling(df_filtered, price_changes)
        
        df_with_prices.to_csv(folders['filtered_data'] / 'price_filtered_data.csv', index=False)
        print(f"💾 Saved price filtered data to: {folders['filtered_data'] / 'price_filtered_data.csv'}")

        ex.validate_revenue_category_exclusivity(df_with_prices)

        mrr_waterfall_df = waterfall_calculations(df_with_prices)
        mrr_waterfall_df.to_csv(folders['mrr_waterfall'] / 'mrr_waterfall_results.csv', index=False)
        print(f"💾 Saved MRR waterfall to: {folders['mrr_waterfall'] / 'mrr_waterfall_results.csv'}")

        combined_df = combined_mrr_parpu_calculations(mrr_waterfall_df, df_with_prices)
        combined_df.to_csv(folders['combined_results'] / 'combined_mrr_arpu_results.csv', index=False)
        print(f"💾 Saved combined results to: {folders['combined_results'] / 'combined_mrr_arpu_results.csv'}")

        arr = combined_df['Ending MRR'].iloc[-2] * 12
        print(f"📊 ARR for {performance_tier}: ${arr:,.2f}")
    else:
        df_without_prices = df_filtered
        
        df_without_prices.to_csv(folders['filtered_data'] / 'baseline_filtered_data.csv', index=False)
        print(f"💾 Saved baseline filtered data to: {folders['filtered_data'] / 'baseline_filtered_data.csv'}")

        ex.validate_revenue_category_exclusivity(df_without_prices)

        mrr_waterfall_df = waterfall_calculations(df_without_prices)
        mrr_waterfall_df.to_csv(folders['mrr_waterfall'] / 'baseline_mrr_waterfall_results.csv', index=False)
        print(f"💾 Saved baseline MRR waterfall to: {folders['mrr_waterfall'] / 'baseline_mrr_waterfall_results.csv'}")

        combined_df = combined_mrr_parpu_calculations(mrr_waterfall_df, df_without_prices)
        combined_df.to_csv(folders['combined_results'] / 'baseline_combined_mrr_arpu_results.csv', index=False)
        print(f"💾 Saved baseline combined results to: {folders['combined_results'] / 'baseline_combined_mrr_arpu_results.csv'}")

        arr = combined_df['Ending MRR'].iloc[-2] * 12
        print(f"📊 Baseline ARR for {performance_tier}: ${arr:,.2f}")

    print(f"✅ Pipeline completed for {performance_tier}\n")


def batch_run_pipeline(selected_keys=None):
    """Updated batch pipeline for performance tiers"""
    
    key_map = {
        0: 'Full',
        1: 'high_performers',
        2: 'moderate_performers', 
        3: 'underperformers'
        
    }

    if selected_keys is None:
        selected_codes = list(key_map.values())
    else:
        selected_codes = [key_map[k] for k in selected_keys if k in key_map]

    print("🚀 Starting batch pipeline run for performance tiers...")
    print("="*60)
    
    for code in selected_codes:
        print(f"\n📋 Running pipeline for: {code}")
        main_pipeline(performance_tier=code)

    print("\n🎉 All batch runs complete!")
    print("📁 Check the 'output' folder for performance tier results:")
    print("   - output/high_performers/")
    print("   - output/moderate_performers/") 
    print("   - output/underperformers/")

if __name__ == "__main__":
    batch_run_pipeline()  

