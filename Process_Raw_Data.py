import pandas as pd
import numpy as np
from revenue_helpers import revenue as rev
from revenue_helpers import EndingMRR as mrr
from revenue_helpers import price_changes as pc
from revenue_helpers import parpu

from data_prep import filtering_tagging as ft
from data_prep import exclusivity as ex
from setup_output import setup_output_structure


def base_added_per_month(df_filtered_customers, performance_tier='Full'):
    # Get sites and convert to clean strings (remove .0 decimals)
    relevant_sites = []
    for site in df_filtered_customers['Site'].dropna().unique():
        try:
            # Convert to int first to remove decimals, then to string
            relevant_sites.append(str(int(float(site))))
        except (ValueError, TypeError):
            # Keep as string for non-numeric sites
            relevant_sites.append(str(site))
    
    # Load and clean site data
    sites_df = pd.read_csv('raw_data/Site_Segment.csv')
    sites_df = sites_df.dropna(subset=['Site', 'Date'])
    
    # Ensure reference sites are clean strings
    sites_df['Site'] = sites_df['Site'].astype(str)
    
    segment_sites = sites_df[sites_df['Site'].isin(relevant_sites)].copy()
    
    if segment_sites.empty:
        return pd.DataFrame(columns=["Month", "Base_Added"])
    
    segment_sites["Site Release Month"] = pd.to_datetime(segment_sites["Date"]).dt.to_period("M")
    
    if performance_tier == 'Full':
        segment_columns = [str(i) for i in range(1, 54)]
        available_columns = [col for col in segment_columns if col in segment_sites.columns]
        base_added = segment_sites.groupby("Site Release Month")[available_columns].sum().sum(axis=1).reset_index()
    else:
        tier_mapping = ft.map_segments_to_performance_tiers()
        relevant_segments = tier_mapping[performance_tier]
        available_segments = [seg for seg in relevant_segments if seg in segment_sites.columns]
        base_added = segment_sites.groupby("Site Release Month")[available_segments].sum().sum(axis=1).reset_index()
    
    base_added.columns = ["Month", "Base"]
    base_added['Month'] = base_added['Month'].astype(str)
    
    return base_added


def site_opens_per_month(df):
    # Get sites and convert to clean strings (remove .0 decimals)
    relevant_sites = []
    for site in df['Site'].dropna().unique():
        try:
            # Convert to int first to remove decimals, then to string
            relevant_sites.append(str(int(float(site))))
        except (ValueError, TypeError):
            # Keep as string for non-numeric sites
            relevant_sites.append(str(site))
    
    # Load and clean site data
    sites_df = pd.read_csv('raw_data/Site_Segment.csv')
    sites_df = sites_df.dropna(subset=['Site', 'Date'])
    
    # Ensure reference sites are clean strings
    sites_df['Site'] = sites_df['Site'].astype(str)
    
    segment_sites = sites_df[sites_df['Site'].isin(relevant_sites)].copy()
    
    if segment_sites.empty:
        return pd.DataFrame(columns=["Month", "Site_Opens"])
    
    segment_sites["Site Release Month"] = pd.to_datetime(segment_sites["Date"]).dt.to_period("M")
    site_opens = segment_sites.groupby("Site Release Month")["Site"].nunique().reset_index()
    site_opens.columns = ["Month", "Site_Opens"]
    
    # Convert Period to string for merging
    site_opens['Month'] = site_opens['Month'].astype(str)

    return site_opens


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
    
    # Extract site metrics from the same dataframe
    site_opens = site_opens_per_month(df_for_arpu)
    
    # Direct merge since Month formats match
    combined_mrr_arpu_df = combined_mrr_arpu_df.merge(site_opens, on='Month', how='left')
    
    # Fill missing with 0
    combined_mrr_arpu_df['Site_Opens'] = combined_mrr_arpu_df['Site_Opens'].fillna(0)
    
    return combined_mrr_arpu_df


def main_pipeline(performance_tier='Full'):
    """Pipeline to process performance tiers instead of lifestage codes"""
    
    # Setup organized folder structure
    folders = setup_output_structure(performance_tier)
    
    # Process data with folder structure
    df_deduped = ft.filtering_tagging('raw_data/Packages_withSites.csv', folders)

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

        base_added = base_added_per_month(df_with_prices, performance_tier)
        combined_df = combined_df.merge(base_added, on='Month', how='left')
        combined_df['Base'] = combined_df['Base'].fillna(0)
        
        combined_df["Cumulative_Base"] = combined_df["Base"].cumsum()
        combined_df['Take_Rate'] = np.where(
            combined_df['Cumulative_Base'] > 0,
            combined_df['Ending Count'] / combined_df['Cumulative_Base'],
            0  
        )

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

        base_added = base_added_per_month(df_without_prices, performance_tier)

        combined_df = combined_df.merge(base_added, on='Month', how='left')
        combined_df['Base'] = combined_df['Base'].fillna(0)
        
        combined_df["Cumulative_Base"] = combined_df["Base"].cumsum()
        combined_df['Take_Rate'] = np.where(
            combined_df['Cumulative_Base'] > 0,
            combined_df['Ending Count'] / combined_df['Cumulative_Base'], 0)

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
    batch_run_pipeline([0])