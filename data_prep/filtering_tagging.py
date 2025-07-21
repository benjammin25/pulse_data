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

def filter_on_performance_tier(df, tier):
    """Filter dataframe by performance tier based on CXN Name"""
    
    # Define performance tier mappings based on visual color coding (green/yellow/red)
    performance_mapping = {
        'high_performers': [
            # GREEN segments from your data
            "You & I Tunes",
            "WiFi Warriors", 
            "The Pragmatics",
            "Plugged-In Families",
            "Cyber Sophisticates",
            "Tech Nests",
            "Tech Skeptics",
            "Video Vistas",
            "Dial-Up Duos",
        ],
        'moderate_performers': [
            # YELLOW segments from your data  
            "Low-Speed Boomers",
            "Gadgets Galore",
            "New Technorati",
            "Technovators",
            "Multimedia Families",
            "Kids & Keyboards",
            "Time Shifters",
            "Broadband Boulevards",
            "Opting Out",
            "Bundled Burbs",
            "Analoggers",
            "Smart Gamers",
            "Connected Country",
            "Calling Circles",
            "IM Nation",
            "Plug & Play",
            "Cyber Strivers",
            "Discounts & Deals",
            "Early-Bird TV",
            "Internet Hinterlands",
            "Antenna Land",
            "High-Tech Society",
        ],
        'underperformers': [
            # RED segments from your data
            "Big City, Small Tech",
            "Old-Time Media",
            "Digital Dreamers",
            "Bucolic Basics",
            "Gearing Up",
            "Satellites & Silos",
            "Landline Living",
            "The Unconnected",
            "Low-Tech Country",
            "Satellite Seniors",
            "Generation WiFi",
            "Rural Transmissions",
            "Leisurely Adopters",
            "New Kids on the Grid",
            "Video Homebodies",
            "Last to Adopt",
            "Techs and the City",
            "Family Dishes",
            "Dish Country",
            "Tech-Free Frontier",
            "Cinemaniacs",
            "Techtown Lites",
            "Unassigned",
        ]
    }
    
    if tier not in performance_mapping:
        raise ValueError(f"Unknown performance tier: {tier}")
    
    cxn_names = performance_mapping[tier]
    filtered_df = df[df['CXN Name'].isin(cxn_names)].copy()
    
    print(f"📊 Filtered for {tier}:")
    print(f"   CXN Names included: {cxn_names}")
    print(f"   Rows before filtering: {len(df)}")
    print(f"   Rows after filtering: {len(filtered_df)}")
    
    return filtered_df

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