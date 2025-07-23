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



def map_segments_to_performance_tiers():
    """Map performance tiers to their segment numbers"""
    
    return {
        'high_performers': [
            "5",   # You & I Tunes
            "11",  # WiFi Warriors 
            "14",  # The Pragmatics
            "2",   # Plugged-In Families
            "13",  # Cyber Sophisticates
            "3",   # Tech Nests
            "42",  # Tech Skeptics
            "27",  # Video Vistas
            "48"   # Dial-Up Duos
        ],
        'moderate_performers': [
            "25",  # Low-Speed Boomers
            "19",  # Gadgets Galore
            "18",  # New Technorati
            "1",   # Technovators
            "21",  # Multimedia Families
            "16",  # Kids & Keyboards
            "17",  # Time Shifters
            "35",  # Broadband Boulevards
            "36",  # Opting Out
            "15",  # Bundled Burbs
            "22",  # Analoggers
            "10",  # Smart Gamers
            "6",   # Connected Country
            "8",   # Calling Circles
            "29",  # IM Nation
            "31",  # Plug & Play
            "23",  # Cyber Strivers
            "47",  # Discounts & Deals
            "50",  # Early-Bird TV
            "24",  # Internet Hinterlands
            "41",  # Antenna Land
            "4"    # High-Tech Society
        ],
        'underperformers': [
            "28",  # Big City, Small Tech
            "46",  # Old-Time Media
            "33",  # Digital Dreamers
            "43",  # Bucolic Basics
            "24",  # Gearing Up
            "12",  # Satellites & Silos
            "45",  # Landline Living
            "52",  # The Unconnected
            "40",  # Low-Tech Country
            "49",  # Satellite Seniors
            "7",   # Generation WiFi
            "26",  # Rural Transmissions
            "44",  # Leisurely Adopters
            "38",  # New Kids on the Grid
            "39",  # Video Homebodies
            "53",  # Last to Adopt
            "34",  # Techs and the City
            "32",  # Family Dishes
            "9",   # Dish Country
            "51",  # Tech-Free Frontier
            "30",  # Cinemaniacs
            "27"   # Techtown Lites
        ]
    }

def filter_on_performance_tier(df, tier):
    """Filter dataframe by performance tier based on CXN Code (segment numbers)"""
    
    # Get segment numbers for this performance tier
    tier_mapping = map_segments_to_performance_tiers()
    
    if tier not in tier_mapping:
        raise ValueError(f"Unknown performance tier: {tier}")
    
    # Convert segment numbers to integers for matching
    segment_numbers = [int(seg) for seg in tier_mapping[tier]]
    
    # Filter by CXN Code instead of CXN Name
    filtered_df = df[df['CXN Code'].isin(segment_numbers)].copy()
    
    print(f"📊 Filtered for {tier}:")
    print(f"   Segment numbers included: {segment_numbers}")
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