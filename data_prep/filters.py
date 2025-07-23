import pandas as pd

def apply_business_filters(df):
    # Drop unneeded columns
    drop_cols = [
        'Account Name', 'Full Address', 'GIS LOCID', 'User Name', 
        'BH HP ID', 'City', 'Zip', 'User', 'Week Number (concurrent from 2024)',
        'Week Number', 'System Flag', 'Flag Code', 'Flag Name', 'Period ARPU'
    ]
    df.drop(columns=drop_cols, inplace=True)

    core_mrr_packages = [
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

    filtered = df[df['Package Group Condensed'].isin(core_mrr_packages)].copy()
    filtered = filtered[filtered["Test Flag"] == 0]  # Keep only non-test accounts
    
    # Filter out invalid dates
    filtered['Package Start Date'] = pd.to_datetime(filtered['Package Start Date'])
    filtered['Package End Date'] = pd.to_datetime(filtered['Package End Date'], errors='coerce')
    
    # Remove rows where start > end
    filtered = filtered[
        (filtered['Package Start Date'] <= filtered['Package End Date']) |
        (filtered['Package End Date'].isna())
    ]

    return filtered

"""
def get_lifestage_filter():
    key_map = {
        1: 'Y1', 2: 'Y2', 3: 'Y3', 4: 'F1', 5: 'F2', 
        6: 'F3', 7: 'M1', 8: 'M2', 9: 'M3', 10: 'M4'
    }

    options_str = ", ".join([f"{num}: {code}" for num, code in key_map.items()])
    
    prompt = (
        f"Enter the Lifestage Code to filter on (1-10) or press Enter / type 'No' to skip filtering:\n"
        f"Options -> {options_str}\n"
        "Your choice: "
    )
    
    while True:
        user_input = input(prompt).strip()
        if user_input == "" or user_input.lower() == "no":
            return None
        
        try:
            choice = int(user_input)
            if choice in key_map:
                return key_map[choice]
            else:
                print("Invalid number. Please enter a number between 1 and 10, or 'No' to skip.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 10, or 'No' to skip.")
"""
