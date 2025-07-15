from pathlib import Path

def setup_output_structure(lifestage_code='Full'):
    """Setup organized folder structure for outputs"""
    base_path = Path('output') / lifestage_code
    
    folders = {
        'filtered_data': base_path / 'filtered_data',
        'mrr_waterfall': base_path / 'mrr_waterfall', 
        'combined_results': base_path / 'combined_results',
        'raw_cleaned': base_path / 'raw_cleaned'
    }
    
    # Create all folders
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created output structure for: {lifestage_code}")
    print(f"   Base path: {base_path}")
    
    return folders