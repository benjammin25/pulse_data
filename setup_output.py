from pathlib import Path

def setup_output_structure(performance_tier='Full'):
    """Setup organized folder structure for performance tier outputs"""
    base_path = Path('output') / performance_tier
    
    folders = {
        'filtered_data': base_path / 'filtered_data',
        'mrr_waterfall': base_path / 'mrr_waterfall', 
        'combined_results': base_path / 'combined_results',
        'raw_cleaned': base_path / 'raw_cleaned'
    }
    
    # Create all folders
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created output structure for: {performance_tier}")
    print(f"   Base path: {base_path}")
    
    return folders