import pandas as pd

def create_lifecycle_grouped_features():
    """Create 6 Ending MRR features based on your business groupings"""
    
    # Define your 6 groupings
    groupings = {
        'young_early': ["Y1", "Y2"],      # Young customers, early lifecycle
        'family_early': ["F1", "F2"],     # Family customers, early lifecycle  
        'mature_early': ["M1", "M2"],     # Mature customers, early lifecycle
        'mature_late': ["M3"]       # Mature customers, late lifecycle

    }
    
    # Load all segment data
    segment_data = {}
    all_segments = ['Y1', 'Y2', 'F1', 'F2', 'M1', 'M2', 'Y3', 'F3', 'M3', 'M4']
    
    for segment in all_segments:
        file_path = f'output/{segment}/combined_results/baseline_combined_mrr_arpu_results.csv'
        segment_data[segment] = pd.read_csv(file_path)[:-1]
    
    # Calculate Ending MRR for each grouping
    features = {}
    for group_name, segments in groupings.items():
        features[f'{group_name}_net_new_mrr'] = sum(
            segment_data[seg]["Net New MRR"] for seg in segments
        )
    
    return features