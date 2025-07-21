from models.segment_models import large_segment_model, full_dataset_model
from pathlib import Path
import pandas as pd



def auto_select_performance_tier_model(csv_file_path, months_to_project=12):
    """
    Auto-selects the appropriate model based on performance_tier
    """
    performance_tier = Path(csv_file_path).parts[-3]


    # Auto-selection logic based on performance_tier and customer counts
    if performance_tier in ['Full']:
        print(f"🔍 Auto-selecting Full Dataset Model for {performance_tier}")
        return full_dataset_model(csv_file_path, months_to_project)
    else:  
        print(f"🔍 Auto-selecting Small Segment Model for {performance_tier}")
        return large_segment_model(csv_file_path, months_to_project)
    
if __name__ == "__main__":
    # Example usage with auto-selection
    csv_file_path = 'output/Full/combined_results/baseline_combined_mrr_arpu_results.csv'

    result = auto_select_performance_tier_model(csv_file_path)
    
    print(f"\n📊 Results for {result['performance_tier']}:")
    print(f"Model Type: {result['model_type']}")
    print(f"Features Used: {len(result['feature_names'])}")
    print(f"Growth Projection: {result['current_metrics']['total_growth']:.1f}%")
    
    # Or use specific models directly:
    # result = large_segment_model('output/full/combined_results/baseline_combined_mrr_arpu_results.csv')
    # result = medium_segment_model('output/F1/combined_results/baseline_combined_mrr_arpu_results.csv')
    # result = small_segment_model('output/Y3/combined_results/baseline_combined_mrr_arpu_results.csv')