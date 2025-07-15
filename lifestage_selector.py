from models.segment_models import large_segment_model, medium_segment_model, small_segment_model, full_dataset_model
from pathlib import Path
import pandas as pd

def get_customer_count(raw_data_file_path):
    """
    Returns the number of customers in the dataset.
    """
    df = pd.read_csv(raw_data_file_path, encoding='latin-1', low_memory=False)
    df = df[:-1]  # Remove last row if it's a summary

    return df.shape[0]

def auto_select_lifestage_model(csv_file_path, raw_data_file_path, months_to_project=12):
    """
    Auto-selects the appropriate model based on lifestage
    """
    lifestage = Path(csv_file_path).parts[-3]
    customer_count = get_customer_count(raw_data_file_path)

    # Auto-selection logic based on lifestage and customer counts
    if lifestage in ['Full']:
        print(f"🔍 Auto-selecting Full Dataset Model for {lifestage}")
        return full_dataset_model(csv_file_path, months_to_project)
    elif customer_count >= 2500:
        print(f"🔍 Auto-selecting Large Segment Model for {lifestage}")
        return large_segment_model(csv_file_path, months_to_project)
    elif customer_count >= 1600 and customer_count <= 2499:
        print(f"🔍 Auto-selecting Medium Segment Model for {lifestage}")
        return medium_segment_model(csv_file_path, months_to_project)
    else:  
        print(f"🔍 Auto-selecting Small Segment Model for {lifestage}")
        return small_segment_model(csv_file_path, months_to_project)
    
if __name__ == "__main__":
    # Example usage with auto-selection
    csv_file_path = 'output/Full/combined_results/baseline_combined_mrr_arpu_results.csv'
    raw_data_file_path = "output/Full/filtered_data/baseline_filtered_data.csv"

    result = auto_select_lifestage_model(csv_file_path, raw_data_file_path)
    
    print(f"\n📊 Results for {result['lifestage']}:")
    print(f"Model Type: {result['model_type']}")
    print(f"Features Used: {len(result['feature_names'])}")
    print(f"Growth Projection: {result['current_metrics']['total_growth']:.1f}%")
    
    # Or use specific models directly:
    # result = large_segment_model('output/full/combined_results/baseline_combined_mrr_arpu_results.csv')
    # result = medium_segment_model('output/F1/combined_results/baseline_combined_mrr_arpu_results.csv')
    # result = small_segment_model('output/Y3/combined_results/baseline_combined_mrr_arpu_results.csv')