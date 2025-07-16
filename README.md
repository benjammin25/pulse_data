# Revenue Analytics & Forecasting Pipeline

A comprehensive Python-based analytics system for subscription business revenue forecasting and MRR (Monthly Recurring Revenue) analysis.

## 📊 Overview

This pipeline provides end-to-end revenue analytics for subscription-based businesses, featuring automated data processing, MRR waterfall calculations, and intelligent forecasting models that adapt to different customer segment sizes.

## 🚀 Key Features

### Revenue Analysis
- **MRR Waterfall Calculations**: Track starting MRR, new revenue, expansion, contraction, and churn
- **Price Change Handling**: Automatic processing of both new contracts and mid-contract price adjustments
- **ARPU/PARPU Metrics**: Average Revenue Per User and Per Account Revenue Per User calculations
- **ARR Projections**: Annual Recurring Revenue forecasting with growth metrics

### Machine Learning Models
- **Adaptive Model Selection**: Auto-selects appropriate forecasting models based on segment size
- **Polynomial Lasso Regression**: Uses degree-2 polynomial features with regularization
- **Cross-Validation**: Comprehensive model performance assessment
- **Multiple Complexity Tiers**: From 3-feature simple models to 11-feature comprehensive models

### Customer Lifecycle Segmentation
- **Lifestage Analysis**: Supports Y1-Y3 (Year 1-3), F1-F3 (Phase 1-3), M1-M4 (Month 1-4) segments
- **Batch Processing**: Run analysis across all segments simultaneously
- **Organized Output Structure**: Structured folder hierarchy for results

## 🏗️ Architecture

### Main Components

1. **Data Processing Pipeline** (`main_pipeline`)
   - Data filtering and tagging
   - Price change application
   - MRR waterfall generation
   - Combined metrics calculation

2. **Model Selection System** (`auto_select_lifestage_model`)
   - Intelligent model selection based on customer count
   - Adaptive feature selection
   - Performance optimization

3. **Forecasting Models** (`segment_models`)
   - Full Dataset Model (11 features)
   - Large Segment Model (7 features)
   - Medium Segment Model (5 features)
   - Small Segment Model (3 features)

## 📁 Project Structure

```
revenue_analytics/
├── main_pipeline.py          # Main execution pipeline
├── models/
│   ├── segment_models.py     # Model implementations
│   └── core.py              # Common model execution logic
├── revenue_helpers/         # Revenue calculation modules
├── data_prep/              # Data preprocessing utilities
├── setup_output/           # Output structure setup
├── raw_data/               # Input data files
└── output/                 # Organized results by lifestage
    ├── Full/
    ├── Y1/, Y2/, Y3/
    ├── F1/, F2/, F3/
    └── M1/, M2/, M3/, M4/
```

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib pathlib
```

### Required Data Files
- `raw_data/Added_Packages.csv` - Main package data
- `raw_data/fake_base_per_month.csv` - Base metrics data

## 🎯 Usage

### Basic Usage - Single Lifestage
```python
from main_pipeline import main_pipeline

# Run pipeline for specific lifestage
main_pipeline(lifestage_code='Y1')
```

### Batch Processing - All Lifestages
```python
from main_pipeline import batch_run_pipeline

# Run for all lifestages
batch_run_pipeline()

# Run for specific lifestages
batch_run_pipeline(selected_keys=[0, 1, 2])  # Full, Y1, Y2
```

### Model Selection & Forecasting
```python
from models.auto_select import auto_select_lifestage_model

# Auto-select and run appropriate model
result = auto_select_lifestage_model(
    csv_file_path='output/Full/combined_results/baseline_combined_mrr_arpu_results.csv',
    raw_data_file_path='output/Full/filtered_data/baseline_filtered_data.csv',
    months_to_project=12
)

print(f"Growth Projection: {result['current_metrics']['total_growth']:.1f}%")
```

## 🤖 Model Selection Logic

The system automatically selects the most appropriate forecasting model based on segment characteristics:

| Segment Type | Customer Count | Features | Alpha | Use Case |
|-------------|---------------|----------|-------|----------|
| **Full Dataset** | Any | 11 | 1.0 | Comprehensive analysis with all available features |
| **Large Segment** | 2500+ | 7 | 2.0 | Complex pattern detection for large datasets |
| **Medium Segment** | 1600-2499 | 5 | 5.0 | Balanced approach for medium-sized segments |
| **Small Segment** | <1600 | 3 | 50.0 | Stable projections for limited data |

## 📈 Model Features by Tier

### Full Dataset Model (11 Features)
- Month_Sequential, Month_Number
- New MRR, Churn, Starting MRR
- Ending Count, Ending PARPU, Starting PARPU
- ARPU, Churn_Rate, Base

### Large Segment Model (7 Features)
- Month_Sequential, Month_Number
- New MRR, Churn
- Ending Count, Ending PARPU, Churn_Rate

### Medium Segment Model (5 Features)
- Month_Sequential, New MRR, Churn
- Ending Count, Ending PARPU

### Small Segment Model (3 Features)
- Month_Sequential, New MRR, Churn

## 📊 Output Files

### Generated Reports
- `filtered_data/` - Processed and filtered datasets
- `mrr_waterfall/` - MRR waterfall analysis results
- `combined_results/` - Combined MRR and ARPU metrics
- Forecasting visualizations and projections

### Key Metrics
- **Current MRR**: Latest monthly recurring revenue
- **Projected MRR**: Future revenue projections
- **Growth Rate**: Monthly and total growth percentages
- **Model Performance**: R², MAE, RMSE, Cross-validation scores

## 🔍 Lifestage Codes

| Code | Description | Code | Description |
|------|-------------|------|-------------|
| `Full` | Complete dataset | `Y1` | Year 1 customers |
| `Y2` | Year 2 customers | `Y3` | Year 3 customers |
| `F1` | Phase 1 customers | `F2` | Phase 2 customers |
| `F3` | Phase 3 customers | `M1` | Month 1 customers |
| `M2` | Month 2 customers | `M3` | Month 3 customers |
| `M4` | Month 4 customers | | |

## 🎨 Visualization Features

- **MRR Trend Analysis**: Historical and projected revenue curves
- **Polynomial Trend Lines**: Degree-2 polynomial fitting
- **Residuals Analysis**: Model accuracy assessment
- **MRR Drivers**: New revenue vs. churn visualization
- **Performance Metrics**: Comprehensive model evaluation

## 🚦 Best Practices

1. **Data Quality**: Ensure clean, consistent input data
2. **Regular Updates**: Refresh models with new data monthly
3. **Model Validation**: Review cross-validation scores before deployment
4. **Segment Analysis**: Compare performance across different lifestages
5. **Conservative Projections**: Higher alpha values provide more stable forecasts

## 🔧 Configuration

### Price Changes
The system prompts for price changes during execution:
- New contract pricing updates
- Mid-contract price adjustments
- Effective date specifications

### Model Parameters
- **Projection Period**: Default 12 months (configurable)
- **Polynomial Degree**: Fixed at 2 for stability
- **Alpha Values**: Automatically selected based on segment size

## 📋 Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0

## 🤝 Contributing

When extending the pipeline:
1. Follow the modular architecture
2. Add comprehensive logging
3. Include performance metrics
4. Update documentation
5. Test with various segment sizes

## 📞 Support

For questions or issues:
- Check the organized output structure in `output/`
- Review model performance metrics
- Validate input data format and completeness

## 🏆 Success Metrics

- **R² Score**: Model fit quality (target: >0.8)
- **Cross-Validation MAE**: Prediction accuracy
- **Growth Rate Stability**: Consistent month-over-month projections
- **ARR Forecasting**: Annual revenue planning accuracy