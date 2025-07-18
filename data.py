import pandas as pd
from datetime import datetime
"""df_base = pd.read_csv("raw_data/fake_base_per_month.csv", encoding='latin-1', low_memory=False)
df_combined = pd.read_csv("output/Full/combined_results/baseline_combined_mrr_arpu_results.csv", encoding='latin-1', low_memory=False)
print(df_combined.info())"""



df = pd.read_csv("output/Full/filtered_data/baseline_filtered_data.csv")
print(df["Lifestage Code"].value_counts())


