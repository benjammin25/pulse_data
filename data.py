import pandas as pd
from datetime import datetime

df_base = pd.read_csv("raw_data/fake_base_per_month.csv", encoding='latin-1', low_memory=False)
df_combined = pd.read_csv("output/Full/combined_results/baseline_combined_mrr_arpu_results.csv", encoding='latin-1', low_memory=False)
df_combined = df_combined.merge(df_base, on='Month_Sequential', how='left')
print(df_combined.tail())
