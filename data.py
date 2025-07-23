import pandas as pd
from datetime import datetime
df = pd.read_csv("output/Full/combined_results/baseline_combined_mrr_arpu_results.csv", encoding='latin-1', low_memory=False)
print(df.shape)
print(df.info())

