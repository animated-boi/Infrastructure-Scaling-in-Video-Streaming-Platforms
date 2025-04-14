# src/check_columns.py
import pandas as pd

df = pd.read_csv("results/aggregated_timeseries.csv")
print("✅ Available columns:")
print(df.columns)
