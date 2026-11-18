"""Joins review_data_1.csv ... review_data_15.csv back into Reviews.csv.
Run once before opening the notebook:  python combine_data.py
"""
import glob
import pandas as pd

# sort numerically (1, 2, ... 10) so the original row order is kept
files = sorted(glob.glob("review_data_*.csv"),
               key=lambda f: int(f.split("_")[-1].split(".")[0]))

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df.to_csv("Reviews.csv", index=False)
print(f"Created Reviews.csv with {len(df):,} rows from {len(files)} files")
