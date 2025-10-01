import pandas as pd
import statsmodels.api as sm
import numpy as np

df = pd.read_csv("data_soc_imm.csv", thousands=",") # number parsing
df["_csv_line"] = df.index + 2   # +2 for header offset

cols = ["soc_cap", "imm", "pop", "gdp_per_capita", "u_rate", "land_area", "pop_den"]

# Parse (str -> num)
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace inf with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Check for NaN
bad_rows = df[df[cols].isna().any(axis=1)]
if not bad_rows.empty:
    print("Found problematic rows in your CSV (NaN or invalid values):")
    print(bad_rows[["_csv_line"] + cols])
    exit()

# OLS
y = df["soc_cap"]
X = df[["imm", "pop", "gdp_per_capita", "u_rate", "land_area", "pop_den"]]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
