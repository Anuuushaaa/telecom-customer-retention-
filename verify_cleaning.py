import pandas as pd
from src.cleaning import clean_data

df_raw = pd.read_csv("data/raw/telco_churn.csv")  # adjust filename as needed
df_clean = clean_data(df_raw)

# TotalCharges should now be fully numeric, no NaNs
print(df_clean["TotalCharges"].dtype)
print(df_clean["TotalCharges"].isna().sum())

# Confirm the 11 zero-tenure rows are now 0, not NaN
print(df_clean.loc[df_raw["tenure"] == 0, "TotalCharges"].tolist())

# Confirm "No internet service" / "No phone service" are gone
for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]:
    print(col, df_clean[col].unique())