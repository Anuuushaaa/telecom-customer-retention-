import pandas as pd
from src.cleaning import clean_data
from src.preprocessing import split_data, fit_transform_features

df_raw = pd.read_csv("data/raw/telco_churn.csv")  # adjust filename if needed
df_clean = clean_data(df_raw)

train_df, test_df = split_data(df_clean)
data = fit_transform_features(train_df, test_df)

print("X_train shape:", data.X_train.shape)
print("X_test shape:", data.X_test.shape)
print("y_train churn rate:", data.y_train.mean().round(4))
print("y_test churn rate:", data.y_test.mean().round(4))
print("\nFeature columns:", data.feature_columns)
print("\nAny NaNs in X_train?", data.X_train.isna().sum().sum())
print("Any NaNs in X_test?", data.X_test.isna().sum().sum())