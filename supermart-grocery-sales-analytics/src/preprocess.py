import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

os.makedirs("data", exist_ok=True)
df = pd.read_csv("data/Supermart Grocery Sales - Retail Analytics Dataset.csv")

# Basic cleaning
df = df.drop_duplicates().copy()
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# Date features
df['Order_Day'] = df['Order Date'].dt.day
df['Order_Month'] = df['Order Date'].dt.month
df['Order_Year'] = df['Order Date'].dt.year

# Encode categorical columns
cols_to_encode = ['Category','Sub Category','City','Region','State']
le = LabelEncoder()
for col in cols_to_encode:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Save processed
df.to_csv("data/processed.csv", index=False)
print("Preprocessing done. Saved to data/processed.csv")