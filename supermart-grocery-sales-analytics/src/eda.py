import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("visuals", exist_ok=True)

df = pd.read_csv("data/Supermart Grocery Sales - Retail Analytics Dataset.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# 1) Sales by Category
sales_cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sales_cat.plot(kind='bar')
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visuals/sales_by_category.png")
plt.close()

# 2) Sales by Month (using month number)
df['Month'] = df['Order Date'].dt.month
monthly = df.groupby("Month")["Sales"].sum().reindex(range(1,13), fill_value=0)
plt.figure(figsize=(10,6))
plt.plot(monthly.index, monthly.values, marker='o')
plt.title("Sales by Month")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.tight_layout()
plt.savefig("visuals/sales_by_month.png")
plt.close()

# 3) Correlation heatmap (numeric only)
plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (numeric columns)")
plt.tight_layout()
plt.savefig("visuals/correlation_heatmap.png")
plt.close()

print("EDA complete. Plots saved to visuals/")