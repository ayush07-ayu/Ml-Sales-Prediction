import pandas as pd

print("🚀 Cleaning Started...")

df = pd.read_csv("sales.csv", encoding="latin1")

# clean column names
df.columns = df.columns.str.strip()

# select useful columns
df = df[["Sales", "Quantity", "Discount", "Profit", "Order Date"]]

# convert date
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"] = df["Order Date"].dt.month

# final save
df.to_csv("clean_sales.csv", index=False)

print("✅ Clean data saved!")