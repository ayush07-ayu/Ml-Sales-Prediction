import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

print("🚀 Training Started...")

df = pd.read_csv("clean_sales.csv")

# features & target
X = df[["Quantity", "Discount", "Month"]]
y = df["Sales"]

model = RandomForestRegressor()
model.fit(X, y)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained & saved!")