from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# Base directory (deploy safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load ML model
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# Load datasets
df_model = pd.read_csv(os.path.join(BASE_DIR, "clean_sales.csv"))  # ML
df = pd.read_csv(os.path.join(BASE_DIR, "sales.csv"), encoding='latin1')  # Dashboard

# Fix column spacing (important safety)
df.columns = df.columns.str.strip()
df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')
df["Month"] = df["Order Date"].dt.month

def get_analytics_data():
    """All dashboard analytics"""

    # 📊 Monthly Sales Trend
    monthly_data = df.groupby("Month")["Sales"].mean()

    # 📊 Category-wise Sales
    cat_sales = df.groupby("Category")["Sales"].sum()

    # 🌍 Region-wise Sales
    reg_sales = df.groupby("Region")["Sales"].sum()

    # 📉 Discount vs Profit (scatter)
    scatter_df = df[['Discount', 'Profit']].sample(n=min(500, len(df)))
    scatter_data = scatter_df.to_dict(orient='records')

    # 🪑 Sub-category performance
    subcat_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)

    # 🏆 Top 10 products
    top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)

    return {
        "labels": json.dumps(list(monthly_data.index)),
        "values": json.dumps(list(monthly_data.values)),

        "cat_labels": json.dumps(list(cat_sales.index)),
        "cat_values": json.dumps(list(cat_sales.values)),

        "reg_labels": json.dumps(list(reg_sales.index)),
        "reg_values": json.dumps(list(reg_sales.values)),

        "subcat_labels": json.dumps(list(subcat_sales.index)),
        "subcat_values": json.dumps(list(subcat_sales.values)),

        "top_prod_labels": json.dumps(list(top_products.index)),
        "top_prod_values": json.dumps(list(top_products.values)),

        "scatter_data": json.dumps(scatter_data)
    }

# Home route
@app.route("/")
def home():
    data = get_analytics_data()
    return render_template("index.html", **data)

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        quantity = int(request.form["quantity"])
        discount = float(request.form["discount"]) / 100
        month = int(request.form["month"])

        features = np.array([[quantity, discount, month]])
        prediction = model.predict(features)[0]

        data = get_analytics_data()

        return render_template(
            "index.html",
            prediction=round(prediction, 2),
            **data
        )

    except Exception as e:
        return f"Error: {e}"

# Run app
if __name__ == "__main__":
    app.run(debug=True)
