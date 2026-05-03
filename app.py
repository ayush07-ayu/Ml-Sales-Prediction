import os
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

# base directory fix
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load model
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

# load data
df = pd.read_csv(os.path.join(BASE_DIR, "clean_sales.csv"))

@app.route("/")
def home():
    monthly_data = df.groupby("Month")["Sales"].mean()

    labels = json.dumps(list(monthly_data.index))
    values = json.dumps(list(monthly_data.values))

    return render_template("index.html", labels=labels, values=values)

@app.route("/predict", methods=["POST"])
def predict():
    quantity = int(request.form["quantity"])
    discount = float(request.form["discount"]) / 100
    month = int(request.form["month"])

    features = np.array([[quantity, discount, month]])
    prediction = model.predict(features)[0]

    monthly_data = df.groupby("Month")["Sales"].mean()
    labels = json.dumps(list(monthly_data.index))
    values = json.dumps(list(monthly_data.values))

    return render_template(
        "index.html",
        prediction=round(prediction, 2),
        labels=labels,
        values=values
    )

if __name__ == "__main__":
    app.run()
