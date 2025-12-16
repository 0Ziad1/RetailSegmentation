from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import os

app = Flask(__name__)

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler safely
model = joblib.load(os.path.join(BASE_DIR, "svm_ova_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

@app.route("/")
def home():
    return send_file(os.path.join(BASE_DIR, "index.html"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ⚠️ FEATURE ORDER MUST MATCH TRAINING
    features = np.array([
        float(data["annual_income"]),
        float(data["num_web_purchases"]),
        float(data["num_store_purchases"])
    ]).reshape(1, -1)

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)

    return jsonify({
        "segment": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
