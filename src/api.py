from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from pymongo import MongoClient

from src.ml_model import predict_risk
from src.reliability_model import predict_reliability
from src.price_deviation import calculate_price_deviation, get_official_price

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=TEMPLATE_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reliability', methods=['GET', 'POST'])
def reliability_page():
    if request.method == 'POST':
        try:
            name = request.form['name']
            price = float(request.form['price'])
            manufacturer = request.form['manufacturer']
            pack = request.form['pack']
            composition = request.form['composition']

            score = predict_reliability(name, price, manufacturer, pack, composition)
            return render_template('result_reliability.html', score=score)
        except Exception as e:
            return render_template('result_reliability.html', score=f"❌ Error: {e}")
    return render_template('reliability_form.html')

@app.route('/price-deviation', methods=['GET', 'POST'])
def price_deviation_page():
    if request.method == 'POST':
        try:
            med_name = request.form['med_name']
            user_price = float(request.form['user_price'])

            deviation = calculate_price_deviation(user_price, med_name)
            official_price = get_official_price(med_name)

            if deviation is None or official_price is None:
                return render_template('result_price.html', deviation="❌ Medicine not found in price list.")

            return render_template('result_price.html', deviation=deviation, official_price=official_price)

        except Exception as e:
            return render_template('result_price.html', deviation=f"❌ Error: {e}")
    return render_template('price_deviation_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_path = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\models\rf_iso_model.joblib'
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model not found at {model_path}'}), 500

        model = joblib.load(model_path)
        data = request.get_json()

        required_fields = ['batch_id', 'price_deviation', 'manufacturer_reliability', 'is_rural']
        if not all(key in data for key in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        input_df = pd.DataFrame([data])
        risk_score = predict_risk(model, input_df)[0]

        status = (
            "Safe" if risk_score < 0.3 else
            "Moderate" if risk_score < 0.6 else
            "Unsafe"
        )

        client = MongoClient("mongodb://localhost:27017/")
        db = client["medpredict"]
        collection = db["batches"]

        batch_id = data["batch_id"].strip().lower()
        existing = collection.find_one({"batch_id": batch_id})
        if not existing:
            record = {
                "batch_id": batch_id,
                "risk_score": float(risk_score),
                "status": status,
                "price_deviation": float(data["price_deviation"]),
                "manufacturer_reliability": float(data["manufacturer_reliability"]),
                "is_rural": int(data["is_rural"])
            }
            collection.insert_one(record)

        return jsonify({
            'batch_id': batch_id,
            'risk_score': float(risk_score),
            'status': status
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def run_api(debug=False):
    app.run(host='0.0.0.0', port=5000, debug=debug)
