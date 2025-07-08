from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from src.ml_model import predict_risk
TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=TEMPLATE_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define model path
        model_path = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\models\rf_iso_model.joblib'
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model file not found at {model_path}'}), 500
            
        # Load model
        model = joblib.load(model_path)
        
        # Get input data
        data = request.get_json()
        if not all(key in data for key in ['batch_id', 'price_deviation', 'manufacturer_reliability', 'is_rural']):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Convert input to DataFrame
        input_data = pd.DataFrame([data])
        
        # Predict risk score
        risk_score = predict_risk(model, input_data)[0]
        
        return jsonify({'batch_id': data['batch_id'], 'risk_score': float(risk_score)})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def run_api(debug=False):
    app.run(host='0.0.0.0', port=5000, debug=debug)