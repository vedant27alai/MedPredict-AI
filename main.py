import pandas as pd
from src.ml_model import train_model, predict_risk
from src.nlp_analysis import analyze_complaints
from src.utils import save_to_mongo, send_alert
from src.api import run_api
import joblib
import os

def main():
    # Load processed data
    data_file = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\data\processed\processed_data.csv'
    try:
        processed_data = pd.read_csv(data_file)
        print(f"Loaded {len(processed_data)} rows from {data_file}")
    except FileNotFoundError:
        print(f"Error: Processed data not found at {data_file}. Run test_parse.py first.")
        return

    # Train ML model
    try:
        model = train_model(processed_data)
        os.makedirs(r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\models', exist_ok=True)
        model_path = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\models\rf_iso_model.joblib'
        joblib.dump(model, model_path)
        print(f"Model trained and saved to {model_path}")
    except Exception as e:
        print(f"Error training model: {e}")
        return

    # Predict risk scores
    try:
        risk_scores = predict_risk(model, processed_data)
        print("Risk Scores (first 5):", risk_scores[:5])
    except Exception as e:
        print(f"Error predicting risk scores: {e}")
        return

    # Analyze complaints
    try:
        complaint_scores = analyze_complaints(processed_data['complaints'].tolist())
        print("Complaint Scores (first 5):", complaint_scores[:5])
    except Exception as e:
        print(f"Error analyzing complaints: {e}")
        return

    # Save to MongoDB
    try:
        
        for idx, row in processed_data.iterrows():
            
            save_to_mongo({
                'batch_id': row['batch_id'],
                'risk_score': float(risk_scores[idx]),
                'complaint_score': float(complaint_scores[idx])
            })
        print("Data saved to MongoDB")
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

    # Send alerts
    try:
        for idx, score in enumerate(risk_scores):
            if score > 0.8:
                batch_id = processed_data.loc[idx, 'batch_id']
                alert_message = f"🚨 High risk batch {batch_id}: Score {score:.2f}"
                send_alert(alert_message)
    except Exception as e:
        print(f"Error sending alerts: {e}")


    # Run Flask API
    try:
        print("Starting Flask API at http://localhost:5000/predict")
        run_api(debug=False)  # Disable debug mode to prevent restarts
    except Exception as e:
        print(f"Error running API: {e}")

if __name__ == "__main__":
    main()