import pandas as pd
import joblib
import os
from pymongo import MongoClient
from src.ml_model import train_model, predict_risk
from src.nlp_analysis import analyze_complaints
from src.utils import send_alert
from src.api import run_api  # ✅ Ensures Flask runs after processing

def save_to_mongo(record):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['medpredict']
    collection = db['batches']

    batch_id = record['batch_id']
    if not collection.find_one({'batch_id': batch_id}):
        collection.insert_one(record)
        print(f"✅ Added batch {batch_id}")
    else:
        print(f"⚠️ Batch {batch_id} already exists. Skipping.")

def main():
    data_file = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\data\processed\processed_data.csv'
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("❌ Processed data file not found.")
        return

    try:
        model = train_model(df)
        model_path = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\models\rf_iso_model.joblib'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"✅ Model trained and saved to {model_path}")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return

    try:
        risk_scores = predict_risk(model, df)
        complaint_scores = analyze_complaints(df['complaints'].tolist())
    except Exception as e:
        print(f"❌ Error during prediction or complaint analysis: {e}")
        return

    for idx, row in df.iterrows():
        risk_score = float(risk_scores[idx])
        status = (
            "Safe" if risk_score < 0.3 else
            "Moderate" if risk_score < 0.6 else
            "Unsafe"
        )

        record = {
            "batch_id": str(row['batch_id']).strip().lower(),
            "medicine_name": row.get('medicine_name', 'Unknown'),
            "risk_score": risk_score,
            "status": status,
            "complaint_score": float(complaint_scores[idx]),
            "price_deviation": float(row.get('price_deviation', 0.0)),
            "manufacturer_reliability": float(row.get('manufacturer_reliability', 0.0)),
            "is_rural": int(row.get('is_rural', 0))
        }

        save_to_mongo(record)

    print("✅ All data processed and saved to MongoDB.")

    try:
        for idx, score in enumerate(risk_scores):
            if score > 0.9:
                batch_id = df.loc[idx, 'batch_id']
                alert_message = f"🚨 High risk batch {batch_id}: Score {score:.2f}"
                send_alert(alert_message)
    except Exception as e:
        print(f"❌ Error sending alerts: {e}")

    # ✅ Start Flask API
    try:
        print("🌐 Starting Flask API at http://localhost:5000")
        run_api(debug=False)
    except Exception as e:
        print(f"❌ Error running API: {e}")

if __name__ == "__main__":
    main()
