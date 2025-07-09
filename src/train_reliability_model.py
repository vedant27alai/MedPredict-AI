import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

def extract_quantity(label):
    try:
        return int([x for x in label.split() if x.isdigit()][0])
    except:
        return 1

def train_reliability_model():
    file_path = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\data\processed\A_Z_medicines_dataset_of_India.csv'

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required = ['name', 'price(₹)', 'manufacturer_name', 'pack_size_label', 'short_composition1']
    df = df.dropna(subset=required)

    # 🧠 MOCK reliability score (just for this model)
    trusted = ['Cipla', 'Sun Pharma', 'Glaxo SmithKline', 'Dr. Reddy', 'Alembic']
    df['reliability_score'] = df['manufacturer_name'].apply(lambda x: 1.0 if any(t in x for t in trusted) else 0.4)

    # Drop rows if score wasn't assigned (rare case)
    df = df.dropna(subset=['reliability_score'])

    # 📦 Calculate price per unit
    df['quantity'] = df['pack_size_label'].apply(extract_quantity)
    df['price_per_unit'] = df['price(₹)'] / df['quantity']

    # 🔤 Encode text features
    le_name = LabelEncoder()
    le_manufacturer = LabelEncoder()
    le_composition = LabelEncoder()

    df['name_encoded'] = le_name.fit_transform(df['name'])
    df['manufacturer_encoded'] = le_manufacturer.fit_transform(df['manufacturer_name'])
    df['composition_encoded'] = le_composition.fit_transform(df['short_composition1'])

    # 🎯 Feature & Target
    features = ['price_per_unit', 'name_encoded', 'manufacturer_encoded', 'composition_encoded']
    target = 'reliability_score'

    X = df[features]
    y = df[target]

    # 📊 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🧠 Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 📈 Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"✅ Model Trained — R2: {r2:.4f}, RMSE: {rmse:.4f}")

    # 💾 Save Model & Encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/reliability_predictor.joblib')
    joblib.dump(le_name, 'models/encoder_name.joblib')
    joblib.dump(le_manufacturer, 'models/encoder_manufacturer.joblib')
    joblib.dump(le_composition, 'models/encoder_composition.joblib')

    print("📦 Model and encoders saved successfully in /models")

if __name__ == "__main__":
    train_reliability_model()
