import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def train_reliability_model():
    file_path = r'C:\Users\vedan\OneDrive\Desktop\language\PROJECT\MedPredict AI\MedPredict-AI\data\processed\manufacturer_data.csv'
    data = pd.read_csv(file_path)

    selected_features = [
        'lead_time_days',
        'fuel_consumption_rate',
        'traffic_congestion_level',
        'order_fulfillment_status',
        'weather_condition_severity',
        'delivery_time_deviation',
        'delay_probability'
    ]
    target = 'supplier_reliability_score'

    # Check if all selected features exist
    missing = [f for f in selected_features + [target] if f not in data.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return

    X = data[selected_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=42)
    }

    best_model = None
    best_r2 = -999

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"🔍 {name} → R2: {r2:.4f}, RMSE: {rmse:.4f}")

        if r2 > best_r2:
            best_model = model
            best_r2 = r2
            best_model_name = name

    # Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/reliability_predictor.joblib')
    print(f"✅ Best model: {best_model_name} saved at models/reliability_predictor.joblib")

if __name__ == "__main__":
    train_reliability_model()
