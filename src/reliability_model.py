import joblib
import numpy as np
import os

# Use the absolute path to models directory
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

model = joblib.load(os.path.join(MODEL_DIR, 'reliability_predictor.joblib'))
le_name = joblib.load(os.path.join(MODEL_DIR, 'encoder_name.joblib'))
le_manufacturer = joblib.load(os.path.join(MODEL_DIR, 'encoder_manufacturer.joblib'))
le_composition = joblib.load(os.path.join(MODEL_DIR, 'encoder_composition.joblib'))

def extract_quantity(pack_label):
    try:
        return int([x for x in pack_label.split() if x.isdigit()][0])
    except:
        return 1

def predict_reliability(name, price, manufacturer, pack_size_label, composition):
    quantity = extract_quantity(pack_size_label)
    price_per_unit = price / quantity

    try:
        name_enc = le_name.transform([name])[0]
    except:
        name_enc = 0
    try:
        manu_enc = le_manufacturer.transform([manufacturer])[0]
    except:
        manu_enc = 0
    try:
        comp_enc = le_composition.transform([composition])[0]
    except:
        comp_enc = 0

    features = np.array([[price_per_unit, name_enc, manu_enc, comp_enc]])
    return round(model.predict(features)[0], 3)
