# 💊 MedPredict AI

**MedPredict AI** is a Machine Learning-powered system designed to **detect counterfeit medicine batches** using:

- Price deviation analysis  
- Manufacturer reliability prediction  
- NLP-based complaint analysis  
- Real-time risk scoring  

All served via a **Flask API** and stored securely in **MongoDB**.

---

## 📌 Features

- ✅ Predicts **risk score** for a medicine batch (0 to 1)
- ✅ Performs **manufacturer reliability prediction** using regression models
- ✅ Analyzes **customer complaints** using NLP
- ✅ Sends **real-time alerts** (Telegram integrated)
- ✅ Stores results in **MongoDB**
- ✅ RESTful API via **Flask**
- ✅ Fully customizable and modular ML pipeline

---

## 🗂️ Key Project Files

- `main.py` – Orchestrates the full pipeline (training + API)
- `src/api.py` – Flask server with all route endpoints
- `src/ml_model.py` – Machine learning models for risk prediction
- `src/train_reliability_model.py` – Reliability prediction model
- `src/utils.py` – MongoDB storage and alert integrations
- `src/nlp_analysis.py` – NLP-based complaint analysis
- `data/processed/` – Contains cleaned datasets (CSV)
- `models/` – Stores trained models (`.joblib`)
- `requirements.txt` – All Python package dependencies
- `.env` – Store sensitive keys like Mongo URI, Telegram token

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vedant27alai/MedPredict-AI.git
cd MedPredict-AI
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory and add:

```env
MONGO_URI=mongodb://localhost:27017
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_USER_ID=your_user_id
```

### 4. Run the Project

```bash
python main.py
```

---

## 👨‍⚕️ Usage Highlights

- **Risk Score Prediction:** Uses ML model to determine whether a batch is counterfeit or not.
- **Reliability Score:** Trains a model to predict how trustworthy a manufacturer is based on real features.
- **Complaint Analysis:** NLP pipeline analyzes user-submitted complaints for sentiment and severity.
- **Telegram Alerts:** Sends alerts for high-risk batches instantly to your Telegram chat.
- **Search Batch Feature:** Allows users to check if their batch ID is marked as safe or unsafe.

---

## 🙌 Built by Vedant Alai
