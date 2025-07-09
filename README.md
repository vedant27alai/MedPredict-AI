# 💊 MedPredict AI

**MedPredict AI** is a Machine Learning-powered system designed to **detect counterfeit medicine batches** through intelligent analysis and prediction. It integrates:

- 🧪 Price Deviation Analysis  
- 🏭 Manufacturer Reliability Prediction  
- 💬 NLP-based Complaint Sentiment Analysis  
- ⚠️ Real-time Risk Scoring with Alerts  

All powered via a Flask API and stored securely in MongoDB.

---

## 📌 Features

- ✅ Predicts **risk score** for a medicine batch (0 to 1)
- ✅ Performs **manufacturer reliability prediction** using regression models
- ✅ Analyzes **customer complaints** using NLP sentiment analysis
- ✅ Sends **real-time alerts** for unsafe batches (Telegram integrated)
- ✅ Stores results in **MongoDB**
- ✅ RESTful API with **Flask**
- ✅ Batch status search feature (Safe / Moderate / Unsafe)
- ✅ Supports **English + Hindi UI toggle**
- ✅ Fully modular ML pipeline

---

## 📊 Models Trained

- 🎯 `RandomForestRegressor` — for batch risk scoring  
- 🏭 `ReliabilityPredictor` — predicts manufacturer trustworthiness  
- 🧠 NLP Sentiment Model — complaint text scoring  
- 🧪 Price Deviation Analyzer — compares real vs market price  

✅ All models are trained and stored as `.joblib` files.

---

## 🗂️ Project Structure

```
MedPredict-AI/
├── data/
│   └── processed/
│       ├── processed_data.csv
│       └── manufacturer_data.csv
├── models/
│   ├── rf_iso_model.joblib
│   ├── reliability_predictor.joblib
│   └── encoder_*.joblib
├── src/
│   ├── api.py
│   ├── ml_model.py
│   ├── nlp_analysis.py
│   ├── utils.py
│   └── train_reliability_model.py
├── templates/
│   └── *.html (frontend pages)
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technology Stack

- 🧠 Scikit-learn
- 🔍 NLTK / TextBlob (for NLP)
- 💻 Flask (API backend)
- ☁️ MongoDB (database)
- 📬 Telegram API (alerts)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/vedant27alai/MedPredict-AI.git
cd MedPredict-AI
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Configure `.env` File

Create a `.env` file and add:

```env
MONGO_URI=mongodb://localhost:27017
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_USER_ID=your_user_id
```

### 4. Train the Models + Start API

```bash
python main.py
```

> This will:
> - Train risk & reliability models
> - Analyze complaints
> - Save results to MongoDB
> - Launch the Flask API at `http://localhost:5000`

---

## 🔎 Usage Highlights

- Predict **Risk Score** using intelligent features
- Identify **Unsafe Manufacturers**
- Check **Price Deviation** of any medicine
- Analyze **user complaints** with NLP
- Get alerts for **high-risk batches**

---

## 🧪 Example Risk Classification

| Risk Score Range | Status     |
|------------------|------------|
| 0.0 - 0.3        | ✅ Safe     |
| 0.3 - 0.6        | ⚠️ Moderate |
| > 0.6            | ❌ Unsafe   |

---

## Made by Vedant Alai  


