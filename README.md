# 💊 MedPredict AI

MedPredict AI is a Machine Learning-powered system designed to **detect counterfeit medicine batches** using a combination of price deviation analysis, manufacturer reliability prediction, NLP-based complaint analysis, and real-time risk scoring — served via a Flask API and MongoDB storage.

---

## 📌 Features

- ✅ Predicts **risk score** for a medicine batch (0 to 1).
- ✅ Performs **manufacturer reliability prediction** using regression models.
- ✅ Analyzes **customer complaints** using NLP.
- ✅ Sends **real-time alerts** (Telegram integrated).
- ✅ Stores results in **MongoDB**.
- ✅ RESTful API via **Flask**.
- ✅ Fully customizable and modular ML pipeline.

---

## 🗂️ Project Structure

MedPredict-AI/
├── data/
│ └── processed/
│ ├── processed_data.csv
│ └── manufacturer_data.csv
├── models/ # Contains saved .joblib models
├── src/
│ ├── api.py # Flask REST API
│ ├── ml_model.py # Risk prediction models
│ ├── nlp_analysis.py # Complaint analysis using NLP
│ ├── utils.py # MongoDB + alert functions
│ └── train_reliability_model.py
├── main.py # Runs full pipeline + API
├── .env # Environment variables (MONGO_URI, etc.)
├── requirements.txt
└── README.md


## 🛠️ Setup Instructions

### 🔗 1. Clone Repository

```bash
git clone https://github.com/vedant27alai/MedPredict-AI.git
cd MedPredict-AI

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Configure .env file
MONGO_URI=mongodb://localhost:27017
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_USER_ID=your_user_id

### 4. Run the Project
python main.py


```bash

