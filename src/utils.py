import os
from pymongo import MongoClient
from dotenv import load_dotenv
import requests

load_dotenv()

def send_telegram_alert(message):
    """Send alert message to Telegram bot chat."""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_USER_ID')
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message}

    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"[Telegram ERROR] {response.json()}")
        else:
            print("[Telegram] Alert sent successfully!")
    except Exception as e:
        print(f"[Telegram Exception] {e}")

def save_to_mongo(data):
    """Save data to MongoDB."""
    try:
        client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
        db = client['medpredict']
        collection = db['batches']
        collection.insert_one(data)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

def send_alert(message):
    print(f"[ALERT] {message}")
    send_telegram_alert(message)  
