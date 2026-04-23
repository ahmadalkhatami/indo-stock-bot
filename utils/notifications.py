import requests
import urllib.parse
from utils.logger import logger

def send_telegram_signal(token, chat_id, message):
    """
    Sends a Telegram message.
    """
    if not token or token == "YOUR_BOT_TOKEN_HERE":
        logger.warning("Telegram Bot Token not set. Skipping notification.")
        return False
        
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(f"Telegram signal sent to {chat_id} successfully.")
            return True
        else:
            logger.error(f"Failed to send Telegram. Status: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram notification error: {e}")
        return False
