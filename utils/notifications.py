import requests
import urllib.parse
from utils.logger import logger

def send_whatsapp_signal(phone, apikey, message):
    """
    Sends a WhatsApp message via CallMeBot API.
    """
    if not apikey or apikey == "YOUR_API_KEY_HERE":
        logger.warning("WhatsApp API Key not set. Skipping notification.")
        return False
        
    try:
        # CallMeBot expects international format without '+'
        clean_phone = phone.replace("+", "").replace(" ", "").replace("-", "")
        # If starts with 0, replace with 62
        if clean_phone.startswith("0"):
            clean_phone = "62" + clean_phone[1:]
            
        encoded_msg = urllib.parse.quote(message)
        url = f"https://api.callmebot.com/whatsapp.php?phone={clean_phone}&text={encoded_msg}&apikey={apikey}"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info(f"WhatsApp signal sent to {clean_phone} successfully.")
            return True
        else:
            logger.error(f"Failed to send WhatsApp. Status: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"WhatsApp notification error: {e}")
        return False
