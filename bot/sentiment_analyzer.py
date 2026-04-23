import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from utils.logger import logger

class SentimentAnalyzer:
    """
    Enhanced Framework to analyze market sentiment from news headlines.
    Uses context-aware financial keywords for the Indonesian market.
    """
    def __init__(self):
        self.sources = {
            'cnbc': 'https://www.cnbcindonesia.com/market/indeks/5',
            'kontan': 'https://www.kontan.co.id/indeks'
        }
        # Institutional grade financial dictionary for Indonesian market
        self.positive_keywords = [
            'naik', 'untung', 'laba', 'positif', 'melesat', 'bullish', 'optimis', 
            'dividen', 'rekor', 'ekspansi', 'tumbuh', 'akuisisi', 'buy', 'beli', 
            'meningkat', 'melonjak', 'prospek', 'cerah', 'penguatan', 'terdongkrak',
            'overweight', 'rebound', 'melaju', 'berjaya', 'solid', 'kinerja apik'
        ]
        self.negative_keywords = [
            'turun', 'rugi', 'negatif', 'anjlok', 'bearish', 'pesimis', 'lemah', 
            'pangkas', 'phk', 'korupsi', 'merosot', 'sell', 'jual', 'tertekan', 
            'defisit', 'beban', 'melemah', 'inflasi', 'suspend', 'waspada', 
            'underweight', 'crash', 'ambles', 'gonjang-ganjing', 'pailit', 'krisis'
        ]

    def fetch_headlines(self, limit=25):
        headlines = []
        try:
            # Scrape CNBC Indonesia Market
            resp = requests.get(self.sources['cnbc'], timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.find_all('article')
                for art in articles[:limit]:
                    title = art.find('h2')
                    if title:
                        headlines.append(title.get_text().strip())
            
            # Scrape Kontan Index (Secondary Source)
            resp_k = requests.get(self.sources['kontan'], timeout=10)
            if resp_k.status_code == 200:
                soup_k = BeautifulSoup(resp_k.text, 'html.parser')
                titles_k = soup_k.find_all('div', class_='indeks-list')
                for t in titles_k[:limit]:
                    h2 = t.find('h2')
                    if h2:
                        headlines.append(h2.get_text().strip())
        except Exception as e:
            logger.error(f"Failed to fetch headlines: {e}")
        
        return list(set(headlines)) # Unique headlines

    def get_market_sentiment_score(self):
        """
        Returns a score between -1 and 1 representing overall market sentiment.
        """
        headlines = self.fetch_headlines()
        if not headlines:
            return 0.0
        
        score = 0
        hits = 0
        
        for h in headlines:
            h_lower = h.lower()
            h_score = 0
            for pos in self.positive_keywords:
                if pos in h_lower:
                    h_score += 1
            for neg in self.negative_keywords:
                if neg in h_lower:
                    h_score -= 1
            
            if h_score != 0:
                score += (1 if h_score > 0 else -1)
                hits += 1
        
        if hits == 0:
            return 0.0
            
        final_score = score / hits
        logger.info(f"Analyzed {len(headlines)} headlines ({hits} hits). Sentiment Score: {final_score:.2f}")
        return final_score
