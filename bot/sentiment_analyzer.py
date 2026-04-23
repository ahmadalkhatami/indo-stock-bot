import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from utils.logger import logger

class SentimentAnalyzer:
    """
    Framework to analyze market sentiment from news headlines.
    """
    def __init__(self):
        self.sources = {
            'cnbc': 'https://www.cnbcindonesia.com/market/indeks/5',
            'kontan': 'https://www.kontan.co.id/indeks'
        }
        self.positive_keywords = ['naik', 'menguat', 'rebound', 'bullish', 'investasi', 'laba', 'ekspansi', 'dividen']
        self.negative_keywords = ['turun', 'melemah', 'anjlok', 'bearish', 'rugi', 'inflasi', 'tekanan', 'krisis']

    def fetch_headlines(self, limit=20):
        headlines = []
        try:
            # Prototype scraping for CNBC Indonesia Market
            resp = requests.get(self.sources['cnbc'], timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                articles = soup.find_all('article')
                for art in articles[:limit]:
                    title = art.find('h2')
                    if title:
                        headlines.append(title.get_text().strip())
        except Exception as e:
            logger.error(f"Failed to fetch headlines: {e}")
        
        return headlines

    def get_market_sentiment_score(self):
        """
        Returns a score between -1 and 1 representing overall market sentiment.
        """
        headlines = self.fetch_headlines()
        if not headlines:
            return 0.0
        
        score = 0
        total_keywords = 0
        
        for h in headlines:
            h_lower = h.lower()
            for pos in self.positive_keywords:
                if pos in h_lower:
                    score += 1
                    total_keywords += 1
            for neg in self.negative_keywords:
                if neg in h_lower:
                    score -= 1
                    total_keywords += 1
        
        if total_keywords == 0:
            return 0.0
            
        final_score = score / total_keywords
        logger.info(f"Analyzed {len(headlines)} headlines. Sentiment Score: {final_score:.2f}")
        return final_score

# Usage example:
# analyzer = SentimentAnalyzer()
# print(analyzer.get_market_sentiment_score())
