import requests
import pandas as pd
from io import StringIO
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://id.wikipedia.org/wiki/LQ45"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers, verify=False, timeout=15)
tables = pd.read_html(StringIO(resp.text))

for i, t in enumerate(tables):
    print(f"\nTable {i} columns: {list(t.columns)}")
    if 'Kode' in t.columns:
        print("Found Ticker Table!")
        print(t.head(3))
