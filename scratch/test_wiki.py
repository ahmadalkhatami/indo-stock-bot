from data.data_loader import fetch_ticker_metadata
import pandas as pd

print("Testing LIVE Wikipedia Sector Fetch...")
df = fetch_ticker_metadata()

if df is not None and not df.empty:
    print(f"SUCCESS! Fetched {len(df)} tickers.")
    print("Sample Data:")
    print(df.head())
    
    unique_sectors = df['Sektor'].unique()
    print(f"\nDetected Sectors ({len(unique_sectors)}):")
    print(unique_sectors)
    
    if 'Lainnya' in unique_sectors and len(unique_sectors) == 1:
        print("\nWARNING: Still hitting fallback (only 'Lainnya' found).")
    else:
        print("\nREAL LIVE DATA DETECTED!")
else:
    print("FAILED: No data returned.")
