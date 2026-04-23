import pandas as pd
import os
import pytest
from datetime import datetime

def test_logbook_initialization(tmp_path):
    """
    Test the automatic creation and date conversion of the trading logbook.
    """
    logbook_path = tmp_path / "trading_logbook.csv"
    
    # Simulate initial creation
    if not os.path.exists(logbook_path):
        initial_data = pd.DataFrame({
            "Tanggal": [pd.Timestamp.now().strftime("%Y-%m-%d")],
            "Ticker": ["Test.JK"],
            "Aksi": ["BUY"],
            "Harga": [1000],
            "Lot": [1],
            "Total_Nilai": [100000],
            "Catatan": ["Test"]
        })
        initial_data.to_csv(logbook_path, index=False)
        
    assert os.path.exists(logbook_path)
    
    # Simulate the read process in app.py
    df_log = pd.read_csv(logbook_path)
    assert 'Tanggal' in df_log.columns
    
    # Test date conversion logic used in app.py
    df_log['Tanggal'] = pd.to_datetime(df_log['Tanggal'], errors='coerce').dt.date
    
    # Verify the type is correctly translated to datetime.date
    import datetime as dt
    assert isinstance(df_log['Tanggal'].iloc[0], dt.date)
