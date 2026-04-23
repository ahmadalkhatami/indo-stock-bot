import pandas as pd
import numpy as np
import pytest
from features.feature_engineering import add_features_and_labels, FEATURE_COLS

def test_add_features_and_labels_basic():
    # Create mock data
    dates = pd.date_range(start="2023-01-01", periods=60)
    data = []
    for ticker in ["ASII.JK", "BBCA.JK"]:
        for date in dates:
            data.append({
                "Date": date,
                "Ticker": ticker,
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
                "Volume": 1000,
                "USD_IDR": 15000.0,
                "SP500": 4000.0,
                "F_Net": 100
            })
    df = pd.DataFrame(data)
    
    # Run engineering
    df_out = add_features_and_labels(df)
    
    # Assertions
    assert not df_out.empty
    assert "return_1d" in df_out.columns
    assert "rsi_14" in df_out.columns
    assert "target_binary" in df_out.columns
    
    # Check that FEATURE_COLS exist
    for col in FEATURE_COLS:
        assert col in df_out.columns

def test_feature_cols_consistency():
    # Verify all expected columns are in FEATURE_COLS
    expected = [
        'return_1d', 'rsi_14', 'foreign_flow_ratio'
    ]
    for col in expected:
        assert col in FEATURE_COLS
