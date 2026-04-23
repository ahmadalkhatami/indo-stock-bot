import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')

class StockPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        self.model_path = model_path
        self.features = [
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff', 
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'BB_High', 'BB_Low', 'BB_Width', 
            'Return_1d', 'Return_3d', 'Return_7d',
            'Vol_7d', 'Vol_14d', 'Volume_Change_1d', 'Momentum_20',
            'USD_IDR_Return', 'SP500_Return', 'Close_ZScore_20'
        ]

    def prepare_data(self, df: pd.DataFrame, training: bool = True):
        # Drop rows with NaN in features
        df_clean = df.dropna(subset=self.features).copy()
        
        if training:
            # For training, drop rows where Future_Return_3d is NaN (the last 3 days)
            df_clean = df_clean.dropna(subset=['Future_Return_3d']).copy()
            # Sort by date for time series
            df_clean = df_clean.sort_values(by='Date').reset_index(drop=True)
            
        X = df_clean[self.features]
        y = df_clean['Target'] if 'Target' in df_clean.columns else None
        return X, y, df_clean

    def train_and_evaluate(self, df: pd.DataFrame):
        X, y, train_df = self.prepare_data(df, training=True)
        
        # Time Series Cross Validation
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'roc_auc': []}
        
        print("Running Time Series Cross-Validation...")
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
            
        print("--- Validation Metrics (Mean) ---")
        print(f"Accuracy:  {np.mean(metrics['accuracy']):.4f}")
        print(f"Precision: {np.mean(metrics['precision']):.4f}")
        print(f"Recall:    {np.mean(metrics['recall']):.4f}")
        print(f"ROC-AUC:   {np.mean(metrics['roc_auc']):.4f}")
        
        # Train final model on all available training data
        print("\nTraining final model on all data...")
        self.model.fit(X, y)
        
        # Save model
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def predict_today(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probability for the most recent day for each ticker.
        """
        try:
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}. Please train first.")
            return pd.DataFrame()
            
        # Get the latest row for each ticker
        latest_data = df.dropna(subset=self.features).groupby('Ticker').tail(1).copy()
        
        if latest_data.empty:
            return pd.DataFrame()
            
        X_latest = latest_data[self.features]
        latest_data['Probability'] = self.model.predict_proba(X_latest)[:, 1]
        
        # Return sorted by highest probability
        return latest_data[['Date', 'Ticker', 'Close', 'Probability']].sort_values(by='Probability', ascending=False)
