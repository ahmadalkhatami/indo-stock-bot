import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit

from features.feature_engineering import FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')


class StockPredictor:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        target: str = 'target_binary',
        confidence_threshold: float = 0.6,
    ):
        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
        )
        self.model_path = model_path
        self.target = target
        self.confidence_threshold = confidence_threshold
        self.features = FEATURE_COLS
        self.metrics: dict = {}
        self.roc_points = None
        self.feature_importances: pd.Series | None = None
        self.oof_predictions: pd.DataFrame | None = None

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d[self.features] = d[self.features].replace([np.inf, -np.inf], np.nan)
        return d.dropna(subset=self.features + [self.target])

    def train_and_evaluate(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df_train = self._clean(df).sort_values('Date').reset_index(drop=True)

        X = df_train[self.features]
        y = df_train[self.target]

        tscv = TimeSeriesSplit(n_splits=5)
        metrics = {'precision': [], 'recall': [], 'roc_auc': []}
        oof_rows = []
        last_roc = None

        print("Running Time Series Cross-Validation...")
        for fold, (tr, te) in enumerate(tscv.split(X), 1):
            self.model.fit(X.iloc[tr], y.iloc[tr])
            y_prob = self.model.predict_proba(X.iloc[te])[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            y_true = y.iloc[te]

            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['roc_auc'].append(roc_auc_score(y_true, y_prob))

            fold_df = df_train.iloc[te][['Date', 'Ticker']].copy()
            fold_df['probability'] = y_prob
            oof_rows.append(fold_df)

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            last_roc = (fpr, tpr)

            print(
                f"  Fold {fold}: AUC={metrics['roc_auc'][-1]:.4f} | "
                f"Prec={metrics['precision'][-1]:.4f} | "
                f"Rec={metrics['recall'][-1]:.4f}"
            )

        self.metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        self.roc_points = last_roc
        self.oof_predictions = pd.concat(oof_rows, ignore_index=True)

        print("\n--- Mean Validation Metrics ---")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"ROC-AUC:   {self.metrics['roc_auc']:.4f}")

        print("\nTraining final model on all available data...")
        self.model.fit(X, y)
        self.feature_importances = pd.Series(
            self.model.feature_importances_, index=self.features
        ).sort_values(ascending=False)

        joblib.dump(
            {
                'model': self.model,
                'features': self.features,
                'target': self.target,
                'threshold': self.confidence_threshold,
                'metrics': self.metrics,
                'feature_importances': self.feature_importances,
            },
            self.model_path,
        )
        print(f"Model saved to {self.model_path}")

    def load(self) -> None:
        bundle = joblib.load(self.model_path)
        self.model = bundle['model']
        self.features = bundle['features']
        self.target = bundle['target']
        self.confidence_threshold = bundle['threshold']
        self.metrics = bundle.get('metrics', {})
        self.feature_importances = bundle.get('feature_importances')

    def predict_today(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for the latest row per ticker and flag BUY if prob >= threshold."""
        latest = df.copy()
        latest[self.features] = latest[self.features].replace([np.inf, -np.inf], np.nan)
        latest = latest.dropna(subset=self.features)
        latest = latest.groupby('Ticker').tail(1).copy()

        cols = ['Date', 'Ticker', 'Close', 'probability', 'signal']
        if latest.empty:
            return pd.DataFrame(columns=cols)

        latest['probability'] = self.model.predict_proba(latest[self.features])[:, 1]
        latest['signal'] = np.where(
            latest['probability'] >= self.confidence_threshold, 'BUY', 'NONE'
        )
        return latest[cols].sort_values('probability', ascending=False).reset_index(drop=True)
