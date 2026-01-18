import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple
from .factors import AlphaFactors
from .config import Config
import os

class LightGBMStrategy:
    def __init__(self):
        self.model = None
        self.features = []
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }

    def prepare_data(self, data_map: Dict[str, pd.DataFrame], is_training: bool = True) -> pd.DataFrame:
        """
        Convert dictionary of {symbol: OHLCV} to a single DataFrame with features and targets.
        """
        all_data = []
        
        for symbol, df in data_map.items():
            min_len = max(Config.LOOKBACK_WINDOW * 2, 120)
            if len(df) < min_len:
                continue
            
            # Calculate features
            feats = AlphaFactors.get_alpha158_lite(df)
            
            # Combine features with original data for alignment
            # We need 'close' to calculate target if training
            combined = feats.copy()
            combined['close'] = df['close'] # Ensure close is available for labeling
            
            if is_training:
                # Add Target (Next Day Return)
                combined = AlphaFactors.add_labels(combined, horizon=1)
            
            combined['symbol'] = symbol

            combined['date'] = df.index
            
            all_data.append(combined)
            
        if not all_data:
            return pd.DataFrame()
            
        full_df = pd.concat(all_data)
        
        # If training, drop rows with NaN target (last row usually)
        if is_training:
            full_df = full_df.dropna(subset=['target'])
            
        return full_df

    def train(self, data_map: Dict[str, pd.DataFrame]):
        """
        Train the LightGBM model.
        """
        # print("Preparing training data...")
        df = self.prepare_data(data_map, is_training=True)
        
        if df.empty:
            print("No training data available.")
            return
        
        self.features = [c for c in df.columns if c not in ['target', 'symbol', 'date', 'close']]
        if not self.features:
            print("No feature columns for LightGBM. Skipping training.")
            return
        X = df[self.features]
        y = df['target']
        dates = df['date'].unique()
        if len(dates) == 0:
            print("No dates available for LightGBM training.")
            return
        if len(dates) == 1:
            if len(df) < 2:
                print("Not enough samples for LightGBM training.")
                return
            split_date = df['date'].iloc[-1]
        else:
            split_date = dates[int(len(dates) * 0.8)]
        train_mask = df['date'] < split_date
        val_mask = df['date'] >= split_date
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        if X_train.empty or X_val.empty:
            print("Train/validation split produced empty dataset. Skipping LightGBM training.")
            return
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0) # Silence
            ]
        )

        
        print("Training complete.")
        
        # Feature Importance (Optional)
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        print("Top 5 Features:", importance.head(5).to_dict('records'))

    def predict_top_k(self, current_data_map: Dict[str, pd.DataFrame], k: int = 3) -> List[str]:
        """
        Predict scores for all symbols and return Top K.
        Uses the *last* available row of data for each symbol.
        """
        if not self.model:
            print("Model not trained!")
            return []
            
        scores = []
        
        for symbol, df in current_data_map.items():
            if df.empty:
                continue
                
            # Calculate features
            feats = AlphaFactors.get_alpha158_lite(df)
            
            # Take the last row (latest data)
            latest_feats = feats.iloc[[-1]] 
            
            # Ensure columns match training
            X_pred = latest_feats[self.features]
            
            pred_ret = self.model.predict(X_pred)[0]
            scores.append((symbol, pred_ret))
            
        # Sort by predicted return descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select Top K
        top_k_symbols = [s[0] for s in scores[:k]]
        
        # Debug output
        print(f"Top {k} Predictions: {scores[:k]}")
        
        return top_k_symbols
