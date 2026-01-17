import pandas as pd
import numpy as np

class AlphaFactors:
    """
    Implements a subset of Alpha158 factors and other common technical indicators.
    Designed for LightGBM input.
    """
    
    @staticmethod
    def get_alpha158_lite(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a robust set of alpha factors (Extended to mimic Alpha158).
        Input df must have columns: ['open', 'high', 'low', 'close', 'volume']
        Returns a DataFrame with only the feature columns.
        """
        # Ensure data is sorted
        df = df.sort_index()
        
        # Avoid modifying original
        data = df.copy()
        features = pd.DataFrame(index=data.index)
        
        # --- 1. Basic Price/Volume Ratios (K-Line) ---
        features['open_close_ratio'] = data['close'] / data['open']
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_high_ratio'] = data['close'] / data['high']
        features['close_low_ratio'] = data['close'] / data['low']
        features['close_prev_close'] = data['close'] / data['close'].shift(1)
        
        # Log Returns
        features['log_ret'] = np.log(data['close'] / data['close'].shift(1))
        
        # --- 2. Rolling Window Features (5, 10, 20, 30, 60) ---
        windows = [5, 10, 20, 30, 60]
        
        for w in windows:
            # ROC (Rate of Change)
            features[f'roc_{w}'] = data['close'].pct_change(w)
            
            # MA (Moving Average)
            ma = data['close'].rolling(w).mean()
            features[f'ma_{w}'] = ma / data['close'] # Normalized MA
            
            # Std (Standard Deviation)
            std = data['close'].rolling(w).std()
            features[f'std_{w}'] = std / data['close']
            
            # Beta (Slope of Close vs Index) - Approximated by simply price slope here
            # Ideally needs a market index. We use simple linear regression slope of price.
            # features[f'beta_{w}'] = ... (Skipped for performance in pure pandas)
            
            # RSI-like: Up/Down volatility
            # KSFT (Karl-Sigma-Feature-Thing? No, just generic stats)
            
            # Volume MA
            vol_ma = data['volume'].rolling(w).mean()
            features[f'vol_ma_{w}_ratio'] = data['volume'] / vol_ma
            
            # Volatility of High-Low Range
            features[f'range_std_{w}'] = (data['high'] - data['low']).rolling(w).std() / data['close']
            
            # VWAP (Volume Weighted Average Price) for window
            vwap = (data['close'] * data['volume']).rolling(w).sum() / data['volume'].rolling(w).sum()
            features[f'vwap_{w}_dist'] = data['close'] / vwap
            
            # Max/Min in window
            features[f'max_{w}'] = data['high'].rolling(w).max() / data['close']
            features[f'min_{w}'] = data['low'].rolling(w).min() / data['close']
            
            # Qlib Alpha158 specific style features (simplified):
            # CORR(close, volume, d)
            # features[f'corr_price_vol_{w}'] = data['close'].rolling(w).corr(data['volume'])
            
            # RANK(close) in window (Percentile)
            # features[f'rank_{w}'] = data['close'].rolling(w).apply(lambda x: pd.Series(x).rank().iloc[-1]) # Slow!

        # --- 3. Technical Indicators ---
        
        # RSI (Relative Strength Index) - 14 period
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands Width (20)
        sma20 = data['close'].rolling(20).mean()
        std20 = data['close'].rolling(20).std()
        features['bb_width'] = (4 * std20) / sma20
        features['bb_pos'] = (data['close'] - sma20 + 2*std20) / (4*std20) # Position within bands
        
        # KDJ (Stochastic Oscillator)
        low_min = data['low'].rolling(9).min()
        high_max = data['high'].rolling(9).max()
        rsv = (data['close'] - low_min) / (high_max - low_min) * 100
        features['kdj_k'] = rsv.ewm(com=2).mean()
        features['kdj_d'] = features['kdj_k'].ewm(com=2).mean()
        features['kdj_j'] = 3 * features['kdj_k'] - 2 * features['kdj_d']
        
        # ATR (Average True Range)
        tr1 = data['high'] - data['low']
        tr2 = (data['high'] - data['close'].shift(1)).abs()
        tr3 = (data['low'] - data['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean() / data['close']

        # CCI (Commodity Channel Index)
        tp = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = tp.rolling(14).mean()
        mad_tp = tp.rolling(14).apply(lambda x: np.abs(x - x.mean()).mean()) # Mean Absolute Deviation
        features['cci_14'] = (tp - sma_tp) / (0.015 * mad_tp)
        
        # --- 4. Lagged Features ---
        features['ret_lag1'] = features['log_ret'].shift(1)
        features['ret_lag2'] = features['log_ret'].shift(2)
        features['ret_lag3'] = features['log_ret'].shift(3)
        features['vol_lag1'] = features['vol_ma_5_ratio'].shift(1)
        
        # Fill NaN (caused by rolling windows) with 0 or forward fill
        features = features.ffill().fillna(0)
        
        # Replace infs
        features = features.replace([np.inf, -np.inf], 0)
        
        return features

    @staticmethod
    def add_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Add target labels for training.
        Target: Future Return (N days ahead).
        """
        df['target'] = df['close'].shift(-horizon) / df['close'] - 1
        return df
