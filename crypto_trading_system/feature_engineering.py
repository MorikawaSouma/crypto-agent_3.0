import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice

class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame
        """
        df = df.copy()
        
        # Ensure no missing values before calculation (fill or drop)
        # df.dropna(inplace=True) # Maybe handled later
        
        # --- Trend ---
        # MACD
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # SMA / EMA
        df['SMA_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['SMA_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        
        df['EMA_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # ADX
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['ADX'] = adx.adx()
        df['ADX_pos'] = adx.adx_pos()
        df['ADX_neg'] = adx.adx_neg()
        
        # Ichimoku
        ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()
        df['Ichimoku_base_line'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
        
        # --- Momentum ---
        # RSI
        df['RSI'] = RSIIndicator(close=df['close']).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()
        
        # ROC
        df['ROC'] = ROCIndicator(close=df['close']).roc()
        
        # Williams %R
        df['WilliamsR'] = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
        
        # --- Volatility ---
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
        df['BB_width'] = bb.bollinger_wband()
        
        # ATR
        df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        # Keltner Channel
        kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
        df['KC_high'] = kc.keltner_channel_hband()
        df['KC_low'] = kc.keltner_channel_lband()
        
        # --- Volume ---
        # OBV
        df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # CMF
        df['CMF'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
        df['VWAP'] = vwap.volume_weighted_average_price()
        
        return df

    @staticmethod
    def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features
        """
        df = df.copy()
        
        # Returns
        df['return_1d'] = df['close'].pct_change(1)
        df['return_3d'] = df['close'].pct_change(3)
        df['return_7d'] = df['close'].pct_change(7)
        df['return_30d'] = df['close'].pct_change(30)
        
        # Volatility (Rolling std dev of returns)
        df['volatility_7d'] = df['return_1d'].rolling(window=7).std()
        df['volatility_30d'] = df['return_1d'].rolling(window=30).std()
        
        # Skewness & Kurtosis
        df['skew_30d'] = df['return_1d'].rolling(window=30).skew()
        df['kurt_30d'] = df['return_1d'].rolling(window=30).kurt()
        
        # Rolling Max Drawdown (simplified)
        rolling_max = df['close'].rolling(window=30, min_periods=1).max()
        df['drawdown_30d'] = (df['close'] - rolling_max) / rolling_max
        
        return df

    @staticmethod
    def process(df: pd.DataFrame) -> pd.DataFrame:
        df = FeatureEngineer.add_technical_indicators(df)
        df = FeatureEngineer.add_statistical_features(df)
        # Drop NaN created by indicators
        df.dropna(inplace=True)
        return df
