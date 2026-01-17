import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from .ml_strategy import LightGBMStrategy
from .config import Config

class RollingModelManager:
    """
    Manages rolling training of LightGBM models.
    """
    def __init__(self):
        self.strategy = LightGBMStrategy()
        self.last_train_date = None
        self.train_interval = Config.ROLLING_TRAIN_INTERVAL
        
    def check_and_retrain(self, current_date: datetime, universe_data: Dict[str, pd.DataFrame]):
        """
        Check if we need to retrain the model based on current_date.
        If yes, select data window [current_date - TRAIN_WINDOW, current_date] and train.
        """
        # If never trained, train immediately
        if self.last_train_date is None:
            self._train(current_date, universe_data)
            return

        # Check interval
        days_since = (current_date - self.last_train_date).days
        if days_since >= self.train_interval:
            self._train(current_date, universe_data)

    def _train(self, current_date: datetime, universe_data: Dict[str, pd.DataFrame]):
        print(f"[{current_date.date()}] Retraining Model (Rolling Window)...")
        
        # Define Training Window
        start_date = current_date - timedelta(days=Config.TRAIN_WINDOW)
        
        # Prepare Data Slice
        train_data_map = {}
        for sym, df in universe_data.items():
            # Slice [start_date, current_date)
            # We strictly exclude future data. The model should learn from past up to yesterday.
            mask = (df.index >= start_date) & (df.index < current_date)
            sliced_df = df[mask].copy()
            if not sliced_df.empty:
                train_data_map[sym] = sliced_df
        
        if not train_data_map:
            print("Warning: Not enough data to train.")
            return
            
        self.strategy.train(train_data_map)
        self.last_train_date = current_date
        
    def predict(self, current_data_map: Dict[str, pd.DataFrame], k: int = 3) -> List[str]:
        return self.strategy.predict_top_k(current_data_map, k)
