import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from .rolling_model import RollingModelManager
from .config import Config
from .feature_engineering import FeatureEngineer

class MLStrategy(bt.Strategy):
    params = (
        ('universe', []),
        ('rolling_manager', None),
        ('top_k', 3),
        ('rebalance_days', 1),
        ('full_universe_data', {}),
        ('use_agents', False),
        ('master_agent', None)
    )

    def __init__(self):
        self.inds = {}
        self.days_counter = 0
        self.last_rebalance = -1
        self.rolling_manager = self.params.rolling_manager
        self.master_agent = self.params.master_agent
        self.last_rebalance_date = None
        self.last_top_k = []
        self.last_weights = {}
        self.last_decisions = {}
        
    def next(self):
        self.days_counter += 1
        dt = self.datas[0].datetime.date(0)
        dt_ts = pd.Timestamp(dt)
        
        # Check Rebalance
        if self.days_counter % self.params.rebalance_days != 0:
            return
            
        print(f"\n--- {dt} Rebalancing ---")
        print(f"Portfolio Value: {self.broker.getvalue():.2f}")
        
        self.rolling_manager.check_and_retrain(dt_ts, self.params.full_universe_data)
        
        current_data_map = {}
        lookback_start = dt_ts - timedelta(days=Config.FEATURE_WINDOW + 10)
        
        for sym in self.params.universe:
            if sym in self.params.full_universe_data:
                full_df = self.params.full_universe_data[sym]
                # Slice [lookback, today]
                # Note: 'today' in backtrader usually means 'Close of today'.
                # So we can use today's close for prediction of tomorrow's return.
                mask = (full_df.index >= lookback_start) & (full_df.index <= dt_ts)
                sub_df = full_df[mask]
                if not sub_df.empty:
                    current_data_map[sym] = sub_df
                    
        top_k_symbols = self.rolling_manager.predict(current_data_map, k=self.params.top_k)
        print(f"ML Selected Top {self.params.top_k}: {top_k_symbols}")
        
        weights = {}
        decisions = {}
        if top_k_symbols:
            if self.params.use_agents and self.master_agent is not None:
                dalio_weights = {}
                for sym in top_k_symbols:
                    full_df = self.params.full_universe_data.get(sym)
                    if full_df is None or full_df.empty:
                        continue
                    mask = full_df.index <= dt_ts
                    df_agent = full_df[mask]
                    df_proc = FeatureEngineer.process(df_agent.copy())
                    if df_proc.empty:
                        continue
                    try:
                        decision = self.master_agent.analyze_symbol(sym, df_proc)
                        decisions[sym] = decision
                    except Exception as e:
                        print(f"Agent analysis failed for {sym}: {e}")
                        continue
                    dalio_res = {}
                    ad = decision.get('agent_details')
                    if isinstance(ad, dict):
                        dalio_res = ad.get('dalio', {})
                    w = dalio_res.get('weight_suggestion', 0)
                    try:
                        wv = max(0.0, float(w))
                    except Exception:
                        wv = 0.0
                    dalio_weights[sym] = wv
                total_w = sum(dalio_weights.values())
                if total_w > 0:
                    for sym, wv in dalio_weights.items():
                        weights[sym] = wv / total_w
                else:
                    eq = 1.0 / len(top_k_symbols)
                    for sym in top_k_symbols:
                        weights[sym] = eq
            else:
                eq = 1.0 / len(top_k_symbols)
                for sym in top_k_symbols:
                    weights[sym] = eq

        self.last_rebalance_date = dt_ts
        self.last_top_k = list(top_k_symbols) if top_k_symbols else []
        self.last_weights = dict(weights)
        self.last_decisions = decisions

        for i, d in enumerate(self.datas):
            name = d._name
            if name in weights:
                self.order_target_percent(d, target=weights[name] * 0.95)
            else:
                self.order_target_percent(d, target=0.0)

class BenchmarkStrategy(bt.Strategy):
    """
    Buy and Hold Benchmark (e.g. BTC)
    """
    def next(self):
        # Buy on first day
        if len(self) == 1:
            self.order_target_percent(self.datas[0], target=0.99)
