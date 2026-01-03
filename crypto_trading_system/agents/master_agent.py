from typing import Dict, List, Any
from .base_agent import BaseAgent
from .strategy_agents import WarrenBuffettAgent, GeorgeSorosAgent, RayDalioAgent, JimSimonsAgent, CryptoSentimentAgent
from ..data_loader import MarketDataManager
from ..feature_engineering import FeatureEngineer
from ..real_data_provider import RealDataProvider
from ..llm_client import DeepSeekClient
import numpy as np
import concurrent.futures
import pandas as pd

class MasterAgent(BaseAgent):
    def __init__(self, llm_client: DeepSeekClient):
        super().__init__("MasterAgent", llm_client)
        self.buffett = WarrenBuffettAgent("WarrenBuffett", llm_client)
        self.soros = GeorgeSorosAgent("GeorgeSoros", llm_client)
        self.dalio = RayDalioAgent("RayDalio", llm_client)
        self.simons = JimSimonsAgent("JimSimons", llm_client)
        self.sentiment = CryptoSentimentAgent("SentimentAnalyzer", llm_client)
        
        self.agents = [self.buffett, self.soros, self.dalio, self.simons, self.sentiment]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
        self.real_data_provider = RealDataProvider()
        
    def prepare_data(self, start_time: str, end_time: str):
        """Prepare real auxiliary data for the backtest period"""
        self.real_data_provider.fetch_all_data(start_time, end_time)

    def analyze_symbol(self, symbol: str, df_processed: Any) -> Dict[str, Any]:
        """
        Coordinate analysis for a single symbol
        """
        # Prepare context data
        # Take the latest data point
        latest = df_processed.iloc[-1]
        current_ts = latest.name # Datetime index
        recent_history = df_processed.iloc[-30:] # Last 30 periods
        
        # Get REAL auxiliary data for this timestamp
        macro_data = self.real_data_provider.get_macro_at(current_ts)
        sentiment_data_point = self.real_data_provider.get_sentiment_at(current_ts)
        onchain_data_point = self.real_data_provider.get_onchain_at(current_ts)
        
        # Construct specific data views for agents
        
        # 1. Buffett Data
        buffett_data = {
            "fundamentals": onchain_data_point, # Using onchain as proxy for fundamentals
            "market_data_summary": {
                "current_price": latest['close'],
                "avg_price_30d": recent_history['close'].mean(),
                "volatility": latest.get('volatility_30d', 0)
            },
            "valuation": {
                "pe_ratio": "N/A (Crypto)",
                "mvrv": onchain_data_point.get('mvrv_ratio', 1.5)
            }
        }
        
        # 2. Soros Data
        soros_data = {
            "trend_data": {
                "price": latest['close'],
                "ma_50": latest.get('SMA_50', 0),
                "ma_200": latest.get('SMA_200', 0),
                "macd": latest.get('MACD', 0),
                "rsi": latest.get('RSI', 50)
            },
            "macro_data": macro_data,
            "sentiment_data": sentiment_data_point,
            "liquidity_data": {
                "volume": latest['volume'],
                "obv": latest.get('OBV', 0)
            }
        }
        
        # 3. Dalio Data
        dalio_data = {
            "economic_cycle": "Data Driven (Real)",
            "correlation_matrix": f"Market Correlation: {macro_data.get('nasdaq_correlation', 0.5)}",
            "risk_metrics": {
                "volatility": latest.get('volatility_30d', 0),
                "drawdown": latest.get('drawdown_30d', 0),
                "vix": macro_data.get('vix', 20)
            },
            "portfolio_allocation": "Current holdings: 0%"
        }
        
        # 4. Simons Data
        simons_data = {
            "statistical_features": {
                "skewness": latest.get('skew_30d', 0),
                "kurtosis": latest.get('kurt_30d', 0),
                "sharpe_ratio": "Calc in backtest (>1.5 target)"
            },
            "technical_factors": {
                "rsi": latest.get('RSI', 50),
                "stoch_k": latest.get('Stoch_k', 50),
                "roc": latest.get('ROC', 0),
                "williams_r": latest.get('WilliamsR', -50)
            },
            "volatility_model": {
                "atr": latest.get('ATR', 0),
                "bb_width": latest.get('BB_width', 0),
                "volatility_30d": latest.get('volatility_30d', 0)
            }
        }
        
        # 5. Sentiment Data
        sentiment_data = {
            "social_metrics": {
                "twitter": sentiment_data_point.get('twitter_sentiment'),
                "reddit": sentiment_data_point.get('reddit_sentiment'),
                "google_trends": sentiment_data_point.get('google_trends')
            },
            "fear_greed": sentiment_data_point.get('fear_greed_index'),
            "news_sentiment": sentiment_data_point.get('news_sentiment_score'),
            "whale_activity": {
                "net_flow": onchain_data_point.get('net_flow'),
                "active_addresses": onchain_data_point.get('active_addresses')
            }
        }
        
        # Get opinions (Sequential Direct)
        print(f"--- Asking Agents about {symbol} (Sequential Direct) ---")
        
        buffett_res = self._run_agent(self.buffett, buffett_data, "Buffett")
        soros_res = self._run_agent(self.soros, soros_data, "Soros")
        dalio_res = self._run_agent(self.dalio, dalio_data, "Dalio")
        simons_res = self._run_agent(self.simons, simons_data, "Simons")
        sentiment_res = self._run_agent(self.sentiment, sentiment_data, "Sentiment")
        
        print(f"Buffett: {buffett_res.get('action')} (Score: {buffett_res.get('score')})")
        print(f"Soros: {soros_res.get('action')} (Score: {soros_res.get('score')})")
        print(f"Dalio: {dalio_res.get('action')} (Weight: {dalio_res.get('weight_suggestion')})")
        print(f"Simons: {simons_res.get('action')} (Score: {simons_res.get('score')})")
        print(f"Sentiment: {sentiment_res.get('action')} (Score: {sentiment_res.get('score')})")
        
        # Aggregation (Weighted Average)
        # Weights: Buffett (20%), Soros (20%), Dalio (10%), Simons (30%), Sentiment (20%)
        # Simons gets higher weight for pure quant approach
        
        b_score = float(buffett_res.get('score', 50))
        s_score = float(soros_res.get('score', 50))
        d_score = float(dalio_res.get('weight_suggestion', 50) if dalio_res.get('weight_suggestion') else 50)
        sim_score = float(simons_res.get('score', 50))
        sent_score = float(sentiment_res.get('score', 50))
        
        final_score = (b_score * 0.2) + (s_score * 0.2) + (d_score * 0.1) + (sim_score * 0.3) + (sent_score * 0.2)
        
        # Decision Logic
        action = "HOLD"
        # More aggressive thresholds as requested
        if final_score > 60: 
            action = "BUY"
        elif final_score < 40:
            action = "SELL"
            
        return {
            "symbol": symbol,
            "final_score": final_score,
            "action": action,
            "agent_details": {
                "buffett": buffett_res,
                "soros": soros_res,
                "dalio": dalio_res,
                "simons": simons_res,
                "sentiment": sentiment_res
            },
            "timestamp": str(latest.name)
        }

    def _run_agent(self, agent, data, name):
        """Helper to run agent with logging"""
        print(f"-> {name} analyzing...")
        try:
            res = agent.analyze(data)
            print(f"<- {name} finished.")
            return res
        except Exception as e:
            print(f"!! {name} failed: {e}")
            return {"action": "HOLD", "score": 50, "reasoning": "Error"}

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Master Agent main entry point. 
        Expects 'market_data' in input.
        """
        # This is just a placeholder to satisfy abstract method
        return {}
