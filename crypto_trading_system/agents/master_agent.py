from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent
from .strategy_agents import WarrenBuffettAgent, GeorgeSorosAgent, RayDalioAgent, JimSimonsAgent, CryptoSentimentAgent
from ..data_loader import MarketDataManager
from ..feature_engineering import FeatureEngineer
from ..real_data_provider import RealDataProvider
from ..llm_client import DeepSeekClient
from ..config import Config
import numpy as np
import concurrent.futures
import pandas as pd
import json
import os
import sys

extra_lib = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".ext_libs")
if os.path.isdir(extra_lib) and extra_lib not in sys.path:
    sys.path.append(extra_lib)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

class TorchMLPWeightingModel:
    def __init__(self, input_dim: int, n_agents: int, lr: float):
        self.input_dim = input_dim
        self.n_agents = n_agents
        self.agent_order = ["buffett", "soros", "dalio", "simons", "sentiment"]
        hidden_dim = max(16, min(input_dim * 2, 128))
        self.temperature = float(np.sqrt(input_dim))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_agents)
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=10, verbose=False
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            scores = self.model(xt).squeeze(0) / self.temperature
            w = F.softmax(scores, dim=0)
            return w.detach().cpu().numpy()

    def save(self, path: str):
        if not TORCH_AVAILABLE:
            return
        state = {
            "model": self.model.state_dict()
        }
        torch.save(state, path)

    def load(self, path: str):
        if not TORCH_AVAILABLE:
            return
        if not os.path.isfile(path):
            return
        state = torch.load(path, map_location="cpu")
        model_state = state.get("model")
        if model_state is None:
            return
        self.model.load_state_dict(model_state)

    def update(self, x: np.ndarray, agent_details: Dict[str, Any], system_action: str, outcome: str):
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        scores = self.model(xt).squeeze(0) / self.temperature
        w = F.softmax(scores, dim=0)
        agree = []
        for i, name in enumerate(self.agent_order):
            a = agent_details.get(name, {})
            act = str(a.get("action", "HOLD")).upper()
            if system_action == "BUY" and act == "BUY":
                agree.append(i)
            if system_action == "SELL" and act == "SELL":
                agree.append(i)
        if len(agree) == 0:
            return
        sel = w[agree]
        if outcome == "profit":
            loss = -sel.mean()
        elif outcome == "loss":
            loss = sel.mean()
        else:
            return
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        try:
            self.scheduler.step(float(loss.detach().cpu().item()))
        except Exception:
            pass

class MasterAgent(BaseAgent):
    def __init__(self, llm_client: DeepSeekClient, mode: Optional[str] = None):
        super().__init__("MasterAgent", llm_client)
        self.buffett = WarrenBuffettAgent("WarrenBuffett", llm_client)
        self.soros = GeorgeSorosAgent("GeorgeSoros", llm_client)
        self.dalio = RayDalioAgent("RayDalio", llm_client)
        self.simons = JimSimonsAgent("JimSimons", llm_client)
        self.sentiment = CryptoSentimentAgent("SentimentAnalyzer", llm_client)
        
        self.agents = [self.buffett, self.soros, self.dalio, self.simons, self.sentiment]
        self.agents_map = {
            "buffett": self.buffett,
            "soros": self.soros,
            "dalio": self.dalio,
            "simons": self.simons,
            "sentiment": self.sentiment
        }
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.real_data_provider = RealDataProvider()
        if mode is None:
            self.mode = getattr(Config, "AGENT_MODE", "voting")
        else:
            self.mode = mode
        try:
            self.debate_rounds = int(getattr(Config, "DEBATE_ROUNDS", 3))
        except Exception:
            self.debate_rounds = 3
        self.weight_model = None
        self.weight_model_loaded = False

    def set_mode(self, mode: str):
        if mode in ["voting", "debate"]:
            self.mode = mode
            print(f"MasterAgent mode switched to: {mode}")
        else:
            print(f"Invalid mode: {mode}. Keeping {self.mode}")

    def set_debate_rounds(self, rounds: int):
        try:
            r = int(rounds)
            if r < 1:
                r = 1
            if r > 10:
                r = 10
            self.debate_rounds = r
            print(f"MasterAgent debate rounds set to: {self.debate_rounds}")
        except Exception:
            print("Invalid rounds value")

    def reset_all_memories(self):
        """Clear memories for all sub-agents"""
        print("Resetting all agent memories...")
        for agent in self.agents:
            agent.reset_memory()
        print("✓ All memories cleared.")

    def prepare_data(self, start_time: str, end_time: str):
        """Prepare real auxiliary data for the backtest period"""
        self.real_data_provider.fetch_all_data(start_time, end_time)

    def analyze_symbol(self, symbol: str, df_processed: Any) -> Dict[str, Any]:
        """
        Coordinate analysis for a single symbol
        """
        # Prepare context data
        latest = df_processed.iloc[-1]
        current_ts = latest.name
        recent_history = df_processed.iloc[-30:]
        
        macro_data = self.real_data_provider.get_macro_at(current_ts)
        sentiment_data_point = self.real_data_provider.get_sentiment_at(current_ts)
        onchain_data_point = self.real_data_provider.get_onchain_at(current_ts, symbol)
        defillama_data = self.real_data_provider.get_defillama_at(symbol, current_ts)
        news_items = self.real_data_provider.get_news_at(symbol, current_ts)
        
        # Data Construction
        buffett_data = {
            "fundamentals": {
                "onchain": onchain_data_point if onchain_data_point else "N/A",
                "defi_tvl": defillama_data.get('tvl', 'N/A')
            },
            "market_data_summary": {
                "current_price": latest['close'],
                "avg_price_30d": recent_history['close'].mean(),
                "volatility_30d": latest.get('volatility_30d', 0),
                "drawdown_30d": latest.get('drawdown_30d', 0)
            },
            "valuation": {
                "pe_ratio": "N/A (Crypto)",
                "mvrv": onchain_data_point.get('mvrv_ratio', 'N/A'),
                "risk_free_rate": macro_data.get('us_10y_yield', 'N/A'),
                "defi_tvl_trend": "Analyze TVL growth if available"
            }
        }
        
        soros_data = {
            "trend_data": {
                "price": latest['close'],
                "ma_50": latest.get('SMA_50', 0),
                "ma_200": latest.get('SMA_200', 0),
                "macd": latest.get('MACD', 0),
                "adx": latest.get('ADX', 0),
                "ichimoku_base": latest.get('Ichimoku_base_line', 0)
            },
            "macro_data": {
                "dxy": macro_data.get('dxy', 'N/A'),
                "us_10y_yield": macro_data.get('us_10y_yield', 'N/A'),
                "vix": macro_data.get('vix', 'N/A')
            },
            "sentiment_data": sentiment_data_point if sentiment_data_point else "N/A",
            "recent_news": news_items if news_items else "No specific news available",
            "liquidity_data": {
                "volume": latest['volume'],
                "obv": latest.get('OBV', 0),
                "vwap": latest.get('VWAP', 0)
            },
            "fear_greed": sentiment_data_point.get('fear_greed_index', 'N/A'),
            "news_sentiment": sentiment_data_point.get('news_sentiment_score', 'N/A'),
            "whale_activity": {
                "net_flow": onchain_data_point.get('net_flow', 'N/A'),
                "active_addresses": onchain_data_point.get('active_addresses', 'N/A')
            }
        }
        
        dalio_data = {
            "economic_cycle": {
                "dxy": macro_data.get('dxy', 'N/A'),
                "yields": macro_data.get('us_10y_yield', 'N/A'),
                "vix": macro_data.get('vix', 'N/A')
            },
            "correlation_matrix": f"Market Correlation: {macro_data.get('nasdaq_correlation', 'N/A')}",
            "risk_metrics": {
                "volatility": latest.get('volatility_30d', 0),
                "drawdown": latest.get('drawdown_30d', 0),
                "atr": latest.get('ATR', 0)
            },
            "portfolio_allocation": "Current holdings: 0%"
        }
        
        simons_data = {
            "statistical_features": {
                "skewness": latest.get('skew_30d', 0),
                "kurtosis": latest.get('kurt_30d', 0),
                "return_7d": latest.get('return_7d', 0),
                "return_30d": latest.get('return_30d', 0)
            },
            "technical_factors": {
                "rsi": latest.get('RSI', 50),
                "stoch_k": latest.get('Stoch_k', 50),
                "cmf": latest.get('CMF', 0),
                "vwap_deviation": (latest['close'] - latest.get('VWAP', latest['close'])) / latest['close']
            },
            "volatility_model": {
                "atr": latest.get('ATR', 0),
                "bb_width": latest.get('BB_width', 0),
                "volatility_30d": latest.get('volatility_30d', 0)
            }
        }
        
        sentiment_data = {
            "social_metrics": {
                "twitter": sentiment_data_point.get('twitter_sentiment', 'N/A'),
                "reddit": sentiment_data_point.get('reddit_sentiment', 'N/A'),
                "google_trends": sentiment_data_point.get('google_trends', 'N/A')
            },
            "fear_greed": sentiment_data_point.get('fear_greed_index', 'N/A'),
            "news_sentiment": sentiment_data_point.get('news_sentiment_score', 'N/A'),
            "whale_activity": {
                "net_flow": onchain_data_point.get('net_flow', 'N/A'),
                "active_addresses": onchain_data_point.get('active_addresses', 'N/A'),
                "tx_count": onchain_data_point.get('transaction_count', 'N/A')
            }
        }

        data_map = {
            "buffett": buffett_data,
            "soros": soros_data,
            "dalio": dalio_data,
            "simons": simons_data,
            "sentiment": sentiment_data
        }

        print(f"--- Asking Agents about {symbol} (Mode: {self.mode}) ---")

        if self.mode == "debate":
            agent_results = self._run_debate(data_map)
        else:
            agent_results = self._run_parallel(data_map)

        b_res = agent_results.get("buffett", {})
        s_res = agent_results.get("soros", {})
        d_res = agent_results.get("dalio", {})
        sim_res = agent_results.get("simons", {})
        sent_res = agent_results.get("sentiment", {})
        b_score = float(b_res.get('score', 50))
        s_score = float(s_res.get('score', 50))
        d_score = float(d_res.get('weight_suggestion', 50) if d_res.get('weight_suggestion') else 50)
        sim_score = float(sim_res.get('score', 50))
        sent_score = float(sent_res.get('score', 50))
        vix = float(macro_data.get('vix', 20)) if isinstance(macro_data.get('vix', 20), (int, float)) else 20.0
        vol30 = float(latest.get('volatility_30d', 0)) if isinstance(latest.get('volatility_30d', 0), (int, float)) else 0.0
        fg = float(sentiment_data_point.get('fear_greed_index', 50)) if isinstance(sentiment_data_point.get('fear_greed_index', 50), (int, float)) else 50.0
        wn = int(getattr(Config, "WEIGHT_WINDOW_N", 64))
        rh = df_processed.iloc[-wn:]
        c = rh['close']
        r = c.pct_change().dropna()
        v = rh['volume'] if 'volume' in rh.columns else pd.Series([0]*len(rh), index=rh.index)
        def stats(s):
            try:
                return [float(s.mean()), float(s.std(ddof=0)), float(s.skew()), float(s.kurt())]
            except Exception:
                return [0.0, 0.0, 0.0, 0.0]
        c_stats = stats(c)
        r_stats = stats(r)
        v_stats = stats(v)
        x = np.array([b_score, s_score, d_score, sim_score, sent_score, vix, vol30, fg] + c_stats + r_stats + v_stats, dtype=float)
        if self.weight_model is None:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for the dynamic weighting model but is not available")
            self.weight_model = TorchMLPWeightingModel(input_dim=len(x), n_agents=5, lr=float(getattr(Config, "TRANSFORMER_LR", 0.001)))
            path = getattr(Config, "WEIGHT_MODEL_PATH", None)
            if path and isinstance(path, str):
                try:
                    self.weight_model.load(path)
                    self.weight_model_loaded = True
                    print(f"Loaded weight model from {path}")
                except Exception as e:
                    print(f"Failed to load weight model from {path}: {e}")
        w = self.weight_model.predict(x)
        
        # Print Real-time Weights
        print(f"Dynamic Weights: B={w[0]:.3f}, S={w[1]:.3f}, D={w[2]:.3f}, Sim={w[3]:.3f}, Sent={w[4]:.3f}")
        
        final_score = (w[0]*b_score) + (w[1]*s_score) + (w[2]*d_score) + (w[3]*sim_score) + (w[4]*sent_score)
        
        action = "HOLD"
        if final_score > 60: 
            action = "BUY"
        elif final_score < 40:
            action = "SELL"
            
        # Synthesize Final Reasoning
        top_agent_idx = np.argmax(w)
        top_agent_name = ["WarrenBuffett", "GeorgeSoros", "RayDalio", "JimSimons", "Sentiment"][top_agent_idx]
        top_agent_key = ["buffett", "soros", "dalio", "simons", "sentiment"][top_agent_idx]
        top_reason = agent_results.get(top_agent_key, {}).get('reasoning', 'N/A')
        final_reasoning = f"Dominant View ({top_agent_name}, w={w[top_agent_idx]:.2f}): {top_reason[:200]}..."
            
        return {
            "symbol": symbol,
            "final_score": final_score,
            "action": action,
            "reasoning": final_reasoning,
            "agent_details": agent_results,
            "timestamp": str(latest.name),
            "model_input": x.tolist(),
            "dynamic_weights": w.tolist()
        }

    def _run_parallel(self, data_map):
        results = {}
        # Could use ThreadPoolExecutor here but for debug clarity keeping sequential or use existing
        # Using existing executor pattern:
        futures = {}
        for name, agent in self.agents_map.items():
            futures[self.executor.submit(self._run_agent, agent, data_map[name], name)] = name
            
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"Agent {name} crashed: {e}")
                results[name] = {"score": 50, "action": "HOLD", "reasoning": "Error"}
        return results

    def _run_debate(self, data_map):
        round_summaries = []
        combined_context = ""
        for r in range(1, int(self.debate_rounds) + 1):
            if r == 1:
                print(">>> Starting Debate Round 1 (Initial Thoughts)...")
            else:
                print(f">>> Starting Debate Round {r} (Rebuttal & Refinement)...")
                self._inject_context(data_map, combined_context)
            
            results = self._run_parallel(data_map)
   
            print(f"--- Round {r} Discussion ---")
            for name, res in results.items():
                reasoning = res.get('reasoning', 'No reasoning provided')
                action = res.get('action', 'HOLD')
                score = res.get('score', res.get('weight_suggestion', 'N/A'))
                print(f"[{name}] Action: {action} | Score: {score}")
                print(f"Reasoning: {reasoning}\n")

            summary = self._summarize_round(results, r)
            round_summaries.append(summary)
            combined_context = "\n\n".join(round_summaries)
        
        return results

    def _summarize_round(self, results, round_num):
        summary = f"Round {round_num} Opinions:\n"
        for name, res in results.items():
            score = res.get('score', res.get('weight_suggestion', 0))
            reasoning = str(res.get('reasoning', ''))[:150].replace('\n', ' ')
            action = res.get('action', 'N/A')
            summary += f"- {name}: {action} (Score: {score}). Reasoning: {reasoning}...\n"
        return summary

    def _inject_context(self, data_map, context_str):
        for name in data_map:
            data_map[name]['debate_context'] = context_str

    def _aggregate_scores(self, results):
        b_res = results.get("buffett", {})
        s_res = results.get("soros", {})
        d_res = results.get("dalio", {})
        sim_res = results.get("simons", {})
        sent_res = results.get("sentiment", {})
        
        b_score = float(b_res.get('score', 50))
        s_score = float(s_res.get('score', 50))
        d_score = float(d_res.get('weight_suggestion', 50) if d_res.get('weight_suggestion') else 50)
        sim_score = float(sim_res.get('score', 50))
        sent_score = float(sent_res.get('score', 50))
        
        print(f"Scores: B={b_score}, S={s_score}, D={d_score}, Sim={sim_score}, Sent={sent_score}")
        
        return (b_score * 0.2) + (s_score * 0.2) + (d_score * 0.1) + (sim_score * 0.3) + (sent_score * 0.2)

    def _run_agent(self, agent, data, name):
        """Helper to run agent with logging"""
        # print(f"-> {name} analyzing...")
        try:
            res = agent.analyze(data)
            # print(f"<- {name} finished.")
            return res
        except Exception as e:
            print(f"!! {name} failed: {e}")
            return {"action": "HOLD", "score": 50, "reasoning": "Error"}

    def reflect_on_trade(self, trade_details: Dict[str, Any]):
        """
        Called after a trade is closed (or at end of backtest) to provide feedback to agents.
        trade_details: {
            "symbol": str,
            "action": str,
            "outcome": "profit" | "loss",
            "pnl": float,
            "agent_details": dict (what each agent said at that time)
        }
        """
        outcome = trade_details.get("outcome", "unknown")
        pnl = trade_details.get("pnl", 0.0)
        
        agent_details = trade_details.get("agent_details", {})
        x = trade_details.get("model_input", None)
        system_action = trade_details.get("action", "HOLD")
        try:
            if isinstance(x, list):
                x_arr = np.array(x, dtype=float)
            else:
                x_arr = None
        except Exception:
            x_arr = None
        if x_arr is not None:
            self.weight_model.update(x_arr, agent_details, system_action, outcome)
        if outcome == "loss" or pnl < 0:
            print(f"--- Reflecting on Loss ({pnl:.2f}%) for {trade_details['symbol']} ---")
            for name, agent in self.agents_map.items():
                agent_res = agent_details.get(name)
                if not agent_res:
                    continue
                system_action = trade_details.get("action")
                agent_action = agent_res.get("action", "HOLD")
                was_wrong = False
                if system_action == "BUY" and agent_action == "BUY":
                    was_wrong = True
                elif system_action == "SELL" and agent_action == "SELL":
                    was_wrong = True
                if was_wrong:
                    context = f"Symbol: {trade_details['symbol']}, Action: {agent_action}, PnL: {pnl:.2f}%"
                    reasoning = f"I recommended {agent_action} because {agent_res.get('reasoning')}, but it resulted in a loss."
                    agent.reflect(context, agent_action, "loss", reasoning)

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {}
