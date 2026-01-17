import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading_system.config import Config
from crypto_trading_system.data_loader import MarketDataManager
from crypto_trading_system.mock_data import MockDataProvider
from crypto_trading_system.strategy_backtrader import MLStrategy
from crypto_trading_system.rolling_model import RollingModelManager
from crypto_trading_system.llm_client import DeepSeekClient
from crypto_trading_system.agents.master_agent import MasterAgent
from crypto_trading_system.feature_engineering import FeatureEngineer

def run_backtrader_backtest():
    print("=== Starting Backtrader ML Pipeline ===")
    
    # 1. Determine Data Range
    data_lookback_days = max(Config.FEATURE_WINDOW, Config.TRAIN_WINDOW + Config.LOOKBACK_WINDOW)
    total_days = data_lookback_days + Config.TEST_WINDOW
    if getattr(Config, "START_TIME", None) and getattr(Config, "END_TIME", None):
        try:
            test_start_dt = pd.to_datetime(Config.START_TIME)
        except Exception:
            test_start_dt = datetime.now() - timedelta(days=Config.TEST_WINDOW)
        try:
            end_dt = pd.to_datetime(Config.END_TIME)
        except Exception:
            end_dt = datetime.now()
        fetch_start_dt = test_start_dt - pd.Timedelta(days=data_lookback_days)
        start_date = fetch_start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")
    else:
        end_dt = datetime.now()
        fetch_start_dt = end_dt - timedelta(days=total_days)
        start_date = fetch_start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")
        test_start_dt = end_dt - timedelta(days=Config.TEST_WINDOW)
    
    print(f"Fetching Universe Data: {Config.UNIVERSE}")
    data_manager = MarketDataManager()
    universe_data = {}
    
    for symbol in Config.UNIVERSE:
        df = data_manager.get_historical_data(symbol, start_date, end_date, timeframe='1d')
        if df.empty:
            print(f"Using Mock Data for {symbol}")
            df = MockDataProvider.generate_price_history(symbol, days=total_days)
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        universe_data[symbol] = df
        print(f"Loaded {symbol}: {len(df)} rows")

    use_mock = getattr(Config, "LLM_MOCK", False)
    llm = DeepSeekClient(mock=use_mock)
    master_agent = MasterAgent(llm, mode=getattr(Config, "AGENT_MODE", "debate"))
    master_agent.set_debate_rounds(getattr(Config, "DEBATE_ROUNDS", 3))
    if getattr(Config, "RESET_MEMORY_ON_BACKTEST", False):
        master_agent.reset_all_memories()
    if getattr(Config, "USE_REAL_AUX_DATA", False):
        master_agent.prepare_data(start_date, end_date)

    cerebro = bt.Cerebro()
    test_start_date = test_start_dt
    print(f"Backtest Start Date: {test_start_date.date()}")
    
    data_added = 0
    for symbol, df in universe_data.items():
        # Slice for Backtrader Feed (Simulation Phase)
        mask = df.index >= test_start_date
        sim_df = df[mask]
        
        if sim_df.empty:
            print(f"Warning: No simulation data for {symbol}")
            continue
            
        data = bt.feeds.PandasData(dataname=sim_df, name=symbol)
        cerebro.adddata(data)
        data_added += 1

    if data_added == 0:
        print("Error: No data added to Cerebro. Exiting.")
        return

    rolling_manager = RollingModelManager()
    
    cerebro.addstrategy(
        MLStrategy, 
        universe=Config.UNIVERSE,
        rolling_manager=rolling_manager,
        top_k=Config.TOP_K,
        full_universe_data=universe_data,
        use_agents=True,
        master_agent=master_agent
    )
    
    # 4. Settings
    cerebro.broker.setcash(getattr(Config, "INITIAL_CAPITAL", 100000.0))
    cerebro.broker.setcommission(commission=0.001) # 0.1% comm
    
    print(f"\nStarting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    
    # 5. Add Analyzers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, timeframe=bt.TimeFrame.Days)

    # 6. Run
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    print(f"Final Portfolio Value: {final_value:.2f}")
    
    # Extract Metrics
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    dd_analysis = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd_analysis.get('max', {}).get('drawdown', 0.0)
    
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

    # 7. Plot
    try:
        # Extract TimeReturn Series
        tret = strat.analyzers.timereturn.get_analysis()
        
        if tret:
            # Convert to Series
            s_ret = pd.Series(tret).sort_index()
            
            # Calculate Cumulative Return
            cum_ret = (1 + s_ret).cumprod()
            
            # Initial Value (from Config.INITIAL_CAPITAL)
            start_value = getattr(Config, "INITIAL_CAPITAL", 100000.0)
            equity_curve = cum_ret * start_value
            
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve.index, equity_curve.values, label='Backtrader Strategy')
            plt.title(f'Backtrader ML Strategy Result')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.grid(True)
            
            output_file = 'backtrader_results.png'
            plt.savefig(output_file)
            print(f"Chart saved to {output_file}")
        else:
            print("No return data to plot.")
            
    except Exception as e:
        print(f"Error plotting results: {e}")

    try:
        last_date = getattr(strat, "last_rebalance_date", None)
        last_top_k = getattr(strat, "last_top_k", [])
        last_weights = getattr(strat, "last_weights", {})
        last_decisions = getattr(strat, "last_decisions", {})
        if not last_top_k:
            print("No Top-K selection recorded during backtest; skipping live signal generation.")
        else:
            signal_date = end_dt + timedelta(days=1)
            print(f"\nLIVE SIGNAL BASED ON LAST BACKTEST REBALANCE ({last_date.date() if last_date is not None else 'N/A'})")
            print(f"Signal applies to next day: {signal_date.date()}")
            rows = []
            for sym in last_top_k:
                dec = last_decisions.get(sym, {})
                action = dec.get("action", "HOLD")
                score = dec.get("final_score", dec.get("score", 0.0))
                alloc = last_weights.get(sym, 0.0)
                rows.append({
                    "symbol": sym,
                    "action": action,
                    "final_score": score,
                    "allocation_pct": alloc,
                    "rebalance_date": last_date,
                    "signal_date": signal_date
                })
            if rows:
                df_live = pd.DataFrame(rows)
                print("\nLIVE SIGNAL REPORT (From Backtest Last Top-K):")
                print("-" * 70)
                print(f"{'Symbol':<10} | {'Score':<6} | {'Action':<10} | {'Allocation':<10}")
                print("-" * 70)
                for _, row in df_live.iterrows():
                    print(f"{row['symbol']:<10} | {row['final_score']:<6.1f} | {row['action']:<10} | {row['allocation_pct']:.2%}")
                print("-" * 70)
                df_live.to_csv("live_trading_signal.csv", index=False)
                print("Live signal report saved to live_trading_signal.csv")
            else:
                print("No rows constructed for live signal.")
    except Exception as e:
        print(f"Error during live signal generation: {e}")

if __name__ == "__main__":
    run_backtrader_backtest()
