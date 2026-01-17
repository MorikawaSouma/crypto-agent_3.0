import sys
import os
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from crypto_trading_system.run_backtrader import run_backtrader_backtest
 
def main():
    run_backtrader_backtest()
 
if __name__ == "__main__":
    main()
