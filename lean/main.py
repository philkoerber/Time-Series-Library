"""
Simple GLD Trading Algorithm for QuantConnect LEAN
"""
from AlgorithmImports import *

class GLDTradingAlgorithm(QCAlgorithm):
    """Simple moving average crossover strategy for GLD"""
    
    def Initialize(self):
        """Initialize the algorithm"""
        self.SetStartDate(2016, 1, 4)  # Start from first trading day in data
        self.SetEndDate(2017, 1, 1)    # One year backtest
        self.SetCash(100000)  # Starting capital
        
        # Add GLD equity
        self.symbol = self.AddEquity("GLD", Resolution.Minute).Symbol
        
        # Track if we've placed the order
        self.order_placed = False
        
        self.Log(f"Algorithm initialized. Trading {self.symbol}")
    
    def OnData(self, data):
        """Called on each data update"""
        # Place buy order on first valid data point
        if not self.order_placed and self.symbol in data:
            bar = data[self.symbol]
            if bar is not None and bar.Price > 0:
                # Buy 100% of portfolio
                self.SetHoldings(self.symbol, 1.0)
                self.order_placed = True
                self.Log(f"BUY ORDER PLACED: {self.symbol} at ${bar.Price:.2f} - 100% allocation")
                self.Log(f"Portfolio value: ${self.Portfolio.TotalPortfolioValue:.2f}")

