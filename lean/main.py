"""
Simple GLD Trading Algorithm for QuantConnect LEAN
Uses custom data class to load GLD from local files
"""
from AlgorithmImports import *
from gld_data import GLDData

class GLDTradingAlgorithm(QCAlgorithm):
    """Simple moving average crossover strategy for GLD"""
    
    def Initialize(self):
        """Initialize the algorithm"""
        self.SetStartDate(2016, 1, 4)  # Start from first trading day in data
        self.SetEndDate(2017, 1, 1)    # One year backtest
        self.SetCash(100000)  # Starting capital
        
        # Add GLD as custom data (minute resolution)
        self.symbol = self.AddData(GLDData, "GLD", Resolution.Minute).Symbol
        
        # Track if we've placed the order
        self.order_placed = False
        
        self.Log(f"Algorithm initialized. Trading {self.symbol} as custom data")
    
    def OnData(self, data):
        """Called on each data update"""
        # Debug: Log first few data points
        if not hasattr(self, 'data_count'):
            self.data_count = 0
        
        self.data_count += 1
        if self.data_count <= 3:
            self.Log(f"OnData called #{self.data_count}, data type: {type(data)}")
            if hasattr(data, 'Keys'):
                self.Log(f"  Data keys: {list(data.Keys)}")
            if self.symbol in data:
                self.Log(f"  Found {self.symbol} in data!")
        
        # Place buy order on first valid data point
        if not self.order_placed:
            if self.symbol in data:
                gld_data = data[self.symbol]
                self.Log(f"GLD data received: {type(gld_data)}, Value: {getattr(gld_data, 'Value', 'N/A')}")
                
                if gld_data is not None:
                    # Get price from Value or close field
                    price = getattr(gld_data, 'Value', 0)
                    if hasattr(gld_data, '__getitem__') and 'close' in gld_data:
                        price = gld_data['close']
                    
                    self.Log(f"Extracted price: {price}")
                    
                    if price > 0:
                        # Buy 100% of portfolio using MarketOrder
                        quantity = int(self.Portfolio.Cash / price)
                        if quantity > 0:
                            order = self.MarketOrder(self.symbol, quantity)
                            self.order_placed = True
                            self.Log(f"✅ BUY ORDER PLACED: {self.symbol} - {quantity} shares at ${price:.2f}")
                            self.Log(f"   Order ID: {order}")
                            self.Log(f"   Portfolio cash: ${self.Portfolio.Cash:.2f}")
                            self.Log(f"   Portfolio value: ${self.Portfolio.TotalPortfolioValue:.2f}")
                    else:
                        self.Log(f"⚠️ Price is 0, cannot place order")
            else:
                if self.data_count <= 5:
                    self.Log(f"❌ {self.symbol} not in data. Available: {list(data.Keys) if hasattr(data, 'Keys') else 'N/A'}")

