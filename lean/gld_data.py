"""
Custom GLD Data Class for LEAN
Loads GLD data from local CSV files
Based on: https://www.quantconnect.com/docs/v2/lean-cli/datasets/custom-data
"""
from AlgorithmImports import *
from datetime import datetime
import os

class GLDData(PythonData):
    """Custom data class for GLD that reads from local CSV files"""
    
    def get_source(self, config, date, is_live):
        """Return the source file path for the given date"""
        # Format: Data/custom/gld/YYYYMMDD.csv
        # Use forward slashes only as per documentation
        date_str = date.strftime('%Y%m%d')
        # Globals.data_folder should point to the Data directory
        file_path = f"{Globals.data_folder}/custom/gld/{date_str}.csv"
        self.Log(f"Loading data from: {file_path}") if hasattr(self, 'Log') else None
        return SubscriptionDataSource(file_path, SubscriptionTransportMedium.LOCAL_FILE)
    
    def reader(self, config, line, date, is_live):
        """Parse a line of CSV data"""
        if not (line and line.strip()):
            return None
        
        try:
            # Format: yyyyMMdd HHmmss,open,high,low,close,volume
            parts = line.split(',')
            if len(parts) < 6:
                return None
            
            data = GLDData()
            data.Symbol = config.Symbol
            data.Time = datetime.strptime(parts[0], '%Y%m%d %H%M%S')
            data.Value = float(parts[4])  # Use close price as value
            data["open"] = float(parts[1])
            data["high"] = float(parts[2])
            data["low"] = float(parts[3])
            data["close"] = float(parts[4])
            data["volume"] = float(parts[5])
            
            return data
        except Exception as e:
            return None

