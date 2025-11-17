"""
Convert GLD.csv to LEAN data format
LEAN expects data in: Data/equity/[market]/[resolution]/[symbol]/
For minute data: Data/equity/usa/minute/gld/ as YYYYMMDD_trade.zip files
"""
import pandas as pd
import os
import zipfile
import io
from pathlib import Path

def convert_gld_to_lean():
    """Convert GLD.csv to LEAN minute bar format (zip files)"""
    
    # Paths
    source_file = Path(__file__).parent.parent / "dataset" / "trading" / "GLD.csv"
    # Use custom data directory instead of equity directory
    lean_data_dir = Path(__file__).parent / "Data" / "custom" / "gld"
    
    # Create directory
    lean_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove old CSV files if they exist
    for csv_file in lean_data_dir.glob("*.csv"):
        csv_file.unlink()
    
    # Read source data
    print(f"Reading {source_file}...")
    df = pd.read_csv(source_file)
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # LEAN expects: time,open,high,low,close,volume
    # Our data has: date,open,high,low,volume,close
    lean_df = pd.DataFrame({
        'time': df['date'],
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    })
    
    # Group by date for LEAN's file structure (one zip file per day)
    lean_df['date'] = lean_df['time'].dt.date
    
    # Save files by date as CSV files (simpler for custom data)
    csv_count = 0
    for date, group in lean_df.groupby('date'):
        date_str = date.strftime('%Y%m%d')
        csv_file = lean_data_dir / f"{date_str}.csv"
        
        # Prepare data for LEAN format: yyyyMMdd HHmmss,open,high,low,close,volume
        group_to_save = group[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
        group_to_save['time'] = group_to_save['time'].dt.strftime('%Y%m%d %H%M%S')
        
        # Save as CSV without header
        group_to_save.to_csv(csv_file, index=False, header=False, sep=',')
        
        csv_count += 1
    
    print(f"Converted {len(lean_df)} rows to {csv_count} CSV files in {lean_data_dir}")
    print(f"First file: {lean_data_dir / lean_df['date'].min().strftime('%Y%m%d')}.csv")
    print(f"Last file: {lean_data_dir / lean_df['date'].max().strftime('%Y%m%d')}.csv")

if __name__ == "__main__":
    convert_gld_to_lean()

