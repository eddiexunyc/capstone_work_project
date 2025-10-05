# This python file will be extracting daily historical stock prices of 10 years for top 25 companies in the S&P500

# load libraries
import numpy as np
import pandas as pd
import yfinance as yf
import datetime

def pull_data(start_date, end_date):
    
    # define the top 25 companies in the S&P 500 by weight based on the link below: https://www.slickcharts.com/sp500
    ticker_list = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META',
                   'AVGO', 'GOOGL', 'TSLA', 'JPM', 'ORCL', 
                   'WMT', 'LLY', 'V', 'MA', 'NFLX', 
                   'XOM', 'JNJ', 'ABBV', 'PLTR', 'COST', 
                   'HD', 'BAC', 'PG', 'UNH', 'GE']
    
    data_field='Adj Close'

    # check if the ticker exists
    if not ticker_list:
        print("No ticker found.")
        return None

    print(f"\nDownloading {data_field} data for the top {len(ticker_list)} companies...")

    try:
        data = yf.download(ticker_list, start=start_date, end=end_date, progress=True, group_by='ticker')
        
        # Filter by specific data field (e.g., 'Adj Close')
        if data_field:
            if data_field in data.columns.levels[0]:
                data = data[data_field]
            else:
                print(f"Warning: '{data_field}' not found in downloaded data. Returning full dataset.")

    

        # Save to CSV
        data.to_csv('Resources/historical_stock_data.csv', index = False)

        return data

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
