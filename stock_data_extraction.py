# This python file will be extracting daily historical stock prices of 10 years for top 25 companies in the S&P500

# load packages
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import warnings

warnings.filterwarnings('ignore')

def pull_data(start_date, end_date, file_name):
    
    # define the top 25 companies in the S&P 500 by weight based on the link below: https://www.slickcharts.com/sp500
    ticker_list = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META',
                   'AVGO', 'GOOGL', 'TSLA', 'JPM', 'ORCL', 
                   'WMT', 'LLY', 'V', 'MA', 'NFLX', 
                   'XOM', 'JNJ', 'ABBV', 'PLTR', 'COST', 
                   'HD', 'BAC', 'PG', 'UNH', 'GE']


    # check if the ticker exists
    if not ticker_list:
        print("No ticker found.")
        return None

    print(f"\nDownloading data for the top {len(ticker_list)} companies...")

    try:
        data = yf.download(ticker_list, start=start_date, end=end_date, auto_adjust=False, group_by='ticker')
    
        # change the multiIndex format to single column data frame for ease of usage
        fin_tidy_data = data.stack(level=0).reset_index()
        fin_tidy_data = fin_tidy_data.rename(columns={'level_1': 'Ticker'})
        fin_tidy_data['Date'] = pd.to_datetime(fin_tidy_data['Date'])
        fin_tidy_data = fin_tidy_data.sort_values(by=['Ticker', 'Date'])

        # save to CSV
        fin_tidy_data.to_csv(file_name,index = False)

        return data

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
def main():

    # set the date interval of 10 years (March 2015 - April 2025)
    start_date = datetime.datetime(2015, 3, 31)
    end_date = datetime.datetime(2025, 4, 30)
    historical_file_name = 'Resources/Data/historical_stock_data_revised.csv'

    # extract the data for historical and actual stock data
    historical_data = pull_data(start_date, end_date, historical_file_name)

if __name__=="__main__":
    main()

