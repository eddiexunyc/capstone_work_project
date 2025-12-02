# This python will define the pre-processing

# load packages
import numpy as np
import pandas as pd
import datetime
import warnings
import matplotlib.pyplot as plt
import datetime as dt
import plotly.express as px
import matplotlib.dates as mdates
from pandas.plotting import table
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def stock_preprocessing(data):

    # calculate the lagged return
    data['Lagged_Returns'] = data.groupby('Ticker')['Adj Close'].shift(1).pct_change(fill_method = None) 

    # calculate the 1-day forward return
    data['Return_1d'] = data.groupby('Ticker')['Adj Close'].pct_change()

    # calculate the 5-days forward return
    data['Return_5d'] = data.groupby('Ticker')['Adj Close'].pct_change(5)

    # calculate the 5 and 21 Days rolling standard deviation
    data['Volatility_5d'] = data.groupby('Ticker')['Return_1d'].rolling(5).std().reset_index(level=0, drop=True)
    data['Volatility_21d'] = data.groupby('Ticker')['Return_1d'].rolling(21).std().reset_index(level=0, drop=True)

    # calculate the relative strength index (RSI) of past 14 periods with EMA smoothing to provide weighted average.
    rsi_window = 14
    delta = data.groupby('Ticker')['Adj Close'].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(data['Ticker']).transform(lambda x: x.ewm(alpha=1/rsi_window, min_periods=rsi_window).mean())
    avg_loss = loss.groupby(data['Ticker']).transform(lambda x: x.ewm(alpha=1/rsi_window, min_periods=rsi_window).mean())
    rs = avg_gain / avg_loss
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))

    # fill out nan values for RSI edge cases
    data['RSI'] = data['RSI'].fillna(50)

    # calculate the simple moving average (SMA)
    sma_window = 20
    data['SMA_20'] = data.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=sma_window).mean())

    # calculate the moving average convergence divergence
    ema12 = data.groupby('Ticker')['Adj Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = data.groupby('Ticker')['Adj Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    data['MACD'] = ema12 - ema26

    # calculate the simple moving average ratio
    data['SMA20_Ratio'] = data['Adj Close'] / data['SMA_20']
    data['SMA20_Ratio'] = data['SMA20_Ratio'].replace([np.inf, -np.inf], np.nan)

    # calculate the log of trading volume
    data['log_Volume'] = np.log1p(data['Volume'])

    # fill missing data using forward and backward fill methods and 
    fill_cols = ['Lagged_Returns', 'Return_1d', 'Return_5d', 'Volatility_5d', 'Volatility_21d',
                'RSI', 'SMA_20', 'MACD', 'SMA20_Ratio']
    data[fill_cols] = data.groupby('Ticker')[fill_cols].transform(lambda x: x.bfill().ffill())

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)

    return data

def main():

    # pull in data
    history_data_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/Data/historical_stock_data.csv'
    history_data = pd.read_csv(history_data_url)

    # perform the pre-processing
    pre_process_data = stock_preprocessing(history_data)
    pre_process_file_name = 'Resources/Data/pre_process_data_final_revised.csv'

    # create a subplot without frame for the VIF result
    plt.figure(figsize=(10, 6))
    plot = plt.subplot(111, frame_on=False)

    # remove axis
    plot.xaxis.set_visible(False) 
    plot.yaxis.set_visible(False) 

    # save to CSV
    pre_process_data.to_csv(pre_process_file_name, index = False)

if __name__=="__main__":
    main()