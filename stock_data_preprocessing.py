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

    # group by company ticker before pre-processing
    grouped_data = data.groupby('Ticker')

    # calculate the lagged return
    data['Lagged_Returns'] = grouped_data['Adj Close'].transform(lambda x: x.pct_change().shift(1) * 100)

    # calculate the relative strength index (RSI) of past 14 periods with EMA smoothing to provide weighted average.
    rsi_window = 14
    delta = grouped_data['Adj Close'].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.groupby(data['Ticker']).transform(lambda x: x.ewm(alpha=1/rsi_window, min_periods=rsi_window).mean())
    avg_loss = loss.groupby(data['Ticker']).transform(lambda x: x.ewm(alpha=1/rsi_window, min_periods=rsi_window).mean())
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # calculate the simple moving average (SMA)
    sma_window = 20
    sma_window = 20
    data['SMA_20'] = grouped_data['Adj Close'].transform(lambda x: x.rolling(window=sma_window).mean())

    # calculate the moving average convergence divergence
    ema12 = grouped_data['Adj Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = grouped_data['Adj Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    data['MACD'] = ema12 - ema26

    # fill missing data using forward and backward fill methods
    fill_cols = ['Lagged_Returns', 'RSI', 'SMA_20', 'MACD']
    data[fill_cols] = data.groupby('Ticker')[fill_cols].transform(lambda x: x.bfill().ffill())

    return data

def vif_calculation(data):
    feature_data = data[['Lagged_Returns', 'RSI', 'SMA_20', 'MACD']]
    vif_data = pd.DataFrame()
    vif_data['feature'] = feature_data.columns
    vif_data['VIF'] = [variance_inflation_factor(feature_data.values, i) for i in range(feature_data.shape[1])]

    return vif_data

def main():

    # pull in data
    history_data_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/Data/historical_stock_data.csv'
    history_data = pd.read_csv(history_data_url)

    # perform the pre-processing
    pre_process_data = stock_preprocessing(history_data)
    pre_process_file_name = 'Resources/Data/pre_process_data_final.csv'

    # calculate the VIF
    vif_result = vif_calculation(pre_process_data)
    vif_result_data_file_name = 'Resources/Results/vif_result_data.csv'

    # create a subplot without frame for the VIF result
    plt.figure(figsize=(10, 6))
    plot = plt.subplot(111, frame_on=False)

    # remove axis
    plot.xaxis.set_visible(False) 
    plot.yaxis.set_visible(False) 

    # create the table plot and position it in the upper left corner
    table(plot, vif_result,loc='upper right')

    # save the plot as a png file
    plt.savefig('Resources/Results/vif_result_plot.png')

    # save to CSV
    pre_process_data.to_csv(pre_process_file_name, index = False)

if __name__=="__main__":
    main()