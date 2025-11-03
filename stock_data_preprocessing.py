# This python will define the pre-processing

# load packages
import numpy as np
import pandas as pd
import datetime
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def stock_preprocessing(data):
    
    # calculate the lagged return
    data['Lagged_Returns'] = data['Adj Close'].pct_change() * 100

    # calculate the relative strength index (RSI)
    window = 14
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # calculate the simple moving average (SMA)
    data['SMA_20'] = data['Adj Close'].rolling(window=20).mean()

    # calculate the moving average convergence divergence
    ema12 = data['Adj Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Adj Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    return data

def vif_calculation(data):
    feature_data = data[['Lagged_Returns', 'RSI', 'SMA_20', 'MACD']]
    vif_data = pd.DataFrame()
    vif_data['feature'] = feature_data.columns
    vif_data['VIF'] = [variance_inflation_factor(feature_data.values, i) for i in range(feature_data.shape[1])]

    return vif_data

def main():

    # pull in data
    history_data_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/historical_stock_data.csv?token=GHSAT0AAAAAADKDZNSYNLHMK3JS6HWNDL4Y2IIDJKQ'
    history_data = pd.read_csv(history_data_url)

    # perform the pre-processing
    pre_process_data = stock_preprocessing(history_data)

    # calculate the VIF
    vif_result = vif_calculation(pre_process_data)
    print(vif_result)

if __name__=="__main__":
    main()