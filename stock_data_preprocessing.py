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

def vif_calculation(data):
    x = data

