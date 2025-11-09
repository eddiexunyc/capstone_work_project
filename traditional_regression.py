# This python file will define all traditional regression models

# load core packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# load linear and polynomial regression packages
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score

# load random forest regression and s
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# load upport vector regression packages
from sklearn.svm import SVR


# create multivariate linear regression function
def stock_price_linear_reg(data):
    
    # define x and y variables
    x=data[['Lagged_Returns', 'RSI', 'SMA_20', 'MACD']].values
    y=data[['Close']].values

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp = train_test_split(x,y, test_size=0.3,random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # deploy linear regression
    lm=LinearRegression()
    lm.fit(x_train,y_train)

    # calculate predictions on validation and test sets
    valid_prediction = lm(x_val)
    test_prediction = lm(x_test)

    # calculate mean absolute error on test set
    mse = mean_squared_error(y_test, test_prediction)

    # calculate root mean squared error on test set
    rmse = np.sqrt(mse)
    
    # calculate r-squared on test set
    r2_score(y_test, test_prediction)

    # calculate the score and coeffiecient
    train_score = lm.score(x_train, y_train)
    coef = lm.coef_

    return {
        'validation_predictions': valid_prediction,
        'test_predictions': test_prediction,
        'coefficients': coef,
        'train_score': train_score,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2_score': r2
    }


# create multivariate polynomial regression function
def stock_price_poly_reg(data):

    x = data

def stock_price_rf_reg(data):
    x = data

def stock_price_svm_reg(data):
    x = data


def main():
    
    
if __name__=="__main__":
    main()
