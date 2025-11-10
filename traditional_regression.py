# This python file will define all traditional regression models

# load core packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# load linear and polynomial regression packages
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# load random forest regression packages
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# load upport vector regression packages
from sklearn.svm import SVR

# create multivariate linear regression function
def stock_price_linear_reg(data):
    
    # define x and y variables
    x = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Returns', 'RSI', 'SMA_20', 'MACD']].values
    y = data[['Adj Close']].values

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3,random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # deploy linear regression
    lm = LinearRegression()
    lm.fit(x_train,y_train)

    # calculate predictions on validation and test sets
    valid_prediction = lm.predict(x_val)
    test_prediction = lm.predict(x_test)

    # calculate mean squared error on test set
    mse = mean_squared_error(y_test, test_prediction)

    # calculate root mean squared error on test set
    rmse = np.sqrt(mse)
    
    # calculate mean absolute error on test set
    mae = mean_absolute_error(y_test, test_prediction) 

    # calculate r-squared on test set
    r2 = r2_score(y_test, test_prediction)

    # calculate the score and coeffiecient
    train_score = lm.score(x_train, y_train)
    coef = lm.coef_

    # return value need to fix later
    return {
        'validation_predictions': valid_prediction,
        'test_predictions': test_prediction,
        'coefficients': coef,
        'train_score': train_score,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2_score': r2,
        'test_mae': mae
    }


# create multivariate polynomial regression function
def stock_price_poly_reg(data):

    # define x and y variables
    x = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Returns', 'RSI', 'SMA_20', 'MACD']].values
    y = data[['Adj Close']].values

    # set the degree to 2 to reduce overfitting
    poly_feature = PolynomialFeatures(degree=2)

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3,random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # train the data
    x_train_quad = poly_feature.fit_transform(x_train)
    x_valid_quad = poly_feature.transform(x_val)
    x_test_quad = poly_feature.transform(x_test)

    # deploy polynomial regression
    poly_lm = LinearRegression()
    poly_lm.fit(x_train_quad, y_train)

    # calculate predictions on validation and test sets
    valid_prediction = poly_lm.predict(x_valid_quad)
    test_prediction = poly_lm.predict(x_test_quad)

    # calculate mean squared error on test set
    mse = mean_squared_error(y_test, test_prediction)

    # calculate root mean squared error on test set
    rmse = np.sqrt(mse)
    
    # calculate r-squared on test set
    r2 = r2_score(y_test, test_prediction)

    # calculate mean absolute error on test set
    mae = mean_absolute_error(y_test, test_prediction) 

    # calculate the score and coeffiecient
    train_score = poly_lm.score(x_train_quad, y_train)
    coef = poly_lm.coef_

    # return value need to fix later
    return {
        'validation_predictions': valid_prediction,
        'test_predictions': test_prediction,
        'coefficients': coef,
        'train_score': train_score,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2_score': r2_score,
        'test_mae': mae
    }

def stock_price_rf_reg(data):

    # define x and y variables
    x = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Returns', 'RSI', 'SMA_20', 'MACD']].values
    y = data['Adj Close'].values

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3,random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # normalize the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_val_scaled = scaler.transform(x_val)

    # define the large ensembles
    rf100 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf100.fit(x_train_scaled, y_train)

    # calculate predictions on validation and test sets
    valid_prediction = rf100.predict(x_val_scaled)
    test_prediction = rf100.predict(x_test_scaled)

    # calculate mean squared error on test set
    mse = mean_squared_error(y_test, test_prediction)

    # calculate root mean squared error on test set
    rmse = np.sqrt(mse)
    
    # calculate r-squared on test set
    r2 = r2_score(y_test, test_prediction)

    # calculate mean absolute error on test set
    mae = mean_absolute_error(y_test, test_prediction) 

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    print(f"RSME: {rmse}")

def stock_price_svr_reg(data):
    
    # define x and y variables
    x = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Returns', 'RSI', 'SMA_20', 'MACD']].values
    y = data['Adj Close'].values

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3,random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # normalize the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_val_scaled = scaler.transform(x_val)

    # define the support vector regression with RBF kernel
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr.fit(x_train_scaled, y_train)

    # calculate predictions on validation and test sets
    valid_prediction = svr.predict(x_val_scaled)
    test_prediction = svr.predict(x_test_scaled)

    # calculate mean squared error on test set
    mse = mean_squared_error(y_test, test_prediction)

    # calculate root mean squared error on test set
    rmse = np.sqrt(mse)
    
    # calculate r-squared on test set
    r2 = r2_score(y_test, test_prediction)

    # calculate mean absolute error on test set
    mae = mean_absolute_error(y_test, test_prediction) 

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    print(f"RSME: {rmse}")



def main():

    # read in data
    pre_process_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/pre_process_data_v2.csv'
    pre_process_data = pd.read_csv(pre_process_url)

    # run the mulitvariate linear regression
    # lm_result = stock_price_linear_reg(pre_process_data)
    # print(lm_result)

    # run the multivariate polynomial regression
    # poly_result = stock_price_poly_reg(pre_process_data)
    # print(poly_result)

    # run the random forest regression
    rf_result = stock_price_rf_reg(pre_process_data)

    # run the support vector regression
    svr_result = stock_price_svr_reg(pre_process_data)

    
if __name__=="__main__":
    main()
