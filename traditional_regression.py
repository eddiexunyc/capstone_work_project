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

# define features that will be used on all regression models
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Returns', 'RSI', 'SMA_20', 'MACD']

# create a 30 days forecasting dataframe function
def forecasting_dataframe(data):

    # define the list and forecasting date
    future_df_list = []
    forecast_dates = pd.date_range(start='2025-04-01', end='2025-04-30', freq='B')
    
    for company in data['Ticker'].unique():
        last_row = data[data['Ticker'] == company].iloc[-1]
        for fdate in forecast_dates:
            row = last_row.copy()
            row['Date'] = fdate
            future_df_list.append(row)

    return pd.DataFrame(future_df_list)

# create multivariate linear regression function
def stock_price_linear_reg(data):
    
    # define x and y variables
    x = data[features].values
    y = data['Adj Close'].values

    ticker_labels = data['Ticker'].values
    dates = data['Date'].values if 'Date' in data.columns else np.arange(len(data))

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp, ticker_train, ticker_temp, date_train, date_temp = train_test_split(
        x, y, ticker_labels, dates, test_size=0.3, random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test, ticker_val, ticker_test, date_val, date_test = train_test_split(
        x_temp, y_temp, ticker_temp, date_temp, test_size=0.5, random_state=42)

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

    # create dataframe for linear regression metric
    result_lm = pd.DataFrame([{
        'Model': 'Multivariate Linear Regression',
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae}])
    
    # provide the future predicted value
    future_df = forecasting_dataframe(data)
    x_future = future_df[features].values

    lm_prediction = future_df[['Ticker', 'Date']].copy()
    lm_prediction['Predicted_AdjClose'] = lm.predict(x_future)
    lm_prediction = lm_prediction.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # save predictions to CSV
    lm_file_name = 'Resources/Predictions/lm_prediction.csv'
    lm_prediction.to_csv(lm_file_name, index=False)

    return result_lm


# create multivariate polynomial regression function
def stock_price_poly_reg(data):

    # define x and y variables
    x = data[features].values
    y = data['Adj Close'].values

    ticker_labels = data['Ticker'].values
    dates = data['Date'].values if 'Date' in data.columns else np.arange(len(data))

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp, ticker_train, ticker_temp, date_train, date_temp = train_test_split(
        x, y, ticker_labels, dates, test_size=0.3, random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test, ticker_val, ticker_test, date_val, date_test = train_test_split(
        x_temp, y_temp, ticker_temp, date_temp, test_size=0.5, random_state=42)
    
    # set the degree to 2 to reduce overfitting
    poly_feature = PolynomialFeatures(degree=2)

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
    
    # calculate mean absolute error on test set
    mae = mean_absolute_error(y_test, test_prediction) 

    # create a dataframe for polynonimal regression metric
    result_poly = pd.DataFrame([{
        'Model': 'Multivariate Polynomial Regression',
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae}])
    
    # provide the future predicted value
    future_df = forecasting_dataframe(data)
    x_future = future_df[features].values
    x_future_quad = poly_feature.transform(x_future)

    poly_prediction = future_df[['Ticker', 'Date']].copy()
    poly_prediction['Predicted_AdjClose'] = poly_lm.predict(x_future_quad)
    poly_prediction = poly_prediction.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # save predictions to CSV
    poly_file_name = 'Resources/Predictions/poly_prediction.csv'
    poly_prediction.to_csv(poly_file_name, index=False)
    
    return result_poly

# create random forest regression function
def stock_price_rf_reg(data):

    # define x and y variables
    x = data[features].values
    y = data['Adj Close'].values

    ticker_labels = data['Ticker'].values
    dates = data['Date'].values if 'Date' in data.columns else np.arange(len(data))

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp, ticker_train, ticker_temp, date_train, date_temp = train_test_split(
        x, y, ticker_labels, dates, test_size=0.3, random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test, ticker_val, ticker_test, date_val, date_test = train_test_split(
        x_temp, y_temp, ticker_temp, date_temp, test_size=0.5, random_state=42)

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

    # create a dataframe for random forest regression metric
    result_rf = pd.DataFrame([{
        'Model': 'Random Forest Regression',
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae}])
    
    # provide the predicted value
    future_df = forecasting_dataframe(data)
    x_future = scaler.transform(future_df[features].values)

    rf_prediction = future_df[['Ticker', 'Date']].copy()
    rf_prediction['Predicted_AdjClose'] = rf100.predict(x_future)
    rf_prediction = rf_prediction.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # save predictions to CSV
    rf_file_name = 'Resources/Predictions/rf_prediction.csv'
    rf_prediction.to_csv(rf_file_name, index=False)

    return result_rf


# create support vector regression function
def stock_price_svr_reg(data):
    
    # define x and y variables
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Returns', 'RSI', 'SMA_20', 'MACD']
    x = data[features].values
    y = data['Adj Close'].values

    ticker_labels = data['Ticker'].values
    dates = data['Date'].values if 'Date' in data.columns else np.arange(len(data))

    # split train data into 70% for training and 30% for tuning and performance
    x_train, x_temp, y_train, y_temp, ticker_train, ticker_temp, date_train, date_temp = train_test_split(
        x, y, ticker_labels, dates, test_size=0.3, random_state=42)

    # split the 30% for tuning and performance into 15% validation and 15% test
    x_val, x_test, y_val, y_test, ticker_val, ticker_test, date_val, date_test = train_test_split(
        x_temp, y_temp, ticker_temp, date_temp, test_size=0.5, random_state=42)

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

    # create a dataframe for support vector regression metric
    result_svr = pd.DataFrame([{
        'Model': 'Support Vector Regression',
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae}])
    
    # provide the predicted values
    future_df = forecasting_dataframe(data)
    x_future = scaler.transform(future_df[features].values)

    svr_prediction = future_df[['Ticker', 'Date']].copy()
    svr_prediction['Predicted_AdjClose'] = svr.predict(x_future)
    svr_prediction = svr_prediction.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # save predictions to CSV
    svr_file_name = 'Resources/Predictions/svr_prediction.csv'
    svr_prediction.to_csv(svr_file_name, index=False)

    return result_svr

def main():

    # read in data
    pre_process_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/pre_process_data_v2.csv'
    pre_process_data = pd.read_csv(pre_process_url)

    # run the mulitvariate linear regression
    lm_result = stock_price_linear_reg(pre_process_data)

    # run the multivariate polynomial regression
    poly_result = stock_price_poly_reg(pre_process_data)

    # run the random forest regression
    rf_result = stock_price_rf_reg(pre_process_data)

    # run the support vector regression
    svr_result = stock_price_svr_reg(pre_process_data)

    # combine the evaluation metrics from all regressions
    summary = pd.concat([
        pd.DataFrame(lm_result, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error']),
        pd.DataFrame(poly_result, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error']),
        pd.DataFrame(rf_result, columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error']),
        pd.DataFrame(svr_result,columns=['Model', 'Root Mean Squared Error', 'Mean Absolute Error'])])
    
    summary = summary.sort_values(by="Model", ascending=False).reset_index(drop=True)
    summary.to_csv('Resources/Predictions/summary_metric.csv', index=False)
    print(summary)

    
if __name__=="__main__":
    main()
