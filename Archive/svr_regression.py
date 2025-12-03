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

# load support vector regression packages
from sklearn.svm import SVR

# define engineered features that will be used on all regression models
features = ['Lagged_Returns', 'Return_1d', 'Return_5d', 'Volatility_5d', 'Volatility_21d',
                'RSI', 'SMA_20', 'MACD', 'SMA20_Ratio', 'log_Volume']

# define a diviation function to address the inf/nan issues for poly
def safe_div(a, b, fallback=0):
        if b is None or b == 0 or np.isnan(b) or np.isinf(b):
            return fallback
        return a / b

# define a function that update the indicators so the indicators won't be constant and allow for an accurate prediction
def recursive_forecast(data, model, features, scaler=None, transformer=None):

    forecast_dates = pd.date_range('2025-05-01', '2025-05-31', freq='B')
    all_predictions = []

    for ticker in data['Ticker'].unique():

        df = data[data['Ticker'] == ticker].copy()
        df = df.sort_values("Date").reset_index(drop=True)

        for fdate in forecast_dates:

            last_row = df.iloc[-1].copy()

            # predict the future price
            x_input = np.array([[last_row[f] for f in features]])

            x_input = np.clip(x_input, -10, 10)
            x_input = np.nan_to_num(x_input, nan=0.0, posinf=0.0, neginf=0.0)

            # apply transformer
            if transformer:
                x_input = transformer.transform(x_input)
                x_input = np.nan_to_num(x_input, nan=0.0, posinf=0.0, neginf=0.0)

            # apply scaler
            if scaler:
                x_input = scaler.transform(x_input)
                x_input = np.nan_to_num(x_input, nan=0.0, posinf=0.0, neginf=0.0)

            predicted_price = model.predict(x_input)[0]

            # build the future row
            new = last_row.copy()
            new['Date'] = fdate
            new['Adj Close'] = predicted_price

            # update the adjusted closed price
            prices = list(df["Adj Close"]) + [predicted_price]

            # update lagged return
            prev_price = last_row['Adj Close']
            new['Lagged_Returns'] = safe_div(predicted_price - prev_price, prev_price)

            # update the 1-day forward return
            new['Return_1d'] = new['Lagged_Returns']

            # update the 5-days forward return
            if len(prices) >= 6:
                new['Return_5d'] = safe_div(prices[-1] - prices[-6], prices[-6])
            else:
                new['Return_5d'] = last_row['Return_5d']

            # update the 5-day volatility
            if len(prices) >= 6:
                new['Volatility_5d'] = np.std(np.diff(prices[-6:]))
            else:
                new['Volatility_5d'] = last_row['Volatility_5d']

            # update the 21-day volatility
            if len(prices) >= 22:
                new['Volatility_21d'] = np.std(np.diff(prices[-22:]))
            else:
                new['Volatility_21d'] = last_row['Volatility_21d']

            # update SMA20
            if len(prices) >= 20:
                new['SMA_20'] = np.mean(prices[-20:])
            else:
                new['SMA_20'] = last_row['SMA_20']
            
            # update RSI
            if len(prices) >= 15:
                deltas = np.diff(prices[-15:])
                ups = deltas[deltas > 0].sum()
                downs = -deltas[deltas < 0].sum()
                rs = ups / downs if downs != 0 else 0
                new['RSI'] = 100 - (100 / (1 + rs))
            else:
                new['RSI'] = last_row["RSI"]

            # update MACD
            ema12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
            new['MACD'] = ema12 - ema26

            # update the SMA20 Ratio
            new['SMA20_Ratio'] = safe_div(predicted_price, new['SMA_20'], fallback=1)

            # update log volume
            new['log_Volume'] = last_row['log_Volume']

            # define the final safety
            for col in features:
                if not np.isfinite(new[col]):
                    new[col] = 0

            # append predicted row
            df = pd.concat([df, new.to_frame().T], ignore_index=True)

            all_predictions.append({
                "Ticker": ticker,
                "Date": fdate,
                "Predicted_AdjClose": predicted_price
            })

    return pd.DataFrame(all_predictions)

def stock_price_svr_reg(data):
    
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
    print('Working on Support Vector Regression Prediction')

    svr_prediction = recursive_forecast(data, svr, features, scaler=scaler)
    svr_prediction = svr_prediction.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # save predictions to CSV
    svr_file_name = 'Resources/Predictions/svr_prediction_revised.csv'
    svr_prediction.to_csv(svr_file_name, index=False)

    print('Support Vector Regression is completed')

    return result_svr

def main():

    # read in data
    pre_process_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/Data/pre_process_data_final_revised.csv'
    pre_process_data = pd.read_csv(pre_process_url)

    # run the support vector regression
    svr_result = stock_price_svr_reg(pre_process_data)

    
    
if __name__=="__main__":
    main()