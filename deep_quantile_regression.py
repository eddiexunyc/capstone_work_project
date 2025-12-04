# load core python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# load deep quantile regression packages
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# define the recursive forecasting for deep quantile regression
def recursive_quantile_forecast(data, models, quantiles, scaler, features):
    
    forecast_dates = pd.date_range('2025-04-01','2025-04-30',freq='B')
    all_predictions = []

    for ticker in data["Ticker"].unique():

        df = data[data["Ticker"]==ticker].copy().sort_values("Date").reset_index(drop=True)

        for fdate in forecast_dates:

            last = df.iloc[-1].copy()

            x = np.array([[ last[f] for f in features ]])
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = scaler.transform(x)

            preds = []
            for q in quantiles:
                preds.append(models[q].predict(x)[0][0])

            all_predictions.append({
                "Ticker": ticker,
                "Date": fdate,
                "Q10": preds[0],
                "Q50": preds[1],
                "Q90": preds[2],
                "Q99": preds[3]
            })

            next_price = preds[1]
            prices = list(df["Adj Close"]) + [next_price]

            last['Date'] = fdate
            last['Adj Close'] = next_price
            last['Lagged_Returns'] = (next_price - df['Adj Close'].iloc[-1]) / df['Adj Close'].iloc[-1]
            last['Return_1d'] = last['Lagged_Returns']

            last['Return_5d'] = (next_price - prices[-6])/prices[-6] if len(prices)>=6 else last['Return_5d']
            last['Volatility_5d']  = np.std(np.diff(prices[-6:])) if len(prices)>=6 else last['Volatility_5d']
            last['Volatility_21d'] = np.std(np.diff(prices[-22:])) if len(prices)>=22 else last['Volatility_21d']
            last['SMA_20'] = np.mean(prices[-20:]) if len(prices)>=20 else last['SMA_20']
            last['SMA20_Ratio'] = next_price / last['SMA_20'] if last['SMA_20']!=0 else last['SMA20_Ratio']

            df = pd.concat([df,last.to_frame().T], ignore_index=True)

    return pd.DataFrame(all_predictions)

# define the deep quantile regression
def deep_quantile_regression(data):

    # define features and quantiles
    features = ['Lagged_Returns', 'Return_1d', 'Return_5d', 'Volatility_5d', 'Volatility_21d',
                'RSI', 'SMA_20', 'MACD', 'SMA20_Ratio', 'log_Volume']
    
    quantiles=[0.1, 0.5, 0.9, 0.99]

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

    # define the quantile loss
    def q_loss(q):
        return lambda y_true,y_pred: tf.reduce_mean(tf.maximum(q*(y_true-y_pred),(q-1)*(y_true-y_pred)))

    models = {} 
    quantile_metrics = {}

    # train the model per quantile
    for q in quantiles:

        m = tf.keras.Sequential([
            tf.keras.layers.Dense(64,activation='relu',input_shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.Dense(32,activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        m.compile(optimizer='adam', loss=q_loss(q))
        m.fit(x_train_scaled,y_train,validation_data=(x_val_scaled,y_val),
              epochs=40,batch_size=32,verbose=1)

        models[q] = m

       # evaluate test quantile loss
        q_pred = m.predict(x_test_scaled).flatten()
        ql = np.mean(np.maximum(q*(y_test-q_pred),(q-1)*(y_test-q_pred)))

        # define PICP(q) = P(y >= predicted quantile)
        picp_q = np.mean(y_test >= q_pred)

        quantile_metrics[q] = {
            "Quantile Loss": round(ql,6),
            "PICP": round(picp_q,4)
        }

        print(f"✓ Quantile {q} → Loss={ql:.6f}, PICP={picp_q:.4f}")

    lower = models[0.1].predict(x_test_scaled).flatten()
    upper = models[0.9].predict(x_test_scaled).flatten()

    interval_picp = np.mean((y_test>=lower)&(y_test<=upper))
    print('PICP', round(interval_picp,4))

    # predict 30 days future horizon
    dqr_prediction = recursive_quantile_forecast(data,models,quantiles,scaler,features)

    # save predictions to CSV
    dqr_file_name = 'Resources/Predictions/dqr_prediction_revised.csv'
    dqr_prediction.to_csv(dqr_file_name, index=False)
    
    return models, dqr_prediction, quantile_metrics, interval_picp

def main():

     # read in data
    pre_process_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/Data/pre_process_data_final_revised.csv'
    pre_process_data = pd.read_csv(pre_process_url)

    models, forecast, metrics, interval_picp = deep_quantile_regression(pre_process_data)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    summary_metric_quantile = pd.DataFrame([metrics])
    summary_metric_quantile.to_csv('Resources/Data/summary_metric_quantile.csv', index=False)

if __name__=="__main__":
    main()