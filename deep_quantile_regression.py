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
def recursive_quantile_forecast(data, model, quantiles, scaler, features, y_scaler):
    
    forecast_dates = pd.date_range('2025-04-01','2025-04-30',freq='B')
    all_predictions = []

    for ticker in data["Ticker"].unique():

        df = data[data["Ticker"]==ticker].copy().sort_values("Date").reset_index(drop=True)

        # skip if the data is too small
        if df.shape[0] == 0:
            continue

        for fdate in forecast_dates:

            last = df.iloc[-1].copy()

            x = np.array([[ last[f] for f in features ]])
            x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            x = scaler.transform(x)
    
            # inverse transform to price-scale
            q_scaled = model.predict(x, verbose=0)[0]  # shape (4,)
            q_vals = y_scaler.inverse_transform(q_scaled.reshape(1, -1)).flatten()

            # append predictions
            all_predictions.append({
                "Ticker": ticker,
                "Date": fdate,
                "Q10": q_vals[0],
                "Q50": q_vals[1],
                "Q90": q_vals[2],
                "Q99": q_vals[3]
            })

            next_price = q_vals[1]
            prices = list(df["Adj Close"].values) + [next_price]

            # update features for next recursive step
            last['Date'] = fdate
            last['Adj Close'] = next_price

            # compute lagged returns relative to most recent actual price in df
            try:
                prev_price = df['Adj Close'].iloc[-1]
                last['Lagged_Returns'] = (next_price - prev_price) / prev_price if prev_price != 0 else 0.0
            except Exception:
                last['Lagged_Returns'] = 0.0

            # updates for multi-day features
            if len(prices) >= 6:
                last['Return_5d'] = (next_price - prices[-6]) / prices[-6] if prices[-6] != 0 else last.get('Return_5d', 0.0)
                last['Volatility_5d'] = np.std(np.diff(prices[-6:]))
            else:
                last['Return_5d'] = last.get('Return_5d', 0.0)
                last['Volatility_5d'] = last.get('Volatility_5d', 0.0)

            if len(prices) >= 22:
                last['Volatility_21d'] = np.std(np.diff(prices[-22:]))
            else:
                last['Volatility_21d'] = last.get('Volatility_21d', 0.0)

            if len(prices) >= 20:
                last['SMA_20'] = np.mean(prices[-20:])
            else:
                last['SMA_20'] = last.get('SMA_20', last.get('Adj Close', next_price))

            last['SMA20_Ratio'] = next_price / last['SMA_20'] if last['SMA_20'] != 0 else last.get('SMA20_Ratio', 1.0)

            # append new row to df for further recursive steps
            df = pd.concat([df, last.to_frame().T], ignore_index=True)
        
    return pd.DataFrame(all_predictions)


# define the monotonic deep quantile regression to address the quantile crossing issue
def build_monotonic_dqr_model(input_dim, quantiles=[0.1,0.5,0.9,0.99]):
    
    inp = tf.keras.Input(shape=(input_dim,))
    
    h = tf.keras.layers.Dense(64, activation='relu')(inp)
    h = tf.keras.layers.Dense(32, activation='relu')(h)

    # set the ase quantile q10
    q10 = tf.keras.layers.Dense(1, name="q10")(h)

    # define the increments for higher quantiles
    d50 = tf.keras.layers.Dense(1, activation=tf.nn.softplus, name="d50")(h)
    d90 = tf.keras.layers.Dense(1, activation=tf.nn.softplus, name="d90")(h)
    d99 = tf.keras.layers.Dense(1, activation=tf.nn.softplus, name="d99")(h)

    q50 = q10 + d50
    q90 = q50 + d90
    q99 = q90 + d99

    outputs = tf.keras.layers.Concatenate(name="quantiles")([q10, q50, q90, q99])

    model = tf.keras.Model(inputs=inp, outputs=outputs)

    return model

# define the quantile loss function
def multi_quantile_loss(quantiles):
    def loss(y_true, y_pred):
        losses = []
        for i, q in enumerate(quantiles):
            e = y_true - y_pred[:, i]
            losses.append(tf.reduce_mean(tf.maximum(q*e, (q-1)*e)))
        return tf.reduce_sum(losses)
    return loss

# define the deep quantile regression
def deep_quantile_regression(data):

    # define features and quantiles
    features = ['Lagged_Returns', 'Return_1d', 'Return_5d', 'Volatility_5d', 'Volatility_21d',
                'RSI', 'SMA_20', 'MACD', 'SMA20_Ratio', 'log_Volume']
    
    quantiles=[0.1, 0.5, 0.9, 0.99]

    # define x and y variables
    x = data[features].values
    y_raw = data['Adj Close'].values.reshape(-1,1)

    # scale the target variables
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y_raw).flatten()

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

    # build the monotonic model
    models = build_monotonic_dqr_model(input_dim=x_train_scaled.shape[1], quantiles=quantiles)

    models.compile(optimizer='adam', loss=multi_quantile_loss(quantiles))

    # train the model
    models.fit(x_train_scaled, y_train,
               validation_data=(x_val_scaled, y_val),
               epochs=40,
               batch_size=32,
               verbose=1)
    
    # predict the value using the test set
    preds_scaled = models.predict(x_test_scaled)\

    # inverse transform the prediction
    preds = y_scaler.inverse_transform(preds_scaled)
    y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    quantile_metrics = {}

    for i, q in enumerate(quantiles):
        pred_q = preds[:, i]
        e = y_test_orig - pred_q
        ql = np.mean(np.maximum(q*e, (q-1)*e))

        picp = np.mean(y_test_orig <= pred_q)
        quantile_metrics[q] = {"Quantile Loss": round(float(ql),6), "PICP": round(float(picp),4)}
        print(f"✓ Quantile {q}: Loss={ql:.6f}, PICP={picp:.4f}")

    # calculate the PICP
    lower = preds[:, 0]
    upper = preds[:, 2]
    interval_picp = np.mean((y_test_orig >= lower) & (y_test_orig <= upper))
    print('Interval (Q10-Q90) PICP: ', round(float(interval_picp),4))

    # predict 30 days future horizon
    dqr_prediction = recursive_quantile_forecast(data,models,quantiles,scaler,features, y_scaler)

    # save predictions to CSV
    dqr_file_name = 'Resources/Predictions/dqr_prediction_revised_monotonic.csv'
    dqr_prediction.to_csv(dqr_file_name, index=False)
    
    return models, dqr_prediction, quantile_metrics, interval_picp

def main():

     # read in data
    pre_process_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/Data/pre_process_data_final_revised.csv'
    pre_process_data = pd.read_csv(pre_process_url)

    models, forecast, metrics, interval_picp = deep_quantile_regression(pre_process_data)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # save the summary metric
    summary_metric_quantile = (pd.DataFrame.from_dict(metrics, orient='index')
                                 .reset_index()
                                 .rename(columns={'index':'Quantile'}))
    summary_metric_quantile.to_csv('Resources/Data/summary_metric_quantile_revised_mono.csv', index=False)

if __name__=="__main__":
    main()