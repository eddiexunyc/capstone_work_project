# load core python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# load deep quantile regression packages
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def deep_quantile_regression(data):

    # define x and y variables
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Lagged_Returns', 'RSI', 'SMA_20', 'MACD']
    x = data[features].values
    y = data['Adj Close'].values

    ticker_labels = data['Ticker'].values
    dates = data['Date'].values if 'Date' in data.columns else np.arange(len(data))

    # define the qunatile
    quantiles=[0.1, 0.5, 0.9, 0.99]

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


    # define Quantile Loss
    def quantile_loss(q, y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * err, (q - 1) * err))

    # run a loop for each quantile
    prediction_result = {}

    for q in quantiles:

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(q, y_true, y_pred))

        # stop to prevent overfitting
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        # train the model
        history = model.fit(
            x_train_scaled, y_train,
            validation_data=(x_val_scaled, y_val),
            epochs=50,
            batch_size=32,
            verbose=1,
            callbacks = [early_stop]
        )

        # evaluate model
        val_loss = model.evaluate(x_val_scaled, y_val, verbose=0)
        test_loss = model.evaluate(x_test_scaled, y_test, verbose=0)
        print(f"Quantile {q}: Validation Loss = {val_loss:.4f}, Test Loss = {test_loss:.4f}")

        # predictions
        y_pred = model.predict(x_test_scaled).flatten()

        # store results
        prediction_result[q] = {
            'model': model,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'history': history
        }


    # provide the future predicted value
    future_df_list = []
    forecast_dates = pd.date_range(start='2025-04-01', end='2025-04-30', freq='B')

    for company in data['Ticker'].unique():
        last_row = data[data['Ticker'] == company].iloc[-1]
        for fdate in forecast_dates:
            row = last_row.copy()
            row['Date'] = fdate
            future_df_list.append(row)

    future_df = pd.DataFrame(future_df_list)
    x_future_scaled = scaler.transform(future_df[features])


    # combine predictions and sort it by dates
    dqr_prediction = future_df[['Ticker', 'Date']].copy()
    for q in quantiles:
        y_pred = prediction_result[q]['model'].predict(x_future_scaled).flatten()
        dqr_prediction[f'Predicted_Quantile_{q}'] = y_pred

    dqr_prediction = dqr_prediction.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # save predictions to CSV
    dqr_file_name = 'Resources/Predictions/dqr_prediction.csv'
    dqr_prediction.to_csv(dqr_file_name, index=False)
    
    return prediction_result, scaler, dqr_prediction

def main():

     # read in data
    pre_process_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/pre_process_data_v2.csv'
    pre_process_data = pd.read_csv(pre_process_url)

    results, scaler, pred_df = deep_quantile_regression(pre_process_data)

if __name__=="__main__":
    main()