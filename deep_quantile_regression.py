# load core python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load deep quantile regression packages
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def deep_quantile_regression(data):

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


    # define Quantile Loss
    def quantile_loss(q, y_true, y_pred):
        err = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * err, (q - 1) * err))

    # build and train Model for a specific quantile (e.g., 0.5 for median)
    q_value = 0.5

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1) # Output for the specific quantile
    ])

    model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(0.5, y_true, y_pred))

    history = model.fit(
        x_train_scaled, y_train,
        validation_data=(x_val_scaled, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # 7. Evaluate and predict
    y_pred = model.predict(x_test_scaled)

    # 8. Compute validation and test losses
    val_loss = model.evaluate(x_val_scaled, y_val, verbose=0)
    test_loss = model.evaluate(x_test_scaled, y_test, verbose=0)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    return model, scaler, y_pred, history

def main():

     # read in data
    pre_process_url = 'https://raw.githubusercontent.com/eddiexunyc/capstone_work_project/refs/heads/main/Resources/pre_process_data_v2.csv'
    pre_process_data = pd.read_csv(pre_process_url)

    deep_quantile_prediction = deep_quantile_regression(pre_process_data)

if __name__=="__main__":
    main()