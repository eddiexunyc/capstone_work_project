# ulitities python file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
import datetime
from sklearn.model_selection import train_test_split

# 1. Data Generation (Example)
np.random.seed(0)
X = np.random.rand(1000, 5)
y = 2 * X[:, 0] + 3 * X[:, 1]**2 + np.random.normal(0, 0.5 + X[:, 0], 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define Quantile Loss
def quantile_loss(q, y_true, y_pred):
    err = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * err, (q - 1) * err))

# 3. Build and Train Model for a specific quantile (e.g., 0.5 for median)
q_value = 0.5

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1) # Output for the specific quantile
])

model.compile(optimizer='adam', loss=lambda y_true, y_pred: quantile_loss(q_value, y_true, y_pred))

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 4. Make Predictions
y_pred_median = model.predict(X_test)
print(y_pred_median)

def stock_data_pull():
    #
    start_date = datetime.datetime(2019, 5, 31)
    end_date = datetime.datetime(2021, 1, 30)
    meta = yf.Ticker("META")
    data = meta.history(start=start_date, end=end_date)
    print(data.to_string())