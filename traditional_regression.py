# Traditional Regression Models

# load core packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# load linear and polynomial regression packages
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# load random forest regression packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# load support vector regression packages
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def stock_price_linear_reg(data):
    
    # define x and y variables
    x=data[['High','Low','Last','Open','Total Trade Quantity','Turnover (Lacs)']].values
    y=data[['Close']].values

    # split train data
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=0)

    # deploy linear regression
    lm=LinearRegression()
    lm.fit(x_train,y_train)

    # calculate the coefficient and the score
    lm.coef_
    lm.score(x_train,y_train)
    predictions = lm.predict(x_test)

    # calculate r-squared
    r2_score(y_test, predictions)
