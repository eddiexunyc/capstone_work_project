
# load core packages
import datetime
import numpy as np
import pandas as pd

# load other python modules
import stock_data_extraction
import traditional_regression
import deep_quantile_regression

def main():

    # set the date interval of 10 years (2015 - 2025)
    start_date = datetime.datetime(2015, 9, 30)
    end_date = datetime.datetime(2025, 9, 30)   

    # extract the data
    data = stock_data_extraction.pull_data(start_date, end_date)

if __name__=="__main__":
    main()