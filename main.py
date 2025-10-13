
# load core packages
import datetime
import numpy as np
import pandas as pd

# load other python modules
import stock_data_extraction
# import traditional_regression (numpy issues)
# import deep_quantile_regression (numpy issues)

def main():

    # set the date interval of 10 years (March 2015 - March 2025)
    start_date = datetime.datetime(2015, 3, 31)
    end_date = datetime.datetime(2025, 3, 31)
    historical_file_name = 'Resources/pre_processing_data.csv'

    # set the data interval of 6 months (March 2025 - September 2025)
    actual_start_date = datetime.datetime(2025, 3, 31)
    actual_end_date = datetime.datetime(2025, 9, 30)
    actual_file_name = 'Resources/actual_stock_data.csv'


    # extract the data for historical and actual stock data
    historical_data = stock_data_extraction.pull_data(start_date, end_date, historical_file_name)
    actual_data = stock_data_extraction.pull_data(actual_start_date, actual_end_date, actual_file_name)

if __name__=="__main__":
    main()