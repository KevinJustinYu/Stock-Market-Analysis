'''
This file will implement functionality for backtesting various trading algorithms on past data.
'''


# ---------- Imports ---------- 
from market_tests import *
from market_screener import *
from datetime import timedelta
import os.path

if os.path.isfile('filename.txt'):
    print ("File exist")
else:
    print ("File not exist")


# ---------- Functions ----------

def backtest_algo(end_date, num_days, path=''):
	# Get the dates we want to backtest for
    date_list = pd.date_range(end=end_date, periods = numdays, freq='B')
    
    # Get the corresponding files for the dates
    csvs = []
    dates = []
    # Go through the dates and add files if they exist
    for d in date_list:
    	if os.path.isfile(path + 'csv_files/company_stats_'+ str(d.date()) + '.csv'):
        	csvs.append('company_stats_' + str(d.date()) + '.csv')
        	dates.append(str(d.date()))


    portfolio = {}

    # Run trading algo for each day, store in csv
    for i, csv in enumerate(csvs):
    	tickers = get_tickers(file_name=csv, path=path)
    	print('Starting number of tickers: ' + str(len(tickers)))
		print('Filtering tickers...')
		# Filter by price
		price_filtered = price_filter(tickers, 10, date[i], path=path)
		# Filter by PE ratio
		filtered_tickers = pe_ratio_to_industry_filter(price_filtered, 0.85, date[i], path=path)
		print('Number of tickers after filtering: ' + str(len(filtered_tickers)))


    # Evaluate csv
    