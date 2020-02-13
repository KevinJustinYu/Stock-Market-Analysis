'''
This file will implement functionality for backtesting various trading algorithms on past data.
'''


# ---------- Imports ---------- 
from market_tests import *
from market_screener import *
from datetime import timedelta
import os.path
import os


# ---------- Functions ----------

def backtest_algo(end_date, num_days, trading_algo_file_name, buy_alpha=0.005, short_alpha=0.0001, price_thresh=10, time_averaged_period=5, path=''):
    # Get the dates we want to backtest for
    date_list = pd.date_range(end=end_date, periods=num_days, freq='B')
    
    # Get the corresponding files for the dates
    csvs = []
    dates = []
    
    # Go through the dates and add files if they exist
    for d in date_list:
        if os.path.exists(path + 'csv_files/company_stats_'+ str(d.date()) + '.csv'):
            csvs.append('company_stats_' + str(d.date()) + '.csv')
            dates.append(str(d.date()))

    assert len(csvs) != 0, 'Failed to located the csvs. Check the date and the csvs.'
    portfolio = {}
    clear_csv = True

    # Run trading algo for each day, store in csv
    for i, csv in enumerate(csvs):
        print('Running trading algo for ' + dates[i])
        
        tickers = get_tickers(file_name=csv, path=path)
        
        print('Starting number of tickers: ' + str(len(tickers)))
        print('Filtering tickers...')
        
        # Apply filters
        price_filtered = price_filter(tickers, price_thresh, dates[i], path=path)
        pb_filtered = price_to_book_filter(price_filtered, 6, dates[i], path=path)
        de_filtered = debt_to_equity_filter(pb_filtered, 4, dates[i], path=path)
        filtered_tickers = pe_ratio_to_industry_filter(de_filtered, 0.85, dates[i], path=path)

        print('Number of tickers after filtering: ' + str(len(filtered_tickers)))

        # Run trading algo for this date
        transactions = run_trading_algo(
                                            filtered_tickers, portfolio, 
                                            time_averaged=True, 
                                            time_averaged_period=time_averaged_period,
                                            min_price_thresh=price_thresh, 
                                            buy_alpha=buy_alpha, 
                                            short_alpha=short_alpha,
                                            append_to_csv=True, 
                                            file_name=trading_algo_file_name, 
                                            path=path,
                                            clear_csv=clear_csv, 
                                            in_csv=True, 
                                            date=dates[i]
                                        )

        clear_csv = False
        portfolio = get_portfolio_from_csv(file_name=trading_algo_file_name, path=path)
        assert portfolio != {}

    assert os.path.exists(path + 'csv_files/trading_algos/' + trading_algo_file_name), "The trading algo file was not made. There must have been a problem."
    # See how we did
    compute_returns(filename=trading_algo_file_name, capital=500000, path=path)