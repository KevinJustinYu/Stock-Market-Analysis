from market_tests import *
from datetime import date, timedelta

yesterday = date.today() #- timedelta(1)
fname = 'company_stats_' + str(yesterday) + '.csv'
update_csv(csv_name=fname)

with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_statistics.csv', 'w', newline='') as dest:   
    writer = csv.writer(dest)
    with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_stats_' + str(yesterday) + '.csv', 'r', newline='') as source:
        reader = csv.reader(source)
        for row in reader:
            writer.writerow(row) 

path = 'C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/'
tickers = list(get_tickers(path=path))
#selection = [tickers[i] for i in range(len(tickers)) if i % 100 == 0] # Test trading
portfolio = get_portfolio_from_csv(file_name='transactions_alpha05_00001.csv', path=path)
transactions = run_trading_algo(tickers, portfolio, time_averaged=True, time_averaged_period=5,
                                min_price_thresh=10, buy_alpha=0.05, short_alpha=0.00001,
                                append_to_csv=True, file_name='transactions_alpha0_05_0_00001.csv', path=path,
                                clear_csv=True)