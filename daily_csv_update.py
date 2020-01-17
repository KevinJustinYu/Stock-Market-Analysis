from market_tests import *
from market_screener import *
from datetime import timedelta

# Get today's date as a string
today = str(datetime.date.today())

# ---------- Procedure to update csv ----------
fname = 'company_stats_' + today + '.csv'
update_csv(csv_name=fname)

with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_statistics.csv', 'w', newline='') as dest:   
    writer = csv.writer(dest)
    with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_stats_' + today + '.csv', 'r', newline='') as source:
        reader = csv.reader(source)
        for row in reader:
            writer.writerow(row) 


# ---------- Run trading algorithm ----------
path = 'C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/'
tickers = list(pd.read_csv(path + 'csv_files/company_statistics.csv', encoding='cp1252')['Ticker']) #list(get_tickers(path=path))
#selection = [tickers[i] for i in range(len(tickers)) if i % 100 == 0] # Test trading
print('Getting industry averages...')
industry_averages = get_industry_averages()

# Filter tickers to speed up 
print('Starting number of tickers: ' + str(len(tickers)))
print('Filtering tickers...')
filtered_tickers = pe_ratio_to_industry_filter(tickers, 0.9, today, industry_averages=industry_averages, path=path)
print('Number of tickers after filtering: ' + str(len(filtered_tickers)))

portfolio = {} # get_portfolio_from_csv(file_name='transactions_a05_b001.csv', path=path)
print('Current Portfolio: ')
print(portfolio)
# Time averaged is temporarily set to false becuase csv features have changed
transactions = run_trading_algo(filtered_tickers, portfolio, time_averaged=True, time_averaged_period=3,
                                min_price_thresh=10, buy_alpha=0.01, short_alpha=0.001,
                                append_to_csv=True, file_name='transactions_a01_b001.csv', path=path,
                                clear_csv=True, in_csv=True, date=today)