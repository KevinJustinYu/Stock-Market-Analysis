from market import *


def test_market():
	test_parse()
	test_get_summary_statistics()

def test_parse():
	companytests = ['AAPL', 'TSLA', 'BMY', 'SNE']
	expected_keys = ['Previous Close', 'Open', 'Bid', 'Ask', "Day's Range", '52 Week Range', 'Volume', 'Avg. Volume', 'Market Cap', 'Beta (3Y Monthly)', 'PE Ratio (TTM)', 'EPS (TTM)', 'Earnings Date', 'Forward Dividend & Yield', 'Ex-Dividend Date', '1y Target Est', 'EPS Beat Ratio', 'ticker', 'url']
	parsed = []
	for ticker in companytests:
		try:
			ticker_parsed = parse(ticker)
			# Check that we are getting all the expected keys
			if set(ticker_parsed.keys()) != set(expected_keys):
				print("Expected key values for dictionary returned by parse do not match with actual key values.")
				print("Expected: " + str(expected_keys))
				print("Actual: " + str(ticker_parsed.keys()))
			parsed.append(ticker_parsed)
		# If parsing fails then throw error
		except:
			print('Parsing ticker ' + ticker + ' failed.')

def test_get_summary_statistics():
	companytests = ['AAPL', 'TSLA', 'BMY', 'SNE']
	expected_keys = ['Market Cap (intraday)', 'Enterprise Value', 'Trailing P/E', 'Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales', 'Price/Book', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA', 'Beta (3Y Monthly)', '52-Week Change', 'S&P500 52-Week Change', '52 Week High', '52 Week Low', '50-Day Moving Average', '200-Day Moving Average', 'Avg Vol (3 month)', 'Avg Vol (10 day)', 'Shares Outstanding', 'Float', '% Held by Insiders', '% Held by Institutions', 'Shares Short (Sep 30, 2019)', 'Short Ratio (Sep 30, 2019)', 'Short % of Float (Sep 30, 2019)', 'Short % of Shares Outstanding (Sep 30, 2019)', 'Shares Short (prior month Aug 30, 2019)', 'Forward Annual Dividend Rate', 'Forward Annual Dividend Yield', 'Trailing Annual Dividend Rate', 'Trailing Annual Dividend Yield', '5 Year Average Dividend Yield', 'Payout Ratio', 'Dividend Date', 'Ex-Dividend Date', 'Last Split Factor (new per old)', 'Last Split Date', 'Fiscal Year Ends', 'Most Recent Quarter', 'Profit Margin', 'Operating Margin', 'Return on Assets', 'Return on Equity', 'Revenue', 'Revenue Per Share', 'Quarterly Revenue Growth', 'Gross Profit', 'EBITDA', 'Net Income Avi to Common', 'Diluted EPS', 'Quarterly Earnings Growth', 'Total Cash', 'Total Cash Per Share', 'Total Debt', 'Total Debt/Equity', 'Current Ratio', 'Book Value Per Share', 'Operating Cash Flow', 'Levered Free Cash Flow']
	parsed = []
	for ticker in companytests:
		try:
			ticker_parsed = get_summary_statistics(ticker)
			# Check that we are getting all the expected keys
			if set(ticker_parsed.keys()) != set(expected_keys):
				print("Expected key values for dictionary returned by get_summary_statistics do not match with actual key values.")
				print("Expected: " + str(expected_keys))
				print("Actual: " + str(ticker_parsed.keys()))
			parsed.append(ticker_parsed)
		# If parsing fails then throw error
		except:
			print('Getting stats for ticker ' + ticker + ' failed.')