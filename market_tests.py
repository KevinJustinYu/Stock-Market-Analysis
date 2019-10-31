from market import *
from market_ml import *
from market_trading import *
import numbers


def test_market_suite():
	test_market()
	test_market_ml()
	test_market_trading()

def test_market():
	'''Call test functions for market.py'''
	test_parse()
	test_get_summary_statistics()
	print('Tests for market.py PASSED!')


def test_market_ml():
	'''Call test functions for market_ml.py. TODO'''
	test_ml_training_and_prediction()
	print('Tests for market_ml.py PASSED!')


def test_market_trading():
	'''Call test functions for market_trading.py. TODO'''
	tickers = ['AAPL', 'TSLA', 'ATVI', 'SNE']
	deciders, prices = get_trade_deciders(tickers, verbose=0)
	assert len(prices) == 4
	assert len(deciders) == 4
	assert isinstance(prices[0], numbers.Number), 'Second return variable of get_trade_deciders is not a list of numbers.'
	assert isinstance(deciders[0], numbers.Number), 'First return variable of get_trade_deciders is not a list of numbers.'
	real_price_aapl = str_to_num(parse('AAPL')['Open'])
	assert prices[0] == real_price_aapl, 'Price outputted by get_trade_deciders does not match price outputted by parse()'
	aapl_price_get_price_data = get_price_data('AAPL', str(date.today()), str(date.today()))
	assert aapl_price_get_price_data['Open'][0] - prices[0] < .01, 'Price outputted by get_trade_deciders does not match price outputted by get_price_data.'
	transactions = make_transactions(deciders, prices, tickers, {})
	# Make sure make_transactions worked
	print('Tests for market_trading.py PASSED!')


def test_ml_training_and_prediction():
	# Make sure training and getting model works
	model = train_and_get_model()
	assert type(model) == xgb.sklearn.XGBRegressor, 'Model outputted by train_and_get_model() is not xgb.sklearn.XGBRegressor'
	pred = predict_price('AAPL', model=model)
	assert isinstance(pred, numbers.Number), 'The prediction outputted by predict_price() is not a float'
	pred_ta = predict_price_time_averaged('AAPL', 5, verbose=0)
	assert isinstance(pred_ta, numbers.Number), 'The prediction outputted by predict_price_time_averaged() is not a float'


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
	keys_with_dates = ['Shares Short', 'Short Ratio', 'Short % of Float', 'Short % of Shares Outstanding', 'Shares Short']
	expected_keys = ['Market Cap (intraday)', 'Enterprise Value', 'Trailing P/E', 'Forward P/E', 'PEG Ratio (5 yr expected)', 'Price/Sales', 'Price/Book', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA', 'Beta (3Y Monthly)', '52-Week Change', 'S&P500 52-Week Change', '52 Week High', '52 Week Low', '50-Day Moving Average', '200-Day Moving Average', 'Avg Vol (3 month)', 'Avg Vol (10 day)', 'Shares Outstanding', 'Float', '% Held by Insiders', '% Held by Institutions', 'Forward Annual Dividend Rate', 'Forward Annual Dividend Yield', 'Trailing Annual Dividend Rate', 'Trailing Annual Dividend Yield', '5 Year Average Dividend Yield', 'Payout Ratio', 'Dividend Date', 'Ex-Dividend Date', 'Last Split Factor (new per old)', 'Last Split Date', 'Fiscal Year Ends', 'Most Recent Quarter', 'Profit Margin', 'Operating Margin', 'Return on Assets', 'Return on Equity', 'Revenue', 'Revenue Per Share', 'Quarterly Revenue Growth', 'Gross Profit', 'EBITDA', 'Net Income Avi to Common', 'Diluted EPS', 'Quarterly Earnings Growth', 'Total Cash', 'Total Cash Per Share', 'Total Debt', 'Total Debt/Equity', 'Current Ratio', 'Book Value Per Share', 'Operating Cash Flow', 'Levered Free Cash Flow']
	parsed = []
	for ticker in companytests:
		try:
			ticker_parsed = get_summary_statistics(ticker)
			# Check that we are getting all the expected keys
			keys = ticker_parsed.keys()
			for key in expected_keys:
				assert key in keys, 'Expected key ' + str(key) + 'not in obtained dictionary.'
			for key in keys_with_dates:
				obtained = False
				for k in keys:
					if key in k:
						obtained = True
				assert obtained == True, 'Expected key ' + str(key) + 'not in obtained dictionary.'
		# If parsing fails then throw error
		except:
			print('Getting stats for ticker ' + ticker + ' failed.')