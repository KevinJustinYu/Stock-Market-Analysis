'''
This file stores functions for trading strategy/ trading algorithms
and the implementation of those algorithms. 
'''

# Imports 
from market_ml import *
import scipy.stats
import os


# TODO: The length of decisions does not equal the length of actual
def get_trade_deciders(tickers, time_averaged=False, time_averaged_period=5, thresh=25, 
                            buy_alpha=0.05, short_alpha=0.00001, min_price_thresh=10, verbose=1, path='',
                            in_csv=False, date_str=None):
    '''
    This function loops through tickers, makes price predictions, and then outputs decisions
    for each ticker. 
    Input:
            tickers: list of strings, representing company tickers available on yahoo finance
            
            time_average: Default = False. When set to true, will make predictions based on mean
            of multiple day predictions. This helps with mitigating daily noise.
            
            time_averaged_period: Default = 5. When time_average= True, this value is the number of
            days to average over.
            
            thresh: This is the percent value that is used to buy/sell stocks. Only when a stock is 
            undervalued or overvalued by thresh will the trade happen.
            
            min_price_thresh: Default = 10. Only makes trades on stocks that are worth more than this value.
    '''
    predictions = []
    actual = []
    decisions = [0] * len(tickers)
    for i, ticker in enumerate(tickers):
        #print('Getting decider for ' + ticker)
        if time_averaged:
            pred, stdev = predict_price_time_averaged(ticker, time_averaged_period, verbose=0, path=path, in_csv=in_csv)
            if pred == None: # if it doesnt work then skip
                print('Getting decider for ' + ticker + ' failed because price prediction failed. Skipping this ticker ...')
                continue
        else:
            if date_str != None:
                today_date = date_str
            else:
                today_date = str(datetime.date.today()) #- datetime.timedelta(1))
            try:
                model = get_model_from_date(today_date, path=path)
            except:
                model = train_and_get_model(path=path)
            pred = predict_price(ticker, model=model, in_csv=in_csv, path=path, date=today_date)
            if pred == None:
                print('Getting decider for ' + ticker + ' failed because price prediction failed. Skipping this ticker ...')
                continue

        # If ticker in csv, then dont call parse. Otherwise do so.
        if in_csv:
            if date_str == None:
                df = pd.read_csv(path + "csv_files/company_statistics.csv", encoding='cp1252')
            else:
                df = pd.read_csv(path + "csv_files/company_stats_" + date_str + ".csv", encoding='cp1252')
            summary = {"error":"Failed to parse json response"} # Default value for summary
            assert ticker in list(df['Ticker']), 'in_csv set to true, but could not find ' + ticker + ' in csv_files/company_statistics.csv' 
            if ticker in list(df['Ticker']):
                p = list(df[df.Ticker == ticker]['Price'])[0]
                assert isinstance(p, numbers.Number), 'Price is not numeric.'
                summary = {'Open': p}
        else: 
            summary = parse(ticker)
        
        # Handle the case where parsing fails
        if summary == {"error":"Failed to parse json response"}:
            actual.append(float('nan'))
            decisions[i] = float('nan')
            continue

        # Handle case when we can't get ticker price even when summary doesnt fail
        try:
            real = str_to_num(summary['Open'])
        except KeyError:
            actual.append(float('nan'))
            decisions[i] = float('nan')
            continue

        # Now we have pred and actual, so we can proceed with obtaining decider.
        predictions.append(pred)
        actual.append(real)

        # Handle case for when prediction fails
        if pred == -1:
            print('Warning: Predicted price for ' + ticker + ' is -1. Skipping this ticker...')
            decisions[i] = float('nan')
            continue

        # Handle the case for when the stock's price is under the thresh
        if real < min_price_thresh:
            decisions[i] = float('nan')
            if verbose == 1:
                print(ticker + "'s price is under the minimun price thresh of " + str(min_price_thresh))
            continue

        # Predicted and real prices have now been obtained and pass the price thresh

        # Get percent difference, negative means overvalued, positive means undervalued
        percent = str(round(abs(pred - real) / real * 100, 2)) + '%'
        
        # Run t-test if time averaged
        if time_averaged:
            n = time_averaged_period
            print('Calculating Decider for ' + ticker)
            # Calculate t-statistic
            t = (pred - real) / (stdev / np.sqrt(n))
            print('Predicted Price: ' + str(pred) + '. Actual Price: ' + str(real))
            print('t-statistic: ' + str(t))
            # The null hypoth. is that pred == actual
            critical_vals = [scipy.stats.t.ppf(short_alpha/2, n), scipy.stats.t.ppf(1 - buy_alpha/2, n)]
            # We claim stock is undervalued, we reject the null
            if t > critical_vals[1]:
                valuation = 'undervalued'
                assert pred > real, 'Predicted value is not greater than actual price but it is marked as undervalued'
                decisions[i] = (pred - real) / real * 100
                if verbose == 1:
                    print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
            elif t < critical_vals[0]:
                valuation = 'overvalued'
                assert pred < real, 'Predicted value is not less than actual price but it is marked as overvalued'
                decisions[i] = -1 * (pred - real) / real * 100
                if verbose == 1:
                    print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
            # We accept the null
            else:
                if verbose == 1:
                    print('The predicted value of ' + str(pred)
                        + ' for ' + ticker + 
                        ' is too close to actual price of ' + str(real) +
                        '. We assume correct valuation for the given alpha values.')
        
        # If not time averaged
        else: 
            # Handle the undervalued case 
            if pred - real > 0:
                valuation = 'undervalued'
                percent_undervalued = abs(pred - real) / real * 100
                if percent_undervalued > thresh:
                    decisions[i] = percent_undervalued
                    if verbose == 1:
                        print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
                elif verbose == 1:
                    print('The predicted value of ' + str(pred)
                        + ' for ' + ticker + 
                        ' is too close to actual price of ' + str(real) +
                        '. We assume correct valuation for the given alpha values.')
            # Handle the overvalued case 
            elif pred - real < 0:
                valuation = 'overvalued'
                percent_overvalued = abs(pred - real) / real * 100
                if percent_overvalued > thresh:
                    decisions[i] = -1 * percent_overvalued
                    if verbose == 1:
                        print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
                elif verbose == 1:
                    print('The predicted value of ' + str(pred)
                        + ' for ' + ticker + 
                        ' is too close to actual price of ' + str(real) +
                        '. We assume correct valuation for the given alpha values.')
    print(decisions)
    print()
    print()
    print()
    print(actual)
    assert len(decisions) == len(actual), 'The length of decisions does not match the length of actual.'
    return decisions, actual


def make_transactions(deciders, actual, tickers, portfolio, thresh=15, min_price_thresh=10):
    '''
    This function takes deciders generated from get_trade_deciders along with portfolio
    current portfolio information and outputs specific transactions to make. 
    '''
    transactions = [] # Each entry will be ticker, price, amount, sell/buy
    for i, ticker in enumerate(tickers):
        if actual[i] != float('nan'): # Actual prices will be 'nan' if ticker cant be parsed
            in_portfolio = False
            for position in portfolio.items():
                if ticker == position[0]:
                    in_portfolio = True
                    amount = position[1]
            # If the ticker is already in our portfolio, then just adjust holding
            if in_portfolio:
                if deciders[i] == 0:
                    if amount > 0:
                        transactions.append([ticker, actual[i], -amount, 'sell'])
                    else:
                        transactions.append([ticker, actual[i], -amount, 'buy'])
                else:
                    adjustment = deciders[i] - amount
                    # Nudge holding in the positive direction TODO
                    if adjustment > 0:
                        if amount < 0 and deciders[i] > 0:
                            transactions.append([ticker, actual[i], round(amount), 'cover short'])
                            transactions.append([ticker, actual[i], round(deciders[i]), 'buy'])
                        elif amount > 0 and adjustment > 0: # Nudge position long by buying
                            transactions.append([ticker, actual[i], round(adjustment), 'buy'])
                    # Nudge holding in the negative direction TODO
                    elif adjustment < 0:
                        if deciders[i] < 0 and amount > 0:
                            transactions.append([ticker, actual[i], round(amount), 'sell'])
                            transactions.append([ticker, actual[i], round(deciders[i]), 'short'])
                        elif amount < 0 and adjustment < 0:
                            transactions.append([ticker, actual[i], round(adjustment), 'short'])
            # If ticker is not in portfolio, buy or short stock
            else: 
                if deciders[i] > 0:
                    transactions.append([ticker, actual[i], round(deciders[i]), 'buy'])
                if deciders[i] < 0:
                    transactions.append([ticker, actual[i], round(deciders[i]), 'short'])
    return transactions


def write_transactions(transactions, file_name='transactions.csv', path=''):
    '''
    This function takes transactions outputted by make_transactions and 
    appends them to a csv. 
    '''
    assert os.path.exists(path + 'csv_files/trading_algos/' + file_name), 'The specidifed path does not exist for writing transactions: ' + path + 'csv_files/trading_algos/' + file_name
    with open(path + 'csv_files/trading_algos/' + file_name, 'a', newline='') as f:
        writer = csv.writer(f)
        today = str(datetime.date.today())
        for t in transactions:
            row = list(t)
            fields = row.insert(0, today)
            writer.writerow(row)


def run_trading_algo(tickers, portfolio, time_averaged=False,
                    time_averaged_period=3, thresh=15, min_price_thresh=10,
                    buy_alpha=0.05, short_alpha=0.00001,
                    verbose=1, path='', append_to_csv=False, file_name='transactions.csv', clear_csv=False,
                    in_csv=True, date=None):
    '''
    This algorithm takes a list of tickers to consider and an existing portfolio,
    and makes trades based on current valuation. 
    '''

    # Compute decisions
    decisions, actual = get_trade_deciders(tickers, time_averaged=time_averaged,
                                                   time_averaged_period=time_averaged_period,
                                                   thresh=thresh,
                                                   buy_alpha=buy_alpha,
                                                   short_alpha=short_alpha,
                                                   min_price_thresh=min_price_thresh,
                                                   path=path, 
                                                   in_csv=in_csv, date_str=date)

    # Get transactions from the decisions
    transactions = make_transactions(decisions, actual, tickers, portfolio)
    if verbose == 1:
        print('Here is a list of transactions that were made: ')
        print(transactions)

    # Clear csv and add header if specified
    if clear_csv:
        try:
            os.remove(path + 'csv_files/trading_algos/' + file_name)
        except:
            print('Clear CSV was set to true but the csv file with path ' + 
                path + 'csv_files/trading_algos/' + file_name + ' does not exist.')
        with open(path + 'csv_files/trading_algos/' + file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Ticker', 'Price', 'Amount', 'Action'])

    # Append the transactions to csv to store a log
    if append_to_csv:
        write_transactions(transactions, file_name=file_name, path=path)
    return transactions


def get_portfolio_from_csv(file_name='transactions.csv', path=''):
    '''
    Runs through a csv file storing transactions and gets the current portfolio.
    '''
    portfolio = {}
    try:
        with open(path + 'csv_files/trading_algos/' + file_name, 'r', newline='') as f:
            for line in f:
                transaction = line.strip().split(',')
                date_str, ticker, price, amount, action = transaction
                if action == 'buy':
                    portfolio[ticker] = float(amount)
                if action == 'sell':
                    portfolio[ticker] = -1 * float(amount)
                if action == 'no position':
                    del portfolio[ticker]
    except:
        assert len(portfolio.items()) > 0, 'No portfolio obtained at: ' + path + 'csv_files/trading_algos/' + file_name
        return portfolio

    assert len(portfolio.items()) > 0, 'No portfolio obtained at: ' + path + 'csv_files/trading_algos/' + file_name
    return portfolio


def compute_returns(filename='transactions.csv', capital=None, path=''):
    '''Runs through the csv file and computes the returns make on the transactions'''
    if capital == None:
        capital = 500000
    print('Starting amount: $' + str(capital))
    portfolio = {}
    with open(path + 'csv_files/trading_algos/' + filename, 'r', newline='') as f:
        for line in f:
            transaction = line.strip().split(',')
            date_str, ticker, price, amount, action = transaction
            if date_str == '': # Stop when we hit the end of the csv
                break
            # Convert to numeric, take the absolute value to avoid confusion
            price, amount = str_to_num(price), abs(str_to_num(amount))
            if action == 'buy':
                capital -= price * amount
                print('Buying '+ str(amount) + ' shares of '+
                 ticker + ' for $' + str(amount) + ', totalling $' + 
                 str(price * amount) + '. Capital is now $' + str(capital))
                if ticker not in portfolio:
                    portfolio[ticker] = [price, amount]
                else: 
                    # Buy more shares of company we already own
                    prev = portfolio[ticker]
                    # New price should be the average price
                    portfolio[ticker] = [prev[0] + ((price - prev[0]) / (prev[1] + amount)), prev[1] + amount]
            elif action == 'sell':
                capital += price * amount
                print('Selling '+ str(amount) + ' shares of '+
                 ticker + ' for $' + str(amount) + ', totalling $' + 
                 str(price * amount) + '. Capital is now $' + str(capital))
                assert ticker in portfolio.keys(), 'Cannot sell ' + ticker + ' because it is not in portfolio.'
                prev = portfolio[ticker]
                portfolio[ticker] = [prev[0], prev[1] - amount] 
            elif action == 'short':
                continue # Ignore shorts for now
                print('SHORTING ' + ticker)
                capital += price * amount
                if ticker not in portfolio:
                    portfolio[ticker] = [price, -1*amount]
                else:
                    # New price should be the average price
                    prev = portfolio[ticker]
                    portfolio[ticker] = [prev[0] + ((price - prev[0]) / abs(abs(prev[1]) - amount)), prev[1] - amount]
            elif action == 'cover short':
                continue # Ignore for now
                amount_shorted = portfolio[ticker][1] # This value should be negative
                assert amount_shorted > 0, 'Amount shorted for ' + ticker + ' is a positive value.'
                average_price = portfolio[ticker][0]
                assert average_price > 0, 'Average price for shorted ' + ticker + ' is not positive.'
                capital += (average_price - price) * amount
                assert ticker in portfolio.keys(), 'Cannot cover short for ' + ticker + ' because it is not in portfolio.'
                assert amount_shorted - amount >= 0, 'Cannot cover short for more shares than shorted.'
                if amount_shorted + amount != 0:
                    portfolio[ticker] = portfolio[average_price, amount_shorted + amount] # Addition here because amount is always positive
                else:
                    del portfolio[ticker]
                    assert ticker not in portfolio.keys()

    # Compute the current value of the portfolio.
    value = capital
    sum_of_returns = 0
    spent = 0
    for ticker, [av_price, amount] in portfolio.items():
        assert av_price > 0, 'Average price for ' + ticker + ' is 0.'
        assert amount > 0, 'Amount of shares of ' + ticker + ' is 0. Make sure it is removed from the portfolio.'

        try:
            cur_price = str_to_num(parse(ticker)['Open'])
            roi = (cur_price - av_price) / av_price
            print('Return on investment for ' + ticker + ' is ' + str(round(roi * 100, 2)) + '%, or $' + str(round((cur_price - av_price) * amount, 2)))
            sum_of_returns += (cur_price - av_price) * amount
            spent += av_price * amount
            value += cur_price * amount 
        except:
            print('Failed to find Open in parse dictionary for ' +  ticker)
    print('Value of portfolio: $' + str(round(value, 2)))
    percent_return = round(100 * (sum_of_returns) / spent, 2)
    print('Return on investment: ' + str(percent_return) + '%')
    print('Sum of returns: $' + str(round(sum_of_returns, 2)))


