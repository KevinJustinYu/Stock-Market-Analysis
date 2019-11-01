'''
This file stores functions for trading strategy/ trading algorithms
and the implementation of those algorithms. 
'''

# Imports 
from market_ml import *
import scipy.stats


def get_trade_deciders(tickers, time_averaged=False, time_averaged_period=5, thresh=15, alpha=0.05, min_price_thresh=10, verbose=1):
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
    model = train_and_get_model()
    for i, ticker in enumerate(tickers):
        if time_averaged:
            pred, stdev = predict_price_time_averaged(ticker, time_averaged_period, verbose=0)

        else:
            pred = predict_price(ticker, model=model)
        summary = parse(ticker)

        # Continue if parsing succeeds
        if summary != {"error":"Failed to parse json response"}:
            try:
                real = float(summary['Open'])
            except KeyError:
                i += 1
                continue
            predictions.append(pred)
            actual.append(real)
            if pred != -1:
                if real >= min_price_thresh:
                    percent = str(round(abs(pred - real) / real * 100, 2)) + '%'
                    # Run t-test if time averaged
                    if time_averaged:
                        alpha = 0.05 # Make this tunable?
                        n = time_averaged_period
                        # Calculate t-statistic
                        t = (pred - real) / (stdev / np.sqrt(n))
                        # The null hypoth. is that pred == actual
                        critical_vals = [scipy.stats.t.ppf(alpha/2, n), 
                                        scipy.stats.t.ppf(1 - (alpha/2), n)]
                        # We claim stock is undervalued, we reject the null
                        if t < critical_vals[0]:
                            valuation = 'undervalued'
                            decisions[i] = (pred - real) / real * 100
                            if verbose == 1:
                                print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
                        elif t > critical_vals[1]:
                            valuation = 'overvalued'
                            decisions[i] = (pred - real) / real * 100
                            if verbose == 1:
                                print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
                        # We accept the null
                        else:
                            if verbose == 1:
                                print('The predicted value of ' + str(pred)
                                    + ' for ' + ticker + 
                                    ' is too close to actual price of ' + str(actual) +
                                    '. We assume correct valuation at alpha = ' + str(alpha))
                    else: 
                        if pred - real > 0:
                            valuation = 'undervalued'
                            percent_undervalued = abs(pred - real) / real * 100
                            if percent_undervalued > thresh:
                                decisions[i] = percent_undervalued
                        elif pred - real < 0:
                            valuation = 'overvalued'
                            percent_overvalued = abs(pred - real) / real * 100
                            if percent_overvalued > thresh:
                                decisions[i] = -1 * percent_overvalued
                        if verbose == 1:
                            print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
                else:
                    if verbose == 1:
                        print(ticker + "'s price is under the minimun price thresh of " + str(min_price_thresh))
        else: 
            actual.append(float('nan'))
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
            # If the ticker is already in our portfolio, then just adjust holding
            if in_portfolio:
                if deciders[i] == 0:
                    if position > 0:
                        transactions.append([ticker, actual[i], -position, 'sell'])
                    else:
                        transactions.append([ticker, actual[i], -position, 'buy'])
                else:
                    adjustment = deciders[i] - position
                    # Nudge holding in the positive direction TODO
                    if adjustment > 0: 
                        transactions.append([ticker, actual[i], round(adjustment), 'buy'])
                    # Nudge holding in the negative direction TODO
                    elif adjustment < 0:
                        if deciders[i] < 0:
                            transactions.append([ticker, actual[i], round(adjustment), 'sell'])
                            transactions.append([ticker, actual[i], round(adjustment), 'short'])
                        transactions.append([ticker, actual[i], round(adjustment), 'sell'])
            # If ticker is not in portfolio, buy or short stock
            else: 
                if deciders[i] > 0:
                    transactions.append([ticker, actual[i], round(deciders[i]), 'buy'])
                if deciders[i] < 0:
                    transactions.append([ticker, actual[i], round(deciders[i]), 'short'])
    return transactions


def write_transactions(transactions):
    '''
    This function takes transactions outputted by make_transactions and 
    appends them to a csv. 
    '''
    with open('csv_files/trading_algos/transactions.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        today = str(date.today())
        for t in transactions:
            row = list(t)
            fields = row.insert(0, today)
            writer.writerow(row)


def run_trading_algo(tickers, portfolio, time_averaged=False,
                    time_averaged_period=5, thresh=15, min_price_thresh=10,
                    verbose=True, append_to_csv=True):
    '''
    This algorithm takes a list of tickers to consider and an existing portfolio,
    and makes trades based on current valuation. 
    '''
    # Compute decisions
    decisions, actual = get_trade_deciders(tickers, time_averaged=time_averaged,
                                                   time_averaged_period=time_averaged_period,
                                                   thresh=thresh,
                                                   min_price_thresh=min_price_thresh)
    # Get transactions from the decisions
    transactions = make_transactions(decisions, actual, tickers, portfolio)
    if verbose:
        print('Here is a list of transactions that were made: ')
        print(transactions)
    # Append the transactions to csv to store a log
    if append_to_csv:
        write_transactions(transactions)
    return transactions


def get_portfolio_from_csv(filename='transactions.csv'):
    '''
    Runs through a csv file storing transactions and gets the current portfolio.
    '''
    portfolio = {}
    with open('csv_files/trading_algos/' + filename, 'r', newline='') as f:
        for line in f:
            transaction = line.strip().split(',')
            date, ticker, price, amount, action = transaction
            if action == 'buy':
                portfolio[ticker] = float(amount)
            if action == 'sell':
                portfolio[ticker] = -1 * float(amount)
            if action == 'no position':
                del portfolio[ticker]
    return portfolio