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
    for each ticker, along with actual prices. Positive decisions = buy, negative = sell. 
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

        if time_averaged:
            pred, stdev = predict_price_time_averaged(ticker, time_averaged_period, verbose=0, path=path, in_csv=in_csv)
            
            # If it doesn't work then skip
            if pred == None:
                actual.append(float('nan'))
                print('Getting decider for ' + ticker + ' failed because price prediction failed. Skipping this ticker ...')
                continue
        # Get decider based on only one day of data
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
                actual.append(float('nan'))
                print('Getting decider for ' + ticker + ' failed because price prediction failed. Skipping this ticker ...')
                continue

        # If ticker in csv, then dont call parse 
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
        # Otherwise call parse
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
                    assert decisions[i] < 0, 'This decider should be negative because ' + ticker + ' is overvalued.'
                    if verbose == 1:
                        print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
                elif verbose == 1:
                    print('The predicted value of ' + str(pred)
                        + ' for ' + ticker + 
                        ' is too close to actual price of ' + str(real) +
                        '. We assume correct valuation for the given alpha values.')

        if len(actual) != i + 1:
            print('Warning! len(actual) != i+1 for ' + ticker + '. Adding nan for actual[i] for this ticker')
            actual.append(float('nan'))
        assert len(actual) == i + 1, 'Actual did not get appended for ' + ticker

    assert len(decisions) == len(actual), 'The length of decisions does not match the length of actual.'
    return decisions, actual


def make_transactions(deciders, actual, tickers, portfolio, thresh=15, min_price_thresh=10):
    '''
    This function takes deciders generated from get_trade_deciders along with portfolio
    current portfolio information and outputs specific transactions to make. 
    '''

    transactions = [] # Each entry will be ticker, price, amount, sell/buy
    
    # For each ticker compute transactions
    for i, ticker in enumerate(tickers):
        if actual[i] != float('nan'): # Actual prices will be 'nan' if ticker cant be parsed
            in_portfolio = False
            
            # Get the amount already in portfolio, if any
            if ticker in portfolio.keys():
                
                if portfolio[ticker] == 0:
                    del portfolio[ticker]
                else:
                    in_portfolio = True
                    amount = portfolio[ticker]
            
            # If the ticker is already in our portfolio, then just adjust holding
            if in_portfolio:
                if round(deciders[i]) == 0:
                    if amount > 0:
                        transactions.append([ticker, actual[i], -amount, 'sell'])
                        del portfolio[ticker]
                    else:
                        transactions.append([ticker, actual[i], -amount, 'buy'])
                        del portfolio[ticker]
                else:
                    adjustment = deciders[i] - amount
                    assert isinstance(deciders[i], numbers.Number), 'deciders[i] not a number.'
                    assert isinstance(amount, numbers.Number), 'amount not a number.'
                    assert isinstance(adjustment, numbers.Number), 'adjustment not a number.'

                    # Progress if we need to make an adjustment
                    if adjustment != 0:
                        # We shorted the stock but we want to buy it so we cover short and buy
                        if amount < 0 and deciders[i] > 0:
                            transactions.append([ticker, actual[i], abs(round(amount)), 'cover short'])
                            transactions.append([ticker, actual[i], abs(round(deciders[i])), 'buy'])
                            portfolio[ticker] = round(deciders[i])
                            assert portfolio[ticker] != 0, 'Should not be 0. Check logic.'
                        
                        # We own stock but want to buy more
                        elif amount > 0 and adjustment > 0 and deciders[i] > 0: # Nudge position long by buying
                            transactions.append([ticker, actual[i], abs(round(adjustment)), 'buy'])
                            portfolio[ticker] = round(deciders[i])
                            assert portfolio[ticker] != 0, 'Should not be 0. Check logic.'
                            
                        # We own stock but we want to short
                        elif deciders[i] < 0 and amount > 0:
                            transactions.append([ticker, actual[i], abs(round(amount)), 'sell'])
                            transactions.append([ticker, actual[i], abs(round(deciders[i])), 'short'])
                            portfolio[ticker] = round(deciders[i])
                            assert portfolio[ticker] != 0, 'Should not be 0. Check logic.'
                        
                        # We have shorted this stock but want to short less
                        elif amount < 0 and adjustment < 0 and deciders[i] < 0:
                            transactions.append([ticker, actual[i], abs(round(adjustment)), 'cover short'])
                            portfolio[ticker] = round(deciders[i])
                            assert portfolio[ticker] != 0, 'Should not be 0. Check logic.'

                        # We have shorted this stock but want to short more
                        elif amount < 0 and adjustment > 0 and deciders[i] < 0:
                            transactions.append([ticker, actual[i], abs(round(adjustment)), 'short'])
                            portfolio[ticker] = round(deciders[i]) # Add more negative value
                            assert portfolio[ticker] != 0, 'Should not be 0. Check logic.'

                        # We own the stock but want to own less of it
                        elif adjustment < 0 and amount > 0 and deciders[i] > 0:
                            transactions.append([ticker, actual[i], abs(round(adjustment)), 'sell'])
                            portfolio[ticker] = round(deciders[i])
                            assert portfolio[ticker] != 0, 'Should not be 0. Check logic.'
                        
                        # Anything that goes here was unhandled, we missed a case
                        else:
                            print('Case was not handled! Amount = ' + str(amount) + '. Adjustment = ' + str(adjustment) + '. deciders[i] = ' + str(deciders[i]) + '.')

                        assert portfolio[ticker] != 0, 'The portfolio value for this ticker should not be zero, check logic above.'

            # If ticker is not in portfolio, buy or short stock, and add to portfolio
            else: 
                if deciders[i] > 0:
                    transactions.append([ticker, actual[i], abs(round(deciders[i])), 'buy'])
                    assert deciders[i] > 0
                    portfolio[ticker] = round(deciders[i])

                if deciders[i] < 0:
                    transactions.append([ticker, actual[i], abs(round(deciders[i])), 'short'])
                    assert round(deciders[i]) < 0
                    portfolio[ticker] = round(deciders[i]) # Deciders[i] would be negative here
    return transactions


def write_transactions(transactions, file_name='transactions.csv', path=''):
    '''
    This function takes transactions outputted by make_transactions and 
    appends them to a csv. 
    '''
    assert os.path.exists(path + 'csv_files/trading_algos/' + file_name), 'The specified path does not exist for writing transactions: ' + path + 'csv_files/trading_algos/' + file_name
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

    # Choose starting capital, defaults to 500k
    if capital == None:
        capital = 500000
    print('Starting amount: $' + str(capital))

    # Initialize portfolio, which will be used to compute returns 
    portfolio = {}

    # Make sure the csv for trading algo exists
    assert os.path.exists(path + 'csv_files/trading_algos/' + filename), 'The speciifed trading algo transaction csv file could not be found at: ' + path + 'csv_files/trading_algos/' + filename
    
    # Access the proper trading algo csv file
    with open(path + 'csv_files/trading_algos/' + filename, 'r', newline='') as f:
        # Go through each transaction
        for line in f:

            # Parse transaction
            transaction = line.strip().split(',')
            date_str, ticker, price, amount, action = transaction

            # Stop when we hit the end of the csv
            if date_str == '': 
                break

            # Convert to numeric, take the absolute value to avoid confusion
            price, amount = str_to_num(price), abs(str_to_num(amount))

            # Temporarily don't handle shorts or covering shorts
            #assert action != 'short' and action != 'cover short' 

            # Handle each case (buy, sell, short, cover short)
            if action == 'buy' and amount != 0:
                # Subtract the money from buying 
                capital -= price * amount
                print('Buying '+ str(amount) + ' shares of '+
                 ticker + ' for $' + str(price) + ', totalling $' + 
                 str(price * amount) + '. Capital is now $' + str(capital))

                # Add the correct amount of shares of ticker to the portfolio
                if ticker not in portfolio:
                    portfolio[ticker] = [price, amount]
                else: 
                    # Buy more shares of company we already own
                    prev = portfolio[ticker]
                        
                    # New price should be the average price
                    portfolio[ticker] = [prev[0] + ((price - prev[0]) / (prev[1] + amount)), prev[1] + amount]

            elif action == 'sell' and amount != 0:
                # Add the money from selling
                capital += price * amount
                print('Selling '+ str(amount) + ' shares of '+
                 ticker + ' for $' + str(price) + ', totalling $' + 
                 str(price * amount) + '. Capital is now $' + str(capital))
                assert ticker in portfolio.keys(), 'Cannot sell ' + ticker + ' because it is not in portfolio.'
                
                # Adjust the amount of shares in portfolio
                prev = portfolio[ticker]
                if prev[1] - amount == 0:
                    del portfolio[ticker]
                else:
                    portfolio[ticker] = [prev[0], prev[1] - amount]

            elif action == 'short' and amount != 0:
                print('SHORTING ' + ticker)
                capital += price * amount
                if ticker not in portfolio:
                    portfolio[ticker] = [price, -1*amount]
                else:
                    # New price should be the average price
                    prev = portfolio[ticker]
                    portfolio[ticker] = [prev[0] + ((price - prev[0]) / abs(abs(prev[1]) - amount)), prev[1] - amount]
                    
            elif action == 'cover short' and amount != 0:
                amount_shorted = portfolio[ticker][1] # This value should be negative
                assert amount_shorted > 0, 'Amount shorted for ' + ticker + ' is a positive value.'
                average_price = portfolio[ticker][0]
                assert average_price > 0, 'Average price for shorted ' + ticker + ' is not positive.'
                capital += (average_price - price) * amount
                assert ticker in portfolio.keys(), 'Cannot cover short for ' + ticker + ' because it is not in portfolio.'
                assert amount_shorted - amount >= 0, 'Cannot cover short for more shares than shorted.'
                if amount_shorted + amount != 0:
                    portfolio[ticker] = [average_price, amount_shorted + amount] # Addition here because amount is always positive
                else:
                    del portfolio[ticker]
                    assert ticker not in portfolio.keys()

    # Compute the current value of the portfolio.
    value = capital
    sum_of_returns = 0
    spent = 0
    for ticker, [av_price, amount] in portfolio.items():
        assert av_price > 0, 'Average price for ' + ticker + ' is 0.'
        # The assertion below failed for this input: backtest_algo('2020-01-28', 50, 'algo_test_3_a005_b0001.csv', buy_alpha=0.005, short_alpha=0.0001, price_thresh=15, time_averaged_period=5, path='')
        # AssertionError: Amount of shares of YNDX is 0. Make sure it is removed from the portfolio.
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
