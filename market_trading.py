'''
This file stores functions for trading strategy/ trading algorithms
and the implementation of those algorithms. 
'''

# Imports 
from market_tests import *
from market_ml import *
import numpy as np


def decide_transaction(tickers,  time_averaged=False, time_averaged_period=5, thresh=15, min_price_thresh=10):
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
			

	'''
    predictions = []
    actual = []
    decisions = [0] * len(tickers)
    i = 0
    model = train_and_get_model()
    for ticker in tickers:
        if time_averaged:
            pred = predict_price_time_averaged(ticker, time_averaged_period, verbose=0)
        else:
            pred = predict_price(ticker, model=model)
        summary = parse(ticker)
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
                    if pred - real > 0:
                        valuation = 'undervalued'
                        percent_undervalued = abs(pred - real) / real * 100
                        if percent_undervalued > thresh:
                            decisions[i] = round(percent_undervalued)
                    elif pred - real < 0:
                        valuation = 'overvalued'
                        percent_overvalued = abs(pred - real) / real * 100
                        if percent_overvalued > thresh:
                            decisions[i] = -1 * round(percent_overvalued)
                    percent = str(round(abs(pred - real) / real * 100, 2)) + '%'
                    print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')
                else:
                    print(ticker + "'s price is under the minimun price thresh of " + str(min_price_thresh))
        else: 
            actual.append(float('nan'))
        i += 1
    return decisions, actual, tickers
