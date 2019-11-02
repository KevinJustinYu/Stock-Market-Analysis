# Stock-Market-Analysis
Tool that fetches company data, statistics, and financials for valuation and analysis, and then performs automated valuation analysis. A trading algorithm is currently in development. 

# Layout
The bulk of the data clobbering and scraping functionalities are defined in market.py.

ML models are trained through functions that are defined in market_ml.py

Trading features/functionality are defined in market_trading.py. This primarily consists of functions that make valuations and output transactions to make. 

Backtesting.ipynb currently contains prediction analytics and the current version of the trading algorithm.

market_tests.py contains the test suite for the python functions defined in the python files.

Testing.ipynb is used to run the test suite to check for bugs. 

The ml_models folder stores any XGboost models that are trained. 

The csv_files folder stores csv files that contains company information that the ML models use. 

daily_update.bat is a windows batch file that can be used by windows task scheduler to automatically schedule daily updating of company statistics. 

# Projects in Development

A webapp is being designed and built to display results. Users should be able to create a profile, input their portfolio, see automated insights + screened stocks that are undervalued/transaction recommendations.  

Infrastructure is being build for testing valuation. Daily models are being tested on historical data/prices. One goal is to see how long the horizon is for the actual prices to reach the predicted prices. Ideally we run the trading algorithm on historical prices and returns get outputted, so we can see how the algorithm compares to the market. 

A neural net is being implemented using PyTorch, to allow for continuous daily training rather than training on the entire market for a single day. This could potentially outperform XGBoost. 


# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

