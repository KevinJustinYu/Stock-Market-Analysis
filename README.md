# Stock-Market-Analysis
Tool that fetches company data, statistics, and financials for valuation and analysis, and then performs automated valuation analysis.

# Layout
The bulk of the data clobbering and scraping functionalities are defined in market.py.

ML models are trained through functions that are defined in market_ml.py

Backtesting.ipynb currently contains prediction analytics and the current version of the trading algorithm. 

The ml_models folder stores any XGboost models that are trained. 

The csv_files folder stores csv files that contains company information that the ML models use. 

daily_update.bat is a windows batch file that can be used by windows task scheduler to automatically schedule daily updating of company statistics. 

# TO DO

Design and build webapp to display results. Users should be able to create a profile, input their portfolio, see automated insights + screened stocks that are undervalued. 

Build methodology for testing. Test daily models on historic prices. See how long the horizon is for the actual prices to reach the predicted prices. Run trading stragegy on historic data and validate results. (IN PROGRESS)

Implement neural net using PyTorch, so allow for continuous daily training rather than training on the entire market for a single day. (IN PROGRESS)


# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

