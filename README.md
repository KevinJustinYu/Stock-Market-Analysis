# Stock-Market-Analysis
Tool that fetches company data, statistics, and financials for valuation and analysis, and then performs automated valuation analysis.

# Layout
The bulk of the data clobbering and scraping functionalities are defined in market.py.
ML models are trained through functions that are defined in market_ml.py
The ml_models folder stores any XGboost models that are trained. 
The csv_files folder stores csv files that contains company information that the ML models use. 
daily_update.bat is a windows batch file that can be used by windows task scheduler to automatically schedule daily updating of company statistics. 
