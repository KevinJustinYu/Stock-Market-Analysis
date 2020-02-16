# Stock-Market-Analysis
Tool that fetches company data, statistics, and financials for valuation and analysis, and then performs automated valuation analysis. A trading algorithm is currently in development. 

# Usage
If you want to get data from a company, all you have to do is this: 
```python
from market import *

get_summary_statistics('AAPL') # This will output a dictionary
```

Check out market.py for many more options. 


If you want to analyze a company:
```python
from market import *

analyze('COST')
```



If you want to get the predicted price of a company using an XGboost model:

```python
from market_ml import *

predict_price('COST') # Predicts the price for the most recent day

predict_price_time_averages('COST') # Average predicted price over 5 days

```

# Layout

## market.py
The bulk of the data clobbering and scraping functionalities are defined in this file. 
If you want a function that will get you company data, it will be here.
This file also contains functions to write data to csv files locally. 

### market_ml.py
ML models are trained through functions that are defined in this file. This file contains functions that predict the price of stocks using XGboost regression. This file also contains analyze(), which returns useful information about a given ticker.

### market_trading.py
Trading features/functionality are defined in market_trading.py. This primarily consists of functions that make valuations and output transactions to make. In addition there is functionality to write and read trancasctions that are written to csv. 

### market_tests.py
This file contains the test suite for the functions defined in the python files.  

### /ml_models
The ml_models folder stores any XGboost models that are trained. 

### /csv_files
The csv_files folder stores csv files that contains company information that the ML models use. 

### daily.bat
daily.bat is a windows batch file that can be used by windows task scheduler to automatically schedule daily updating of company statistics. 

# Projects in Development

A webapp is being designed and built to display results. Users should be able to create a profile, input their portfolio, see automated insights + screened stocks that are undervalued/transaction recommendations.  

Infrastructure is being build for testing valuation methods. Daily models are being tested on historical data/prices. One goal is to see how long the horizon is for the actual prices to reach the predicted prices. Ideally we run the trading algorithm on historical prices and returns get outputted, so we can see how the algorithm compares to the market. 

A neural net is being implemented using PyTorch, to allow for continuous daily training rather than training on the entire market for a single day. This could potentially outperform XGBoost. 


# License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

