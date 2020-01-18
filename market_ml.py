

# Imports
from market import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle as pkl
from sklearn import metrics 
import pickle
import datetime
from pandas_datareader import data
import numbers
from pprint import pprint
import warnings
#import robin_stocks

def get_model_from_date(date, verbose=0, path=''):
    '''
    Gets the XGB model from date, first by checking for the pretrained 
    model in ml_models, then then checking for the csv and training the model
    from scratch. If the csv is not located, we print an error.
    '''

    # First, check the ml_models folder for the correponding model
    model_string = 'xgbr_' + date + '.dat'
    try:
        model = pkl.load(open(path + 'ml_models/' + model_string, "rb"))
        return model
    except:
        print('The model could not be found at: ' + 'ml_models/' + model_string)

    # Second, see if we have a csv for the date, and train a model for that csv
    csv_string = 'company_stats_' + date + '.csv'
    try:
        model = train_and_get_model(filename=csv_string, verbose=0, save_to_file=True, saved_model_name=model_string, path=path)
        return model
    except:
        print('Could not train model for the data located in ' + csv_string + '. Check that this file exists.')
    warnings.warn('get_model_from_date(' + date + ') is getting called but date is invalid', RuntimeWarning)


def train_and_get_model(filename='company_statistics.csv', verbose=0, save_to_file=False, saved_model_name='xgbr_latest.dat', path=''):
    '''
    Given a csv file (defaults to company_statistics.csv), trains an XGBoost model and saves the model 
    to saved_model_name (defaults to xgbr_latest.dat).
    '''
    if verbose != 0:
        print('Training XGB model with hyperparameter tuning... Make sure csv is updated.')

    # Extract the data from the csv as a dataframe
    financial_data = pd.read_csv(path + "csv_files/" + filename, encoding='cp1252')
    # Clobber the data (remove uneccesary columns)
    to_remove = ['Ticker', 'Price']
    categoricals = ['Sector', 'Industry']
    feature_cols = [x for x in financial_data.columns if x not in to_remove]
    X = financial_data[feature_cols]

    attributes = ['Market Cap (intraday)','Trailing P/E','Forward P/E','PEG Ratio (5 yr expected)','Price/Sales','Price/Book',
                  'Enterprise Value/Revenue','Enterprise Value/EBITDA','Profit Margin','Operating Margin',
                  'Return on Assets','Return on Equity','Revenue','Revenue Per Share',
                  'Quarterly Revenue Growth','Gross Profit','EBITDA','Diluted EPS', 'EPS Beat Ratio',
                  'Quarterly Earnings Growth','Total Cash','Total Cash Per Share','Total Debt',
                  'Total Debt/Equity','Current Ratio','Book Value Per Share','Operating Cash Flow',
                  'Levered Free Cash Flow','Beta (3Y Monthly)','Shares Outstanding','Forward Annual Dividend Rate',
                  'Forward Annual Dividend Yield','Trailing Annual Dividend Rate','Trailing Annual Dividend Yield',
                  '5 Year Average Dividend Yield','Payout Ratio', 'Net Income Avi to Common', 'Enterprise Value']

    assert set(attributes) == set(feature_cols).difference(set(categoricals))

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categoricals)
    Y = financial_data['Price']

    # Get training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

    # Hyperparameter tune an XGBoost model
    param_test = {
        'max_depth':[3],
        'min_child_weight':[4],
        'learning_rate':[.25],
        'gamma':[0],
        'reg_alpha':[ 0.1, .12, .14]
    }
    xgbr = xgb.XGBRegressor(objective='reg:squarederror') 
    gsearch = GridSearchCV(estimator = xgbr , 
    param_grid = param_test,n_jobs=4,iid=False, cv=5)
    
    gsearch.fit(X_train, y_train)

    if verbose != 0:
        print("Test Score: " + str(gsearch.best_estimator_.score(X_test, y_test)))
    model = gsearch.best_estimator_
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    if verbose != 0:
        print("RMSE: %f" % (rmse))

    # Save model to file if desired
    if save_to_file:
        save_model(model, name=saved_model_name)
    return model


def save_model(model, name=None):
    '''
    Saves XGBoost model in C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/ml_models/ .
    File name defaults to xgbr_latest.dat if not specified.    
    '''
    if name == None:
        name = 'xgbr_latest.dat'
    # save model to file
    pkl.dump(model, open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/ml_models/' + name, 'wb'))
    today = datetime.date.today()
    pkl.dump(model, open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/ml_models/xgbr_' + 
        str(today) + '.dat', 'wb'))


def predict_price(ticker, model=None, model_type='xgb', verbose=0, path='', in_csv=False, date=None): # Next Step: Compareto actual price and output how much its overvalued or undervalued by
    attributes = ['Market Cap (intraday)','Trailing P/E','Forward P/E','PEG Ratio (5 yr expected)','Price/Sales','Price/Book',
                  'Enterprise Value/Revenue','Enterprise Value/EBITDA','Profit Margin','Operating Margin',
                  'Return on Assets','Return on Equity','Revenue','Revenue Per Share',
                  'Quarterly Revenue Growth','Gross Profit','EBITDA','Diluted EPS', 'EPS Beat Ratio',
                  'Quarterly Earnings Growth','Total Cash','Total Cash Per Share','Total Debt',
                  'Total Debt/Equity','Current Ratio','Book Value Per Share','Operating Cash Flow',
                  'Levered Free Cash Flow','Beta (3Y Monthly)','Shares Outstanding','Forward Annual Dividend Rate',
                  'Forward Annual Dividend Yield','Trailing Annual Dividend Rate','Trailing Annual Dividend Yield',
                  '5 Year Average Dividend Yield','Payout Ratio', 'Net Income Avi to Common', 'Enterprise Value']
    # Check that arguments are valid
    if in_csv == True:
        assert date != None
        assert "company_stats_" + date + ".csv" in os.listdir(path + "csv_files/"), 'Could not find the specified csv file for ' + date
    elif date != None:
        print('Warning: in_csv is set to False but date has been specified. date will not be used')

    if model == None and verbose != 0:
        print('This instance of predict price for ' + ticker + ' is getting called with model = None')
    elif verbose != 0:
        print('This instance of predict price for ' + ticker + ' is getting called with a specified model. Check that model is valid.')
    
    # Get financial data
    if date == None:
        financial_data = pd.read_csv(path + "csv_files/company_statistics.csv", encoding='cp1252')
    else:
        financial_data = pd.read_csv(path + "csv_files/company_stats_" + date + ".csv", encoding='cp1252')
    
    to_remove = ['Price']
    categoricals = ['Sector', 'Industry']
    feature_cols = [x for x in financial_data.columns if x not in to_remove]
    X = financial_data[feature_cols]
    
    # One hot encode categorical columns
    X = pd.get_dummies(X, columns=categoricals)
    if in_csv:
        row_of_interest = X[X.Ticker == ticker]
        del row_of_interest['Ticker']
    del X['Ticker']

    # Update attributes with the new column names
    attributes = X.columns
    assert 'Beta (3Y Monthly)' in X.columns

    if model_type != 'xgb':
        financial_data = financial_data.fillna(-1)

    x = [] # Data point we predict

    # Fill x with scraped data
    if in_csv == False:
        stats = get_summary_statistics(ticker)
        summary = parse(ticker)
        # TODO: try, except for sector, industry maybe?
        sector, industry = get_sector_industry(ticker)
        if 'error' in summary.keys() or 'error' in stats.keys():
            return -1
        eps_beat_ratio = summary["EPS Beat Ratio"]

        for a in attributes:
            try:
                if a == 'EPS Beat Ratio': # Handle the case with beat ratio because not included in summary stats
                    x.append(str_to_num(eps_beat_ratio))
                elif 'Sector' in a:
                    if a == 'Sector_' + sector:
                        x.append(1)
                    else:
                        x.append(0)
                elif 'Industry' in a:
                    if a == 'Industry_' + industry:
                        x.append(1)
                    else:
                        x.append(0)
                else:
                    x.append(str_to_num(stats[a]))
            except: # If One of the features is not in the parsed dictionary, then use float('nan')
                x.append(float('nan'))
        assert len(x) == X.shape[1]

        X = pd.DataFrame([x], columns=list(X.columns))
        assert X.shape == (1, len(list(X.columns)))
        assert X.shape[0] > 0, 'Could not find ' + ticker + ' in the csv. Try again with in_csv set to False'
    else: # When in_csv, no need to call get_summary_statistics and parse functions
        assert len(X.columns) == len(row_of_interest.columns)
        X = row_of_interest
        if X.shape[0] == 0:
            print('In predic_price function. Could not find ' + ticker + ' in the csv. Make sure csv is updated properly. Returning None for this ticker.')
            return None
        #assert X.shape[0] > 0, 'Could not find ' + ticker + ' in the csv. Try again with in_csv set to False'
    
    assert len(X.columns) == len(attributes), 'Training Data Features: ' + str(list(X.columns)) + '. Attributes: ' + str(list(attributes)) + '.'

    # For now, enforce that model is specified
    assert model != None, 'Model not specified'
    # TODO: use default model, but make sure it is up to date
    # If the model is not specified, use default model
    #if model == None:
        #if verbose != 0:
            #print('Using last saved model.')
        #model = pkl.load(open(path + "ml_models/xgbr_latest.dat", "rb"))

    # If model is loaded from pickle feature names get messed up maybe
    # If X has extra Industry Columns that aren't in the model, then drop them
    extra_x_cols =list(set(list(X.columns)).difference(set(model.get_booster().feature_names)))
    if len(extra_x_cols) > 0:
        X = X.drop(extra_x_cols, axis=1)
        warnings.warn('X has more columns than model features. Dropping extras. Extra columns: ' + str(extra_x_cols), RuntimeWarning)

    # If model has extra Industry Columns that aren't in X, then use float(nan)
    extra_model_features = list(set(model.get_booster().feature_names).difference(set(list(X.columns))))
    if len(extra_model_features) > 0:
        warnings.warn('Model has more features than X contains. Using nan instead. Extra features: ' + str(extra_model_features), RuntimeWarning)
        for f in extra_model_features:
            X[f] = [float('nan')]

    # Make sure feature names match
    assert set(model.get_booster().feature_names) == set(X.columns), 'Feature names do not match the column names in X.'

    # Put the columns of X in the same order as the features in the model (to avoid errors)
    X = X[model.get_booster().feature_names]
    
    # Predict the price
    price = model.predict(X)
    return price[0]


def check_portfolio_valuation(portfolio, time_averaged=False, time_averaged_period=5):
    predictions = []
    actual = []
    for ticker in portfolio:
        if time_averaged:
            pred, _ = predict_price_time_averaged(ticker, time_averaged_period, verbose=0)
        else:
            pred = predict_price(ticker)
        real = float(parse(ticker)['Open'])
        predictions.append(pred)
        actual.append(real)
        valuation = 'overvalued'
        if pred - real > 0:
            valuation = 'undervalued'
        percent = str(round(abs(pred - real) / real * 100, 2)) + '%'
        print(ticker + ' is ' + valuation + ' by ' + str(round(abs(pred - real), 2)) + ', or ' + percent + '.')


def predict_price_time_averaged(ticker, numdays, verbose=1, metric='mean', show_actual=False, start_date=None, path='', in_csv=False):
    if start_date == None: # Use yesterday
        base = str((datetime.datetime.today() - datetime.timedelta(1)).date())
    else:
        base = start_date
    date_list = pd.date_range(end=base, periods = numdays, freq='B')
    csvs = []
    for d in date_list:
        csvs.append('company_stats_' + str(d.date()) + '.csv')
    if show_actual:
        price_data = get_price_data(ticker, date_list[0], date_list[len(date_list) - 1])['Open']
    dates = []
    models = []
    pred_prices = []
    for csv in csvs:
        try:
            date = csv[14:24] # Parse out the date from csv string
            m = get_model_from_date(date, path=path)
            if m != None:
                models.append(m)
                dates.append(date)
        except FileNotFoundError:
            print(csv + ' was not found. Data from that day will be excluded.')

    assert None not in models, 'The models list contains NoneTypes'

    assert len(models) > 1, 'Could only obtain models for 1 of fewer days.'

    for i in range(len(models)):
        # Predict the price given the model, and set in_csv to true to speed up
        p = predict_price(ticker, model=models[i], path=path, in_csv=True, date=date) # Make sure in_csv is false because that assumes current date 
        if p == None:
            return None, None
        pred_prices.append(p)
        if verbose != 0 and show_actual:
            if len(csvs) == len(list(price_data)):
                print("Predicted Price for " + ticker + ': ' + str(p) + '. Actual price is: ' + str(price_data[i]) + '.')
    
    if verbose != 0:
        print('Mean predicted price: ' + str(np.mean(pred_prices)))
        print('Median predicted price: ' + str(np.median(pred_prices)))
        print('Standard Dev. predicted price: ' + str(np.std(pred_prices)))

    #print(ticker + ' is ' + valuation + ' by ' + str(float(abs(pred - real), 2)) + ', or ' + percent + '.')
    if metric == 'mean':
        return np.mean(pred_prices), np.std(pred_prices)
    elif metric == 'median':
        return np.median(pred_prices), np.std(pred_prices)
        
        

# JUST TESTING OUT
def check_robinhood_portfolio(rh_username, rh_password):
        ''' 
        Testing robin snacks API. Takes in robinhood username and password and calls
        check_portfolio valuation on user's portfolio.
        '''
        robin_stocks.login(rh_username, rh_password)
        my_stocks = robin_stocks.build_holdings()
        check_portfolio_valuation(my_stocks.keys())



def plot_feature_importances(clf, X_train, y_train=None, 
                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):
    '''
    plot feature importances of a tree-based sklearn estimator
    
    Note: X_train and y_train are pandas DataFrames
    
    Note: Scikit-plot is a lovely package but I sometimes have issues
              1. flexibility/extendibility
              2. complicated models/datasets
          But for many situations Scikit-plot is the way to go
          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html
    
    Parameters
    ----------
        clf         (sklearn estimator) if not fitted, this routine will fit it
        
        X_train     (pandas DataFrame)
        
        y_train     (pandas DataFrame)  optional
                                        required only if clf has not already been fitted 
        
        top_n       (int)               Plot the top_n most-important features
                                        Default: 10
                                        
        figsize     ((int,int))         The physical size of the plot
                                        Default: (8,8)
        
        print_table (boolean)           If True, print out the table of feature importances
                                        Default: False
        
    Returns
    -------
        the pandas dataframe with the features and their importance
    '''
    
    __name__ = "plot_feature_importances"
    
    import pandas as pd
    import numpy  as np
    import matplotlib.pyplot as plt
    
    from xgboost.core     import XGBoostError
    
    try: 
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                    format(clf.__class__.__name__))
                
    except (XGBoostError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())
            
    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()
    
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))
        
    return feat_imp



def get_price_data(ticker, start_date, end_date):
    '''
    Input: string ticker, string start_date, string end_date
    Output: Pandas Dataframe containing price data for ticker for the specified time period
    '''
    price_data = data.DataReader(ticker, 'yahoo', start_date, end_date)
    return price_data


def analyze(ticker, industry=None):
    '''
    analyze: Analyzes a company, given ticker name and industry_averages dictionary
        Input:
            ticker: company ticker
            industry: string representing industry of ticker, defaults to None
        Output: 
            No output, just prints information
            Prints analysis for company
            Values printed and returned are listed below:
                Company Health: 
                    Current Ratio
                    Debt Ratio
                    Altman Z-Score
                    Assets Per Share
                
                Valuation:
                    Book Value
                    Price to Book Value
                    Revenue Growth and Prediction         
    '''
    ticker = ticker.upper()
    summary_stats = get_summary_statistics(ticker)
    if industry == None:
        industry = get_company_industry(ticker)
    av = get_industry_averages()
            
    # altman_zscore = get_altman_zscore(ticker)
    print("ANALYSIS FOR " + ticker)
    print("Industry: " + str(industry))
    print("Trailing P/E Ratio: " + summary_stats['Trailing P/E'] + ". Industry Average: " + 
      str(round(av['industry_trailing_pe'][industry], 2)) + '.')
    print("Forward P/E Ratio: " + summary_stats['Forward P/E'] + ". Industry Average: " + 
      str(round(av['industry_forward_pe'][industry], 2)) + '.')
    print("Price to Sales Ratio: " + summary_stats['Price/Sales'] + ". Industry Average: " + 
      str(round(av['industry_price_to_sales'][industry], 2)) + '.')
    print("Price to Book Ratio: " + summary_stats['Price/Book'] + ". Industry Average: " + 
      str(round(av['industry_price_to_book'][industry], 2)) + '.')
    print("Enterprise Value to Revenue: " + summary_stats['Enterprise Value/Revenue'] + ". Industry Average: " + 
      str(av['industry_ev_to_rev'][industry]) + '.')
    print("Enterprise Value to EBITDA: " + summary_stats['Enterprise Value/EBITDA'] + ". Industry Average: " + 
      str(round(av['industry_ev_to_ebitda'][industry], 2)) + '.')
    print("Profit Margin: " + summary_stats['Profit Margin'] + ". Industry Average: " + 
      str(round(av['industry_profit_margin'][industry], 2)) + '%.')
    print("Operating Margin: " + summary_stats['Operating Margin'] + ". Industry Average: " + 
      str(round(av['industry_operating_margin'][industry], 2)) + '%.')
    print("Return on Assets: " + summary_stats['Return on Assets'] + ". Industry Average: " + 
      str(round(av['industry_return_on_assets'][industry], 2)) + '%.')
    print("Return on Equity: " + summary_stats['Return on Equity'] + ". Industry Average: " + 
      str(round(av['industry_return_on_equity'][industry], 2)) + '%.')
    print("Quarterly Revenue Growth: " + summary_stats['Quarterly Revenue Growth']) #+ ". Industry Average: " + 
      #str(round(industry_quarterly_rev_growth[industry], 2)) + '%.')
    print("Gross Profit: " + summary_stats['Gross Profit'] + ". Industry Average: " + 
      str(round(av['industry_gross_profit'][industry], 2)) + '.')
    print("Quarterly Earnings Growth: " + summary_stats['Quarterly Earnings Growth']) #+ ". Industry Average: " + 
      #str(round(industry_quarterly_earnings_growth[industry], 2)) + '%.')
    print("Debt to Equity: " + summary_stats['Total Debt/Equity'] + ". Industry Average: " + 
      str(round(av['industry_debt_to_equity'][industry], 2)) + '.')
    print("Current Ratio: " + summary_stats['Current Ratio'] + ". Industry Average: " + 
      str(round(av['industry_current_ratio'][industry], 2)) + '.')
    print("Book Value Per Share: " + summary_stats['Book Value Per Share'] + ". Industry Average: " + 
      str(round(av['industry_bvps'][industry], 2)) + '.')
    #print("Beta: " + summary_stats['Beta (5Y Monthly)'] + ". Industry Average: " + 
    #  str(round(av['industry_beta'][industry], 2)) + '.')
    dividend_yield_raw = get_dividend_yield(ticker)
    isPercent = False
    dividend_yield = ''
    for letter in dividend_yield_raw:
        if letter == "%":
            break;
        elif isPercent:
            dividend_yield += letter
        if letter == "(":
           isPercent = True
    dividend_yield = float(dividend_yield) / 100.0
    print("Forward Dividend & Yield: " + str(round(dividend_yield, 3)))
    pred, std = predict_price_time_averaged(ticker, 5, verbose=0)
    print('Predicted price using XGBoost Regression: ' + str(pred) + '. Stdev: ' + str(std))
    #print("Altman Zscore: " + str(altman_zscore))