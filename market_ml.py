

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
from datetime import date
import datetime
from pandas_datareader import data
#import robin_stocks

def get_model_from_date(date, verbose=0):
    '''
    Gets the XGB model from date, first by checking for the pretrained 
    model in ml_models, then then checking for the csv and training the model
    from scratch. If the csv is not located, we print an error.
    '''

    # First, check the ml_models folder for the correponding model
    model_string = 'xgbr_' + date + '.dat'
    try:
        model = pkl.load(open('ml_models/' + model_string, "rb"))
        return model
    except:
        print('The model could not be found at: ' + 'ml_models/' + model_string)

    # Second, see if we have a csv for the date, and train a model for that csv
    csv_string = 'company_stats_' + date + '.csv'
    try:
        model = train_and_get_model(filename=csv_string, verbose=0, save_to_file=True, saved_model_name=model_string)
        return model
    except:
        print('Could not train model for the data located in ' + csv_string + '. Check that this file exists.')


def train_and_get_model(filename='company_statistics.csv', verbose=0, save_to_file=False, saved_model_name='xgbr_latest.dat'):
    if verbose != 0:
        print('Training XGB model with hyperparameter tuning... Make sure csv is updated.')
    financial_data = pd.read_csv("csv_files/" + filename)
    to_remove = ['Ticker', 'Name', 'Price', 'Sector', 'Industry', 'IPO Year']
    feature_cols = [x for x in financial_data.columns if x not in to_remove]
    X = financial_data[feature_cols]
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
    gsearch.fit(X_train,y_train)
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
    if name == None:
        name = 'xgbr_latest.dat'
    # save model to file
    pkl.dump(model, open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/ml_models/' + name, 'wb'))
    today = date.today()
    pkl.dump(model, open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/ml_models/xgbr_' + 
        str(today) + '.dat', 'wb'))


def predict_price(ticker, model=None, model_type='xgb', verbose=0): # Next Step: Compareto actual price and output how much its overvalued or undervalued by
    attributes = ['Market Cap (intraday)','Trailing P/E','Forward P/E','PEG Ratio (5 yr expected)','Price/Sales','Price/Book',
                  'Enterprise Value/Revenue','Enterprise Value/EBITDA','Profit Margin','Operating Margin',
                  'Return on Assets','Return on Equity','Revenue','Revenue Per Share',
                  'Quarterly Revenue Growth','Gross Profit','EBITDA','Diluted EPS', 'EPS Beat Ratio',
                  'Quarterly Earnings Growth','Total Cash','Total Cash Per Share','Total Debt',
                  'Total Debt/Equity','Current Ratio','Book Value Per Share','Operating Cash Flow',
                  'Levered Free Cash Flow','Beta (3Y Monthly)','Shares Outstanding','Forward Annual Dividend Rate',
                  'Forward Annual Dividend Yield','Trailing Annual Dividend Rate','Trailing Annual Dividend Yield',
                  '5 Year Average Dividend Yield','Payout Ratio']
    stats = get_summary_statistics(ticker)
    summary = parse(ticker)
    if summary == {"error":"Failed to parse json response"} or stats == {"error":"Failed to parse json response"}:
        return -1
    eps_beat_ratio = summary["EPS Beat Ratio"]
    financial_data = pd.read_csv("csv_files/company_statistics.csv")
    if model_type != 'xgb':
        financial_data = financial_data.fillna(-1)
    x = []
    for a in attributes:
        if a == 'EPS Beat Ratio': # Handle the case with beat ratio because not included in summary stats
            x.append(str_to_num(eps_beat_ratio))
        else:
            x.append(str_to_num(stats[a]))
    to_remove = ['Ticker', 'Name', 'Price', 'Sector', 'Industry', 'IPO Year']
    feature_cols = [y for y in financial_data.columns if y not in to_remove]

    X = pd.DataFrame(columns=feature_cols)
    X.loc[-1] = x
    if model == None: # Use default model
        if verbose != 0:
            print('Using last saved model.')
        model = pickle.load(open("ml_models/xgbr_latest.dat", "rb"))
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


def predict_price_time_averaged(ticker, numdays, verbose=1, metric='mean', show_actual=False, start_date=None):
    if start_date == None: # Use yesterday
        base = str((datetime.datetime.today() - datetime.timedelta(1)).date())
    else:
        base = start_date
    date_list = pd.date_range(end=base, periods = numdays, freq='B')
    csvs = []
    for date in date_list:
        csvs.append('company_stats_' + str(date.date()) + '.csv')
    if show_actual:
        price_data = get_price_data(ticker, date_list[0], date_list[len(date_list) - 1])['Open']
    
    models = []
    pred_prices = []
    for csv in csvs:
        try:
            date = csv[14:24] # Parse out the date from csv string
            models.append(get_model_from_date(date))
        except FileNotFoundError:
            print(csv + ' was not found. Data from that day will be excluded.')
    for i in range(len(models)):
        p = predict_price(ticker, model=models[i])
        pred_prices.append(p)
        if verbose != 0 and show_actual:
            if len(csvs) == len(list(price_data)):
                print("Predicted Price for " + ticker + ': ' + str(p) + '. Actual price is: ' + str(price_data[i]) + '.')
    if verbose != 0:
        print('Mean predicted price: ' + str(np.mean(pred_prices)))
        print('Median predicted price: ' + str(np.median(pred_prices)))
        print('Standard Dev. predicted price: ' + str(np.std(pred_prices)))
    if metric == 'mean':
        return np.mean(pred_prices), np.std(pred_prices)
    elif metric == 'median':
        return np.median(pred_prices), np.std(pred_prices)
        print(ticker + ' is ' + valuation + ' by ' + str(float(abs(pred - real), 2)) + ', or ' + percent + '.')
        
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