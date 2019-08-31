# Imports
from market import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tensorflow as  tf
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle as pkl
from sklearn import cross_validation, metrics 
from sklearn.grid_search import GridSearchCV
import pickle


def train_and_get_model():
    print('Training XGB model with hyperparameter tuning... Make sure csv is updated.')
    financial_data = pd.read_csv("company_statistics.csv")
    to_remove = ['Ticker', 'Name', 'Price', 'Sector', 'Industry', 'IPO Year']
    feature_cols = [x for x in financial_data.columns if x not in to_remove]
    X = financial_data[feature_cols]
    Y = financial_data['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    param_test = {
        'max_depth':[3],
        'min_child_weight':[4],
        'learning_rate':[.25],
        'gamma':[0],
        'reg_alpha':[ 0.1, .12, .14]
    }
    xgbr = xgb.XGBRegressor() 
    gsearch = GridSearchCV(estimator = xgbr , 
    param_grid = param_test,n_jobs=4,iid=False, cv=5)
    gsearch.fit(X_train,y_train)
    gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    print("Test Score: " + str(gsearch.best_estimator_.score(X_test, y_test)))
    model = gsearch.best_estimator_
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))
    save_model(model)
    return model


def save_model(model, name=None):
    if name == None:
        name = 'xgbr_latest.dat'
    # save model to file
    pkl.dump(model, open(name, 'wb'))


def predict_price(ticker, model=None, model_type='xgb'): # Next Step: Compareto actual price and output how much its overvalued or undervalued by
    attributes = ['Market Cap (intraday)','Trailing P/E','Forward P/E','PEG Ratio (5 yr expected)','Price/Sales','Price/Book',
                  'Enterprise Value/Revenue','Enterprise Value/EBITDA','Profit Margin','Operating Margin',
                  'Return on Assets','Return on Equity','Revenue','Revenue Per Share',
                  'Quarterly Revenue Growth','Gross Profit','EBITDA','Diluted EPS',
                  'Quarterly Earnings Growth','Total Cash','Total Cash Per Share','Total Debt',
                  'Total Debt/Equity','Current Ratio','Book Value Per Share','Operating Cash Flow',
                  'Levered Free Cash Flow','Beta (3Y Monthly)','Shares Outstanding','Forward Annual Dividend Rate',
                  'Forward Annual Dividend Yield','Trailing Annual Dividend Rate','Trailing Annual Dividend Yield',
                  '5 Year Average Dividend Yield','Payout Ratio']
    stats = get_summary_statistics(ticker)
    financial_data = pd.read_csv("company_statistics.csv")
    if model_type != 'xgb':
        financial_data = financial_data.fillna(-1)
    x = []
    for a in attributes:
        x.append(str_to_num(stats[a]))
    to_remove = ['Ticker', 'Name', 'Price', 'Sector', 'Industry', 'IPO Year']
    feature_cols = [y for y in financial_data.columns if y not in to_remove]
    X = pd.DataFrame(columns=feature_cols)
    X.loc[-1] = x
    if model == None: # Use default model
        print('Using last saved model.')
        model = pickle.load(open("xgbr_latest.dat", "rb"))
    price = model.predict(X)
    return price[0]


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