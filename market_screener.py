'''
This file will include various functions for screening stocks
'''

# Imports
from market_tests import *

# Functions

def price_to_book_filter(tickers, thresh, date):
    ''' 
    This function screens tickers by their price to book ratio.
    Tickers with values under thresh will be returned.
     ''' 

    # Make sure the file exists
    assert "company_stats_" + date + ".csv" in os.listdir(path + "csv_files/"), 'Could not find the specified csv file for ' + date
    # Get the file as a df
    financial_data = pd.read_csv(path + "csv_files/company_stats_" + date + ".csv", encoding='cp1252')

    # Loop through and store each valid ticker in filtered
    filtered = []
    for ticker in tickers:
        p_to_b_ratio = financial_data[financial_data.Ticker == ticker]["Price/Book"]
        assert isinstance(p_to_b_ratio, numbers.Number), "Expected a numeric value. Use str_to_num. "

        if p_to_b_ratio <= thresh:
            filtered.append(ticker)

    return filtered 

def pe_ratio_to_industry_filter(tickers, thresh, date, industry_averages=None):
    '''
    This function screens tickers that have PE ratios less than the industry average.
    The thresh argument should be a ratio (0.9 for instance) of ticker PE ratio to industry average. 
    Returned tickers will have ratios under the thresh. 
    '''

    assert "company_stats_" + date + ".csv" in os.listdir(path + "csv_files/"), 'Could not find the specified csv file for ' + date
    financial_data = pd.read_csv(path + "csv_files/company_stats_" + date + ".csv", encoding='cp1252')

    if industry_averages == None:
        industry_averages = get_industry_averages(date=date)

    # Loop through and store each valid ticker in filtered
    filtered = []
    for ticker in tickers:
        row = financial_data[financial_data.Ticker == ticker]
        industry = row['Industry']
        pe_ratio = row['Forward P/E']
        industry_pe = industry_averages['industry_forward_pe'][industry]

        if pe_ratio / industry_pe < thresh:
            filtered.append(ticker)

    return filtered


def debt_to_equity_filter(tickers, thresh, date):
    '''
    This function screens tickers by debt to equity ratio. 
    Lower debt to equity ratios are better, so tickers with ratios 
    lower than the thresh will be returned.
    '''

    assert "company_stats_" + date + ".csv" in os.listdir(path + "csv_files/"), 'Could not find the specified csv file for ' + date
    financial_data = pd.read_csv(path + "csv_files/company_stats_" + date + ".csv", encoding='cp1252')




