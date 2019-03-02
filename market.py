# Imports
from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import sys
import pandas as pd
from lxml import html  
import requests
from time import sleep
import json
import argparse
from collections import OrderedDict
from time import sleep
from collections import defaultdict
import csv
import statistics


''' ***************************************************
# Functions that get financial data 
    ***************************************************'''

def parse(ticker):
    '''
    parse: This function returns the summary info on the yahoo finance page for "ticker". 
    The information returned is in the form of a dictionary. 
    '''
    url = "https://finance.yahoo.com/quote/%s?p=%s"%(ticker,ticker)
    response = requests.get(url, verify=True)
    #print ("Parsing %s"%(url))
    sleep(4)
    parser = html.fromstring(response.text)
    summary_table = parser.xpath('//div[contains(@data-test,"summary-table")]//tr')
    summary_data = OrderedDict()
    other_details_json_link = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{0}?formatted=true&lang=en-US&region=US&modules=summaryProfile%2CfinancialData%2CrecommendationTrend%2CupgradeDowngradeHistory%2Cearnings%2CdefaultKeyStatistics%2CcalendarEvents&corsDomain=finance.yahoo.com".format(ticker)
    summary_json_response = requests.get(other_details_json_link)
    try:
        json_loaded_summary =  json.loads(summary_json_response.text)
        y_Target_Est = json_loaded_summary["quoteSummary"]["result"][0]["financialData"]["targetMeanPrice"]['raw']
        earnings_list = json_loaded_summary["quoteSummary"]["result"][0]["calendarEvents"]['earnings']
        eps = json_loaded_summary["quoteSummary"]["result"][0]["defaultKeyStatistics"]["trailingEps"]['raw']
        datelist = []
        for i in earnings_list['earningsDate']:
            datelist.append(i['fmt'])
        earnings_date = ' to '.join(datelist)
        for table_data in summary_table:
            raw_table_key = table_data.xpath('.//td[contains(@class,"C(black)")]//text()')
            raw_table_value = table_data.xpath('.//td[contains(@class,"Ta(end)")]//text()')
            table_key = ''.join(raw_table_key).strip()
            table_value = ''.join(raw_table_value).strip()
            summary_data.update({table_key:table_value})
        summary_data.update({'1y Target Est':y_Target_Est,'EPS (TTM)':eps,'Earnings Date':earnings_date,'ticker':ticker,'url':url})
        return summary_data
    except:
        print ("Failed to parse json response")
    return {"error":"Failed to parse json response"}


def get_summary_statistics(ticker):
    '''
    Input: ticker value as a string. Example: 'NVDA'
    Output: Dictionary of summary statistics on the yahoo finance summary stats page
    '''
    url = "https://finance.yahoo.com/quote/%s/key-statistics/?p=%s"%(ticker,ticker)
    response = requests.get(url, verify=True)
    parser = html.fromstring(response.text)
    stats_table = parser.xpath('//table[contains(@class,"table-qsp-stats Mt(10px)")]//tr')
    summary_stats = {}
    try:
        for table_data in stats_table:
            raw_table_key = table_data.xpath('.//td[contains(@class,"")]//text()')[0]
            raw_table_value = table_data.xpath('.//td[contains(@class,"Fz(s)")]//text()')[0]
            summary_stats[raw_table_key] = raw_table_value
        return summary_stats
    except:
        print("Getting summary statistics for " + ticker + " did not work")



def periodic_figure_values(soup, yahoo_figure):
    '''
    periodic_figure_values: Call this function to obtain financial data from a company's financial statements.
        Args: 
            soup: use the function financials_soup("ticker", "is" or "bs" or "cf") to get the correct soup 
            yahoo_figure: The name of the information you want from the financial statement. Ex: Total Current Assets
        Return:
            This function normally returns a list of 4 elements, with numbers pertaining to the last 4 years
    '''
    values = []
    pattern = re.compile(yahoo_figure)

    title = soup.find("strong", text=pattern)    # works for the figures printed in bold
    if title:
        row = title.parent.parent
    else:
        title = soup.find("td", text=pattern)    # works for any other available figure
        if title:
            row = title.parent
        else:
            sys.exit("Invalid figure '" + yahoo_figure + "' passed.")

    cells = row.find_all("td")[1:]    # exclude the <td> with figure name
    for cell in cells:
        if cell.text.strip() != yahoo_figure:    # needed because some figures are indented
            str_value = cell.text.strip().replace(",", "").replace("(", "-").replace(")", "")
            if str_value == "-":
                str_value = 0
            value = int(float(str_value)) * 1000
            values.append(value)

    return values


def get_key_statistic(soup, name):
    value = 0
    pattern = re.compile(yahoo_figure)

    title = soup.find("strong", text=pattern)    # works for the figures printed in bold
    if title:
        row = title.parent.parent
    else:
        title = soup.find("td", text=pattern)    # works for any other available figure
        if title:
            row = title.parent
        else:
            sys.exit("Invalid figure '" + yahoo_figure + "' passed.")

    cells = row.find_all("td")[1:]    # exclude the <td> with figure name
    for cell in cells:
        if cell.text.strip() != yahoo_figure:    # needed because some figures are indented
            str_value = cell.text.strip().replace(",", "").replace("(", "-").replace(")", "")
            if str_value == "-":
                str_value = 0
            value = int(str_value) * 1000
            values.append(value)

    return values


def financials_soup(ticker_symbol, statement="is", quarterly=False):
    '''
    financials_soup: Gets the soup corresponding to the company and the financial statement you want. 
    This is used in the first arg for periodic_figure_values. 
    '''
    if statement == "is" or statement == "cf":
        url = "https://finance.yahoo.com/q/" + statement + "?s=" + ticker_symbol
        if not quarterly:
            url += "&annual"
        return BeautifulSoup(requests.get(url).text, "html.parser")
    if statement == "bs":
        url = "https://finance.yahoo.com/quote/" + ticker_symbol + "/balance-sheet?p=" + ticker_symbol
        if not quarterly:
            url += "&annual"
        return BeautifulSoup(requests.get(url).text, "html.parser")
    if statement == "ks":
        url = "https://finance.yahoo.com/quote/" + ticker_symbol + "/key-statistics?p=" + ticker_symbol
    return sys.exit("Invalid financial statement code '" + statement + "' passed.")

# Get a list of tickers from the csv 'companylist.csv'
def get_tickers():
    with open('companylist.csv', newline='') as f:
        reader = csv.reader(f)
        company_matrix = np.array(list(reader))
        company_matrix = np.delete(company_matrix, (0), axis=0)
    return company_matrix[:,0]


# Returns the industry of a company given its ticker
# USES WIKIPEDIA,NOT CSV
def get_company_industry(ticker):
    # Get S&P500 Tickers with Industry
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = data[1] # Index of data may need to be changed based on additions to the wiki
    sliced_table = table[1:]
    header = table.iloc[0]
    corrected_table = sliced_table.rename(columns=header)
    print(corrected_table.keys())
    tickers = corrected_table['Symbol'].tolist() # Gets a list of the tickers in the S&P500
    industries = corrected_table['GICS Sub Industry'].tolist() # Gets a list of the industries in the S&P500
    
    for i in range(len(corrected_table['Symbol'].tolist())):
        if ticker.lower() == corrected_table['Symbol'].tolist()[i].lower():
            return corrected_table["GICS Sub Industry"].tolist()[i]
    return "failed"


# Returns a dictionary with sectors as keys and companies as values
def get_company_industry_dict():
    with open('companylist.csv', newline='') as f:
        reader = csv.reader(f)
        company_matrix = np.array(list(reader))
        company_matrix = np.delete(company_matrix, (0), axis=0)

    tickers_full = company_matrix[:,0]
    industry = company_matrix[:,7]

    company_industry = defaultdict(list)
    for i in range(len(tickers_full)):
        if industry[i] in company_industry:
            company_industry[industry[i]].append(tickers_full[i])
        else:
            company_industry[industry[i]] = [tickers_full[i]]
    return company_industry # Dictionary with sectors as keys and companies as values

    ''' OLD CODE
    company_industry = defaultdict(list)
    for i in range(len(tickers)):
        if industries[i] in company_industry:
            company_industry[industries[i]].append(tickers[i])
        else:
            company_industry[industries[i]] = [tickers[i]]
    print(company_industry)
    '''

# Returns an array oof dictionaries consisting of averages for each industry
# TODO: This can be optimized
def get_industry_averages():
    industry_dict = get_company_industry_dict()
    industry_current_ratio = {}
    industry_current_assets_per_share = {}
    industry_debt_ratio = {}
    industry_book_value_per_share = {}
    industry_earnings_growth_yoy = {}
    #industry_dividend_yield = {}
    for key in industry_dict.keys():
        current_ratio_av = 0
        current_assets_per_share_av = 0
        debt_ratio_av = 0
        book_value_per_share_av = 0
        earnings_growth_yoy_av = 0
        #dividend_yield_av = 0
        tickers_not_added = 0
        for ticker in industry_dict[key]:
            cur_ratio = get_current_ratio(ticker)
            if len(cur_ratio) == 0:
                add_to_averages = False
                tickers_not_added += 1
                break
            current_ratio_av += get_current_ratio(ticker)[0]
            cur_assets_per_share = get_current_assets_per_share(ticker)
            '''if len(cur_assets_per_share) < 4:
                add_to_averages = False
                tickers_not_added += 1
                break
            else:
                current_assets_per_share_av += cur_assets_per_share[0]
            '''
            try: 
                current_assets_per_share_av += cur_assets_per_share[0]
            except: 
                add_to_averages = False
                tickers_not_added += 1
                break
            debt_ratio_av += get_debt_ratio(ticker)[0]
            bvps = get_book_value_per_share(ticker)[0]
            print("book value per share: " + str(bvps))
            book_value_per_share_av += bvps
            earnings_growth_yoy_av += get_earning_growth_yoy(ticker)
            #dividend_yield_av += get_dividend_yield(ticker)
        #TODO: Check if the division by 0
        if (len(industry_dict[key]) - tickers_not_added) == 0:
            break;
        industry_current_ratio[key] = current_ratio_av / (len(industry_dict[key]) - tickers_not_added)
        industry_current_assets_per_share[key] = current_assets_per_share_av / (len(industry_dict[key]) - tickers_not_added)
        industry_debt_ratio[key] = debt_ratio_av / (len(industry_dict[key]) - tickers_not_added)
        industry_book_value_per_share[key] = book_value_per_share_av / (len(industry_dict[key]) - tickers_not_added)
        industry_earnings_growth_yoy[key] = earnings_growth_yoy_av / (len(industry_dict[key]) - tickers_not_added)
        #industry_dividend_yield[key] = dividend_yield_av / len(industsry_dict[key])
    return [industry_current_ratio, industry_current_assets_per_share, industry_debt_ratio, industry_book_value_per_share,industry_earnings_growth_yoy]#, industry_dividend_yield ]



''' ***************************************************
# Functions that calculate some ratio or metric 
    ***************************************************'''

# Higher the better, preferably greater than 2
def get_current_ratio(ticker):
    try:
        total_current_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Assets")
        total_current_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Liabilities")
        cur_ratio = np.divide(total_current_assets, total_current_liabilities)
    except:
        print("Could not calculate the current ratio for " + ticker)
    return cur_ratio

    
def get_current_assets_per_share(ticker):
    total_current_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Assets")
    total_current_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Liabilities")
    net_income = periodic_figure_values(financials_soup(ticker, "is"), "Net Income")
    try:
        shares_outstanding = np.divide(net_income, parse(ticker)['EPS (TTM)'])
        return np.divide(np.subtract(total_current_assets, total_current_liabilities), shares_outstanding)
    except:
        print("Could not calculate current assets per share for " + ticker)
        return 0


# The lower the better
def get_debt_ratio(ticker):
    try:
        total_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Assets")
        total_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Liabilities")
    except:
        print("Could not calculate debt ratio for " + ticker)
    return np.divide(total_liabilities, total_assets)


def get_book_value_per_share(ticker):
    try:
        total_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Assets")
        total_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Liabilities")
        net_income = periodic_figure_values(financials_soup(ticker, "is"), "Net Income")
        eps = parse(ticker)['EPS (TTM)']
        shares_outstanding = np.divide(net_income, eps)
    except:
        print("Could not calculate the book value per share for " + ticker)
    return np.divide(np.subtract(total_assets, total_liabilities), shares_outstanding)


def get_price_to_book_value(ticker):
    try:
        open_price = float(parse(ticker)['Open'])
        #print("open price")
        #print(open_price)
        bvps = float(get_book_value_per_share(ticker)[0])
        #print("bvps")
        #print(bvps)
    except:
        print("Could not calculate the price to book value for " + ticker)
    return np.divide(open_price , bvps)


def get_altman_zscore(ticker):
    # A = working capital / total assets
    total_cur_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Assets")[0]
    total_cur_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Liabilities")[0]
    total_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Assets")[0]
    a = (total_cur_assets - total_cur_liabilities) / total_assets
    # B = retained earnings / total assets
    net_income = periodic_figure_values(financials_soup(ticker, "is"), "Net Income")[0]
    try:
        dividends_paid = periodic_figure_values(financials_soup(ticker, "cf"), "Dividends Paid")[0]
        retained_earnings = net_income + dividends_paid
    except:
        retained_earnings = net_income
    b = retained_earnings / total_assets
    # C = earnings before interest and tax / total assets
    operating_income = periodic_figure_values(financials_soup(ticker, "is"), "Earnings Before Interest and Taxes")[0]
    c = operating_income / total_assets
    # D = market value of equity / total liabilities
    market_cap = parse(ticker)["Market Cap"] 
    if market_cap[len(market_cap) - 1] == "B":
        market_cap = float(market_cap[0:len(market_cap) - 1]) * 1000000
    elif market_cap[len(market_cap) - 1] == "M":
        market_cap = float(market_cap[0:len(market_cap) - 1]) * 1000
    elif market_cap[len(market_cap) - 1] == "T":
        market_cap = float(market_cap[0:len(market_cap) - 1]) * 1000000000
    else:
        print("Error: Market Cap is " + market_cap[len(market_cap) - 1] + ". Expected M or B.")
    d = market_cap / periodic_figure_values(financials_soup(ticker, "bs"), "Total Liabilities")[0]
    # E = sales / total assets
    e = periodic_figure_values(financials_soup(ticker, "is"), "Total Revenue")[0] / total_assets
    return 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e


def get_earning_growth_yoy(ticker):
    try:
        net_income = periodic_figure_values(financials_soup(ticker, "is"), "Net Income")
        if net_income[0] < 0 and net_income[1] < 0:
            return -1 * (net_income[0] - net_income[1]) / net_income[1]
        elif net_income[1] < 0 and net_income[0] > 0:
            return -1 * (net_income[0] - net_income[1]) / net_income[1]
        else:
            return (net_income[0] - net_income[1]) / net_income[1]
    except: 
        print("Could not calculate the earning growth for " + ticker)
    

def get_dividend_yield(ticker):
    return parse(ticker)['Forward Dividend & Yield']


def get_pe_ratio(ticker):
    return float(parse(ticker)['PE Ratio (TTM)'])


def expected_return_capm(risk_free, beta, expected_market_return):
    # CAPM
    return risk_free + beta(expected_market_return - risk_free)

#def value_company_discounted_cash_flow(revenue_growth_rate):


def multiples_valuation(ticker, comparables, ratio='EV/EBITDA'):
    '''
    Computes the Enterprise Value to EBITDA Multiples Valuation
    or the PE Multiple Valuation, depening on the value of ratio
    '''
    print('Valuation for ' + ticker)
    print('Comparables used: ' + str(comparables))

    if ratio == 'P/E' or ratio == 'PE':
        pe_ratios = []
        for comp in comparables:
            stats = get_summary_statistics(comp)
            ratio = str_to_num(stats['Forward P/E'])
            pe_ratios.append(ratio)
            print('Comparable ' + comp + ' has a P/E of ' + str(ratio))
        multiple_of_comparables = statistics.median(pe_ratios)
        print('Using the median multiple value of ' + str(multiple_of_comparables))
        key_stats = parse(ticker)
        eps = key_stats['EPS (TTM)']
        valuation = eps * multiple_of_comparables
        print('Calculation for ' + ticker + ': ' + 
            str(eps) + ' * ' + str(multiple_of_comparables) +
            ' = ' + str(valuation) + ' (EPS * PE = Price per Share)')
        print('Valuation for share price: ' + str(valuation))
        return valuation
    else:
        ev_to_ebitda_ratios = []
        for comp in comparables:
            stats = get_summary_statistics(comp)
            ratio = str_to_num(stats['Enterprise Value/EBITDA'])
            ev_to_ebitda_ratios.append(ratio)
            print('Comparable ' + comp + ' has a Enterprise Value/EBITDA of ' + str(ratio))
        multiple_of_comparables = statistics.median(ev_to_ebitda_ratios)
        print('Using the median multiple value of ' + str(multiple_of_comparables))
        summary_stats = get_summary_statistics(ticker)
        ebitda = str_to_num(summary_stats['EBITDA'])
        debt = str_to_num(summary_stats['Total Debt'])
        cash = str_to_num(summary_stats['Total Cash'])
        shares_outstanding = str_to_num(summary_stats['Shares Outstanding'])
        ev = ebitda * multiple_of_comparables
        print('Calculated Enterprise Value for ' + ticker + ': ' + 
            str(ebitda) + ' * ' + str(multiple_of_comparables) + 
            ' = ' + str(ev) + ' (EV = EBITDA * Multiple)')
        equity = ev + cash - debt
        print('Calculated Equity for ' + ticker + ': ' + 
            str(ev) + ' + ' + str(cash) + ' - ' + str(debt) + 
            ' = ' + str(equity) + ' (Equity = EV + Cash - Debt)')
        equity_per_share = equity / shares_outstanding
        print('Valuation for share price: ' + str(equity_per_share))
        return equity_per_share


def str_to_num(number_string):
    '''
    Converts string to float
    Handles cases where there is a string
    like '18.04B'. This would return
    18,040,000,000.
    Input: string
    Output: float
    '''
    if number_string[-1] == 'B':
        return float(number_string[0:len(number_string) - 1]) * 1000000000
    elif number_string[-1] == 'M':
        return float(number_string[0:len(number_string) - 1]) * 1000000
    elif number_string[-1] == 'T':
        return float(number_string[0:len(number_string) - 1]) * 1000
    else:
        try:
            return float(number_string)
        except:
            raise Exception('Could not convert ' + number_string + ' to a number')


    
