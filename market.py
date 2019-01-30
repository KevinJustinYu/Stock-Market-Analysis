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
