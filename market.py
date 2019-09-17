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
    #sleep(0.1) # This is used to slow down so blocking doesnt happen. Consider decreasing.
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
        eps_beat_ratio = get_eps_beat_ratio(json_loaded_summary["quoteSummary"]["result"][0]["earnings"]["earningsChart"]["quarterly"])
        datelist = []
        for i in earnings_list['earningsDate']:
            datelist.append(i['fmt'])
        earnings_date = ' to '.join(datelist)
        for table_data in summary_table:
            raw_table_key = table_data.xpath('.//td[contains(@class,"C($primaryColor)")]//text()')
            raw_table_value = table_data.xpath('.//td[contains(@class,"Ta(end)")]//text()')
            table_key = ''.join(raw_table_key).strip()
            table_value = ''.join(raw_table_value).strip()
            summary_data.update({table_key:table_value})
        summary_data.update({'1y Target Est':y_Target_Est,'EPS (TTM)':eps, 'EPS Beat Ratio': eps_beat_ratio, 'Earnings Date':earnings_date,'ticker':ticker,'url':url})
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
    stats_table = parser.xpath('//div[contains(@class,"Mstart(a) Mend(a)")]//tr')
    summary_stats = {}
    try:
        for table_data in stats_table:
            raw_table_key = table_data.xpath('.//td[contains(@class,"")]//text()')[0]
            raw_table_value = table_data.xpath('.//td[contains(@class,"Fz(s)")]//text()')[0]
            summary_stats[raw_table_key] = raw_table_value
        # summary_stats["EPS Beat Ratio"] = parse(ticker)["EPS Beat Ratio"]
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


def get_tickers():
    '''
    Returns a list of tickers from the csv 'companylist.csv'
    '''
    with open('csv_files/company_data.csv', newline='') as f:
        reader = csv.reader(f)
        company_matrix = np.array(list(reader))
        company_matrix = np.delete(company_matrix, (0), axis=0)
    return company_matrix[:,0]

def get_eps_beat_ratio(qtr_eps_chart):
    '''
    Returns the ratio latest quarter EPS divided by the analysts EPS consensus.
    '''
    try:
        return str(round(qtr_eps_chart[-1]["actual"]["raw"]/qtr_eps_chart[-1]["estimate"]['raw'], 4))
    except:
        return "N/A"      
    
def get_company_industry(ticker):
    '''
    Input: ticker of a company (S&P500)
    Returns the industry of an S&P500 company 
    '''
    industries = get_company_industry_dict()
    for key in industries.keys():
        if ticker in industries[key]:
            return key
    print("Failed to find the company industry.")
    return 0
    '''
    OLD CODE
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
    '''


# Returns a dictionary with sectors as keys and companies as values
def get_company_industry_dict():
    with open('csv_files/company_statistics.csv', newline='') as f:
        reader = csv.reader(f)
        company_matrix = np.array(list(reader))
        company_matrix = np.delete(company_matrix, (0), axis=0)

    tickers_full = company_matrix[:,0] # First column is tickers
    industry = company_matrix[:,3] # Third column is industry

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

def get_company_comprables(ticker):
    '''
    Input: Company ticker
    Output: Returns a list of comparable companies. This can be used for multiples valuation
    '''
    industries = get_company_industry_dict()
    industry = get_company_industry(ticker)
    comps = industries[industry].remove(ticker)
    return comps


def get_industry_averages():
    '''
    Returns an array of dictionaries consisting of averages for each industry
    '''
    industry_dict = get_company_industry_dict()
    industry_trailing_pe = {}
    industry_forward_pe = {}
    industry_price_to_sales = {}
    industry_price_to_book = {}
    industry_ev_to_rev = {}
    industry_ev_to_ebitda = {}
    industry_profit_margin = {}
    industry_operating_margin = {}
    industry_return_on_assets = {}
    industry_return_on_equity = {}
    industry_quarterly_rev_growth = {}
    industry_gross_profit = {}
    industry_quarterly_earnings_growth = {}
    industry_debt_to_equity = {}
    industry_current_ratio = {}
    industry_bvps = {}
    industry_beta = {}
    
    stats = pd.read_csv('csv_files/company_statistics.csv')
    
    for key in industry_dict.keys():
        trailing_pe_av = 0
        forward_pe_av = 0
        price_to_sales_av = 0
        price_to_book_av = 0
        ev_to_rev_av = 0
        ev_to_ebitda_av = 0
        profit_margin_av = 0
        operating_margin_av = 0
        return_on_assets_av = 0
        return_on_equity_av = 0
        quarterly_rev_growth_av = 0
        gross_profit_av = 0
        quarterly_earnings_growth_av = 0
        debt_to_equity_av = 0
        current_ratio_av = 0
        bvps_av = 0
        beta_av = 0
        averages = [0]*17
        
        d1 = 0
        d2 = 0
        d3 = 0
        d4 = 0
        d5 = 0
        d6 = 0
        d7 = 0
        d8 = 0
        d9 = 0
        d10 = 0
        d11 = 0
        d12 = 0
        d13 = 0
        d14 = 0
        d15 = 0
        d16 = 0
        d17 = 0
        for ticker in industry_dict[key]:
            #print(ticker)
            cs = stats.loc[stats['Ticker'] == ticker]
            if np.isnan(cs[['Trailing P/E']].values[0][0]) == False:
                d1 += 1
                trailing_pe_av += cs[['Trailing P/E']].values[0][0]
            if np.isnan(cs[['Forward P/E']].values[0][0]) == False:
                d2 += 1
                forward_pe_av += cs[['Forward P/E']].values[0][0]
            if np.isnan(cs[['Price/Sales(ttm)']].values[0][0]) == False:
                d3 += 1
                price_to_sales_av += cs[['Price/Sales(ttm)']].values[0][0]
            if np.isnan(cs[['Price/Book']].values[0][0]) == False:
                d4 += 1
                price_to_book_av += cs[['Price/Book']].values[0][0]
            if np.isnan(cs[['Enterprise Value/Revenue']].values[0][0]) == False:
                d5 += 1
                ev_to_rev_av += cs[['Enterprise Value/Revenue']].values[0][0]
            if np.isnan(cs[['Enterprise Value/EBITDA']].values[0][0]) == False:
                d6 += 1
                ev_to_ebitda_av += cs[['Enterprise Value/EBITDA']].values[0][0]
            if np.isnan(cs[['Profit Margin']].values[0][0]) == False:
                d7 += 1
                profit_margin_av += cs[['Profit Margin']].values[0][0]
            if np.isnan(cs[['Operating Margin(TTM)']].values[0][0]) == False:
                d8 += 1
                operating_margin_av += cs[['Operating Margin(TTM)']].values[0][0]
            if np.isnan(cs[['Return on Assets(TTM)']].values[0][0]) == False:
                d9 += 1
                return_on_assets_av += cs[['Return on Assets(TTM)']].values[0][0]
            if np.isnan(cs[['Return on Equity(TTM)']].values[0][0]) == False:
                d10 += 1
                return_on_equity_av += cs[['Return on Equity(TTM)']].values[0][0]
            if np.isnan(cs[['Quarterly Revenue Growth(YOY)']].values[0][0]) == False:
                d11 += 1
                quarterly_rev_growth_av += cs[['Quarterly Revenue Growth(YOY)']].values[0][0]
            if np.isnan(cs[['Gross Profit(TTM)']].values[0][0]) == False:
                d12 += 1
                gross_profit_av += cs[['Gross Profit(TTM)']].values[0][0]
            if np.isnan(cs[['Quarterly Earnings Growth(YOY)']].values[0][0]) == False:
                d13 += 1
                quarterly_earnings_growth_av += cs[['Quarterly Earnings Growth(YOY)']].values[0][0]
            if np.isnan(cs[['Total Debt/Equity']].values[0][0]) == False:
                d14 += 1
                debt_to_equity_av += cs[['Total Debt/Equity']].values[0][0]
            if np.isnan(cs[['Current Ratio']].values[0][0]) == False:
                d15 += 1
                current_ratio_av += cs[['Current Ratio']].values[0][0]
            if np.isnan(cs[['Book Value Per Share']].values[0][0]) == False:
                d16 += 1
                bvps_av += cs[['Book Value Per Share']].values[0][0]
            if np.isnan(cs[['Beta(3Y Monthly)']].values[0][0]) == False:
                d17 += 1
                beta_av += cs[['Beta(3Y Monthly)']].values[0][0]
        if d1 != 0:
            industry_trailing_pe[key] = trailing_pe_av / d1
        if d2 != 0:
            industry_forward_pe[key] = forward_pe_av / d2
        if d3 != 0:
            industry_price_to_sales[key] = price_to_sales_av / d3
        if d4 != 0:
            industry_price_to_book[key] = price_to_book_av / d4
        if d5 != 0:
            industry_ev_to_rev[key] = ev_to_rev_av / d5
        if d6 != 0:
            industry_ev_to_ebitda[key] = ev_to_ebitda_av / d6
        if d7 != 0:
            industry_profit_margin[key] = profit_margin_av / d7
        if d8 != 0:
            industry_operating_margin[key] = operating_margin_av / d8
        if d9 != 0:
            industry_return_on_assets[key] = return_on_assets_av / d9
        if d10 != 0:
            industry_return_on_equity[key] = return_on_equity_av / d10
        if d11 != 0:
            industry_quarterly_rev_growth[key] = quarterly_rev_growth_av / d11
        if d12 != 0:
            industry_gross_profit[key] = gross_profit_av / d12
        if d13 != 0:
            industry_quarterly_earnings_growth[key] = quarterly_earnings_growth_av / d13
        if d14 != 0:
            industry_debt_to_equity[key] = debt_to_equity_av / d14
        if d15 != 0:
            industry_current_ratio[key] = current_ratio_av / d15
        if d16 != 0:
            industry_bvps[key] = bvps_av / d16
        if d17 != 0:
            industry_beta[key] = beta_av / d17
        #industry_dividend_yield[key] = dividend_yield_av / len(industsry_dict[key])
    return [industry_trailing_pe, industry_forward_pe, industry_price_to_sales, industry_price_to_book, industry_ev_to_rev, 
            industry_ev_to_ebitda, industry_profit_margin, industry_operating_margin, industry_return_on_assets, 
            industry_return_on_equity, industry_quarterly_rev_growth, industry_gross_profit, industry_quarterly_earnings_growth,
            industry_debt_to_equity, industry_current_ratio, industry_bvps, industry_beta]


''' ***************************************************
# Functions that calculate some ratio or metric 
    ***************************************************'''

# Higher the better, preferably greater than 2
def get_current_ratio(ticker):
    '''
    Input: Company ticker
    Output: The current ratio of the company (short term assets / short term debt) as a float
    '''
    try:
        total_current_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Assets")
        total_current_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Liabilities")
        cur_ratio = np.divide(total_current_assets, total_current_liabilities)
    except:
        print("Could not calculate the current ratio for " + ticker)
    return cur_ratio

    
def get_current_assets_per_share(ticker):
    '''
    Input: Company ticker
    Output: The assets per share of the company
    '''
    total_current_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Assets")
    total_current_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Current Liabilities")
    net_income = periodic_figure_values(financials_soup(ticker, "is"), "Net Income")
    try:
        shares_outstanding = np.divide(net_income, parse(ticker)['EPS (TTM)'])
        return np.divide(np.subtract(total_current_assets, total_current_liabilities), shares_outstanding)
    except:
        print("Could not calculate current assets per share for " + ticker)
        return 0


def get_debt_ratio(ticker):
    '''
    Input: Company ticker
    Output: The debt ratio of the company
    '''
    try:
        total_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Assets")
        total_liabilities = periodic_figure_values(financials_soup(ticker, "bs"), "Total Liabilities")
    except:
        print("Could not calculate debt ratio for " + ticker)
    return np.divide(total_liabilities, total_assets)


def get_book_value_per_share(ticker):
    '''
    Input: Company ticker
    Output: The book value per share of the company
    '''
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
    '''
    Input: Company ticker
    Output: The price to book value of the company
    '''
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
    '''
    Input: Company ticker
    Output: The altman z-score of the company
    '''
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


# TODO: Handle verbose optional arg and auto industry comparables
# TODO: Handle optional argument of using the mean rather than median 
def multiples_valuation(ticker, comparables, ratio='EV/EBITDA', verbose=True):
    '''
    Computes the Enterprise Value to EBITDA Multiples Valuation
    or the PE Multiple Valuation, depening on the value of ratio
    '''
    print('Valuation for ' + ticker)
    print('Comparables used: ' + str(comparables))

    if ratio == 'P/E' or ratio.upper() == 'PE':
        pe_ratios = []
        for comp in comparables:
            try:
                stats = get_summary_statistics(comp)
                ratio = str_to_num(stats['Forward P/E'])
                pe_ratios.append(ratio)
                print('Comparable ' + comp + ' has a P/E of ' + str(ratio))
            except:
                print('Could not get the P/E ratio for comparable: ' + comp)
        multiple_of_comparables = np.nanmedian(pe_ratios)
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
            try:
                stats = get_summary_statistics(comp)
                ratio = str_to_num(stats['Enterprise Value/EBITDA'])
                ev_to_ebitda_ratios.append(ratio)
                print('Comparable ' + comp + ' has a Enterprise Value/EBITDA of ' + str(ratio))
            except:
                print('Could not get the Enterprise Value/EBITDA ratio for comparable: ' + comp)
        multiple_of_comparables = np.nanmedian(ev_to_ebitda_ratios)
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
    elif number_string[-1] == '%':
        return float(number_string[0:len(number_string) - 1])
    else:
        try:
            return float(number_string)
        except:
            return float('nan')

    
# Make a function that updates the companylist csv
# Try speeding up: https://stackoverflow.com/questions/2632520/what-is-the-fastest-way-to-send-100-000-http-requests-in-python
def update_csv(csv_name='company_statistics.csv'):
    with open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/company_data.csv', newline='') as f:
        reader = csv.reader(f)
        company_matrix = np.array(list(reader))
        company_matrix = np.delete(company_matrix, (0), axis=0)

    csvFile = open('C:/Users/kevin/Documents/Projects/Coding Projects/Stock Market/Stock-Market-Analysis/csv_files/' + csv_name, "w", newline='')
    writer = csv.writer(csvFile)
    writer.writerow(['Ticker','Name','Sector','Industry','IPO Year','Price',
                    'Market Cap', 'Trailing P/E', 'Forward P/E',
                    'PEG Ratio(5yr Expected)', 'Price/Sales(ttm)', 'Price/Book',
                    'Enterprise Value/Revenue', 'Enterprise Value/EBITDA',
                    'Profit Margin', 'Operating Margin(TTM)', 'Return on Assets(TTM)',
                    'Return on Equity(TTM)', 'Revenue(TTM)', 'Revenue Per Share(TTM)',
                    'Quarterly Revenue Growth(YOY)', 'Gross Profit(TTM)', 'EBITDA',
                    'Diluted EPS(TTM)', 'EPS Beat Ratio', 'Quarterly Earnings Growth(YOY)',
                    'Total Cash', 'Total Cash Per Share', 'Total Debt',
                    'Total Debt/Equity', 'Current Ratio', 'Book Value Per Share',
                    'Operating Cash Flow(TTM)', 'Levered Free Cash Flow(TTM)', 
                    'Beta(3Y Monthly)', 'Shares Outstanding', 'Forward Annual Dividend Rate',
                    'Forward Annual Dividend Yield', 'Trailing Annual Dividend Rate',
                    'Trailing Annual Dividend Yield', '5 Year Average Dividend Yield', 
                    'Payout Ratio'])

    i = 0
    tickers_full = company_matrix[:,0]
    name = company_matrix[:,1]
    sector = company_matrix[:,2]
    industry = company_matrix[:,3]
    ipoYear = company_matrix[:,4]
    company_prices = company_matrix[:,5]
    for ticker in tickers_full:
        print("Getting data for: " + ticker)
        price = 0
        try:
            summary = parse(ticker) # get summary info
            s = get_summary_statistics(ticker) # Get stats
            price = float(summary['Open'])
            try:
                mcap = str_to_num(summary['Market Cap'])
            except:
                mcap = float('nan')
            try:
                tpe = str_to_num(s['Trailing P/E'])
            except:
                tpe = float('nan')
            try:
                fpe = str_to_num(s['Forward P/E'])
            except:
                fpe = float('nan')
            try:
                peg = str_to_num(s['PEG Ratio (5 yr expected)'])
            except:
                peg = float('nan')
            try:
                ps = str_to_num(s['Price/Sales'])
            except:
                ps = float('nan')
            try:
                pb = str_to_num(s['Price/Book'])
            except:
                pb = float('nan')
            try:
                evr = str_to_num(s['Enterprise Value/Revenue'])
            except:
                evr = float('nan')
            try:
                evebitda = str_to_num(s['Enterprise Value/EBITDA'])
            except:
                evebitda = float('nan')
            try:
                pm = str_to_num(s['Profit Margin'])
            except:
                pm = float('nan')
            try:
                om = str_to_num(s['Operating Margin'])
            except:
                om = float('nan')
            try:
                roa = str_to_num(s['Return on Assets'])
            except: 
                roa = float('nan')
            try:
                roe = str_to_num(s['Return on Equity'])
            except:
                roe = float('nan')
            try:
                rev = str_to_num(s['Revenue'])
            except:
                rev = float('nan')
            try:
                revps = str_to_num(s['Revenue Per Share'])
            except:
                revps = float('nan')
            try:
                qrg = str_to_num(s['Quarterly Revenue Growth'])
            except:
                qrg = float('nan')
            try:
                gp = str_to_num(s['Gross Profit'])
            except: 
                gp = float('nan')
            try:
                ebitda = str_to_num(s['EBITDA'])
            except:
                ebitda = float('nan')
            try:
                deps = str_to_num(s['Diluted EPS'])
            except:
                deps = float('nan')
            try:
                epsbr = str_to_num(summary['EPS Beat Ratio'])
            except:
                epsbr = float('nan')
            try:
                qeg = str_to_num(s['Quarterly Earnings Growth'])
            except: 
                qeg = float('nan')
            try:
                totc = str_to_num(s['Total Cash'])
            except:
                totc = float('nan')
            try:
                tcps = str_to_num(s['Total Cash Per Share'])
            except:
                tcps = float('nan')
            try:
                td = str_to_num(s['Total Debt'])
            except:
                td = float('nan')
            try:
                tde = str_to_num(s['Total Debt/Equity'])
            except: 
                tde = float('nan')
            try:
                cr = str_to_num(s['Current Ratio'])
            except:
                cr = float('nan')
            try:
                bvps = str_to_num(s['Book Value Per Share'])
            except:
                bvps = float('nan')
            try:
                ocf = str_to_num(s['Operating Cash Flow'])
            except:
                ocf = float('nan')
            try:
                lfcf = str_to_num(s['Levered Free Cash Flow'])
            except:
                lfcf = float('nan')
            try:
                beta = str_to_num(s['Beta (3Y Monthly)'])
            except:
                beta = float('nan')
            try:
                so = str_to_num(s['Shares Outstanding'])
            except:
                so = float('nan')
            try:
                fadr = str_to_num(s['Forward Annual Dividend Rate'])
            except:
                fadr = float('nan')
            try:
                fady = str_to_num(s['Forward Annual Dividend Yield'])
            except:
                fady = float('nan')
            try:
                tadr = str_to_num(s['Trailing Annual Dividend Rate'])
            except:
                tadr = float('nan')
            try:
                tady = str_to_num(s['Trailing Annual Dividend Yield'])
            except:
                tady = float('nan')
            try:
                fyady = str_to_num(s['5 Year Average Dividend Yield'])
            except:
                fyady = float('nan')
            try:
                pr = str_to_num(s['Payout Ratio'])
            except:
                pr = float('nan')
            writer.writerow([ticker, name[i], sector[i], industry[i], ipoYear[i] ,str(price),
                        mcap, tpe, fpe, peg, ps, pb, evr, evebitda, pm, om, roa, roe, rev, 
                        revps, qrg, gp, ebitda, deps, epsbr, qeg, totc, tcps, td, tde, cr, bvps, 
                        ocf, lfcf, beta, so, fadr, fady, tadr, tady, fyady, pr])
        except:
            print('Ticker: ' + ticker + " did not work.")
        i += 1
    csvFile.close()

# NEXT STEPS:
# Create an industry csv with averages for everything

# Get Asset per share per price per share
def get_asset_per_share_per_price_ratio(ticker):
    total_assets = periodic_figure_values(financials_soup(ticker, "bs"), "Total Assets")[0] * 1000
    shares_outstanding = str_to_num(get_summary_statistics(ticker)["Shares Outstanding"])
    price = float(parse(ticker)['Open'])
    return total_assets / shares_outstanding / price

def get_analysis_text(ticker):
    summary_stats = get_summary_statistics(ticker)
    industry = get_company_industry(ticker)
    [industry_trailing_pe, industry_forward_pe, industry_price_to_sales, industry_price_to_book, industry_ev_to_rev, 
            industry_ev_to_ebitda, industry_profit_margin, industry_operating_margin, industry_return_on_assets, 
            industry_return_on_equity, industry_quarterly_rev_growth, industry_gross_profit, industry_quarterly_earnings_growth,
            industry_debt_to_equity, industry_current_ratio, industry_bvps, industry_beta] = get_industry_averages()
    
    altman_zscore = get_altman_zscore(ticker)
    out = ''
    out += "ANALYSIS FOR " + ticker
    out +="Industry: " + industry
    out +="Trailing P/E Ratio: " + summary_stats['Trailing P/E'] + ". Industry Average: " + str(round(industry_trailing_pe[industry], 2)) + '.'
    out +="Forward P/E Ratio: " + summary_stats['Forward P/E'] + ". Industry Average: " + str(round(industry_forward_pe[industry], 2)) + '.'
    out +="Price to Sales Ratio: " + summary_stats['Price/Sales'] + ". Industry Average: " + str(round(industry_price_to_sales[industry], 2)) + '.'
    out +="Price to Book Ratio: " + summary_stats['Price/Book'] + ". Industry Average: " + str(round(industry_price_to_book[industry], 2)) + '.'
    out +="Enterprise Value to Revenue: " + summary_stats['Enterprise Value/Revenue'] + ". Industry Average: " + str(industry_ev_to_rev[industry]) + '.'
    out +="Enterprise Value to EBITDA: " + summary_stats['Enterprise Value/EBITDA'] + ". Industry Average: " + str(round(industry_ev_to_ebitda[industry], 2)) + '.'
    out +="Profit Margin: " + summary_stats['Profit Margin'] + ". Industry Average: " + str(round(industry_profit_margin[industry], 2)) + '%.'
    out +="Operating Margin: " + summary_stats['Operating Margin'] + ". Industry Average: " + str(round(industry_operating_margin[industry], 2)) + '%.'
    out +="Return on Assets: " + summary_stats['Return on Assets'] + ". Industry Average: " + str(round(industry_return_on_assets[industry], 2)) + '%.'
    out +="Return on Equity: " + summary_stats['Return on Equity'] + ". Industry Average: " + str(round(industry_return_on_equity[industry], 2)) + '%.'
    out +="Quarterly Revenue Growth: " + summary_stats['Quarterly Revenue Growth'] #+ ". Industry Average: " + 
      #str(round(industry_quarterly_rev_growth[industry], 2)) + '%.')
    out +="Gross Profit: " + summary_stats['Gross Profit'] + ". Industry Average: " + str(round(industry_gross_profit[industry], 2)) + '.'
    out +="Quarterly Earnings Growth: " + summary_stats['Quarterly Earnings Growth'] #+ ". Industry Average: " + 
      #str(round(industry_quarterly_earnings_growth[industry], 2)) + '%.')
    out +="Debt to Equity: " + summary_stats['Total Debt/Equity'] + ". Industry Average: " + str(round(industry_debt_to_equity[industry], 2)) + '.'
    out +="Current Ratio: " + summary_stats['Current Ratio'] + ". Industry Average: " + str(round(industry_current_ratio[industry], 2)) + '.'
    out +="Book Value Per Share: " + summary_stats['Book Value Per Share'] + ". Industry Average: " + str(round(industry_bvps[industry], 2)) + '.'
    out +="Beta: " + summary_stats['Beta (3Y Monthly)'] + ". Industry Average: " + str(round(industry_beta[industry], 2)) + '.'
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
    out +="Forward Dividend & Yield: " + str(dividend_yield)
    out +="Altman Zscore: " + str(altman_zscore)
    return out


def analyze(ticker, industry=None):
    '''
    Analyzes a company, given ticker name and industry_averages dictionary
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
    summary_stats = get_summary_statistics(ticker)
    if industry == None:
        industry = get_company_industry(ticker)
    [industry_trailing_pe, industry_forward_pe, industry_price_to_sales, industry_price_to_book, industry_ev_to_rev, 
            industry_ev_to_ebitda, industry_profit_margin, industry_operating_margin, industry_return_on_assets, 
            industry_return_on_equity, industry_quarterly_rev_growth, industry_gross_profit, industry_quarterly_earnings_growth,
            industry_debt_to_equity, industry_current_ratio, industry_bvps, industry_beta] = get_industry_averages()
            
    altman_zscore = get_altman_zscore(ticker)
    print("ANALYSIS FOR " + ticker)
    print("Industry: " + str(industry))
    print("Trailing P/E Ratio: " + summary_stats['Trailing P/E'] + ". Industry Average: " + 
      str(round(industry_trailing_pe[industry], 2)) + '.')
    print("Forward P/E Ratio: " + summary_stats['Forward P/E'] + ". Industry Average: " + 
      str(round(industry_forward_pe[industry], 2)) + '.')
    print("Price to Sales Ratio: " + summary_stats['Price/Sales'] + ". Industry Average: " + 
      str(round(industry_price_to_sales[industry], 2)) + '.')
    print("Price to Book Ratio: " + summary_stats['Price/Book'] + ". Industry Average: " + 
      str(round(industry_price_to_book[industry], 2)) + '.')
    print("Enterprise Value to Revenue: " + summary_stats['Enterprise Value/Revenue'] + ". Industry Average: " + 
      str(industry_ev_to_rev[industry]) + '.')
    print("Enterprise Value to EBITDA: " + summary_stats['Enterprise Value/EBITDA'] + ". Industry Average: " + 
      str(round(industry_ev_to_ebitda[industry], 2)) + '.')
    print("Profit Margin: " + summary_stats['Profit Margin'] + ". Industry Average: " + 
      str(round(industry_profit_margin[industry], 2)) + '%.')
    print("Operating Margin: " + summary_stats['Operating Margin'] + ". Industry Average: " + 
      str(round(industry_operating_margin[industry], 2)) + '%.')
    print("Return on Assets: " + summary_stats['Return on Assets'] + ". Industry Average: " + 
      str(round(industry_return_on_assets[industry], 2)) + '%.')
    print("Return on Equity: " + summary_stats['Return on Equity'] + ". Industry Average: " + 
      str(round(industry_return_on_equity[industry], 2)) + '%.')
    print("Quarterly Revenue Growth: " + summary_stats['Quarterly Revenue Growth']) #+ ". Industry Average: " + 
      #str(round(industry_quarterly_rev_growth[industry], 2)) + '%.')
    print("Gross Profit: " + summary_stats['Gross Profit'] + ". Industry Average: " + 
      str(round(industry_gross_profit[industry], 2)) + '.')
    print("Quarterly Earnings Growth: " + summary_stats['Quarterly Earnings Growth']) #+ ". Industry Average: " + 
      #str(round(industry_quarterly_earnings_growth[industry], 2)) + '%.')
    print("Debt to Equity: " + summary_stats['Total Debt/Equity'] + ". Industry Average: " + 
      str(round(industry_debt_to_equity[industry], 2)) + '.')
    print("Current Ratio: " + summary_stats['Current Ratio'] + ". Industry Average: " + 
      str(round(industry_current_ratio[industry], 2)) + '.')
    print("Book Value Per Share: " + summary_stats['Book Value Per Share'] + ". Industry Average: " + 
      str(round(industry_bvps[industry], 2)) + '.')
    print("Beta: " + summary_stats['Beta (3Y Monthly)'] + ". Industry Average: " + 
      str(round(industry_beta[industry], 2)) + '.')
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
    print("Forward Dividend & Yield: " + str(dividend_yield))
    print("Altman Zscore: " + str(altman_zscore))
