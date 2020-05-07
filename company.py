from portfolio_helpers import *
from security import Security
import yfinance as yf
import io
import sys
from market_ml import plot_val_vs_industry
import matplotlib.pyplot as plt
from market import *
from market_ml import *
from portfolio_helpers import *
from company_helpers import *

class Company(Security):
    def __init__(self, ticker):
        super().__init__(ticker)
        self.ticker = ticker
        self.name = None
        self.industry = None
        self.comparables = None
        self.forward_pe_ratio = None
        self.trailing_pe_ratio = None
        self.peg_ratio = None
        self.eps = None
        self.beta = None
        self.market_cap = None
        self.dividend_yield = None
        self.profit_margin = None
        self.operating_margin = None
        self.roa = None
        self.roe = None
        self.revenue = None
        self.revenue_per_share = None
        self.quarterly_revenue_growth = None
        self.gross_profit = None
        self.ebitda = None
        self.net_income = None
        self.quarterly_earnings_growth = None
        self.cash = None
        self.cash_per_share = None
        self.debt = None
        self.debt_to_equity = None
        self.current_ratio = None
        self.book_value = None
        self.operating_cash_flow = None
        self.free_cash_flow = None
        self.shares_outstanding = None
        self.ev = None
        self.price_to_book = None
        self.yahoo_recommendation = None
        self.two_hundred_day_av = None
        self.fifty_day_av = None
        self.analyst_mean_target = None
        self.analyst_median_target = None
        self.num_analyst_opinions = None
        self.score = None


    def analyze(self, fetched_data=False, verbose=True, market_proxy_ticker="VTI", day_interval=30):
        if fetched_data == False:
            if self.fetch_data() == 'failure':
                print("Failed to fetch data for BMY")
                return 'failure'

        comparables = []
        for comp in self.comparables:
            firm = Company(comp)
            if firm.fetch_data() != 'failure':
                comparables.append(firm)
            else:
                print('fetch data failed for : ' + comp)
        if len(comparables) > 1:
            comparables = self.filter_comparables(comparables, lambda x: x.market_cap)

        # Health Metrics
        plt.style.use('seaborn-dark')
        fig, axs = plt.subplots(1, 5, squeeze=False, figsize=(14,4))
        fig.suptitle('Health Metrics', fontsize=11)
        health_score = 0
        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.profit_margin, "Profit Margin", axs, [0,0])
        if self_metric > comp_metric:
            health_score += 1

        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.operating_margin, "Operating Margin", axs, [0,1])
        if self_metric > comp_metric:
            health_score += 1

        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.roa, "Return On Assets", axs, [0,2])
        if self_metric > comp_metric:
            health_score += 1

        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.roe, "Return On Equity", axs, [0,3])
        if self_metric > comp_metric:
            health_score += 1

        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.current_ratio, "Current Ratio", axs, [0,4])
        if self_metric > comp_metric:
            health_score += 1

        plt.show()
        if verbose:
            print("Health Score: {} / 5".format(health_score))

        # Growth  Metrics
        fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(5,4))
        fig.suptitle('Growth Metrics', fontsize=11)
        growth_score = 0
        growth_score_denom = 0

        if self.quarterly_revenue_growth != None:
            growth_score_denom += 1
            self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.quarterly_revenue_growth, "Quarterly Revenue Growth", axs, [0,0])
            if self_metric > comp_metric:
                growth_score += 1
        if self.quarterly_earnings_growth != None:
            growth_score_denom += 1
            self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.quarterly_earnings_growth, "Quarterly Earnings Growth", axs, [0,1])
            if self_metric > comp_metric:
                growth_score += 1

        plt.show()
        if verbose:
            print("Growth Score: {} / {}".format(growth_score, growth_score_denom))

        # Value Metrics
        fig, axs = plt.subplots(1, 4, squeeze=False, figsize=(11,4))
        fig.suptitle('Value Metrics', fontsize=11)
        value_score = 0
        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.price_to_book, "Price To Book", axs, [0,0])
        if self_metric < comp_metric:
            value_score += 1
        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.trailing_pe_ratio, "Trailing PE Ratio", axs, [0,1])
        if self_metric < comp_metric:
            value_score += 1
        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.forward_pe_ratio, "Forward PE Ratio", axs, [0,2])
        if self_metric < comp_metric:
            value_score += 1
        self_metric, comp_metric = plot_company_vs_comparables(self, comparables, lambda x: x.eps, "Earnings Per Share", axs, [0,3])
        if self_metric > comp_metric:
            value_score += 1

        plt.show()
        if verbose:
            print("Value Score: {} / 4".format(value_score))

        # Analyst Metrics
        if self.num_analyst_opinions > 0:
            analyst_guess = np.mean([self.analyst_mean_target, self.analyst_median_target]) / self.historic_prices["Close"][-1] - 1
            print("Analyst Target ({} analysts): {}%".format(self.num_analyst_opinions, analyst_guess * 100))

        # Past History Analysis
        expected_market_return, expected_market_stdev, proxy = self.get_market_return_and_stdev(market_proxy_ticker, lookback_period=365*10)
        returns = self.get_past_returns(day_interval=day_interval)
        self.calculate_stdev_of_returns(returns)
        systematic_risk = self.beta * expected_market_stdev**2
        # This should be excess returns, we are assuming risk free rate is constant
        firm_specific_risk = self.stdev_of_returns**2
        variance = systematic_risk + firm_specific_risk
        if verbose:
            print("Volatility (Standard Dev.): " + str(round(np.sqrt(variance), 2)))
        price_plot(self.historic_prices.index, self.historic_prices['Close'],
            title=self.ticker + ' Price', horizontal_lines=[self.historic_prices['Close'][-1], self.fifty_day_av, self.two_hundred_day_av],
            horizontal_lines_labels=['Current Price', '50 Day Average', '200 Day Average'])
        print("Current price: " + str(self.historic_prices['Close'][-1]))

        # Perform_multiples_valuation
        multiples_valuation = multiples_analysis(self, comparables, verbose=verbose)
        if multiples_valuation == 'failed':
            print("Multiples valuation failed for " + self.ticker)
        self.score = health_score + growth_score + value_score
        return self.score


    def fetch_data(self, debug=False, none_thresh=0.2):
        if debug:
            print("Fetching Data for " + self.ticker)
        p = re.compile(r'root\.App\.main = (.*);')
        results = {}
        #if debug == False:
            # create a text trap and redirect stdout
            #text_trap = io.StringIO()
            #sys.stdout = text_trap
        yfinance_data = yf.Ticker(self.ticker)

        # get stock info
        try:
            if 'industry' in yfinance_data.info.keys():
                self.industry = yfinance_data.info['industry']
            else:
                print("Failed to get industry for " + self.ticker)
        except:
            print("Failed to get industry for " + self.ticker)
        self.historic_prices = yfinance_data.history(period="max")
        # now restore stdout function
        #sys.stdout = sys.__stdout__

        industries = get_company_industry_dict()
        self.comparables = industries[self.industry]
        if self.ticker in self.comparables:
            self.comparables.remove(self.ticker)

        with requests.Session() as s:
            r = s.get('https://finance.yahoo.com/quote/{}/key-statistics?p={}'.format(self.ticker,self.ticker))
            try:
                data = json.loads(p.findall(r.text)[0])
                key_stats = data['context']['dispatcher']['stores']['QuoteSummaryStore']
            except:
                if debug:
                    print("Parsing failed.")
                return "failure"
            self.shares_outstanding = key_stats['defaultKeyStatistics']['sharesOutstanding'].get('raw') if 'sharesOutstanding' in key_stats['defaultKeyStatistics'].keys() else None
            if self.shares_outstanding == None:
                if debug:
                    print("{} has no shares outstanding.".format(self.ticker))
                return "failure"
            self.forward_pe_ratio = key_stats['summaryDetail']['forwardPE'].get('raw') if 'forwardPE' in key_stats['summaryDetail'].keys() else None
            self.trailing_pe_ratio = key_stats['summaryDetail']['trailingPE'].get('raw') if 'trailingPE' in key_stats['summaryDetail'].keys() else None
            self.peg_ratio = key_stats['defaultKeyStatistics']['pegRatio'].get('raw') if 'pegRatio' in key_stats['defaultKeyStatistics'].keys() else None
            self.eps = key_stats['defaultKeyStatistics']['trailingEps'].get('raw') if 'trailingEps' in key_stats['defaultKeyStatistics'].keys() else None
            self.beta = key_stats['defaultKeyStatistics']['beta'].get('raw') if 'beta' in key_stats['defaultKeyStatistics'].keys() else None
            self.market_cap = key_stats['summaryDetail']['marketCap'].get('raw')
            self.dividend_yield = key_stats['summaryDetail']['dividendYield'].get('raw')
            self.profit_margin = key_stats['defaultKeyStatistics']['profitMargins'].get('raw') if 'profitMargins' in key_stats['defaultKeyStatistics'].keys() else None
            self.operating_margin = key_stats['financialData']['operatingMargins'].get('raw') if 'operatingMargins' in key_stats['financialData'].keys() else None
            self.roa = key_stats['financialData']['returnOnAssets'].get('raw') if 'returnOnAssets' in key_stats['financialData'].keys() else None
            self.roe = key_stats['financialData']['returnOnEquity'].get('raw') if 'returnOnEquity' in key_stats['financialData'].keys() else None
            self.revenue = key_stats['financialData']['totalRevenue'].get('raw') if 'totalRevenue' in key_stats['financialData'].keys() else None
            self.revenue_per_share = key_stats['financialData']['revenuePerShare'].get('raw')
            self.quarterly_revenue_growth = key_stats['financialData']['revenueGrowth'].get('raw') if 'revenueGrowth' in key_stats['financialData'].keys() else None
            self.gross_profit = key_stats['financialData']['grossProfits'].get('raw')
            self.ebitda = key_stats['financialData']['ebitda'].get('raw')
            self.net_income = key_stats['defaultKeyStatistics']['netIncomeToCommon'].get('raw') if 'netIncomeToCommon' in key_stats['defaultKeyStatistics'].keys() else None
            self.quarterly_earnings_growth = key_stats['defaultKeyStatistics']['earningsQuarterlyGrowth'].get('raw') if 'earningsQuarterlyGrowth' in key_stats['defaultKeyStatistics'].keys() else None
            self.cash = key_stats['financialData']['totalCash'].get('raw')
            self.cash_per_share = key_stats['financialData']['totalCashPerShare'].get('raw')
            self.debt = key_stats['financialData']['totalDebt'].get('raw')
            self.debt_to_equity = key_stats['financialData']['debtToEquity'].get('raw') if 'debtToEquity' in key_stats['financialData'].keys() else None
            self.current_ratio = key_stats['financialData']['currentRatio'].get('raw') if 'currentRatio' in key_stats['financialData'].keys() else None
            self.book_value = key_stats['defaultKeyStatistics']['bookValue'].get('raw') if 'bookValue' in key_stats['defaultKeyStatistics'].keys() else None
            self.operating_cash_flow = key_stats['financialData']['operatingCashflow'].get('raw') if 'operatingCashflow' in key_stats['financialData'].keys() else None
            self.free_cash_flow = key_stats['financialData']['freeCashflow'].get('raw')  if 'freeCashflow' in key_stats['financialData'].keys() else None
            self.ev = key_stats['defaultKeyStatistics']['enterpriseValue'].get('raw')
            self.price_to_book = key_stats['defaultKeyStatistics']['priceToBook'].get('raw')
            self.yahoo_recommendation = key_stats['financialData'].get('recommendationKey')
            self.two_hundred_day_av = key_stats['summaryDetail']['twoHundredDayAverage'].get('raw')
            self.fifty_day_av = key_stats['summaryDetail']['fiftyDayAverage'].get('raw')
            self.analyst_mean_target = key_stats['financialData']['targetMeanPrice'].get('raw')
            self.analyst_median_target = key_stats['financialData']['targetMedianPrice'].get('raw')
            self.num_analyst_opinions = key_stats['financialData']['numberOfAnalystOpinions'].get('raw')

        # If more than non_thresh % of data is None then consider it a failure
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        if members.count(None) / len(members) >= none_thresh:
            if debug:
                print("Number of fetched variables that are not none aren't sufficient.")
            return "failure"


    def filter_comparables(self, comparables, metric_obtainer, filter_amount=0.5):
        '''
        Filters a list of Objects along the output of a metric_obtainer function
        filter_amount is the fraction of objects to remove from the list
        '''
        # Get rid of NoneTypes and sort comparables
        comparables = [i for i in comparables if i]
        comparables.sort(key=metric_obtainer)
        n = len(comparables)
        return self.get_k_closest(comparables, self, int(n*(1 - filter_amount)), n, metric_obtainer)


    def get_k_closest(self, arr, x, k, n, key):
        '''
        Gets k closest elements to x along metric obtained by key, which should be
        a function.
        Pass in an array of objects.
        '''
        closest = []
        # Find the crossover point
        l = self.find_crossover(arr, 0, n - 1, x, key)
        r = l + 1 # Right index to search
        count = 0 # To keep track of count of
                  # elements already printed

        # If x is present in arr[], then reduce
        # left index. Assumption: all elements
        # in arr[] are distinct
        if (key(arr[l]) == key(x)):
            l -= 1

        # Compare elements on left and right of crossover
        # point to find the k closest elements
        while (l >= 0 and r < n and count < k):
            if (key(x) - key(arr[l]) < key(arr[r]) - key(x)):
                closest.append(arr[l])
                l -= 1
            else :
                closest.append(arr[r])
                r += 1
            count += 1

        # If there are no more elements on right
        # side, then print left elements
        while (count < k and l >= 0):
            closest.append(arr[l])
            l -= 1
            count += 1

        # If there are no more elements on left
        # side, then print right elements
        while (count < k and r < n):
            closest.append(arr[r])
            r += 1
            count += 1
        return closest

    def find_crossover(self, arr, low, high, x, key):
        # Base cases
        if (key(arr[high]) <= key(x)): # x is greater than all
            return high

        if (key(arr[low]) > key(x)): # x is smaller than all
            return low

        # Find the middle point
        mid = (low + high) // 2 # low + (high - low)// 2

        # If x is same as middle element,
        # then return mid
        if (key(arr[mid]) <= key(x) and key(arr[mid + 1]) > key(x)):
            return mid

        # If x is greater than arr[mid], then
        # either arr[mid + 1] is ceiling of x
        # or ceiling lies in arr[mid+1...high]
        if(key(arr[mid]) < key(x)):
            return self.find_crossover(arr, mid + 1, high, x, key)
        return self.find_crossover(arr, low, mid - 1, x, key)

    # Get the expected market return and stdev, given a market proxy
    def get_market_return_and_stdev(self, proxy_ticker, lookback_period=365*10):
        proxy = Company(proxy_ticker)
        proxy.fetch_data()
        market_returns = proxy.get_past_returns(lookback_period=lookback_period)
        return np.mean(market_returns), np.std(market_returns), proxy
