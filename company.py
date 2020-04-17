from company_helpers import *
from security import Security
import yfinance as yf

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


    def analyze(self, fetched_data=False, verbose=True):
        if fetched_data == False:
            self.fetch_data()
        comparables = []
        for comp in self.comparables:
            firm = Company(comp)
            if firm.fetch_data() != 'failure':
                comparables.append(firm)

        # Health Metrics
        health_score = 0
        if self.profit_margin > np.nanmean(list(map(lambda x: x.profit_margin, comparables))):
            health_score += 1
        if self.operating_margin > np.nanmean(list(map(lambda x: x.operating_margin, comparables))):
            health_score += 1
        if self.roa > np.nanmean([i for i in list(map(lambda x: x.roa, comparables)) if i != None]):
            health_score += 1
        if self.roe > np.nanmean([i for i in list(map(lambda x: x.roe, comparables)) if i != None]):
            health_score += 1
        if verbose:
            print("Health Score: {} / 4".format(health_score))

        # Growth  Metrics
        growth_score = 0
        growth_score_denom = 0
        if self.quarterly_revenue_growth != None:
            growth_score_denom += 1
            if self.quarterly_revenue_growth > np.nanmean([i for i in list(map(lambda x: x.quarterly_revenue_growth, comparables)) if i != None]):
                growth_score += 1
        if self.quarterly_earnings_growth != None:
            growth_score_denom += 1
            if self.quarterly_earnings_growth > np.nanmean([i for i in list(map(lambda x: x.quarterly_earnings_growth, comparables)) if i != None]):
                growth_score += 1
        if verbose:
            print("Growth Score: {} / {}".format(growth_score, growth_score_denom))

        # Value Metrics
        value_score = 0
        if self.price_to_book < np.nanmean([i for i in list(map(lambda x: x.price_to_book, comparables)) if i != None]):
            value_score += 1
        if verbose:
            print("Value Score: {} / 1".format(value_score))

        # Perform_multiples_valuatoin
        multiples_valuation = multiples_analysis(self, comparables, verbose=False)
        self.score = health_score + growth_score + value_score
        return self.score


    def fetch_data(self, debug=False, none_thresh=0.2):
        if debug:
            print("Fetching Data for " + self.ticker)
        p = re.compile(r'root\.App\.main = (.*);')
        results = {}
        self.historic_prices = yf.download(self.ticker)
        self.industry =  get_company_industry(self.ticker)
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
                return "failure"
            self.shares_outstanding = key_stats['defaultKeyStatistics']['sharesOutstanding'].get('raw') if 'sharesOutstanding' in key_stats['defaultKeyStatistics'].keys() else None
            if self.shares_outstanding == None:
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
            return "failure"
