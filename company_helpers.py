from market_tests import *
import matplotlib.pyplot as plt
from company import *

def get_stock_prices(companies, lookback_window=252*3):
    price_data = [c.historic_prices['Close'].tail(lookback_window) for c in companies]
    ticker_names = [c.ticker for c in companies]
    df_dict = {}
    for i, _ in enumerate(price_data):
        df_dict[ticker_names[i]] = price_data[i]
    index = price_data[0].index
    return pd.DataFrame(df_dict, columns=ticker_names)

def scatterplot(x, y, title="", xlabel="", ylabel=""):
    '''
    Used for plotting stock prices
    '''
    if title == "":
        title = ylabel + " vs " + xlabel
    plt.style.use('seaborn-dark')
    plt.scatter(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


# Ranks companies by the score that company.analyze() outputs
def rank_companies_by_score(companies):
    scores = []
    for company in companies:
        scores.append(company.analyze())
    ranked_companies = [[company, score] for score,company in sorted(zip(scores,companies), key = lambda x: x[0])]
    return ranked_companies


# TODO: If there are lots of comparables, use the ones that are more similar (using various metrics)
def multiples_analysis(company, comparable_companies, verbose=True):
    if company.ebitda == None and company.revenue == None and company.net_income == None:
        return "failed"

    if verbose:
        print("Multiples valuation for " + company.ticker)
    ev_to_ebitda_ratios = []
    ev_to_rev_ratios = []
    pe_ratios = []
    for comp in comparable_companies:
        if isinstance(comp.ev, numbers.Number) and isinstance(comp.ebitda, numbers.Number) and comp.ev != float('nan') and comp.ebitda != float('nan'):
            ev_to_ebitda_ratios.append(comp.ev / comp.ebitda)
        if isinstance(comp.ev, numbers.Number) and isinstance(comp.revenue, numbers.Number) and comp.ev != float('nan') and comp.revenue != float('nan'):
            ev_to_rev_ratios.append(comp.ev / comp.revenue)
        if isinstance(comp.trailing_pe_ratio, numbers.Number) and comp.trailing_pe_ratio != float('nan') and comp.trailing_pe_ratio > 0:
            pe_ratios.append(comp.trailing_pe_ratio)

    # EV/EBITDA Analysis
    if company.ebitda is not None:
        median = company.ebitda * np.median(ev_to_ebitda_ratios)
        mean = company.ebitda * np.mean(ev_to_ebitda_ratios)
        equity_per_share_median = (median + company.cash - company.debt) / company.shares_outstanding
        equity_per_share_mean = (mean + company.cash - company.debt) / company.shares_outstanding
        ev_ebitda_valuation = equity_per_share_median
        if verbose:
            print(company.ticker + "'s ebitda is " + str(round(company.ebitda)))
            print("Industry EV/EBITDA: (Mean = " + str(round(np.mean(ev_to_ebitda_ratios))) + ") (Median = " + str(round(np.median(ev_to_ebitda_ratios))) + ")")

    # EV/revenue Analysis
    if company.revenue is not None:
        median = company.revenue * np.median(ev_to_rev_ratios)
        mean = company.revenue * np.mean(ev_to_rev_ratios)
        equity_per_share_median = (median + company.cash - company.debt) / company.shares_outstanding
        equity_per_share_mean = (mean + company.cash - company.debt) / company.shares_outstanding
        ev_rev_valuation = equity_per_share_median
        if verbose:
            print(company.ticker + "'s revenue is " + str(round(company.revenue)))
            print("Industry EV/Revenue: (Mean = " + str(round(np.mean(ev_to_rev_ratios))) + ") (Median = " + str(round(np.median(ev_to_rev_ratios))) + ")")

    # PE Analysis
    if company.net_income is not None:
        median = company.net_income * np.median(pe_ratios)
        mean = company.net_income * np.mean(pe_ratios)
        equity_per_share_median = median / company.shares_outstanding
        equity_per_share_mean = mean / company.shares_outstanding
        pe_valuation = equity_per_share_median
    if verbose:
        print(company.ticker + "'s earnings are " + str(round(company.net_income)))
        print("Industry PE ratios: (Mean = " + str(round(np.mean(pe_ratios))) + ") (Median = " + str(round(np.median(pe_ratios))) + ")")

    if verbose:
        if company.ebitda is not None:
            print("Valuation: " + str(ev_ebitda_valuation) + " (using EV/EBITDA multiple)")
        if company.revenue is not None:
            print("Valuation: " + str(ev_rev_valuation) + " (using EV/revenue multiple)")
        if company.net_income is not None:
            print("Valuation: " + str(pe_valuation) + " (using PE ratio multiple)")

    if company.ebitda != None:
        return ev_ebitda_valuation
    elif company.revenue != None:
        return ev_rev_valuation
    elif company.net_income != None:
        return pe_valuation
    else:
        raise RuntimeError('Failed to get valuation')
