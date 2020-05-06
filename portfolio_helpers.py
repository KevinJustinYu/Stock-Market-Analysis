from portfolio import *
from company import *

def create_portfolio(tickers, name="My Portfolio", proxy_ticker="VTI"):
    # Calculate market info
    expected_market_return, expected_market_stdev, market_proxy = get_market_return_and_stdev(proxy_ticker)
    print("Expected Market Return: {}".format(expected_market_return))
    print("Expected Stdev of market return: {}".format(expected_market_stdev))

    # Create Portfolio
    p = Portfolio(name)
    p.securities = [Company(ticker) for ticker in tickers]

    # Gather porfolio data
    p.fetch_portfolio_data(market_proxy, expected_market_return=expected_market_return, expected_market_stdev=expected_market_stdev)
    return p

def calculate_quantity_shares_per_ticker(portfolio, capital):
    '''
    Given a portfolio, with weights precomputed, and the amount of capital
    provide the number of shares of each security that yields the optimal risky portfolio
    '''
    return [portfolio.weights[i] * capital / list(portfolio.securities[i].historic_prices['Close'])[-1] for i in range(len(portfolio.weights))]
