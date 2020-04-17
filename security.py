import numpy as np

class Security:
    def __init__(self, name):
        self.name = name
        self.expected_return = None
        self.stdev_of_returns = None
        self.beta = None
        self.historic_prices = None
        self.alpha = None

    def update_all_data(self, expected_market_return, expected_market_std, day_interval=30):
        self.fetch_data()
        score = self.analyze(fetched_data=True, verbose=False)
        self.alpha = float(score) / 70
        returns = self.get_past_returns(day_interval)
        self.calculate_stdev_of_returns(returns)
        self.calculate_expected_return(self.alpha, expected_market_return)

    def calculate_stdev_of_returns(self, returns):
        self.stdev_of_returns = np.std(returns)
        return self.stdev_of_returns

    def calculate_expected_return(self, alpha, expected_market_return, set=True):
        '''
        Calculates the expected return using the single index model
        E(R) = alpha + beta * expected_market_return
        '''
        expected_return = alpha + self.beta * expected_market_return
        if set:
            self.expected_return = expected_return
        return expected_return

    def get_past_returns(self, day_interval, lookback_period=730):
        closing_prices = list(self.historic_prices['Adj Close'])
        closing_prices = closing_prices[-lookback_period:]
        returns = []
        cur_day = 0
        for price in closing_prices:
            if cur_day == 0:
                start_price = price
                cur_day += 1
            elif cur_day == day_interval:
                returns.append(price / start_price - 1)
                cur_day == 0
            else:
                cur_day += 1
        return returns
