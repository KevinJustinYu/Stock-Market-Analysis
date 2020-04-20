import numpy as np
from security import Security

class Portfolio:
    def __init__(self, securities):
        self.securities = securities
        self.weights = None
        self.cov_matrix = None


    def fetch_portfolio_data(self, market_comparable, expected_market_return=0.0955, expected_market_stdev=0.17, day_interval=30, verbose=True):
        '''
        Call this function before calculating portfolio information.
        '''
        assert self.securities != None
        if verbose:
            print("Fetching Portfolio Data")
        for s in self.securities:
            s.update_all_data(market_comparable, expected_market_return, expected_market_stdev);
            if verbose:
                print("Getting data for " + s.ticker)
                print("Expected return: " + str(s.expected_return))
                print("Standard Deviation: " + str(s.stdev_of_returns))
                print("Beta: " + str(s.beta))
                print("Alpha: " + str(s.alpha))
        self.get_cov_matrix_using_betas(market_var=expected_market_stdev**2)


    def check_weights_valid(self, weights):
        '''
        Helper function used to check that weights are valid
        '''
        assert len(weights) == len(self.securities), "Length mismatch between weights({}) and securities ({})".format(len(weights), len(self.securities))
        assert sum(weights) == 1 or sum(weights) - 1 < 0.00001, "Sum of weights must equal 1. It is " + str(sum(weights))


    def calculate_optimal_weights(self):
        '''
        Uses strategy on page 267 of Investments (Bodie, Kane, Marcus)
        to calculate optimal weights of a risky portfolio
        '''
        weights = []
        # Compute the initial position of each security
        for security in self.securities:
            weights.append(security.alpha / security.stdev_of_returns**2)

        # Scale initial positions to force portfolio weights to sum to 1
        weights = np.divide(weights, sum(weights))
        self.check_weights_valid(weights)

        self.weights = weights
        return weights


    def calculate_risky_portfolio_weight(self):
        '''
        Calculates optimal position in the active portfolio, vs the market
        Steps 3-10 on page 267 of Investments (Bodie, Kane, Marcus)
        '''
        # Compute the alhpa and resid variance of the active portfolio
        portfolio_alpha = 0
        portfolio_variance = 0
        for i, weight in enumerate(weights):
            portfolio_alpha += weight * self.securities[i].alpha
            portfolio_variance += weight**2 * self.securities[i].stdev_of_returns**2

        # Compute the initial position in the active portfolio
        pass


    def calculate_sharpe_ratio(self, weights, expected_market_return, corr_matrix=None):
        '''
        Calculates the sharpe ratio (expected return / expected risk)
        given weights, expected_market_return, and a covariance matrix.
        Make sure self.cov_matrix is calculated before calling this function,
        or set corr_matrix
        '''
        expected_returns = self.calculate_expected_return(weights, expected_market_return)
        expected_stdev = self.calculate_expected_stdev(weights, cov_matrix=self.cov_matrix, corr_matrix=corr_matrix)
        return expected_returns / expected_stdev


    def calculate_expected_return(self, weights, expected_market_return):
        self.check_weights_valid(weights)
        expected_returns = [r.calculate_expected_return(float(r.score)/70, expected_market_return) for r in self.securities]
        return sum([w*r for w, r in zip(weights, expected_returns)])


    # Covariance(ij) = correlation(ij) * stdev(i) * stdev(j)
    def calculate_expected_stdev(self, weights, cov_matrix=None, corr_matrix=None):
        self.check_weights_valid(weights)
        assert type(cov_matrix) != type(None) or type(corr_matrix) != type(None), "Make sure you pass in a covariance or correlation matrix."
        var = 0
        n = len(weights)
        # Convert to cov_matrix
        if corr_matrix:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        corr_matrix[i][j] = self.securities[i].stdev_of_returns**2
                    else:
                        corr_matrix[i][j] = self.securities[i].stdev_of_returns * self.securities[j].stdev_of_returns * corr_matrix[i][j]
            cov_matrix = corr_matrix
        for i, wi in enumerate(weights):
                for j, wj in enumerate(weights):
                        var += wi * wj * cov_matrix[i][j]
        return np.sqrt(var)


    def get_cov_matrix_using_betas(self, market_var=0.05**2):
        n = len(self.securities)
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    systematic_risk = self.securities[i].beta * market_var
                    # This should be excess returns, we are assuming risk free rate is constant
                    firm_specific_risk = self.securities[i].stdev_of_returns**2
                    cov_matrix[i][j] = systematic_risk + firm_specific_risk
                else:
                    cov_matrix[i][j] = self.securities[i].beta * self.securities[j].beta * market_var
        self.cov_matrix = cov_matrix
        return cov_matrix
