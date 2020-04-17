import numpy as np
from security import Security

class Portfolio:
    def __init__(self, securities):
        self.securities = securities

    def fetch_portfolio_data(self, expected_market_return, expected_market_stdev, day_interval=30, verbose=True):
        assert self.securities != None
        if verbose:
            print("Fetching Portfolio Data")
        for s in self.securities:
            s.update_all_data(expected_market_return, expected_market_stdev);
            if verbose:
                print("Getting data for " + s.ticker)
                print("Expected return: " + str(s.expected_return))
                print("Standard Deviation: " + str(s.stdev_of_returns))
                print("Beta: " + str(s.beta))
                print("Alpha: " + str(s.alpha))

    def check_weights_valid(self, weights):
        assert len(weights) == len(self.securities), "Length mismatch between weights({}) and securities ({})".format(len(weights), len(self.securities))
        assert sum(weights) == 1, "Sum of weights must equal 1. It is " + str(sum(weights))

    def calculate_sharpe_ratio(self, weights, expected_market_return, cov_matrix=None, corr_matrix=None):
        expected_returns = self.calculate_expected_return(weights, expected_market_return)
        expected_stdev = self.calculate_expected_stdev(weights, cov_matrix=cov_matrix, corr_matrix=corr_matrix)
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
        # Convert to cov_matrix
        if corr_matrix:
            for i in range(len(weights)):
                for j in range(len(weights)):
                    if i == j:
                        corr_matrix[i][j] = self.securities[i].stdev_of_returns**2
                    else:
                        corr_matrix[i][j] = self.securities[i].stdev_of_returns * self.securities[j].stdev_of_returns * corr_matrix[i][j]
            cov_matrix = corr_matrix
        for i in range(len(weights)):
                for j in range(len(weights)):
                        var += weights[i] * weights[j] * cov_matrix[i][j]
        return np.sqrt(var)

    def get_cov_matrix_using_betas(self, market_var):
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
        return cov_matrix
