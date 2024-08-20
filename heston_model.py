import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

class HestonModel:
    def __init__(self, S0, K, T, r, kappa, theta, sigma, rho, V0, option_type='call'):
        """
        Initialize the Heston model parameters.

        :param S0: Initial spot price of the underlying asset
        :param K: Strike price of the option
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate
        :param kappa: Mean reversion rate of variance
        :param theta: Long-term variance
        :param sigma: Volatility of variance (vol of vol)
        :param rho: Correlation between asset price and variance
        :param V0: Initial variance
        :param option_type: 'call' or 'put'
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.V0 = V0
        self.option_type = option_type

    def generate_paths(self, N, M):
        """
        Generate asset price paths using the Heston model.
        """
        dt = self.T / N
        S = np.zeros((M, N + 1))
        V = np.zeros((M, N + 1))
        S[:, 0] = self.S0
        V[:, 0] = self.V0

        for t in range(1, N + 1):
            Z1 = np.random.normal(0, 1, M)
            Z2 = np.random.normal(0, 1, M)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

            V[:, t] = np.maximum(
                V[:, t - 1] + self.kappa * (self.theta - V[:, t - 1]) * dt +
                self.sigma * np.sqrt(V[:, t - 1] * dt) * W2, 0)
            
            S[:, t] = S[:, t - 1] * np.exp(
                (self.r - 0.5 * V[:, t - 1]) * dt + np.sqrt(V[:, t - 1] * dt) * W1)

        return S, V

    def option_price(self, N=100, M=10000):
        """
        Calculate the option price using Monte Carlo simulation.
        """
        S, _ = self.generate_paths(N, M)

        if self.option_type == 'call':
            payoff = np.maximum(S[:, -1] - self.K, 0)
        elif self.option_type == 'put':
            payoff = np.maximum(self.K - S[:, -1], 0)

        option_price = np.exp(-self.r * self.T) * np.mean(payoff)
        return option_price

    def implied_volatility(self, market_price, tol=1e-6, max_iter=100):
        """
        Calculate the implied volatility using the Heston model and numerical methods.
        """
        def objective_function(sigma):
            self.sigma = sigma
            return self.option_price() - market_price

        iv = newton(objective_function, self.sigma, tol=tol, maxiter=max_iter)
        return iv
