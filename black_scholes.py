import math
from scipy.stats import norm
from scipy.optimize import newton

class BlackScholesIV:
    def __init__(self, S, K, T, r, q, option_type='call'):
        """
        Initialize the Black-Scholes model parameters.

        :param S: Spot price of the underlying asset
        :param K: Strike price of the option
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate
        :param q: Dividend yield
        :param option_type: 'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.option_type = option_type

    def d1(self, sigma):
        return (math.log(self.S / self.K) + (self.r - self.q + 0.5 * sigma ** 2) * self.T) / (sigma * math.sqrt(self.T))

    def d2(self, sigma):
        return self.d1(sigma) - sigma * math.sqrt(self.T)

    def bs_price(self, sigma):
        """
        Calculate the Black-Scholes price given a volatility sigma.
        """
        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        if self.option_type == 'call':
            price = (self.S * math.exp(-self.q * self.T) * norm.cdf(d1) -
                     self.K * math.exp(-self.r * self.T) * norm.cdf(d2))
        elif self.option_type == 'put':
            price = (self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) -
                     self.S * math.exp(-self.q * self.T) * norm.cdf(-d1))
        return price

    def implied_volatility(self, market_price, tol=1e-6, max_iter=1000):
        """
        Calculate the implied volatility using the Newton-Raphson method.
        """
        def objective_function(sigma):
            return self.bs_price(sigma) - market_price

        initial_guess = 0.2
        iv = newton(objective_function, initial_guess, tol=tol, maxiter=max_iter)
        return iv