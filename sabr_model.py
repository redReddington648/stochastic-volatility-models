import numpy as np
from scipy.stats import norm


class SABRModel:
    def __init__(self, F, K, T, r, alpha, beta, rho, nu, option_type):
        """
        Initialize the SABR model parameters.

        :param F: Forward price
        :param K: Strike price
        :param T: Time to maturity
        :param alpha: Initial volatility
        :param beta: Elasticity parameter
        :param rho: Correlation coefficient between asset price and volatility
        :param nu: Volatility of volatility
        """
        self.F = F
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.option_type = option_type
        self.r = r

    def implied_volatility(self):
        """
        Calculate the implied volatility using the Hagan et al. SABR approximation formula.
        """
        if self.F == self.K:
            # ATM implied volatility
            term1 = self.alpha / (self.F ** (1 - self.beta))
            term2 = ((1 - self.beta) ** 2 / 24) * (self.alpha ** 2 / (self.F ** (2 - 2 * self.beta)))
            term3 = (self.rho * self.beta * self.nu * self.alpha) / (4 * self.F ** (1 - self.beta))
            term4 = (2 - 3 * self.rho ** 2) * (self.nu ** 2 / 24)
            return term1 * (1 + (term2 + term3 + term4) * self.T)
        else:
            # OTM/ITM
            z = (self.nu / self.alpha) * ((self.F * self.K) ** ((1 - self.beta) / 2)) * np.log(self.F / self.K)
            x_z = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))

            term1 = self.alpha / ((self.F * self.K) ** ((1 - self.beta) / 2) * (1 + ((1 - self.beta) ** 2 / 24) * (np.log(self.F / self.K) ** 2) + ((1 - self.beta) ** 4 / 1920) * (np.log(self.F / self.K) ** 4)))
            term2 = z / x_z
            term3 = 1 + (((1 - self.beta) ** 2 / 24) * (self.alpha ** 2 / ((self.F * self.K) ** (1 - self.beta))) + (self.rho * self.beta * self.nu * self.alpha / (4 * (self.F * self.K) ** ((1 - self.beta) / 2))) + ((2 - 3 * self.rho ** 2) * (self.nu ** 2 / 24))) * self.T
            return term1 * term2 * term3

    def option_price(self):
        """
        Calculate the option price using the SABR model.
        """
        vol = self.implied_volatility()
        d1 = (np.log(self.F / self.K) + 0.5 * vol ** 2 * self.T) / (vol * np.sqrt(self.T))
        d2 = d1 - vol * np.sqrt(self.T)

        if self.option_type == 'call':
            price = np.exp(-self.r * self.T) * (self.F * norm.cdf(d1) - self.K * norm.cdf(d2))
        elif self.option_type == 'put':
            price = np.exp(-self.r * self.T) * (self.K * norm.cdf(-d2) - self.F * norm.cdf(-d1))
        
        return price