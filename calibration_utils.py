from scipy.optimize import minimize
import numpy as np

def calibrate_heston(heston_model, market_data, initial_params, bounds=None):
    """
    Calibrate the Heston model parameters to market data.
    """
    def objective_function(params):
        kappa, theta, sigma, rho, V0 = params
        heston_model.kappa = kappa
        heston_model.theta = theta
        heston_model.sigma = sigma
        heston_model.rho = rho
        heston_model.V0 = V0
        
        errors = []
        for K, S, T, r, q, cmid, pmid in market_data:
            heston_model.S = S
            heston_model.K = K
            heston_model.T = T
            heston_model.r = r
            if S > K:
                heston_model.option_type = "call"
                market_price = cmid
            else:
                heston_model.option_type = "put"
                market_price = pmid                
            model_price = heston_model.option_price()
            errors.append((model_price - market_price) ** 2)
        
        return np.sum(errors)

    result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
    
    return result.x


def calibrate_sabr(sabr_model, market_data, initial_params, bounds=None):
    """
    Calibrate the SABR model parameters to market data.
    """
    def objective_function(params):
        alpha, beta, rho, nu = params
        sabr_model.alpha = alpha
        sabr_model.beta = beta
        sabr_model.rho = rho
        sabr_model.nu = nu
        
        errors = []
        for K, S, T, r, q, cmid, pmid in market_data:
            sabr_model.F = S
            sabr_model.K = K
            sabr_model.T = T
            sabr_model.r = r
            if S > K:
                sabr_model.option_type = "call"
                market_price = cmid
            else:
                sabr_model.option_type = "put"
                market_price = pmid                
            model_price = sabr_model.option_price()
            errors.append((model_price - market_price) ** 2)
        
        return np.sum(errors)

    result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
    
    return result.x