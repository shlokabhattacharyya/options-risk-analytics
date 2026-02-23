### IMPORT LIBRARIES
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


### BLACK-SCHOLES OPTION PRICING

# calculate Black-Scholes call option price
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# calculate Black-Scholes put option price
def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# IMPLIED VOLATILITY (BRENT'S METHOD)
def implied_volatility(market_price, S, K, T, r, option_type):
    """
    arguments:
        - market_price: float - observed market price of the option
        - S: float - current stock price
        - K: float - strike price
        - T: float - time to expiration in years
        - r: float - annual risk-free rate
        - option_type: str - 'call' or 'put'

    returns:
        - float - implied volatility, or None if it cannot be calculated
    """
    if T <= 0:
        return None

    # intrinsic value check
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price < intrinsic:
        return None

    def objective(sigma):
        if option_type == 'call':
            return black_scholes_call(S, K, T, r, sigma) - market_price
        else:
            return black_scholes_put(S, K, T, r, sigma) - market_price

    try:
        return brentq(objective, 0.001, 5.0, xtol=1e-8, maxiter=100)
    except Exception:
        return None


### GREEKS
def calculate_greeks(S, K, T, r, sigma, option_type):
    """
    arguments:
        - S: float - current stock price
        - K: float - strike price
        - T: float - time to expiration in years
        - r: float - annual risk-free rate
        - sigma: float - volatility
        - option_type: str - 'call' or 'put'

    returns:
        - dict containing delta, gamma, theta, vega, rho
    """
    if T <= 0:
        return {
            'delta': 1.0 if option_type == 'call' else -1.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega':  0.0,
            'rho':   0.0
        }

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # delta
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1

    # gamma (same for calls and puts)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # theta (per calendar day)
    if option_type == 'call':
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    # vega (per 1% move in volatility)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # rho (per 1% change in interest rate)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega':  vega,
        'rho':   rho
    }