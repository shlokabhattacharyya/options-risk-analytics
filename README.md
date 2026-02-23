# Options Risk Analytics Tool

A quantitative analytics tool that applies machine learning to options pricing and portfolio risk analysis, combining Black-Scholes modeling with a Gaussian Process Regression model for volatility surface estimation.

Given a stock ticker and option type (call or put), the system retrieves live market data and the current risk-free rate via Yahoo Finance, computes implied volatility across strikes and expirations using Brent's method, and trains a GP model with an RBF kernel to learn the spatial structure of the volatility surface — interpolating a smooth, arbitrage-free surface from sparse market observations while quantifying prediction uncertainty at every point. The results are visualized across six panels: a raw IV heatmap, a GP-smoothed IV heatmap, a GP uncertainty heatmap, a volatility smile with confidence bands for the nearest expiration, and an ATM volatility term structure — all with raw market observations overlaid against the GP fit.

The portfolio risk analyzer allows users to define multiple options positions and evaluate aggregate Greeks (delta, gamma, theta, vega, rho) across the book. For positions without an exact market observation, the GP model infers implied volatility through learned surface geometry rather than a simple average, providing interpolated estimates with uncertainty bounds. A scenario analysis engine stress-tests the portfolio across a grid of price and volatility shocks, with results displayed as a P&L heatmap, per-position Greeks breakdown, and P&L profile curves.


## Process

Version 1: As I've been learning more about options theory, I wanted to explore the idea of transformaing the traditional 3D volatility surface into a 2D representation by using a heatmap. Note that this is just Version 1 of the project -- I intend on elevating the UI and even the architecture of the program at some point.

Version 2: The portfolio risk component was added to account for risk management within options trading. Using analytical pricing models (Black-Scholes, Greeks), the system derives position-level risk sensitivities and produces portfolio-level metrics to support scenario-based stress testing across price and volatility shifts.

Version 3: The codebase was refactored into a cleaner modular structure and machine learning was introduced via Gaussian Process Regression. The GP model smooths out the raw volatility surface — which is often patchy due to illiquid contracts and wide bid-ask spreads — producing a continuous, well-behaved surface across all strikes and expirations. It also quantifies how confident it is in each region of the surface, and uses that same model to fill in implied volatility estimates for portfolio positions where no direct market observation exists.


## Installation
1. Clone this repository:
```
git clone https://github.com/shlokabhattacharyya/options-risk-analytics.git
cd options-risk-analytics
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the project:
```
python main.py
```