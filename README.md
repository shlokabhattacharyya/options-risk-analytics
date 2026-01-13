# Options Risk Analytics Tool

A quantitative analytics tool computes implied volatility surfaces, Greeks, and portfolio risk metrics using Black-Scholes modeling with real-time market data integration.

Given a stock ticker and option type (call or put), the system retrives live market data via Yahoo Finance, computes implied volatility across strikes and expirations, and visualizes the resulting volatility surface using a two-dimensional heatmap representation, along with a volatility smile for the nearest expiration and an at-the-money (ATM) volatility term structure across maturities.

This project also includes a portfolio risk analyzer that allows users to define multiple options positions and evaluate their aggregate risk exposures.


## Process

Version 1: As I've been learning more about options theory, I wanted to explore the idea of transformaing the traditional 3D volatility surface into a 2D representation by using a heatmap. Note that this is just Version 1 of the project -- I intend on elevating the UI and even the architecture of the program at some point.

Version 2: The portfolio risk component was added to account for risk management within options trading. Using analytical pricing models (Black-Scholes, Greeks), the system derives position-level risk sensitivities and produces portfolio-level metrics to support scenario-based stress testing across price and volatility shifts.


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