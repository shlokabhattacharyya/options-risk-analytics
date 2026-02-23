### IMPORT LIBARIES
import warnings
import numpy as np
import pandas as pd

from surface import get_volatility_surface, fit_gp_surface
from visualization import plot_volatility_surface
from portfolio import PortfolioRiskAnalyzer

warnings.filterwarnings('ignore')

### RISK FREE RATE (13-WEEK T-BILL YIELD (^IRX))
def get_risk_free_rate(fallback=0.05):
    try:
        import yfinance as yf
        rate = yf.Ticker("^IRX").history(period="1d")['Close'].iloc[-1] / 100
        print(f"Live risk-free rate (13-week T-bill): {rate * 100:.3f}%")
        return rate
    except Exception:
        print(f"Could not fetch live rate — using fallback: {fallback * 100:.2f}%")
        return fallback

RISK_FREE_RATE = get_risk_free_rate()


### USER INPUTS
def get_ticker():
    while True:
        ticker = input("\nEnter stock ticker (e.g., AAPL, TSLA, SPY): ").strip().upper()
        if ticker:
            return ticker
        print("Please enter a valid ticker symbol.")

def get_option_type():
    while True:
        choice = input("Option type? (C / P): ").strip().upper()
        if choice == 'C':
            return 'call'
        if choice == 'P':
            return 'put'
        print("Please enter 'C' for Call or 'P' for Put.")


### LOOK UP IMPLIED VOLATILITY FOR A POSITION
def get_iv_for_position(pos_strike, pos_exp, surface_data):
    """
    priority: exact match in raw data -> GP interpolation -> dataset mean
    """
    df = surface_data['df']
    match = df[(df['strike'] == pos_strike) & (df['expiration'] == pos_exp)]

    if len(match) > 0:
        return match.iloc[0]['iv'] / 100

    gp = surface_data.get('gp_results')
    if gp is not None:
        days = (pd.to_datetime(pos_exp) - pd.Timestamp.now()).days
        X_query = gp['scaler_X'].transform([[pos_strike, days]])
        iv_pred, iv_std = gp['gp_model'].predict(X_query, return_std=True)
        iv = float(np.clip(iv_pred[0], 0.5, 300)) / 100
        print(f"Using GP-interpolated IV: {iv * 100:.1f}%  (+/-{iv_std[0]:.1f}%)")
        return iv

    iv = df['iv'].mean() / 100
    print(f"Using average IV: {iv * 100:.1f}%")
    return iv


### INTERACTIVE LOOP FOR PORTFOLIO ANALYSIS
def run_portfolio_analysis(surface_data):
    portfolio = PortfolioRiskAnalyzer(
        surface_data['ticker'],
        surface_data['stock_price'],
        RISK_FREE_RATE
    )

    print("\nPORTFOLIO RISK ANALYSIS")
    print("Enter positions one per line, or 'done' when finished.")
    print("Format:   [call | put]  [strike]  [YYYY-MM-DD]  [quantity]")
    print("Example:  call 260 2026-06-20 10")
    print("Example:  put 240 2026-06-20 -5")

    while True:
        raw = input("\nPosition: ").strip().lower()
        if raw == 'done':
            break

        try:
            parts = raw.split()
            if len(parts) != 4:
                print("Invalid format. Expected: call 260 2026-06-20 10")
                continue

            pos_type   = parts[0]
            pos_strike = float(parts[1])
            pos_exp    = parts[2]
            pos_qty    = int(parts[3])

            if pos_type not in ('call', 'put'):
                print("Option type must be 'call' or 'put'.")
                continue

            iv = get_iv_for_position(pos_strike, pos_exp, surface_data)
            portfolio.add_position(pos_type, pos_strike, pos_exp, pos_qty, iv)

        except Exception as e:
            print(f"Error: {e}. Please try again.")

    if portfolio.positions:
        portfolio.plot_risk_dashboard()
    else:
        print("No positions added.")


### MAIN
def main():
    print("\nVOLATILITY SURFACE & PORTFOLIO RISK ANALYSER")

    while True:
        ticker = get_ticker()
        option_type = get_option_type()

        print(f"\nTicker:         {ticker}")
        print(f"Option Type:      {option_type.upper()}")
        print(f"Risk-Free Rate:   {RISK_FREE_RATE * 100:.2f}%")

        try:
            surface_data = get_volatility_surface(ticker, option_type, RISK_FREE_RATE)
            surface_data['gp_results'] = fit_gp_surface(
                surface_data['df'], surface_data['stock_price']
            )
            plot_volatility_surface(surface_data)

            if input("\nAnalyse a portfolio? (y/n): ").strip().lower() == 'y':
                run_portfolio_analysis(surface_data)

        except Exception as e:
            print(f"\nError: {e}")
            print("Tip: try a highly liquid ticker (AAPL, TSLA, SPY, MSFT).")

        if input("\nAnalyse another ticker? (y/n): ").strip().lower() != 'y':
            break


if __name__ == "__main__":
    main()