### IMPORT LIBRARIES
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


### BLACK-SCHOLES FUNCTIONS
# calculate Black-Scholes call option price
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# calculate Black-Scholes put option price
def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# calculate implied volatility using Brent's method
def implied_volatility(market_price, S, K, T, r, option_type):
    if T <= 0:
        return None
    
    # intrinsic value check
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price < intrinsic:
        return None

    # define objective function
    def objective(sigma):
        if option_type == 'call':
            return black_scholes_call(S, K, T, r, sigma) - market_price
        else:
            return black_scholes_put(S, K, T, r, sigma) - market_price
        
    try:
        # use Brent's method
        iv = brentq(objective, 0.001, 5.0, xtol = 1e-8, maxiter=100)
        return iv
    except:
        return None
    

### DATA COLLECTION AND PROCESSING (UPDATED)
def get_volatility_surface(ticker, option_type, risk_free_rate=0.05):
    """
    arguments:
        - ticker: str - stock ticker symbol
        - option_type: str - 'call' or 'put'
        - risk_free_rate: float - annual risk-free rate

    returns:
        - dict containing surface data, stock price, and raw option data
    """
    
    print(f"Fetching data for {ticker}...")

    # collect stock data
    stock = yf.Ticker(ticker)
    stock_price = stock.history(period='1d')['Close'].iloc[-1]

    print(f"Current stock price: ${stock_price:.2f}")

    # collect expiration date data
    expirations = stock.options

    if len(expirations) == 0:
        raise ValueError(f"No options data could be found for {ticker}.")
    
    # only using first 6 expirations for cleaner visualization
    expirations = expirations[:6]

    print(f"Found {len(expirations)} expiration dates")

    # collect all option data
    all_options = []
    
    for exp_date in expirations:
        print(f"Processing expiration: {exp_date}")

        opt_chain = stock.option_chain(exp_date)
        options = opt_chain.calls if option_type == 'call' else opt_chain.puts

        exp_datetime = pd.to_datetime(exp_date)
        days_to_exp = (exp_datetime - pd.Timestamp.now()).days
        years_to_exp = days_to_exp / 365.0

        for _, row in options.iterrows():
            strike = row['strike']
            
            # FILTER: Only keep strikes within reasonable range of stock price
            # This prevents the chart from being too wide
            if strike < stock_price * 0.7 or strike > stock_price * 1.3:
                continue

            if pd.notna(row['lastPrice']) and row['lastPrice'] > 0:
                market_price = row['lastPrice']
            elif pd.notna(row['bid']) and pd.notna(row['ask']):
                market_price = (row['bid'] + row['ask']) / 2
            else:
                continue

            # filter out options with low liquidity or extreme prices
            if market_price < 0.05:
                continue

            volume = row.get('volume', 0)
            if pd.isna(volume):
                volume = 0

            iv = implied_volatility(market_price, stock_price, strike, years_to_exp, risk_free_rate, option_type)

            # only keep reasonable iv values
            if iv is not None and 0.01 < iv < 3.0:
                all_options.append({
                    'expiration': exp_date,
                    'days_to_exp': days_to_exp,
                    'strike': strike,
                    'market_price': market_price,
                    'iv': iv * 100, # convert to percentage
                    'volume': volume,
                    'moneyness': stock_price / strike,
                    'bid': row.get('bid', np.nan),
                    'ask': row.get('ask', np.nan)       
                })

    if len(all_options) == 0:
        raise ValueError("Could not calculate implied volatility for any options. Try a more liquid stock.")
    
    print(f"Successfully calculated implied volatility for {len(all_options)} options.")

    # create DataFrame
    df = pd.DataFrame(all_options)

    # create pivot table for heatmap
    pivot = df.pivot_table(
        values = 'iv',
        index = 'days_to_exp',
        columns = 'strike',
        aggfunc = 'mean'
    )

    return{
        'pivot': pivot,
        'df': df,
        'stock_price': stock_price,
        'ticker': ticker,
        'option_type': option_type
    }


### VISUALIZATION
def plot_volatility_surface(surface_data):
    """
    arguments:
        - surface_data: dict - output from get_volatility_surface()
    """

    pivot = surface_data['pivot']
    df = surface_data['df']
    stock_price = surface_data['stock_price']
    ticker = surface_data['ticker']
    option_type = surface_data['option_type']

    fig = plt.figure(figsize = (12, 7))

    ## heatmap ##
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

    # decide whether to show annotations based on data size
    show_annot = len(pivot.columns) <= 30  # only annotate if <= 30 strikes

    # robust color scaling to highlight relative differences
    vmin = np.nanpercentile(pivot.values, 10)  # 10th percentile (ignore outliers)
    vmax = np.nanpercentile(pivot.values, 90)  # 90th percentile (ignore outliers)

    cmap = sns.color_palette('RdYlGn_r', as_cmap=True)
    cmap.set_bad(color='#ebe9e6') 

    sns.heatmap(
        pivot,
        annot = show_annot,  # conditional annotations
        fmt = '.0f', 
        cmap = cmap,
        cbar_kws = {'label': 'Implied Volatility (%)'},
        linewidths = 0.3,
        vmin = vmin, 
        vmax = vmax, 
        robust = True,
        ax = ax1
    )

    ax1.set_title(
        f'{ticker} Volatility Surface - {option_type.upper()}s\n'
        f'Current Price: ${stock_price:.2f}',
        fontsize = 14, fontweight = 'bold', pad = 20
    )
    ax1.set_xlabel('Strike Price', fontsize = 12, fontweight = 'bold')
    ax1.set_ylabel('Days to Expiration', fontsize = 12, fontweight = 'bold')

    # rotate x-axis labels for better readability
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # mark current stock price
    strikes = pivot.columns
    closest_strike_idx = np.argmin(np.abs(strikes - stock_price))
    for i in range(len(pivot)):
        ax1.add_patch(plt.Rectangle(
            (closest_strike_idx, i), 1, 1, 
            fill = False, edgecolor = '#cc66b3', lw = 2.5
        ))

    ## volatility smile ##
    ax2 = plt.subplot(2, 2, 3)
    
    # get the shortest expiration for smile
    shortest_exp = df['days_to_exp'].min()
    smile_data = df[df['days_to_exp'] == shortest_exp].sort_values('strike')
    
    ax2.plot(smile_data['strike'], smile_data['iv'], 'o-', linewidth=2, markersize=6, color='#6d4c9c')
    ax2.axvline(stock_price, color='#cc66b3', linestyle='--', linewidth=2, label='Current Price')
    ax2.set_xlabel('Strike Price', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Implied Volatility (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Volatility Smile ({shortest_exp} days)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ## term structure ##
    ax3 = plt.subplot(2, 2, 4)
    
    # get ATM options (closest to current price)
    df['strike_diff'] = abs(df['strike'] - stock_price)
    atm_data = df.loc[df.groupby('days_to_exp')['strike_diff'].idxmin()]
    atm_data = atm_data.sort_values('days_to_exp')
    
    ax3.plot(atm_data['days_to_exp'], atm_data['iv'], 'o-', linewidth=2, markersize=6, color='#3682cf')
    ax3.set_xlabel('Days to Expiration', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Implied Volatility (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Volatility Term Structure (ATM)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # print summary statistics
    print("SUMMARY STATISTICS")
    print(f"Ticker: {ticker}")
    print(f"Stock Price: ${stock_price:.2f}")
    print(f"Option Type: {option_type.upper()}")
    print(f"Number of options analyzed: {len(df)}")
    print(f"\nIMPLIED VOLATILITY STATISTICS:")
    print(f"Mean IV: {df['iv'].mean():.2f}%")
    print(f"Median IV: {df['iv'].median():.2f}%")
    print(f"Min IV: {df['iv'].min():.2f}%")
    print(f"Max IV: {df['iv'].max():.2f}%")
    print(f"Std Dev: {df['iv'].std():.2f}%")

   
### MAIN
if __name__ == "__main__":

    RISK_FREE_RATE = 0.05 # 5%

    # get ticker from user
    while True:
        ticker = input("\nEnter stock ticker (e.g., AAPL, TSLA, SPY): ").strip().upper()
        if ticker:
            break
        print("Please enter a valid ticker symbol.")
    
    # get option type from user
    while True:
        opt_input = input("Option type? (type C/P): ").strip().upper()
        if opt_input == 'C':
            option_type = 'call'
            break
        elif opt_input == 'P':
            option_type = 'put'
            break
        else:
            print("Please enter 'C' for Call or 'P' for Put.")
    

    print(f"Ticker: {ticker}")
    print(f"Option Type: {option_type.upper()}")
    print(f"Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%")
    
    try:
        # get stock data and calculate surface
        surface_data = get_volatility_surface(
            ticker, 
            option_type=option_type,
            risk_free_rate=RISK_FREE_RATE
        )
        
        # Visualize
        plot_volatility_surface(surface_data)
        
     
        # ask if user wants to analyze another ticker
        again = input("\nAnalyze another ticker? (y/n): ").strip().lower()
        if again == 'y':
            # re-run the script by calling main recursively
            print("\n" * 2)
            if __name__ == "__main__":
                exec(open(__file__).read())
        
    except Exception as e:
        print(f"\nError: {e}")