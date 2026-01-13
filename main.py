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
    

### GREEKS CALUCLATION
# calculate all Greeks for an option
def calculuate_greeks(S, K, T, r, sigma, option_type):
    """
    returns:   
        - dict containing delta, gamma, theta, vega, rho
    """

    if T <= 0: 
        return {
            'delta': 1.0 if option_type == 'call' else '-1.0',
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }

    d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    # delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else: # if option_type == 'put'
        delta = norm.cdf(d1) - 1

    # gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # theta (per day)
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else: # if option_type == 'put'
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # vega (per 1% move in volatility)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # rho (per 1% change in interest rate)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else: # if option_type == 'put'
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


### DATA COLLECTION AND PROCESSING
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
            
            # filter: only keep strikes within reasonable range of stock price
            # this prevents the chart from being too wide
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
    print("\nSUMMARY STATISTICS")
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

   
### PORTFOLIO RISK ANALYZER
# analyze risk for a portfolio of options positions
class PortfolioRiskAnalyzer:
    def __init__(self, ticker, stock_price, risk_free_rate):
        self.ticker = ticker
        self.stock_price = stock_price
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)
        self.positions = []

    # add an option position to the portfolio
    def add_position(self, option_type, strike, expiration, quantity, iv=None):
        exp_date = pd.to_datetime(expiration)
        days_to_exp = (exp_date - pd.Timestamp.now()).days
        years_to_exp = days_to_exp / 365

        if years_to_exp <= 0:
            print (f"Warning: Option {strike} {option_type} already expired")
            return
        
        if iv is None:
            try:
                opt_chain = self.stock.option_chain(expiration)
                options = opt_chain.calls if option_type == 'call' else opt_chain.puts
                closest_idx = (options['strike'] - strike).abs().idxmin()
                iv = options.loc[closest_idx, 'impliedVolatility']
                market_price = options.loc[closest_idx, 'lastPrice']
            except:
                print(f"Could not find IV for {strike} {option_type}, using 30%")
                iv = 0.30
                market_price = None
        else:
            market_price = None

        # calculate Greeks
        greeks = calculuate_greeks(self.stock_price, strike, years_to_exp, self.risk_free_rate, iv, option_type)

        position = {
            'option_type': option_type,
            'strike': float(strike),
            'expiration': expiration,
            'days_to_exp': int(days_to_exp),
            'years_to_exp': float(years_to_exp),
            'quantity': int(quantity),
            'iv': float(iv),
            'market_price': float(market_price) if market_price is not None else None,
            'delta': float(greeks['delta']),
            'gamma': float(greeks['gamma']),
            'theta': float(greeks['theta']),
            'vega': float(greeks['vega']),
            'rho': float(greeks['rho']),
        }

        self.positions.append(position)
        print(f"Added: {quantity:+d} {option_type.upper()} ${strike} exp {expiration}")


    # calculate portfolio-level Greeks
    def get_portfolio_greeks(self):

        df = pd.DataFrame(self.positions)

        portfolio_greeks = {
            'delta': float((df['delta'] * df['quantity'] * 100).sum()),
            'gamma': float((df['gamma'] * df['quantity'] * 100).sum()),
            'theta': float((df['theta'] * df['quantity'] * 100).sum()),
            'vega': float((df['vega'] * df['quantity'] * 100).sum()),
            'rho': float((df['rho'] * df['quantity'] * 100).sum())
        }

        return portfolio_greeks
    
    # run scenario analysis on portfolio
    def scenario_analysis(self, price_changes=None, vol_changes=None):

        if price_changes is None:
            price_changes = [-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]

        if vol_changes is None:
            vol_changes = [-0.10, -0.05, 0, 0.05, 0.10]

        scenarios = []

        for price_change in price_changes:
            for vol_change in vol_changes:
                new_price = self.stock_price * (1 + price_change)
                portfolio_value = 0

                for position in self.positions:
                    new_vol = max(0.01, float(position['iv']) + vol_change)
                    new_time = float(position['years_to_exp'])
                    pos_strike = float(position['strike'])
                    pos_quantity = int(position['quantity'])

                    if position['option_type'] == 'call':
                        new_value = black_scholes_call(new_price, position['strike'], new_time, self.risk_free_rate, new_vol)
                    else:
                        new_value = black_scholes_put(new_price, position['strike'], new_time, self.risk_free_rate, new_vol)

                    portfolio_value += new_value * position['quantity'] * 100

                scenarios.append({
                    'price_change': price_change,
                    'vol_change': vol_change,
                    'new_price': new_price,
                    'portfolio_value': portfolio_value
                })
        
        return pd.DataFrame(scenarios)


    # create comprehensive risk dashboard
    def plot_risk_dashboard(self):

        if len(self.positions) == 0:
            print("No positions in portfolio!")
            return
        
        fig = plt.figure(figsize=(16, 10))

        portfolio_greeks = self.get_portfolio_greeks()
        scenarios = self.scenario_analysis()

        ## summary ##
        ax1 = plt.subplot(3, 3, 1)
        ax1.axis('off')
        
        summary_text = f"PORTFOLIO SUMMARY\n"
        summary_text += f"Ticker: {self.ticker}\n"
        summary_text += f"Stock Price: ${self.stock_price:.2f}\n"
        summary_text += f"Positions: {len(self.positions)}\n"
        summary_text += f"\nPORTFOLIO GREEKS:\n"
        summary_text += f"Delta: {portfolio_greeks['delta']:,.0f} shares\n"
        summary_text += f"Gamma: {portfolio_greeks['gamma']:,.2f}\n"
        summary_text += f"Vega: ${portfolio_greeks['vega']:,.0f} per 1% IV\n"
        summary_text += f"Theta: ${portfolio_greeks['theta']:,.0f} per day\n"
        
        ax1.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        ## delta breakdown ##
        ax2 = plt.subplot(3, 3, 2)

        df_positions = pd.DataFrame(self.positions)
        df_positions['delta_exposure'] = df_positions['delta'] * df_positions['quantity'] * 100
        df_positions['position_label'] = (df_positions['quantity'].astype(str) + 'x ' +
                                        df_positions['option_type'].str.upper() + ' $' +
                                        df_positions['strike'].astype(str))

        colors = plt.cm.Paired(range(len(df_positions)))        
        y_positions = range(len(df_positions))
        ax2.barh(y_positions, df_positions['delta_exposure'], color=colors, height=0.4, edgecolor='black', linewidth=1)
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(df_positions['position_label'])
        ax2.set_xlabel('Delta Exposure (shares)')
        ax2.set_title('Delta by Position', fontweight='bold')
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim(-0.5, len(df_positions) - 0.5) 
        ax2.margins(y=0.2)

        ## vega breakdown ##
        ax3 = plt.subplot(3, 3, 3)
        
        df_positions['vega_exposure'] = df_positions['vega'] * df_positions['quantity'] * 100
        colors = plt.cm.Paired(range(len(df_positions)))
        ax3.barh(range(len(df_positions)), df_positions['vega_exposure'], color=colors, height=0.4, edgecolor='black', linewidth=1)
        ax3.set_yticks(range(len(df_positions)))
        ax3.set_yticklabels(df_positions['position_label'])
        ax3.set_xlabel('Vega Exposure ($)')
        ax3.set_title('Vega by Position', fontweight='bold')
        ax3.axvline(0, color='black', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_ylim(-0.5, len(df_positions) - 0.5)
        ax3.margins(y=0.2)
        
        ## P&L heatmap ##
        ax4 = plt.subplot(3, 3, (4, 6))
        
        pivot_pnl = scenarios.pivot_table(
            values='portfolio_value',
            index='vol_change',
            columns='price_change',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot_pnl / 1000,
            annot=True,
            fmt='.1f',
            cmap='flare',
            center=0,
            cbar_kws={'label': 'Portfolio Value ($1000s)'},
            linewidths=0.5,
            ax=ax4
        )
        
        ax4.set_title('Portfolio Value Scenario Analysis', fontweight='bold', pad=15)
        ax4.set_xlabel('Stock Price Change (%)', fontweight='bold')
        ax4.set_ylabel('IV Change (percentage points)', fontweight='bold')
        ax4.set_xticklabels([f'{x*100:+.0f}%' for x in pivot_pnl.columns], rotation=45)
        ax4.set_yticklabels([f'{y*100:+.0f}pp' for y in pivot_pnl.index], rotation=0)
        
        ## P&L profile ##
        ax5 = plt.subplot(3, 3, (7, 9))
        
        for vol_chg in [-0.10, -0.05, 0, 0.05, 0.10]:
            scenario_subset = scenarios[scenarios['vol_change'] == vol_chg]
            label = f'IV {vol_chg*100:+.0f}pp'
            ax5.plot(scenario_subset['price_change'] * 100,
                    scenario_subset['portfolio_value'] / 1000,
                    marker='o', label=label, linewidth=2)
        
        ax5.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Stock Price Change (%)', fontweight='bold')
        ax5.set_ylabel('Portfolio Value ($1000s)', fontweight='bold')
        ax5.set_title('P&L Profile Across Scenarios', fontweight='bold')
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # print risk metrics
        print("\nRISK METRICS")
        
        base_value = scenarios[(scenarios['price_change'] == 0) &
                              (scenarios['vol_change'] == 0)]['portfolio_value'].iloc[0]
        
        worst_case = scenarios['portfolio_value'].min()
        best_case = scenarios['portfolio_value'].max()
        
        print(f"Current Portfolio Value: ${base_value:,.2f}")
        print(f"Best Case: ${best_case:,.2f} ({(best_case/base_value-1)*100:+.1f}%)")
        print(f"Worst Case: ${worst_case:,.2f} ({(worst_case/base_value-1)*100:+.1f}%)")
        print(f"Max Potential Loss: ${base_value - worst_case:,.2f}")
        print(f"Max Potential Gain: ${best_case - base_value:,.2f}")
        
        print(f"\nHEDGE RECOMMENDATIONS:")
        if abs(portfolio_greeks['delta']) > 100:
            shares_to_hedge = -int(portfolio_greeks['delta'])
            print(f"- Delta hedge: {shares_to_hedge:+,d} shares of {self.ticker}")
        else:
            print(f"- Delta is near neutral ({portfolio_greeks['delta']:.0f} shares)")
        
        if abs(portfolio_greeks['vega']) > 1000:
            print(f"- High vega exposure (${portfolio_greeks['vega']:,.0f})")
            print(f"  Consider opposite vol positions to reduce risk")


### MAIN

if __name__ == "__main__":
    
    print("\nVOLATILITY SURFACE & PORTFOLIO RISK ANALYZER")
    
    RISK_FREE_RATE = 0.05
    
    # get ticker
    while True:
        ticker = input("\nEnter stock ticker (e.g., AAPL, TSLA, SPY): ").strip().upper()
        if ticker:
            break
        print("Please enter a valid ticker symbol.")
    
    # get option type
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
    
    print(f"\nTicker: {ticker}")
    print(f"Option Type: {option_type.upper()}")
    print(f"Risk-Free Rate: {RISK_FREE_RATE*100:.2f}%")
    
    try:
        # fetch data and show surface
        surface_data = get_volatility_surface(ticker, option_type, RISK_FREE_RATE)
        plot_volatility_surface(surface_data)
        
        # portfolio risk analysis
        do_risk = input("\nWould you like to analyze a portfolio? (y/n): ").strip().lower()
        
        if do_risk == 'y':
            print("\nPORTFOLIO RISK ANALYSIS")
            
            portfolio = PortfolioRiskAnalyzer(
                ticker,
                surface_data['stock_price'],
                RISK_FREE_RATE
            )
            
            print("Enter your positions (or type 'done' when finished)")
            print("Format: [option_type] [strike] [expiration] [quantity]")
            print("Example: call 260 2026-02-20 10")
            print("Example: put 240 2026-01-16 -5")
            
            while True:
                pos_input = input("\nPosition: ").strip().lower()
                
                if pos_input == 'done':
                    break
                
                try:
                    parts = pos_input.split()
                    if len(parts) != 4:
                        print("Invalid format. Try: call 260 2026-02-20 10")
                        continue
                    
                    pos_type = parts[0]
                    pos_strike = float(parts[1])
                    pos_exp = parts[2]
                    pos_qty = int(parts[3])
                    
                    if pos_type not in ['call', 'put']:
                        print("Option type must be 'call' or 'put'")
                        continue
                    
                    # get IV from surface data
                    df = surface_data['df']
                    matching = df[
                        (df['strike'] == pos_strike) &
                        (df['expiration'] == pos_exp)
                    ]
                    
                    if len(matching) > 0:
                        pos_iv = matching.iloc[0]['iv'] / 100
                    else:
                        pos_iv = df['iv'].mean() / 100
                        print(f"Using average IV: {pos_iv*100:.1f}%")
                    
                    portfolio.add_position(
                        pos_type, pos_strike, pos_exp, pos_qty, pos_iv
                    )
                    
                except Exception as e:
                    print(f"Error: {e}. Try again.")
            
            if len(portfolio.positions) > 0:
                portfolio.plot_risk_dashboard()
            else:
                print("No positions added.")
        
        # analyze another ?
        again = input("\nAnalyze another ticker? (y/n): ").strip().lower()
        if again == 'y':
            print("\n" * 2)
            exec(open(__file__).read())
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry a highly liquid stock (AAPL, TSLA, SPY, MSFT)")