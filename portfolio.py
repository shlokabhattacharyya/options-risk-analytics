### IMPORT LIBARIRES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from pricing import black_scholes_call, black_scholes_put, calculate_greeks

### MANAGES PORTFOLIO OF OPTIONS (GREEKS, PRICE X VOLATILITY, RISK DASHBOARD)
class PortfolioRiskAnalyzer:

    def __init__(self, ticker, stock_price, risk_free_rate):
        self.ticker = ticker
        self.stock_price = stock_price
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)
        self.positions = []

    # add an option position to the portfolio
    def add_position(self, option_type, strike, expiration, quantity, iv=None):
        """
        arguments:
            - option_type: str - 'call' or 'put'
            - strike: float - strike price
            - expiration: str - expiration date as 'YYYY-MM-DD'
            - quantity: int - number of contracts (negative = short)
            - iv: float (optional) - implied volatility as a decimal
                  if None, fetched live from yfinance
        """
        exp_date = pd.to_datetime(expiration)
        days_to_exp = (exp_date - pd.Timestamp.now()).days
        years_to_exp = days_to_exp / 365

        if years_to_exp <= 0:
            print(f"Warning: Option {strike} {option_type} exp {expiration} has already expired.")
            return

        market_price = None

        if iv is None:
            try:
                opt_chain = self.stock.option_chain(expiration)
                options = opt_chain.calls if option_type == 'call' else opt_chain.puts
                closest_idx = (options['strike'] - strike).abs().idxmin()
                iv = options.loc[closest_idx, 'impliedVolatility']
                market_price = options.loc[closest_idx, 'lastPrice']
            except Exception:
                print(f"Could not fetch IV for {strike} {option_type} - using 30%")
                iv = 0.30

        greeks = calculate_greeks(
            self.stock_price, strike, years_to_exp,
            self.risk_free_rate, iv, option_type
        )

        self.positions.append({
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
        })

        print(f"Added: {quantity:+d} {option_type.upper()} ${strike} exp {expiration}")

    # Greeks aggregation
    def get_portfolio_greeks(self):
        df = pd.DataFrame(self.positions)
        multiplier = df['quantity'] * 100
        return {
            'delta': float((df['delta'] * multiplier).sum()),
            'gamma': float((df['gamma'] * multiplier).sum()),
            'theta': float((df['theta'] * multiplier).sum()),
            'vega': float((df['vega']  * multiplier).sum()),
            'rho': float((df['rho']   * multiplier).sum()),
        }

    # scenario analysis (compute portfolio value across grid of price & volatility shocks)
    def scenario_analysis(self, price_changes=None, vol_changes=None):
        """
        arguments:
            - price_changes: list of float - fractional price moves (e.g. -0.10 = -10%)
            - vol_changes: list of float - absolute IV shifts (e.g. 0.05 = +5pp)

        returns:
            - DataFrame with columns: price_change, vol_change, new_price, portfolio_value
        """
        if price_changes is None:
            price_changes = [-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]
        if vol_changes is None:
            vol_changes = [-0.10, -0.05, 0, 0.05, 0.10]

        scenarios = []

        for price_change in price_changes:
            for vol_change in vol_changes:
                new_price = self.stock_price * (1 + price_change)
                portfolio_value = 0

                for pos in self.positions:
                    new_vol = max(0.01, float(pos['iv']) + vol_change)
                    T = float(pos['years_to_exp'])
                    K = float(pos['strike'])
                    qty = int(pos['quantity'])

                    if pos['option_type'] == 'call':
                        value = black_scholes_call(new_price, K, T, self.risk_free_rate, new_vol)
                    else:
                        value = black_scholes_put(new_price, K, T, self.risk_free_rate, new_vol)

                    portfolio_value += value * qty * 100

                scenarios.append({
                    'price_change': price_change,
                    'vol_change': vol_change,
                    'new_price': new_price,
                    'portfolio_value': portfolio_value
                })

        return pd.DataFrame(scenarios)


    ### RISK DASHBOARD
    def plot_risk_dashboard(self):
        """
        render a comprehensive 3x3 risk dashboard:
            (1,1) portfolio summary text
            (1,2) delta exposure by position
            (1,3) vega exposure by position
            (2,*) P&L heatmap across price x volatility scenarios
            (3,*) P&L profile curves at selected volatility shocks
        """
        if not self.positions:
            print("nNo positions in portfolio!")
            return

        portfolio_greeks = self.get_portfolio_greeks()
        scenarios = self.scenario_analysis()

        fig = plt.figure(figsize=(14, 8))

        ## (1,1): summary ##
        ax1 = plt.subplot(3, 3, 1)
        ax1.axis('off')
        summary = (
            f"PORTFOLIO SUMMARY\n"
            f"Ticker: {self.ticker}\n"
            f"Stock: ${self.stock_price:.2f}\n"
            f"Positions:  {len(self.positions)}\n"
            f"\nPORTFOLIO GREEKS:\n"
            f"Delta: {portfolio_greeks['delta']:,.0f} shares\n"
            f"Gamma: {portfolio_greeks['gamma']:,.2f}\n"
            f"Vega:  ${portfolio_greeks['vega']:,.0f} per 1% IV\n"
            f"Theta: ${portfolio_greeks['theta']:,.0f} per day\n"
        )
        ax1.text(0.1, 0.5, summary, fontsize=10, family='monospace', va='center')

        ## (1,2): delta by position ##
        ax2 = plt.subplot(3, 3, 2)
        df_pos = pd.DataFrame(self.positions)
        df_pos['delta_exposure'] = df_pos['delta'] * df_pos['quantity'] * 100
        df_pos['label'] = (
            df_pos['quantity'].astype(str) + 'x '
            + df_pos['option_type'].str.upper() + ' $'
            + df_pos['strike'].astype(str)
        )
        colors = plt.cm.Paired(range(len(df_pos)))
        ax2.barh(range(len(df_pos)), df_pos['delta_exposure'], color=colors, height=0.4,
                 edgecolor='black', linewidth=1)
        ax2.set_yticks(range(len(df_pos)))
        ax2.set_yticklabels(df_pos['label'])
        ax2.set_xlabel('Delta Exposure (shares)')
        ax2.set_title('Delta by Position', fontweight='bold')
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim(-0.5, len(df_pos) - 0.5)
        ax2.margins(y=0.2)

        ## (1,3): vega by position ##
        ax3 = plt.subplot(3, 3, 3)
        df_pos['vega_exposure'] = df_pos['vega'] * df_pos['quantity'] * 100
        ax3.barh(range(len(df_pos)), df_pos['vega_exposure'], color=colors, height=0.4,
                 edgecolor='black', linewidth=1)
        ax3.set_yticks(range(len(df_pos)))
        ax3.set_yticklabels(df_pos['label'])
        ax3.set_xlabel('Vega Exposure ($)')
        ax3.set_title('Vega by Position', fontweight='bold')
        ax3.axvline(0, color='black', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_ylim(-0.5, len(df_pos) - 0.5)
        ax3.margins(y=0.2)

        ## (2,*): P&L heatmap ##
        ax4 = plt.subplot(3, 3, (4, 6))
        pivot_pnl = scenarios.pivot_table(
            values='portfolio_value', index='vol_change',
            columns='price_change', aggfunc='mean'
        )
        sns.heatmap(
            pivot_pnl / 1000, annot=True, fmt='.1f',
            cmap='flare', center=0,
            cbar_kws={'label': 'Portfolio Value ($1000s)'},
            linewidths=0.5, ax=ax4
        )
        ax4.set_title('Portfolio Value Scenario Analysis', fontweight='bold', pad=15)
        ax4.set_xlabel('Stock Price Change (%)', fontweight='bold')
        ax4.set_ylabel('IV Change (percentage points)', fontweight='bold')
        ax4.set_xticklabels([f'{x * 100:+.0f}%' for x in pivot_pnl.columns], rotation=45)
        ax4.set_yticklabels([f'{y * 100:+.0f}pp' for y in pivot_pnl.index], rotation=0)

        ## (3,*): P&L profile ##
        ax5 = plt.subplot(3, 3, (7, 9))
        for vol_chg in [-0.10, -0.05, 0, 0.05, 0.10]:
            subset = scenarios[scenarios['vol_change'] == vol_chg]
            ax5.plot(
                subset['price_change'] * 100,
                subset['portfolio_value'] / 1000,
                marker='o', label=f'IV {vol_chg * 100:+.0f}pp', linewidth=2
            )
        ax5.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_xlabel('Stock Price Change (%)', fontweight='bold')
        ax5.set_ylabel('Portfolio Value ($1000s)', fontweight='bold')
        ax5.set_title('P&L Profile Across Scenarios', fontweight='bold')
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # risk metrics
        base = scenarios[(scenarios['price_change'] == 0) & (scenarios['vol_change'] == 0)]['portfolio_value'].iloc[0]
        worst = scenarios['portfolio_value'].min()
        best = scenarios['portfolio_value'].max()

        print("\nRISK METRICS")
        print(f"Current Portfolio Value: ${base:,.2f}")
        print(f"Best Case:  ${best:,.2f}  ({(best / base - 1) * 100:+.1f}%)")
        print(f"Worst Case: ${worst:,.2f}  ({(worst / base - 1) * 100:+.1f}%)")
        print(f"Max Potential Loss: ${base - worst:,.2f}")
        print(f"Max Potential Gain: ${best - base:,.2f}")

        print("\nHEDGE RECOMMENDATIONS:")
        delta = portfolio_greeks['delta']
        if abs(delta) > 100:
            print(f"Delta hedge: {-int(delta):+,d} shares of {self.ticker}")
        else:
            print(f"Delta is near neutral ({delta:.0f} shares)")

        vega = portfolio_greeks['vega']
        if abs(vega) > 1000:
            print(f"High vega exposure (${vega:,.0f})")
            print(f"Consider opposite volatility positions to reduce risk")