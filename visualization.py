### IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from surface import fit_gp_surface


### VOLATILITY SURFACE WITH GP SMOOTHING
def plot_volatility_surface(surface_data):
    """
    produces a 3-row, 2-column figure:
        row 1 - raw IV heatmap & GP-smoothed IV heatmap
        row 2 — GP uncertainty heatmap & Volatility smile (raw vs GP)
        row 3 — Term structure: raw ATM vs GP smooth  (full width)

    arguments:
        - surface_data: dict - output from get_volatility_surface(),
          optionally with 'gp_results' pre-populated by fit_gp_surface()
    """
    pivot = surface_data['pivot']
    df = surface_data['df']
    stock_price = surface_data['stock_price']
    ticker = surface_data['ticker']
    option_type = surface_data['option_type']

    # reuse pre-fitted GP if available, otherwise fit now
    if surface_data.get('gp_results') is not None:
        gp_results = surface_data['gp_results']
    else:
        gp_results = fit_gp_surface(df, stock_price)

    pivot_smooth = gp_results['pivot_smooth']
    uncertainty = gp_results['uncertainty']
    grid_strikes = gp_results['grid_strikes']
    grid_days = gp_results['grid_days']

    # shared colour scale across both heatmaps for fair visual comparison
    all_iv_vals = np.concatenate([df['iv'].values, pivot_smooth.values.ravel()])
    vmin = np.nanpercentile(all_iv_vals, 5)
    vmax = np.nanpercentile(all_iv_vals, 95)
    cmap = sns.color_palette('RdYlGn_r', as_cmap=True)
    cmap.set_bad(color='#ebe9e6')

    # GridSpec lets us give the heatmap rows more height than the term structure row
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f'{ticker} Volatility Surface — {option_type.upper()}s   |   '
        f'Current Price: ${stock_price:.2f}',
        fontsize=12, fontweight='bold'
    )

    gs = GridSpec(3, 2, figure=fig, height_ratios=[2.2, 2.2, 1.6], hspace=0.52, wspace=0.35)

    ## row 1: raw heatmap & GP-smoothed heatmap ##
    ax1 = fig.add_subplot(gs[0, 0])
    show_annot = len(pivot.columns) <= 20
    sns.heatmap(
        pivot, annot=show_annot, fmt='.0f',
        cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'IV (%)', 'shrink': 0.8}, linewidths=0.3, ax=ax1
    )
    ax1.set_title('Raw IV Surface (sparse)', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Strike Price', fontsize=8)
    ax1.set_ylabel('Days to Expiration', fontsize=8)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax1.tick_params(axis='y', labelsize=7)

    # highlight ATM column
    raw_atm_idx = np.argmin(np.abs(pivot.columns - stock_price))
    for i in range(len(pivot)):
        ax1.add_patch(plt.Rectangle(
            (raw_atm_idx, i), 1, 1, fill=False, edgecolor='#cc66b3', lw=1.5
        ))

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        pivot_smooth, annot=False,
        cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_kws={'label': 'IV (%)', 'shrink': 0.8}, linewidths=0, ax=ax2
    )
    ax2.set_title('GP-Smoothed IV Surface', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Strike Price', fontsize=8)
    ax2.set_ylabel('Days to Expiration', fontsize=8)

    n_ticks = 6
    tick_idx = np.linspace(0, len(pivot_smooth.columns) - 1, n_ticks, dtype=int)
    dte_tick_idx = np.linspace(0, len(pivot_smooth.index) - 1, 5, dtype=int)
    ax2.set_xticks(tick_idx + 0.5)
    ax2.set_xticklabels([f'{pivot_smooth.columns[i]:.0f}' for i in tick_idx], rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(dte_tick_idx + 0.5)
    ax2.set_yticklabels([f'{pivot_smooth.index[i]}' for i in dte_tick_idx], rotation=0, fontsize=7)

    ## row 2: GP uncertainty heatmap & volatility smile comparison ##
    ax3 = fig.add_subplot(gs[1, 0])
    uncertainty_df = pd.DataFrame(uncertainty, index=pivot_smooth.index, columns=pivot_smooth.columns)
    sns.heatmap(
        uncertainty_df, annot=False,
        cmap='YlOrRd',
        cbar_kws={'label': 'Std Dev (IV %)', 'shrink': 0.8}, linewidths=0, ax=ax3
    )
    ax3.set_title('GP Uncertainty (higher = less data)', fontsize=9, fontweight='bold')
    ax3.set_xlabel('Strike Price', fontsize=8)
    ax3.set_ylabel('Days to Expiration', fontsize=8)
    ax3.set_xticks(tick_idx + 0.5)
    ax3.set_xticklabels([f'{pivot_smooth.columns[i]:.0f}' for i in tick_idx], rotation=45, ha='right', fontsize=7)
    ax3.set_yticks(dte_tick_idx + 0.5)
    ax3.set_yticklabels([f'{pivot_smooth.index[i]}' for i in dte_tick_idx], rotation=0, fontsize=7)

    ax4 = fig.add_subplot(gs[1, 1])
    shortest_exp = df['days_to_exp'].min()
    smile_raw = df[df['days_to_exp'] == shortest_exp].sort_values('strike')
    nearest_dte_idx = np.argmin(np.abs(grid_days - shortest_exp))
    gp_smile_iv = pivot_smooth.iloc[nearest_dte_idx].values
    unc_smile = uncertainty[nearest_dte_idx]

    ax4.scatter(smile_raw['strike'], smile_raw['iv'], color='#6d4c9c', zorder=5, s=25, label='Raw market IV')
    ax4.plot(grid_strikes, gp_smile_iv, color='#e07b39', linewidth=2, label='GP smooth')
    ax4.fill_between(grid_strikes, gp_smile_iv - unc_smile, gp_smile_iv + unc_smile,
                     alpha=0.2, color='#e07b39', label='±1σ confidence')
    ax4.axvline(stock_price, color='#cc66b3', linestyle='--', linewidth=1.5, label='ATM')
    ax4.set_xlabel('Strike Price', fontsize=8)
    ax4.set_ylabel('Implied Volatility (%)', fontsize=8)
    ax4.set_title(f'Volatility Smile: Raw vs GP  ({shortest_exp}d)', fontsize=9, fontweight='bold')
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=7)
    ax4.grid(True, alpha=0.3)

    ## row 3: term structure comparison (full width) ##
    ax5 = fig.add_subplot(gs[2, :])
    df['strike_diff'] = abs(df['strike'] - stock_price)
    atm_raw = df.loc[df.groupby('days_to_exp')['strike_diff'].idxmin()].sort_values('days_to_exp')

    atm_strike_idx = np.argmin(np.abs(grid_strikes - stock_price))
    gp_atm_iv  = pivot_smooth.iloc[:, atm_strike_idx].values
    gp_atm_unc = uncertainty[:, atm_strike_idx]

    ax5.scatter(atm_raw['days_to_exp'], atm_raw['iv'], color='#3682cf', zorder=5, s=25, label='Raw ATM IV')
    ax5.plot(grid_days, gp_atm_iv, color='#e07b39', linewidth=2, label='GP smooth')
    ax5.fill_between(grid_days, gp_atm_iv - gp_atm_unc, gp_atm_iv + gp_atm_unc,
                     alpha=0.2, color='#e07b39', label='±1σ confidence')
    ax5.set_xlabel('Days to Expiration', fontsize=8)
    ax5.set_ylabel('Implied Volatility (%)', fontsize=8)
    ax5.set_title('Term Structure: Raw ATM vs GP Smooth', fontsize=9, fontweight='bold')
    ax5.legend(fontsize=7)
    ax5.tick_params(labelsize=7)
    ax5.grid(True, alpha=0.3)

    plt.show()

    ### SUMMARY STATISTICS
    print("\nSUMMARY STATISTICS")
    print(f"Ticker: {ticker}")
    print(f"Stock Price: ${stock_price:.2f}")
    print(f"Option Type: {option_type.upper()}")
    print(f"Options analysed: {len(df)}")
    print(f"\nIMPLIED VOLATILITY (raw):")
    print(f"  Mean: {df['iv'].mean():.2f}%")
    print(f"  Median {df['iv'].median():.2f}%")
    print(f"  Min: {df['iv'].min():.2f}%")
    print(f"  Max: {df['iv'].max():.2f}%")
    print(f"  Std: {df['iv'].std():.2f}%")
    print(f"\nGP-SMOOTHED SURFACE:")
    print(f"  Mean IV: {pivot_smooth.values.mean():.2f}%")
    print(f"  Grid: {pivot_smooth.shape[0]}×{pivot_smooth.shape[1]} "
          f"({pivot_smooth.shape[0] * pivot_smooth.shape[1]} points from {len(df)} observations)")
    print(f"  Mean ±σ: ±{uncertainty.mean():.2f}%")
    print(f"  Kernel: {gp_results['gp_model'].kernel_}")