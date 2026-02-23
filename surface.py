### IMPORT LIBARIES
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

from pricing import implied_volatility

warnings.filterwarnings('ignore')

### FETCH LIVE OPTIONS DATA & COMPUTE IMPLIED VOLATILITY
def get_volatility_surface(ticker, option_type, risk_free_rate=0.05):
    """
    arguments:
        - ticker: str - stock ticker symbol
        - option_type: str - 'call' or 'put'
        - risk_free_rate: float - annual risk-free rate

    returns:
        - dict with keys: pivot, df, stock_price, ticker, option_type, gp_results
    """
    print(f"Fetching data for {ticker}...")

    stock = yf.Ticker(ticker)
    stock_price = stock.history(period='1d')['Close'].iloc[-1]
    print(f"Current stock price: ${stock_price:.2f}")

    expirations = stock.options
    if len(expirations) == 0:
        raise ValueError(f"No options data could be found for {ticker}.")

    # use first 6 expirations for a clean surface
    expirations = expirations[:6]
    print(f"Found {len(expirations)} expiration dates")

    all_options = []

    for exp_date in expirations:
        print(f"Processing expiration: {exp_date}")

        opt_chain = stock.option_chain(exp_date)
        options = opt_chain.calls if option_type == 'call' else opt_chain.puts

        exp_datetime = pd.to_datetime(exp_date)
        days_to_exp  = (exp_datetime - pd.Timestamp.now()).days
        years_to_exp = days_to_exp / 365.0

        for _, row in options.iterrows():
            strike = row['strike']

            # keep only strikes within 30% of spot to avoid wide, sparse charts
            if strike < stock_price * 0.7 or strike > stock_price * 1.3:
                continue

            if pd.notna(row['lastPrice']) and row['lastPrice'] > 0:
                market_price = row['lastPrice']
            elif pd.notna(row['bid']) and pd.notna(row['ask']):
                market_price = (row['bid'] + row['ask']) / 2
            else:
                continue

            if market_price < 0.05:
                continue

            volume = row.get('volume', 0)
            if pd.isna(volume):
                volume = 0

            iv = implied_volatility(
                market_price, stock_price, strike, years_to_exp,
                risk_free_rate, option_type
            )

            if iv is not None and 0.01 < iv < 3.0:
                all_options.append({
                    'expiration': exp_date,
                    'days_to_exp': days_to_exp,
                    'strike': strike,
                    'market_price': market_price,
                    'iv': iv * 100, # store as percentage
                    'volume': volume,
                    'moneyness': stock_price / strike,
                    'bid': row.get('bid', np.nan),
                    'ask': row.get('ask', np.nan),
                })

    if len(all_options) == 0:
        raise ValueError(
            "Could not calculate implied volatility for any options. "
            "Try a more liquid stock."
        )

    print(f"Successfully calculated implied volatility for {len(all_options)} options.")

    df = pd.DataFrame(all_options)

    pivot = df.pivot_table(
        values='iv',
        index='days_to_exp',
        columns='strike',
        aggfunc='mean'
    )

    return {
        'pivot': pivot,
        'df': df,
        'stock_price': stock_price,
        'ticker': ticker,
        'option_type': option_type,
        'gp_results': None   # populated by fit_gp_surface()
    }


### FIT GAUSSIAN PROCESS REGRESSION MODEL 
def fit_gp_surface(df, stock_price, resolution=30):
    """
    the GP kernel is:
        ConstantKernel * RBF(strike, dte) + WhiteKernel(noise)

    - ConstantKernel: scales overall IV magnitude
    - RBF: encodes smooth spatial correlation; separate lengthscales for
      strike and DTE are learned from data via log-marginal-likelihood
    - WhiteKernel: absorbs bid-ask noise in the raw IV observations

    arguments:
        - df: DataFrame - raw options data from get_volatility_surface()
        - stock_price: float - current stock price
        - resolution: int - grid points per axis on the dense output surface

    returns:
        - dict with keys:
            pivot_smooth: smoothed IV as a DataFrame (rows=dte, cols=strike)
            gp_model: fitted GaussianProcessRegressor
            scaler_X: fitted StandardScaler for (strike, dte) features
            grid_strikes: 1-D array of strike values in the output grid
            grid_days: 1-D array of DTE values in the output grid
            uncertainty: 2-D array of GP prediction std dev (IV %)
    """
    # prepare training data 
    X_raw = df[['strike', 'days_to_exp']].values.astype(float)
    y = df['iv'].values.astype(float)

    # scale features so kernel hyperparameters stay in a sensible numeric range
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    # define GP kernel
    kernel = (
        ConstantKernel(constant_value=1.0, constant_value_bounds=(0.1, 10.0))
        * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(0.1, 10.0))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 2.0))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5, # multiple restarts to find the global optimum
        normalize_y=True, # standardises IV targets internally
        alpha=1e-6 # small diagonal jitter for numerical stability
    )

    print("Fitting Gaussian Process to volatility surface...")
    gp.fit(X_scaled, y)
    print(f"Optimized kernel: {gp.kernel_}")

    # dense prediction grid
    grid_strikes = np.linspace(df['strike'].min(),      df['strike'].max(),      resolution)
    grid_days = np.linspace(df['days_to_exp'].min(), df['days_to_exp'].max(), resolution)

    gs, gd = np.meshgrid(grid_strikes, grid_days)
    X_grid_scaled = scaler_X.transform(np.column_stack([gs.ravel(), gd.ravel()]))

    iv_pred, iv_std = gp.predict(X_grid_scaled, return_std=True)

    iv_surface = np.clip(iv_pred.reshape(resolution, resolution), 0.5, None)
    uncertainty = iv_std.reshape(resolution, resolution)

    pivot_smooth = pd.DataFrame(
        iv_surface,
        index=np.round(grid_days, 0).astype(int),
        columns=np.round(grid_strikes, 2)
    )
    pivot_smooth.index.name = 'days_to_exp'
    pivot_smooth.columns.name = 'strike'

    return {
        'pivot_smooth': pivot_smooth,
        'gp_model': gp,
        'scaler_X': scaler_X,
        'grid_strikes': grid_strikes,
        'grid_days': grid_days,
        'uncertainty': uncertainty
    }