import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def load_stock_data(ticker: str, start: str = "2025-01-01", end: str = None):
    """
    Loads daily OHLCV data for a given stock ticker from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start (str): Start date for data (default: '2015-01-01')
        end (str): End date for data (default: today)

    Returns:
        pd.DataFrame: DataFrame containing stock data
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    data = data.rename(columns=str.lower)
    return data

def print_data_info(data: pd.DataFrame):
    """
    Prints summary statistics and basic info about the stock data.
    """
    print("\n=== Data Info ===")
    print(data.info())
    print("\n=== Head ===")
    print(data.head())
    print("\n=== Tail ===")
    print(data.tail())
    print("\n=== Summary Statistics ===")
    print(data.describe())

def plot_closing_prices(df: pd.DataFrame, ticker: str):
    """
    Plots closing prices for a given ticker using the provided DataFrame.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["close"], label=f"{ticker} Close", color="tab:blue")
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ticker = "V"
    df = load_stock_data(ticker, "2025-05-15")
    print_data_info(df)

    n = 30  # number of days to show
    first_n_closes = df['close'][:n].values.flatten()
    first_n_closes = [round(float(x), 2) for x in first_n_closes]
    print(f"\nFirst {n} closing prices:\n", first_n_closes)

    plot_closing_prices(df, ticker)