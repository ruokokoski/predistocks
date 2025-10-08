import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_loader import load_stock_data

def create_features(data: pd.DataFrame, lookback: int = 30):
    """
    Converts OHLCV data into supervised learning features for next-day prediction.

    Args:
        data (pd.DataFrame): Daily stock data with columns ['open','high','low','close','volume']
        lookback (int): Number of past days used as features

    Returns:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target values (next-day close)
    """
    df = data.copy()

    # compute daily returns
    #df["return"] = df["close"].pct_change()

    df = df.dropna()

    features = []
    targets = []
    target_dates = []
    feature_dates = []

    for i in range(lookback, len(df) - 1):
        window = df.iloc[i - lookback:i]
        next_close = df.iloc[i]["close"]
        next_date = df.index[i]
        window_dates = df.index[i - lookback:i]

        # flatten window data (close & volume)
        feat = np.concatenate([
            window["close"].values.flatten(),
            window["volume"].values.flatten()
        ])
        features.append(feat)
        targets.append(next_close)
        target_dates.append(next_date)
        feature_dates.append(window_dates)

    X = np.array(features)
    y = np.array(targets).flatten()
    target_dates = np.array(target_dates)
    return X, y, target_dates, feature_dates

def plot_sample(X: np.ndarray, y: np.ndarray, target_dates: np.ndarray, feature_dates: list, 
                sample_idx: int = 0, lookback: int = 30, ticker: str = "MSFT"):
    """
    Plots the feature vector (closing prices and volumes) for a given sample,
    and shows the next-day target price.

    Args:
        X (np.ndarray): Feature matrix (n_samples, lookback*2)
        y (np.ndarray): Target array (n_samples,)
        sample_idx (int): Index of the sample to visualize
        lookback (int): Number of past days used in features
        ticker (str): Stock ticker for title
    """
    first_features = X[sample_idx]
    first_target = y[sample_idx]
    target_date = target_dates[sample_idx]
    window_dates = feature_dates[sample_idx]

    closes = first_features[:lookback]
    volumes = first_features[lookback:]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color=color)
    ax1.plot(window_dates, closes, marker='o', color=color, label='Past closes')
    ax1.scatter(target_date, first_target, color='red', label='Next day target')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Volume', color=color)
    ax2.bar(window_dates, volumes, alpha=0.3, color=color, label='Past volumes')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'{ticker} - Sample {sample_idx} features (closes + volumes) vs. next-day target')
    plt.show()


if __name__ == "__main__":
    ticker = "MSFT"
    lookback = 30
    data = load_stock_data(ticker, start="2025-01-01")
    X, y, target_dates, feature_dates = create_features(data, lookback)

    print(f"\nTicker: {ticker}")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Example feature vector (first row): {X[0][:5]} ...")
    print(f"First target: {y[0]:.2f}")
    print(f"First target date: {target_dates[0]}")
    print(f"First feature dates: from {feature_dates[0][0]} to {feature_dates[0][-1]}")

    plot_sample(X, y, target_dates, feature_dates, sample_idx=0, lookback=lookback, ticker=ticker)
