import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def create_features(data: pd.DataFrame, lag: int = 30):
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

    for i in range(lag, len(df)):
        window = df.iloc[i - lag:i]
        next_close = df.iloc[i]["close"]
        next_date = df.index[i]
        window_dates = df.index[i - lag:i]

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

def walk_forward_sliding_window(X, y, target_dates, train_window=200, step=1, params=None):
    """
    Walk-forward validation with a sliding training window for next-day stock prediction.

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        target_dates (np.ndarray): Corresponding dates
        train_window (int): Number of past samples used for training at each step
        step (int): How many steps to move the window forward each iteration
        params (dict): Hyperparameters for XGBRegressor

    Returns:
        results (pd.DataFrame): DataFrame with date, actual, predicted
        metrics (dict): Overall MAE and RMSE
    """
    if params is None:
        params = {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.2,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": 42,
        }

    n_samples = len(X)
    predictions = []
    actuals = []
    dates = []

    start_idx = train_window
    for i in range(start_idx, n_samples, step):
        train_idx = slice(i - train_window, i)
        test_idx = i

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx:test_idx+1], y[test_idx:test_idx+1]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        predictions.append(y_pred[0])
        actuals.append(y_test[0])
        dates.append(target_dates[test_idx])

    results = pd.DataFrame({
        "date": dates,
        "actual": actuals,
        "predicted": predictions
    }).set_index("date")

    mae = mean_absolute_error(results["actual"], results["predicted"])
    rmse = np.sqrt(mean_squared_error(results["actual"], results["predicted"]))
    mape = np.mean(np.abs((results["actual"] - results["predicted"]) / results["actual"])) * 100

    # Directional Accuracy (DA)
    actual_change = np.sign(results["actual"].diff())
    predicted_change = np.sign(results["predicted"].diff())
    da = np.mean(actual_change[1:] == predicted_change[1:]) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional_Accuracy": da
    }

    print(f"\nWalk-forward (sliding window) metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"DA:   {da:.2f}%")

    return results, metrics

def plot_walk_forward_results(results: pd.DataFrame, ticker: str = "MSFT"):
    """
    Plots walk-forward prediction results:
    - Actual vs predicted prices
    - Prediction error over time

    Args:
        results (pd.DataFrame): DataFrame with 'actual' and 'predicted' columns indexed by date
        ticker (str): Stock ticker for title
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw={'height_ratios': [3, 1]})

    ax[0].plot(results.index, results["actual"], label="Actual", color="black")
    ax[0].plot(results.index, results["predicted"], label="Predicted", color="red", alpha=0.7)
    ax[0].set_ylabel("Close Price")
    ax[0].set_title(f"{ticker} — Next-Day Close Prediction (Walk-Forward, Sliding Window)")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, linestyle="--", alpha=0.4)

    errors = results["predicted"] - results["actual"]
    ax[1].plot(results.index, errors, color="blue", alpha=0.6)
    ax[1].axhline(0, color="black", linewidth=1)
    ax[1].set_ylabel("Prediction Error")
    ax[1].set_xlabel("Date")
    ax[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_sample(X: np.ndarray, y: np.ndarray, target_dates: np.ndarray, feature_dates: list, 
                sample_idx: int = 0, lag: int = 30, ticker: str = "MSFT"):
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

    closes = first_features[:lag]
    volumes = first_features[lag:]

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
    lag = 30
    data = load_stock_data(ticker, start="2023-01-01")
    X, y, target_dates, feature_dates = create_features(data, lag)
    results, metrics = walk_forward_sliding_window(
        X, y, target_dates,
        train_window=100,
        step=1 # predict daily
    )
    plot_walk_forward_results(results, ticker)

'''
    plot_sample(X, y, target_dates, feature_dates, sample_idx=0, lag=lag, ticker=ticker)
'''