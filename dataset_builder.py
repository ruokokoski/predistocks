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

def train_xgb_model(X, y, target_dates, train_ratio=0.8, params=None, plot=True):
    """
    Trains and evaluates an XGBoost regression model for next-day stock prediction.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target vector (next-day close prices)
        target_dates (np.ndarray): Corresponding target dates
        train_ratio (float): Fraction of samples for training (chronological split)
        params (dict): Optional hyperparameters for XGBRegressor
        plot (bool): If True, plots predicted vs actual prices

    Returns:
        model (XGBRegressor): Trained model
        results (pd.DataFrame): DataFrame with true & predicted values
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

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = target_dates[:split_idx], target_dates[split_idx:]

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nModel evaluation:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    results = pd.DataFrame({
        "date": dates_test,
        "actual": y_test,
        "predicted": y_pred
    }).set_index("date")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(results.index, results["actual"], label="Actual", color="black")
        plt.plot(results.index, results["predicted"], label="Predicted", color="red", alpha=0.7)
        plt.title("Next-day Close Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model, results

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
    data = load_stock_data(ticker, start="2025-01-01")
    X, y, target_dates, feature_dates = create_features(data, lag)
    model, results = train_xgb_model(X, y, target_dates, train_ratio=0.8)

'''
    print(f"\nTicker: {ticker}")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Example feature vector (first row): {X[0][:5]} ...")
    print(f"First target: {y[0]:.2f}")
    print(f"First target date: {target_dates[0]}")
    print(f"First feature dates: from {feature_dates[0][0]} to {feature_dates[0][-1]}")

    plot_sample(X, y, target_dates, feature_dates, sample_idx=0, lag=lag, ticker=ticker)
'''