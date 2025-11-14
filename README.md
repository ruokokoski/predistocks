# Stock Price Prediction with Machine Learning

This project investigates whether machine learning can extract meaningful signals from noisy financial markets by forecasting next-day closing prices for S&P 500 stocks. The model is built using XGBoost, a gradient boosting method well suited for capturing nonlinear relationships in time series data. Historical OHLCV data is combined with a range of common technical indicators to test whether they improve predictive accuracy beyond raw price and volume information.

The workflow includes data collection with `yfinance`, exploratory data analysis, feature engineering, rolling-window training, and hyperparameter optimization using Optuna. Performance is evaluated through metrics such as MAPE, MAE, RMSE, R², and directional accuracy. Results show that the model can achieve around 1% prediction error and 50–60% directional accuracy, with technical indicators providing consistent improvements across most stocks.

This repository contains all code used for the project, including data processing, model training, evaluation, and visualizations. While the model demonstrates promising predictive power, financial time series remain challenging, and further refinement is needed for real-world use. The full implementation is available in the included Jupyter Notebook.
