# Infosys Stock Price Movement Prediction using Machine Learning

---

## Project Overview

The goal of this project is to predict the next-day price movement (Up/Down) for a NIFTY 50 constituent stock, **Infosys (INFY.NS)**, using historical market data.

The project implements a complete machine learning pipeline, including:
* Data Collection (`yfinance`)
* Extensive Feature Engineering (40+ technical indicators)
* Model Training (Logistic Regression, Random Forest, XGBoost)
* Model Evaluation
* A simple strategy backtest to compare against a "Buy & Hold" benchmark

## Workflow

1.  **Data Collection:** Fetched 2 years of daily OHLCV (Open, High, Low, Close, Volume) data for `INFY.NS` from Yahoo Finance.
2.  **Feature Engineering:** Generated over 40 features to capture market dynamics, including SMAs, EMAs, MACD, RSI, Bollinger Bands, On-Balance Volume (OBV), and lagged returns.
3.  **Model Development:** Defined the target variable as a binary (1 = Up, 0 = Down) based on the next day's close and split the data 80/20 chronologically to prevent lookahead bias.
4.  **Model Training:** Trained and compared three models:
    * Logistic Regression (as a baseline)
    * Random Forest
    * XGBoost (Hyperparameter-tuned with `RandomizedSearchCV`)
5.  **Evaluation:** Assessed models on Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC/AUC curves.
6.  **Backtesting:** Simulated a simple long-only strategy based on the XGBoost model's signals to evaluate its performance against the market's "Buy & Hold" return.

## Key Results & Insights

* **Model Performance:** The Random Forest model achieved the highest test accuracy (**~53.9%**), while Logistic Regression had the best F1-Score for the "Up" class (**0.589**), indicating it was better at correctly identifying positive days (though with many false positives).
* **Feature Importance:** `Lag_Return_1` (yesterday's return) was the most predictive feature for both Random Forest and XGBoost. `Volatility_20` and `EMA_26` were also highly significant for the XGBoost model.
* **Backtest Strategy:** In a simple academic backtest on the test period (which was a downtrend), the XGBoost-based strategy **outperformed the "Buy & Hold" benchmark**.
    * **Strategy Return:** -4.5%
    * **Market (Buy & Hold) Return:** -6.9%

## Libraries Used

This project uses Python 3 and the following libraries:
* `pandas`
* `numpy`
* `yfinance`
* `scikit-learn`
* `xgboost`
* `matplotlib`
* `seaborn`

## Limitations & Future Improvements

This analysis acknowledges several limitations and proposes areas for future work:

**Limitations:**
* The backtest is purely academic and ignores transaction costs, slippage, and market impact.
* Only the XGBoost model was hyperparameter-tuned; a tuned Random Forest might have performed better.
* The test set size (89 days) is small and may not be statistically significant.

**Future Improvements:**
* Implement a more robust time-series cross-validation (e.g., `TimeSeriesSplit`).
* Use feature selection (like RFE or SHAP) to reduce the 40+ features and potentially lessen noise.
* Incorporate alternative data, such as market sentiment from news headlines.
