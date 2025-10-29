-----

# Tesla (TSLA) Stock Price Forecasting

This project explores and compares various time series forecasting models to predict the 'Adjusted Close' price of Tesla (TSLA) stock. It evaluates the performance of classical statistical models, machine learning ensemble methods, and deep learning models to determine the most accurate approach for the given dataset.

## Dataset

The dataset used is the "TESLA.csv" file from the [Tesla Stock Price Prediction dataset on Kaggle](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021). It contains daily stock data (Open, High, Low, Close, Adj Close, Volume) for one year, from September 29, 2021, to September 29, 2022.

## Methodology

The project follows a structured approach to model building and evaluation:

1.  **Data Loading & Preprocessing:** The data was loaded, the 'Date' column was converted to a datetime index, and the dataset was checked for any null values.
2.  **Exploratory Data Analysis (EDA):** Visualized price trends (Open vs. Close), trading volume, and 'Adjusted Close' price over time to identify underlying patterns.
3.  **Feature Engineering:** Created new time-series features from the date index (e.g., `month`, `year`, `dayofweek`, `quarter`) for use in the machine learning model.
4.  **Train/Test Split:** The data was split into training and testing sets using a fixed cutoff date of **2022-07-01**.

### Models Implemented

Four distinct forecasting models were built and evaluated:

1.  **XGBoost:** An `XGBRegressor` model was trained using the engineered date features (`dayofyear`, `month`, `weekofyear`, etc.) to predict the 'Adj Close' price.
2.  **SARIMA (Seasonal AutoRegressive Integrated Moving Average):** A classical statistical model. The optimal parameters `(0,1,0)x(2,1,0,5)` were determined using `pmdarima.auto_arima` on the **training data only** to prevent data leakage.
3.  **LSTM (Long Short-Term Memory):** A recurrent neural network (RNN) built with Keras. The model used a lookback window of 30 days and was trained on data scaled with `MinMaxScaler` (fit only on the training set).
4.  **Prophet:** Facebook's Prophet model was used to forecast the time series, capturing daily, weekly, and yearly seasonality.

## Results & Conclusion

All models were evaluated against the test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

The results show that **XGBoost** provided the most accurate and reliable predictions for this specific dataset.

| Model | MAE | RMSE | MAPE |
| :--- | ---: | ---: | ---: |
| **XGBoost** | 3.79 | 4.83 | **1.37%** |
| **LSTM** | 9.42 | 11.70 | 3.32% |
| **Prophet** | 14.77 | 20.40 | 4.98% |
| **SARIMA** | 96.62 | 105.73 | 33.53% |

**Final Conclusion:** The XGBoost model, trained on engineered date features, was the clear winner. This suggests that for this one-year dataset, the complex, non-linear patterns were better captured by the gradient-boosted tree model than by the sequence-based deep learning (LSTM) or classical statistical (SARIMA) models.

## Libraries & Tools

This project utilizes the following key Python libraries:

  * `pandas`
  * `numpy`
  * `scikit-learn` (for `MinMaxScaler` and metrics)
  * `xgboost`
  * `pmdarima` (for `auto_arima`)
  * `statsmodels` (for `SARIMAX`)
  * `prophet`
  * `keras` / `tensorflow` (for `LSTM`)
  * `matplotlib` & `seaborn` (for visualization)
