## Part 1 - Load and Explore the Dataset
## 1. Load the Dataset: Download the dataset and load it into Python using Pandas.
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('datasetassignment9.csv', parse_dates=['date'], index_col='date')
print("\nThe first 5 rows of the dataset are: \n", df.head())

# Get the dataset information and the types of elements in the columns
print("\n The information of the dataset are:  \n", df.info())


## 2. Check for Missing Values: Ensure there are no missing values in the dataset.
print("\nThe misssing values inthe dataset are: \n", df.isnull().sum())


## 3. Visualize the Time Series Data: Plot the original humidity values to identify trends, seasonality, and patterns.
plt.figure(figsize=(12,6))
plt.plot(df['humidity'], label='Original Wind Speed')
plt.title('humidity Over Time')
plt.xlabel('Date')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()


## Part 2 - Forecasting with Moving Average (3 M.A)

## 1. Calculate 3-Period Moving Average
# Compute the 3-period moving average
df['3_MA'] = df['humidity'].rolling(window=3).mean()
print(df[['humidity', '3_MA']].head())

## 2: Forecast next humidity value using the last 3 observations
forecast_next = df['humidity'].iloc[-3:].mean()
print(f"\nForecasted humidity for the next time point (3-MA): {forecast_next:.2f}")

## 3. Visualize the forecasted values along with the original time series data.
plt.figure(figsize=(12,6))
plt.plot(df['humidity'], label='Original', color='blue')
plt.plot(df['3_MA'], label='3-Period Moving Average', color='red', alpha=0.7)
plt.title('Humidity and 3-Period Moving Average')
plt.xlabel('Date')
plt.ylabel('humidity')
plt.legend()
plt.show()

# Extend forecast one step ahead
#future_date = df.index[-1] + pd.Timedelta(days=1)
#plt.scatter(future_date, forecast_next, color='green', label='Forecast (Next Day)', zorder=5)


## Part3 - Forecasting with Exponential Smoothing

   ##1. Apply Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing
# Apply Simple Exponential Smoothing (alpha = 0.5)
model = SimpleExpSmoothing(df['humidity']).fit(smoothing_level=0.5)
df['Exp_Smoothed'] = model.fittedvalues
print(df[['humidity', 'Exp_Smoothed']].head())

## 2. Generate Forecasts
# Forecast the next X steps ahead in the future, replace X with the number of periods to forecast
X = 10  # Adjust as needed
future_forecast = model.forecast(steps=X)
print(future_forecast)


## 3. Visualize Exponential Smoothing Forecast
plt.figure(figsize=(12,6))
plt.plot(df['humidity'], label='Original')
plt.plot(df['Exp_Smoothed'], label='Exponential Smoothing Forecast', alpha=0.7)
plt.title('humidity and Exponential Smoothing Forecast')
plt.xlabel('Date')
plt.ylabel('humidity')
plt.legend()
plt.show()











