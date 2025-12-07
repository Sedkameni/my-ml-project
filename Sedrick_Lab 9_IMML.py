#dataset
## Part 1 - Load and Explore the Dataset
## 1. Load the Dataset: Download the dataset and load it into Python using Pandas.
import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('dataset.csv', parse_dates=['date'], index_col='date') ##datasetassignment9
print(df.head())

## 2. Check for Missing Values: Ensure there are no missing values in the dataset.
print(df.isnull().sum())


## 3. Visualize the Time Series Data: Plot the original wind speed values to identify trends, seasonality, and patterns.
plt.figure(figsize=(12,6))
plt.plot(df['wind_speed'], label='Original Wind Speed')
plt.title('Wind Speed Over Time')
plt.xlabel('Date')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()

## Part 2 - Forecasting with Moving Average (3 M.A)

## 1. Calculate 3-Period Moving Average
# Compute the 3-period moving average
df['3_MA'] = df['wind_speed'].rolling(window=3).mean()
print(df[['wind_speed', '3_MA']].head())


## 2. Visualize Moving Average Forecast
plt.figure(figsize=(12,6))
plt.plot(df['wind_speed'], label='Original')
plt.plot(df['3_MA'], label='3-Period Moving Average', alpha=0.7)
plt.title('Wind Speed and 3-Period Moving Average')
plt.xlabel('Date')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()


## Part3 - Forecasting with Exponential Smoothing

   ##1. Apply Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing
# Apply Simple Exponential Smoothing (alpha = 0.5)
model = SimpleExpSmoothing(df['wind_speed']).fit(smoothing_level=0.5)
df['Exp_Smoothed'] = model.fittedvalues
print(df[['wind_speed', 'Exp_Smoothed']].head())

## 2. Generate Forecasts
# Forecast the next X steps ahead in the future, replace X with the number of periods to forecast
X = 10  # Adjust as needed
future_forecast = model.forecast(steps=X)
print(future_forecast)


## 3. Visualize Exponential Smoothing Forecast
plt.figure(figsize=(12,6))
plt.plot(df['wind_speed'], label='Original')
plt.plot(df['Exp_Smoothed'], label='Exponential Smoothing Forecast', alpha=0.7)
plt.title('Wind Speed and Exponential Smoothing Forecast')
plt.xlabel('Date')
plt.ylabel('Wind Speed')
plt.legend()
plt.show()












