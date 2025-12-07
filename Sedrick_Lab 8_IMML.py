
## 1 - Load the Dataset: Download the dataset and load it into Python using Pandas & numpy.
import pandas as pd
import numpy as np
# Load dataset
df = pd.read_csv('salary_data.csv')
print(df.head())

## 2 - Check for Missing Values: Ensure there are no missing values in the dataset.
print(df.isnull().sum())

## 3 - Check Data Types: Confirm that "Years Experience" and "Salary" are numerical values.
print(df.dtypes)

## 4 - Compute Summary Statistics

    ## 4.1 - Compute Mean and Variance: Calculate the mean and variance of "Years Experience" and "Salary."
import numpy as np

# Compute mean
x_mean = np.mean(df['YearsExperience'])
y_mean = np.mean(df['Salary'])

# Compute variance
x_var = np.var(df['YearsExperience'], ddof=1)
y_var = np.var(df['Salary'], ddof=1)
print(f"Mean of Years Experience: {x_mean}, Variance: {x_var}")
print(f"Mean of Salary: {y_mean}, Variance: {y_var}")

## 5 - Solve Simple Linear Regression by Hand

  ##  5.1 - Compute Beta (Slope) and Alpha (Intercept): Use the formulas for linear regression:
n = len(df)
numerator = n * sum(df['YearsExperience'] * df['Salary']) - sum(df['YearsExperience']) * sum(df['Salary'])
denominator = n * sum(df['YearsExperience'] ** 2) - (sum(df['YearsExperience'])) ** 2
beta = numerator / denominator
alpha = y_mean - beta * x_mean

print(f"Alpha (Intercept): {alpha}")
print(f"Beta (Slope): {beta}")

## 6 - Build Regression Model Using Scikit-Learn

    ## 6.1 - Train the Model
from sklearn.linear_model import LinearRegression
X = df[['YearsExperience']]
y = df['Salary']
model = LinearRegression()
model.fit(X, y)
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

## 6.2 - Predict Salaries: Use the trained model to make predictions.
y_pred = model.predict(X)
df['Predicted Salary'] = y_pred
print(df.head())

## 7 - Evaluate the Model

    ## 7.1 - Compute R-squared Value: Determine how well the model fits the data.
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print(f"R-Squared Value: {r2}")

## 8 - Visualize the Results

   ## 8.1 - Plot Regression Line
import matplotlib.pyplot as plt
plt.scatter(df['YearsExperience'], df['Salary'], color='blue', label='Actual Data')
plt.plot(df['YearsExperience'], y_pred, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()