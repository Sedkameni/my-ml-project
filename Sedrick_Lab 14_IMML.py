'''Linear Algebra for Machine Learning

Objective:

    Understand vectors, matrices, and their operations in machine learning.
    Use NumPy to implement these operations.

Tasks:

    Create and manipulate vectors and matrices.
    Perform basic operations like matrix addition, multiplication, and transposition.
    Apply matrix operations to feature scaling and data transformation.
'''
import numpy as np

# Creating vectors and matrices
vector = np.array([1, 2, 3])
matrix = np.array([[1, 2], [3, 4]])

# Matrix operations
sum_matrix = matrix + matrix
product_matrix = np.dot(matrix, matrix)  # Matrix multiplication
transposed_matrix = np.transpose(matrix)

print("Vector:", vector)
print("Sum Matrix:\n", sum_matrix)
print("Product Matrix:\n", product_matrix)
print("Transposed Matrix:\n", transposed_matrix)

'''Calculus for Optimization

Objective:

    Understand derivatives and gradients and their role in optimization.
    Implement Gradient Descent using Python.

Tasks:

    Differentiate a function to find the minimum value.
    Implement Gradient Descent to optimize a function.
'''
import numpy as np

# Example function: f(x) = x^2
def function(x):
    return x**2

# Compute derivative (gradient)
def derivative(x):
    return 2*x

# Gradient Descent Algorithm
def gradient_descent(starting_x, learning_rate, iterations):
    x = starting_x
    for i in range(iterations):
        grad = derivative(x)
        x = x - learning_rate * grad  # Update step
    return x

# Run gradient descent
optimized_x = gradient_descent(starting_x=10, learning_rate=0.1, iterations=100)
print("Optimized x:", optimized_x)

'''Probability in Machine Learning

Objective:

    Understand probability distributions and their applications in machine learning.
    Apply Bayes' Theorem for probabilistic predictions.

Tasks:

    Simulate a probability distribution (Normal Distribution).
    Apply Bayes' Theorem to predict class probabilities.
'''

import numpy as np
import scipy.stats as stats

# Generate a normal distribution
data = np.random.normal(loc=50, scale=10, size=1000)

# Probability of a value within one standard deviation
probability = stats.norm.cdf(60, loc=50, scale=10) - stats.norm.cdf(40, loc=50, scale=10)
print("Probability of a value within one standard deviation:", probability)

'''Time Series Analysis

Objective:

    Apply moving averages and exponential smoothing to predict trends.

Tasks:

    Load time-series data and visualize trends.
    Apply a 7-day moving average.
    Use Exponential Smoothing for trend forecasting.
'''
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset
data = {'Date': pd.date_range(start='1/1/2023', periods=30, freq='D'), 'StockPrice': np.random.randint(100, 200, 30)}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Compute 7-day moving average
df['7_MA'] = df['StockPrice'].rolling(window=7).mean()

# Plot the data
plt.figure(figsize=(10,5))
plt.plot(df.index, df['StockPrice'], label='Original')
plt.plot(df.index, df['7_MA'], label='7-day Moving Average', color='red')
plt.title('Stock Price Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
'''Implementing a Machine Learning Model

Objective:

    Train a Supervised Learning Model to predict labels from data.

Tasks:

    Prepare data and preprocess features.
    Train a Logistic Regression Model.
    Evaluate the model using accuracy and confusion matrix.
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

# Load sample dataset
data = {'Feature1': np.random.randn(100), 'Feature2': np.random.randn(100), 'Label': np.random.randint(0, 2, 100)}
df = pd.DataFrame(data)

# Splitting the dataset
X = df[['Feature1', 'Feature2']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(report)