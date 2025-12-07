### Part 1 - Predicting Employee Work Satisfaction

## 1 - Load the Dataset: Download the dataset and load it into Python using Pandas & numpy.
import pandas as pd
import numpy as np
# Load dataset
df = pd.read_excel('Sheet1_DataSet.xlsx')
print(df.head())

## 2 - Check for Missing Values: Ensure there are no missing values in the dataset.
print(df.isnull().sum())

## 3 - Check Data Types: Confirm that "Years Experience" and "Salary" are numerical values.
print(df.dtypes)

## 4 - Compute Summary Statistics

print(df.describe())

## Visualize the relationships: scatter plots of wrksatis vs interest and wrksatis vs wage.
import matplotlib.pyplot as plt
plt.scatter(df['WrkSatis'], df['Interest'], color='blue', label='Actual Data')
plt.xlabel('WrkSatis')
plt.ylabel('Interest')
plt.legend()
plt.show()

plt.scatter(df['WrkSatis'], df['Wage'], color='red', label='Actual Data')
plt.xlabel('WrkSatis')
plt.ylabel('Wage')
plt.show()

## 3 - Build Regression Model Using Scikit-Learn
    ## - Train the Model
#Import the model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['Interest', 'Wage']]     # Independent variables
y = df['WrkSatis']               # Dependent variable

# Initialize the model
model = LinearRegression()
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 ) ## 20% of the data goes to the test set; ensures reproducibility

#Display the shapes to verify
print("Training set size:", X_train.shape, y_train.shape)
print("Testing set size:", X_test.shape, y_test.shape)

## Fit the model on the training data
model.fit(X_train, y_train)

# Step 4: Display the coefficients
print("Intercept (β0):", model.intercept_)
print("Coefficients (β1, β2):", model.coef_)

## Step 5 - Make predictions on the test data.
y_pred = model.predict(X_test)
print("The prediscted values are: ", y_pred)

## 6 - Evaluate the Model
##- Compute R-squared Value: Determine how well the model fits the data.
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"R-Squared Value: {r2}")

## 7 - Visualize the Results
##- Plot Regression Line
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)

# Label the axes and add a title
plt.xlabel('Actual wrksatis')
plt.ylabel('Predicted wrksatis')
plt.title('Predicted vs Actual wrksatis')
plt.show()

