#### 1 - Explore the dataset attached to the bottom of this page by examining its structure, dimensions, and data types.
##  - Importing the Dataset
#Imports the Pandas library for data manipulation.
import pandas as pd
import numpy as np
# Assuming the data is in a CSV file named 'dataset.csv'
df = pd.read_excel('assignment dataset.xlsx') #Reads the dataset into a Pandas DataFrame.

# Tell pandas to display all columns
pd.set_option('display.max_columns', None)

print("\nThe first five rows are:\n", df.head()) #Displays the first five rows of the dataset to understand its structure.

## - Data Exploration
# Check the structure and contents
df.info()

#### 2 - Identify and handle any missing or inconsistent data if present.
# Check for missing values and Counts the number of missing values in each column.
print("\nThe missing values are:\n", df.isnull().sum())

# Filling the missing values in Discount Band with "NaN"
df['Discount Band'] = df['Discount Band'].fillna('NaN')

# Check if the missing values have been filled
print("Number of missing values in Discount Band: ", df['Discount Band'].isna().sum())

#Check if there are other values in Discount Band than NaN
print(df['Discount Band'].unique())

## 3 - Utilize descriptive statistics techniques to summarize the numerical attributes (e.g., Units Sold, Manufacturing Price, Sale Price).
print("\nThe describtive statistics is given by: \n", df[['Units Sold', 'Manufacturing Price', 'Sale Price']].describe())


# 4 - Identify any outliers, anomalies, or potential data quality issues.
## - Detect Outliers Using the IQR Method
Q1 = df[['Units Sold', 'Manufacturing Price', 'Sale Price']].quantile(0.25)
Q3 = df[['Units Sold', 'Manufacturing Price', 'Sale Price']].quantile(0.75)
IQR = Q3 - Q1

outliers = ((df[['Units Sold', 'Manufacturing Price', 'Sale Price']] < (Q1 - 1.5 * IQR)) |
            (df[['Units Sold', 'Manufacturing Price', 'Sale Price']] > (Q3 + 1.5 * IQR)))

print("\nThe number os outliers are: ", outliers.sum())

## We also use a boxplot for visual purpose
import matplotlib.pyplot as plt

df[['Units Sold', 'Manufacturing Price', 'Sale Price']].boxplot(figsize=(8, 5))
plt.title("Boxplots of Numerical Attributes")
plt.show()

## Check for Anomalies or Quality Issues
df.info()
print(df.isna().sum())

### 5 -  Data visualization
# Histogram of units sold
df['Units Sold'].hist(bins=20) # Creates a histogram of units sold with 20 bins.
plt.show()  #Displays the plot using plt.show().

# Bar plot of average Sale Price by Country
df.groupby('Country')['Sale Price'].mean().plot(kind='bar') #Groups data by Country and calculates average Sale Price.
plt.show() # Creates a bar plot to compare Sale Price across different countries.

## 6 - Advanced Data Manipulation
# Group by Segment for average Units Sold and Sale Price
df.groupby('Segment').agg({'Units Sold': 'mean', 'Sale Price': 'mean'}).plot(kind='bar') # Groups the dataset by Segment and calculates the average Units sold and Sale Price for each segment.

# Filter dataset for specific conditions
#Filters the dataset with Sale Price below 15 and Units Sold below $1,000.
df_filtered = df[(df['Sale Price'] < 15) & (df['Units Sold'] < 1000)]
print("\nThe Filtered dataset to Product with Sale Price below $15 and Units Sold below $1,000 are:\n", df_filtered['Product'])
# Groups by Product and calculates the average Manufacturing Price and Sale Price.
# Creates a bar plot to visualize the results.
df.groupby('Product').agg({'Manufacturing Price': 'mean', 'Sale Price': 'mean'}).plot(kind='bar')
plt.show()

# Counts the occurrences of each Product and displays them in a pie chart.
df['Product'].value_counts().plot(kind='pie')
plt.show()



