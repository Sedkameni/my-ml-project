
## 1 - Importing the Dataset
#Imports the Pandas library for data manipulation.
import pandas as pd
# Assuming the data is in a CSV file named 'dataset.csv'
df = pd.read_csv('car.csv') #Reads the dataset into a Pandas DataFrame.
print(df.head()) #Displays the first five rows of the dataset to understand its structure.

## 2 - Data Exploration
# Check the structure and contents
df.info()
# Check for missing values and Counts the number of missing values in each column.
print("The missing values are:", df.isnull().sum())

## 3 - Basic Data Manipulation
# Number of unique car makes and models
unique_makes = df['Make'].nunique() #Counts the unique car makes.
unique_models = df['Model'].nunique() #  Counts the unique car models.

# Average mileage
average_mileage = df['Mileage'].mean() # Computes the average mileage of the cars.

# Calculate the total cost
df['Total Cost'] = df['Price'] + df['Cost'] # Creates a new column Total Cost by adding Price and Cost columns.

## 4 - Data Visualization

import matplotlib.pyplot as plt #Imports Matplotlib for visualization.
# Histogram of car prices
df['Price'].hist(bins=20) # Creates a histogram of car prices with 20 bins.
plt.show()  #Displays the plot using plt.show().


# Bar plot of average mileage by car color
df.groupby('Color')['Mileage'].mean().plot(kind='bar') #Groups data by Color and calculates average mileage.
plt.show() # Creates a bar plot to compare mileage across different colors.

# Scatter plot of mileage vs price
df.plot.scatter(x='Mileage', y='Price') ## Generates a scatter plot to analyze the relationship between mileage and price.
plt.show()

## 5 - Advanced Data Manipulation
# Group by car make for average price and mileage
df.groupby('Make').agg({'Price': 'mean', 'Mileage': 'mean'}) # Groups the dataset by Make and calculates the average Price and Mileage for each make.

# Filter dataset for specific conditions
#Filters the dataset to include cars with mileage below 100,000 and price below $5,000.
df_filtered = df[(df['Mileage'] < 100000) & (df['Price'] < 5000)]

# Sort by price in descending order
df_sorted = df.sort_values(by='Price', ascending=False) #Sorts cars by Price in descending order.
top5_expensive = df_sorted.head(5) #Extracts the top 5 most expensive cars.

## 6 - Custom Data Analysis

# Average cost (Price - Cost) for each car model
# Groups by Model and calculates the average Price and Cost.
# Creates a bar plot to visualize the results.
df.groupby('Model').agg({'Price': 'mean', 'Cost': 'mean'}).plot(kind='bar')
plt.show()
# Most common car color
# Counts the occurrences of each car color and displays them in a pie chart.
df['Color'].value_counts().plot(kind='pie')
plt.show()

# Highest and lowest mileage cars
highest_mileage_car = df[df['Mileage'] == df['Mileage'].max()] #Identifies the car with the highest mileage.
lowest_mileage_car = df[df['Mileage'] == df['Mileage'].min()] #Identifies the car with the lowest mileage.
