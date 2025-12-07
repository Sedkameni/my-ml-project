## 1 - Read the Dataset

    ## Load the provided dataset from a CSV file into a Pandas DataFrame.
import pandas as pd
# Load dataset
df = pd.read_csv('sample_dataset.csv')
print(df.head())

## 2 - Perform Basic Data Manipulation
## Display the first few rows to inspect the data.
# ##Check for missing values and handle them appropriately.
# Display dataset information
print(df.info())
# Check for missing values
print(df.isnull().sum())

## 3 - Calculate Statistical Measures
## Compute the mean, median, minimum, and maximum age from the dataset.
age_mean = df["Age"].mean()
age_median = df["Age"].median()
age_min = df["Age"].min()
age_max = df["Age"].max()
print(f"Mean: {age_mean}, Median: {age_median}, Min: {age_min}, Max: {age_max}")

##4 - Filter Data
## Filter the dataset to include only rows where the age is between 30 and 50.
df_filtered = df[(df["Age"] >= 30) & (df["Age"] <= 50)]
print(df_filtered)

## 5 - Group Data and Compute Aggregates
## Group the dataset by the 'Group' column and calculate the average score for each group.
group_avg_score = df.groupby("Group")["Score"].mean()
print(group_avg_score)

## 6 - Visualize Data

## Generate a histogram of age distribution and a bar plot of average score per group.
import matplotlib.pyplot as plt
# Histogram of age distribution
plt.figure(figsize=(8, 5))
df["Age"].hist(bins=5, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# Bar chart of average score per group
plt.figure(figsize=(8, 5))
df.groupby("Group")["Score"].mean().plot(kind="bar", color="orange", edgecolor="black")
plt.title("Average Score by Group")
plt.xlabel("Group")
plt.ylabel("Average Score")
plt.show()













