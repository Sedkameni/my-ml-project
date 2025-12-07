## 1 - Read the provided dataset from a excel file into a Pandas DataFrame.
## import pandas library
import pandas as pd
# Load dataset
df = pd.read_excel('dataset_assignment10.xlsx')

## 2 - Perform Basic Data Manipulation
## Display the first few rows to inspect the data.
print("The first five rows are:\n", df.head())
# Display dataset information
print("\nInformation related to the dataset are:\n", df.info())
# Check for missing values
print("\nBelow is the summary of the missing values in the dataset: \n", df.isnull().sum())

## 3 - Calculate Statistical Measures: Calculate the mean, median, minimum, and maximum age.
age_mean = df["Age"].mean()
age_median = df["Age"].median()
age_min = df["Age"].min()
age_max = df["Age"].max()
print(f"Mean: {age_mean}, Median: {age_median}, Min: {age_min}, Max: {age_max}")

##4 - Filter the dataset to include only rows where the age is between 30 and 50.
df_filtered = df[(df["Age"] >= 30) & (df["Age"] <= 50)]
print("\n The part of the dataset that include only rows where age is between 30 and 50 is: \n", df_filtered)

## 5 - Group the data by 'Group' column and calculate the average score for each group.
group_avg_score = df.groupby("Group")["Score"].mean()
print("Group column with their respective average score are: \n",  group_avg_score)













