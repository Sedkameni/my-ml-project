## 1 - Explore the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_excel('Sheet2_Dataset.xlsx')
print(df.head())

## 2 - Check for Missing Values: Ensure there are no missing values in the dataset.
print(df.isnull().sum())

## 3 - Check Data Types: Confirm that "Years Experience" and "Salary" are numerical values.
print(df.dtypes)

## 4 - Compute Summary Statistics

print(df.describe())

# 5 - Distribution of the wine quality
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme
axes[0].hist(df['quality'], bins=6, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Quality Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution wine quality', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Boxplot
axes[1].boxplot(df['quality'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
axes[1].set_ylabel('Quality Score', fontsize=12, fontweight='bold')
axes[1].set_title('Boxplot de la Qualité', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('1_distribution_quality.png', dpi=300, bbox_inches='tight')
print("\n The image has been save under the following name: 1_distribution_quality.png")
plt.close()


# 6 - Correlation matrix and heat map
print("\n The Correlation matrix with Quality:")
correlation_matrix = df.corr()
quality_corr = correlation_matrix['quality'].sort_values(ascending=False)
print(quality_corr.round(3))

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation matrix of the Chemical properties',
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
plt.savefig('2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saving of the Correlation matrix and Heatmap under the name: 2_correlation_heatmap.png")
plt.close()

# 7 - Pairplot of the most correlated variables with quality
top_features = quality_corr.abs().nlargest(6).index.tolist()
top_features.remove('quality')
top_features.append('quality')

print(f"\nTop 5 correlated variables with quality: {top_features[:-1]}")
sns.pairplot(df[top_features], hue='quality', palette='viridis', plot_kws={'alpha':0.6})
plt.savefig('3_pairplot_top_features.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saving figure under: 3_pairplot_top_features.png")
plt.close()

## II - Preprocess the data
X = df.drop('quality', axis=1)
y = df['quality']

print(f"\nII.1 - Dimensions:")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# II.2 - Check for missing values
print(f"\nMissing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in y: {y.isnull().sum()}")

# II.3 - Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n# Split the dataset into training and testing sets (80% train, 20% test):")
print(f"Training set: {X_train.shape[0]} sample")
print(f"Testing set: {X_test.shape[0]} sample")

## III - Build a regression model

# Initialize the model
model_original = LinearRegression()
##	Fit the model on the training set.
model_original.fit(X_train, y_train)

##	Use Multiple Linear Regression:
print(f"Equation: quality = β0 + β1*fixed_acidity + β2*volatile_acidity + ... + β11*alcohol + ε")
print(f"\nIntercept (β₀): {model_original.intercept_:.4f}")

 ##  Display the coefficients
coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient (Original)': model_original.coef_,
}).sort_values('Coefficient (Original)', ascending=False)

print(coefficients_df.to_string(index=False))

## IV - Evaluate the model

# - Predictions
y_train_pred_original = model_original.predict(X_train)
y_test_pred_original = model_original.predict(X_test)

##	Metrics: R-squared, MAE, MSE.
print("TRAINING SET:")
r2_train_orig = r2_score(y_train, y_train_pred_original)
mae_train_orig = mean_absolute_error(y_train, y_train_pred_original)
mse_train_orig = mean_squared_error(y_train, y_train_pred_original)
rmse_train_orig = np.sqrt(mse_train_orig)

print(f"  R² Score: {r2_train_orig:.4f} ({r2_train_orig*100:.2f}% of the variance)")
print(f"  MAE (Mean Absolute Error): {mae_train_orig:.4f}")
print(f"  MSE (Mean Squared Error): {mse_train_orig:.4f}")
print(f"  RMSE (Root Mean Squared Error): {rmse_train_orig:.4f}")

print("\nTESTING SET:")
r2_test_orig = r2_score(y_test, y_test_pred_original)
mae_test_orig = mean_absolute_error(y_test, y_test_pred_original)
mse_test_orig = mean_squared_error(y_test, y_test_pred_original)
rmse_test_orig = np.sqrt(mse_test_orig)

print(f"  R² Score: {r2_test_orig:.4f} ({r2_test_orig*100:.2f}% of the variance)")
print(f"  MAE (Mean Absolute Error): {mae_test_orig:.4f}")
print(f"  MSE (Mean Squared Error): {mse_test_orig:.4f}")
print(f"  RMSE (Root Mean Squared Error): {rmse_test_orig:.4f}")


##	Check residual plots to ensure no major patterns (good model fit).
# - Analysis of residuals
residuals_train = y_train - y_train_pred_original
residuals_test = y_test - y_test_pred_original

# Plotting of residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuals vs Predicted (Training)
axes[0, 0].scatter(y_train_pred_original, residuals_train, alpha=0.5, color='blue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Quality', fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontweight='bold')
axes[0, 0].set_title('Residuals vs Predicted (Training Set)', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# 2. Residuals vs Predicted (Testing)
axes[0, 1].scatter(y_test_pred_original, residuals_test, alpha=0.5, color='green')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Quality', fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontweight='bold')
axes[0, 1].set_title('Residuals vs Predicted (Testing Set)', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

## 3. Distribution of residuals (Training)
axes[1, 0].hist(residuals_train, bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Residuals', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Distribution of Residuals (Training)', fontweight='bold')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].grid(alpha=0.3)

# 4. Q-Q Plot (Normal distribution check)
from scipy import stats
stats.probplot(residuals_test, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Testing Set)', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('analysis of residuals', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()
plt.savefig('4_residual_analysis.png', dpi=300, bbox_inches='tight')
print("\nImage saved under the name: 4_residual_analysis.png")
plt.close()


# 4.5 - Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training set
axes[0].scatter(y_train, y_train_pred_original, alpha=0.5, color='blue')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Quality', fontweight='bold')
axes[0].set_ylabel('Predicted Quality', fontweight='bold')
axes[0].set_title(f'Training Set (R²={r2_train_orig:.3f})', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Testing set
axes[1].scatter(y_test, y_test_pred_original, alpha=0.5, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Quality', fontweight='bold')
axes[1].set_ylabel('Predicted Quality', fontweight='bold')
axes[1].set_title(f'Testing Set (R²={r2_test_orig:.3f})', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Actual vs Predicted Quality', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('5_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print(" Image saved under the name: 5_actual_vs_predicted.png")
plt.close()







