"""
Train a logistic regression model on the Iris dataset and save it using pickle.
"""
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("Training Logistic Regression Model on Iris Dataset")
print("=" * 60)

# Load the Iris dataset
print("\nLoading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

print(f" Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Features: {iris.feature_names}")
print(f"  Classes: {iris.target_names.tolist()}")

# Split the data into training and testing sets
print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Training set: {X_train.shape[0]} samples")
print(f" Testing set: {X_test.shape[0]} samples")

# Train a logistic regression model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
print("✓ Model trained successfully!")

# Evaluate the model
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model Performance:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Save the model and metadata to a file using pickle
print("\nSaving model to 'logistic_model.pkl'...")
model_data = {
    'model': model,
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'accuracy': accuracy
}

with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(" Model saved as 'logistic_model.pkl'")
