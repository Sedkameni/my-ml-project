
## 1 - Import Necessary Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

## # Load dataset
df = pd.read_csv('gender_classification.csv')  # Replace 'path_to_your_data.csv' with the actual path
print(df.head())

## Encoding Categorical Data
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  # Male: 1, Female: 0

## Splitting the Dataset
X = df.drop('gender', axis=1)
y = df['gender']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Model Training
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = (accuracy, report)

for model, (accuracy, report) in results.items():
    print(f"{model} Accuracy: {accuracy:.2f}")
    print(report)

