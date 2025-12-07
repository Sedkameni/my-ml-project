# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# import json
# from datetime import datetime
# import os  # Added for path handling
#
# # Ensure 'models' directory exists for saving artifacts
# os.makedirs('models', exist_ok=True)
#
#
# def load_data():
#     """Load preprocessed datasets"""
#     print("Loading datasets...")
#     try:
#         train_df = pd.read_csv('data/train.csv')
#         val_df = pd.read_csv('data/val.csv')
#         test_df = pd.read_csv('data/test.csv')
#         return train_df, val_df, test_df
#     except FileNotFoundError:
#         print("\n🚨 ERROR: Required data files ('data/*.csv') not found.")
#         print("Please run the preprocessing script ('preporcessproj_fpa.py') first.")
#         # Exit gracefully if data is missing
#         exit()
#
#
# def train_model(train_df, val_df):
#     """Train logistic regression model with TF-IDF features"""
#     print("\nVectorizing text with TF-IDF...")
#
#     # Initialize TF-IDF Vectorizer
#     vectorizer = TfidfVectorizer(
#         max_features=10000,
#         ngram_range=(1, 2),
#         min_df=1,            # Keeps tokens that appear at least once (prevents empty vocabulary)
#         max_df=0.8,
#         stop_words=None      # <-- CRITICAL FIX: DO NOT remove standard English stop words
#     )
#
#     # Fit and transform training data
#     X_train = vectorizer.fit_transform(train_df['cleaned_text'].astype(str))
#     y_train = train_df['sentiment'].values
#
#     # Transform validation data
#     X_val = vectorizer.transform(val_df['cleaned_text'].astype(str))
#     y_val = val_df['sentiment'].values
#
#     # Check for empty vocabulary after fit_transform
#     if not vectorizer.vocabulary_:
#         raise ValueError(
#             "Vectorization failed: Vocabulary is empty after transformation. "
#             "Re-check data cleaning in 'preporcessproj_fpa.py'."
#         )
#
#     print(f"Training data shape: {X_train.shape}")
#     print(f"Validation data shape: {X_val.shape}")
#     print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
#
#     # Train Logistic Regression model
#     print("\nTraining Logistic Regression model...")
#     model = LogisticRegression(
#         max_iter=1000,
#         random_state=42,
#         solver='liblinear',
#         C=1.0
#     )
#
#     model.fit(X_train, y_train)
#
#     # Evaluate on training data
#     # (Evaluation code remains the same)
#     print("\nTraining Set Performance:")
#     train_pred = model.predict(X_train)
#     train_acc = accuracy_score(y_train, train_pred)
#     train_f1 = f1_score(y_train, train_pred)
#     print(f"Accuracy: {train_acc:.4f}")
#     print(f"F1-Score: {train_f1:.4f}")
#
#     # Evaluate on validation data
#     print("\nValidation Set Performance:")
#     val_pred = model.predict(X_val)
#     val_acc = accuracy_score(y_val, val_pred)
#     val_f1 = f1_score(y_val, val_pred)
#     print(f"Accuracy: {val_acc:.4f}")
#     print(f"F1-Score: {val_f1:.4f}")
#
#     print("\nClassification Report (Validation):")
#     print(classification_report(y_val, val_pred, target_names=['Negative', 'Positive']))
#
#     print("\nConfusion Matrix (Validation):")
#     print(confusion_matrix(y_val, val_pred))
#
#     return model, vectorizer, {
#         'train_accuracy': train_acc,
#         'train_f1': train_f1,
#         'val_accuracy': val_acc,
#         'val_f1': val_f1
#     }
#
#
# def evaluate_test_set(model, vectorizer, test_df):
#     """Evaluate model on test set"""
#     print("\n" + "=" * 50)
#     print("FINAL TEST SET EVALUATION")
#     print("=" * 50)
#
#     # --- FIX: Ensure test data is also string type ---
#     X_test = vectorizer.transform(test_df['cleaned_text'].astype(str))
#     y_test = test_df['sentiment'].values
#     # --------------------------------------------------
#
#     test_pred = model.predict(X_test)
#     test_acc = accuracy_score(y_test, test_pred)
#     test_f1 = f1_score(y_test, test_pred)
#
#     print(f"Test Accuracy: {test_acc:.4f}")
#     print(f"Test F1-Score: {test_f1:.4f}")
#
#     print("\nClassification Report (Test):")
#     print(classification_report(y_test, test_pred, target_names=['Negative', 'Positive']))
#
#     print("\nConfusion Matrix (Test):")
#     print(confusion_matrix(y_test, test_pred))
#
#     return {
#         'test_accuracy': test_acc,
#         'test_f1': test_f1
#     }
#
#
# def save_model(model, vectorizer, metrics, version='1.0'):
#     """Save model, vectorizer, and metrics"""
#     print(f"\nSaving model version {version}...")
#
#     # Ensure 'models' directory exists (safe re-check)
#     os.makedirs('models', exist_ok=True)
#
#     # Save model
#     with open(f'models/model_v{version}.pkl', 'wb') as f:
#         pickle.dump(model, f)
#
#     # Save vectorizer
#     with open(f'models/vectorizer_v{version}.pkl', 'wb') as f:
#         pickle.dump(vectorizer, f)
#
#     # Save metrics with timestamp
#     metrics['version'] = version
#     metrics['timestamp'] = datetime.now().isoformat()
#
#     with open(f'models/metrics_v{version}.json', 'w') as f:
#         json.dump(metrics, f, indent=4)
#
#     print(f"Model saved successfully as version {version}")
#
#
# if __name__ == "__main__":
#     # Load data
#     train_df, val_df, test_df = load_data()
#
#     # Train model
#     model, vectorizer, metrics = train_model(train_df, val_df)
#
#     # Evaluate on test set
#     test_metrics = evaluate_test_set(model, vectorizer, test_df)
#     metrics.update(test_metrics)
#
#     # Save model
#     save_model(model, vectorizer, metrics, version='1.0')
#
#     print("\nTraining complete!")

# -*- coding: utf-8 -*-
"""trainProj_FPA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_r6XSktL1eaQSYkEAk5VdhkfiNes01oV
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from datetime import datetime

def load_data():
    """Load preprocessed datasets"""
    print("Loading datasets...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    test_df = pd.read_csv('data/test.csv')
    return train_df, val_df, test_df

def train_model(train_df, val_df):
    """Train logistic regression model with TF-IDF features"""
    print("\nVectorizing text with TF-IDF...")

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=5,
        max_df=0.8
    )

    # Fit and transform training data
    #X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    X_train = vectorizer.fit_transform(train_df['cleaned_text'].astype(str))
    y_train = train_df['sentiment'].values

    # Transform validation data
    X_val = vectorizer.transform(val_df['cleaned_text'])
    y_val = val_df['sentiment'].values

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='liblinear',
        C=1.0
    )

    model.fit(X_train, y_train)

    # Evaluate on training data
    print("\nTraining Set Performance:")
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred)
    print(f"Accuracy: {train_acc:.4f}")
    print(f"F1-Score: {train_f1:.4f}")

    # Evaluate on validation data
    print("\nValidation Set Performance:")
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)
    print(f"Accuracy: {val_acc:.4f}")
    print(f"F1-Score: {val_f1:.4f}")

    print("\nClassification Report (Validation):")
    print(classification_report(y_val, val_pred, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix (Validation):")
    print(confusion_matrix(y_val, val_pred))

    return model, vectorizer, {
        'train_accuracy': train_acc,
        'train_f1': train_f1,
        'val_accuracy': val_acc,
        'val_f1': val_f1
    }

def evaluate_test_set(model, vectorizer, test_df):
    """Evaluate model on test set"""
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)

    X_test = vectorizer.transform(test_df['cleaned_text'])
    y_test = test_df['sentiment'].values

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_pred, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, test_pred))

    return {
        'test_accuracy': test_acc,
        'test_f1': test_f1
    }

def save_model(model, vectorizer, metrics, version='1.0'):
    """Save model, vectorizer, and metrics"""
    print(f"\nSaving model version {version}...")

    # Save model
    with open(f'models/model_v{version}.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save vectorizer
    with open(f'models/vectorizer_v{version}.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save metrics with timestamp
    metrics['version'] = version
    metrics['timestamp'] = datetime.now().isoformat()

    with open(f'models/metrics_v{version}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved successfully as version {version}")

if __name__ == "__main__":
    # Load data
    train_df, val_df, test_df = load_data()

    # Train model
    model, vectorizer, metrics = train_model(train_df, val_df)

    # Evaluate on test set
    test_metrics = evaluate_test_set(model, vectorizer, test_df)
    metrics.update(test_metrics)

    # Save model
    save_model(model, vectorizer, metrics, version='1.0')

    print("\nTraining complete!")