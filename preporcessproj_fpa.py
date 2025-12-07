import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import pickle
import os
import kagglehub
from pathlib import Path


def clean_tweet(text):
    """Clean and preprocess tweet text"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (keep the text)
    text = re.sub(r'#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(filepath, sample_size=None):
    """
    Load and preprocess Sentiment140 dataset, with robust file handling.

    Args:
        filepath (str or Path): Path to the Sentiment140 CSV file (often from kagglehub).
        sample_size (int, optional): Number of rows to sample. Defaults to None.

    Returns:
        pd.DataFrame or None: Cleaned DataFrame or None if loading fails.
    """
    print("Loading dataset...")

    # Ensure the path is a string for robust use with os.path.abspath and pd.read_csv
    filepath_str = str(filepath)
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

    # --- ERROR HANDLING ---
    try:
        # Load dataset
        # Pandas can handle the Path object returned by kagglehub, or the string.
        df = pd.read_csv(filepath_str, encoding='latin-1', names=columns)
        print(df.head())
    except FileNotFoundError:
        print(f"\n ERROR: File not found at '{os.path.abspath(filepath_str)}'")
        # Update message to reflect the dynamic nature of the path from kagglehub
        print("Please ensure the Kaggle Hub download process completed successfully.")
        return None
    except pd.errors.ParserError as e:
        print(f"\n WARNING: Data parsing error encountered: {e}")
        print("Try inspecting the CSV file for incorrect formatting or use a different delimiter.")
        return None
    # ----------------------

    # Sample data if specified
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)

    print(f"Dataset loaded: {len(df)} rows")

    # Clean tweets
    print("Cleaning tweets...")
    # NOTE: Assuming clean_tweet is defined globally and available.
    df['cleaned_text'] = df['text'].apply(clean_tweet)

    # Remove empty tweets after cleaning
    initial_rows = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    rows_filtered = initial_rows - len(df)
    print(f"Empty tweets removed: {rows_filtered}")

    # Convert target: 0->0 (negative), 4->1 (positive)
    df['sentiment'] = df['target'].map({0: 0, 4: 1})

    # Drop rows where 'sentiment' is NaN (Crucial for stratification)
    initial_len = len(df)
    df.dropna(subset=['sentiment'], inplace=True)
    rows_dropped = initial_len - len(df)

    # Select relevant columns
    df = df[['cleaned_text', 'sentiment']]

    print(f"Rows with missing sentiment dropped: {rows_dropped}")
    print(f"Dataset after cleaning and filtering: {len(df)} rows")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}")

    return df


def split_data(df, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets"""
    # First split: train+val and test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['sentiment']
    )

    # Second split: train and validation
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=42, stratify=train_val['sentiment']
    )

    print(f"\nData split:")
    print(f"Train: {len(train)} samples")
    print(f"Validation: {len(val)} samples")
    print(f"Test: {len(test)} samples")

    return train, val, test

if __name__ == "__main__":
    # --- 1. DOWNLOAD AND FIND DATASET PATH ---
    print("Downloading and locating Sentiment140 dataset via Kaggle Hub")

    # Download the dataset (returns the directory path where files are stored)
    path = kagglehub.dataset_download("kazanova/sentiment140")

    # Find the largest CSV file in the downloaded directory (the main dataset file)
    dataset_path = Path(path)
    csv_files = list(dataset_path.glob("*.csv"))

    if not csv_files:
        print(" ERROR: No CSV files found in the downloaded directory.")
        exit()

    # Get the path to the main CSV file (usually the largest one)
    main_file = max(csv_files, key=lambda x: x.stat().st_size)

    print(f"Main dataset file found at: {main_file}")

    # --- 2. CALL PREPROCESSING FUNCTION ---
    # Pass the actual Path object to your data loading function.
    df = load_and_preprocess_data(filepath=main_file)

    # Split data
    train_df, val_df, test_df = split_data(df)

    # Save processed datasets
    print("\nSaving processed datasets...")
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print("Preprocessing complete!")