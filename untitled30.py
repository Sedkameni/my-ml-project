
# # ================================================================
# # PROJECT 2 – DATA CLEANING & PREPARATION
# # Builds on Project 1 dataset
# # ================================================================
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ------------------------------------------------
# # 1. LOAD DATA FROM PROJECT 1
# # ------------------------------------------------
#
#
#
# stocks_df = pd.read_csv("historical_stocks.csv")
# prices_df = pd.read_csv("historical_stock_prices.csv")
#
# # Clean whitespace
# prices_df['date'] = prices_df['date'].astype(str).str.strip()
#
# # Convert safely
# prices_df['date'] = pd.to_datetime(prices_df['date'], errors='coerce')
#
# # Drop the bad rows
# prices_df = prices_df.dropna(subset=['date'])
#
#
# # Merge
# merged = prices_df.merge(stocks_df, on="ticker", how="left")
# merged.index = prices_df.index
#
# print("Merged dataset preview:")
# merged.head()
#
# # =================================================================
# # 2. ADVANCED DATA CLEANING
# # =================================================================
#
# # ------------------------------------------------
# # 2.1 ADVANCED MISSING VALUE HANDLING
# # ------------------------------------------------
#
# # Check missing values
# print("\nMissing values before cleaning:")
# print(merged.isnull().sum())
#
# # Forward fill numeric columns
# num_cols = merged.select_dtypes(include=[np.number]).columns
# merged[num_cols] = merged[num_cols].interpolate(method='linear')
# #Missing close = 100 + (110 – 100)/2 = 105 - for linear
#
#
# # Fill remaining missing categorical values with "Unknown"
# cat_cols = merged.select_dtypes(include=['object']).columns
# merged[cat_cols] = merged[cat_cols].fillna("Unknown")
#
# print("\nMissing values AFTER cleaning:")
# print(merged.isnull().sum())
#
# """Step by Step:
#
# Q1 (25th percentile) – the value below which 25% of the data falls.
#
# Q3 (75th percentile) – the value below which 75% of the data falls.
#
# IQR (Interquartile Range) – Q3 - Q1 → measures the middle 50% spread of the data.
#
# Lower bound = Q1 - 1.5 * IQR
#
# Upper bound = Q3 + 1.5 * IQR
#
# Any value outside this range is considered an outlier.
# """
#
# # ------------------------------------------------
# # 2.2 OUTLIER DETECTION & HANDLING
# # Using IQR for 'close' and 'volume'
# # ------------------------------------------------
# def remove_outliers_iqr(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower) & (df[column] <= upper)]
#     #keeps only rows within the acceptable range.
#
# print("Before outlier removal:", merged.shape)
# #First, removes extreme stock prices (close column)
# #Then, removes extreme trading volumes (volume column)
# #After this, your dataset only contains realistic and meaningful data.
# merged = remove_outliers_iqr(merged, 'close')
# merged = remove_outliers_iqr(merged, 'volume')
#
# print("After outlier removal:", merged.shape)
#
# # ------------------------------------------------
# # 2.3 ERROR IDENTIFICATION
# # (1) Negative prices or volume
# # ------------------------------------------------
#
# # Replace negative values with NaN then interpolate
# for col in ['open','high','low','close','volume']:
#     merged.loc[merged[col] < 0, col] = np.nan
#     merged[col] = merged[col].interpolate()
#
# print("Any negative values left?")
# print((merged[['open','high','low','close','volume']] < 0).sum())
#
# # ------------------------------------------------
# # 3.1 FEATURE ENGINEERING
# # ------------------------------------------------
#
# # Rolling averages (technical indicators)
# merged['ma_7'] = merged['close'].rolling(7).mean()
# merged['ma_30'] = merged['close'].rolling(30).mean()
# merged['volatility_30'] = merged['close'].rolling(30).std()
#
# # Daily returns
# merged['daily_return'] = merged['close'].pct_change()
#
# # Future close price (for ML prediction)
# merged['future_close_7'] = merged['close'].shift(-7)
#
# merged.head()
#
# # ------------------------------------------------
# # 3.2 DATA NORMALIZATION / STANDARDIZATION
# # ------------------------------------------------
#
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
#
# scaled_cols = ['open','high','low','close','volume',
#                'ma_7','ma_30','volatility_30','daily_return']
#
# merged[scaled_cols] = scaler.fit_transform(merged[scaled_cols])
#
# merged.head()
#
# # ------------------------------------------------
# # 3.3 ENCODING CATEGORICAL VARIABLES
# # ------------------------------------------------
#
# # One-hot encoding for industry, sector, exchange
# cat_features = ['sector','industry','exchange','name']
#
# merged_encoded = pd.get_dummies(merged, columns=cat_features, drop_first=True)
#
# merged_encoded.head()
#
# # ------------------------------------------------
# # 4.1 CONSOLIDATE CLEAN DATASET
# # ------------------------------------------------
#
# clean_df = merged_encoded.copy()
# clean_df.dropna(inplace=True)
#
# print("Final cleaned dataset shape:", clean_df.shape)
# clean_df.head()
#
# # ------------------------------------------------
# # 4.2 TRAIN / VALIDATION / TEST SPLIT
# # Predict future_close_7
# # ------------------------------------------------
#
# from sklearn.model_selection import train_test_split
#
# X = clean_df.drop(['future_close_7'], axis=1)
# y = clean_df['future_close_7']
#
# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.3, shuffle=False
# )
#
# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, shuffle=False
# )
#
# print("Train shape:", X_train.shape)
# print("Validation shape:", X_val.shape)
# print("Test shape:", X_test.shape)
#
# # ------------------------------------------------
# # 4.3 SAVE CLEAN DATA
# # ------------------------------------------------
#
# clean_df.to_csv("clean_stock_data.csv", index=True)
# X_train.to_csv("train_X.csv")
# y_train.to_csv("train_y.csv")
# X_val.to_csv("val_X.csv")
# y_val.to_csv("val_y.csv")
# X_test.to_csv("test_X.csv")
# y_test.to_csv("test_y.csv")
#
# print("Files saved successfully!")

# Step 1 — Imports and setup
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)


INPUT_CSV = "clean_stock_data.csv"

# Choose a chunk size (100k rows per chunk usually safe)
chunk_size = 100_000

# Initialize empty list if you want to store processed chunk results
chunks = []

# Read in chunks
for chunk in pd.read_csv(
    INPUT_CSV,
    parse_dates=["date"],
    low_memory=True,
    chunksize=chunk_size,
    engine="python"
):
    # OPTIONAL: Process the chunk here instead of loading all into memory
    # Example: keep only needed columns
    # chunk = chunk[["col1", "col2", "date"]]

    chunks.append(chunk)

# Concatenate only if needed
df = pd.concat(chunks, ignore_index=True)

print("Loaded successfully with chunking!")
print(df.info())






# # Directories
# INPUT_CSV = "clean_stock_data.csv"
# OUTPUT_DIR = "outputs"
# MODEL_DIR = "models"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Step 2 — Load CSV
# df = pd.read_csv(INPUT_CSV, parse_dates=["date"], low_memory=False)
#
# # Reset index to avoid alignment issues in groupby operations
# df = df.reset_index(drop=True)
#
# print("Loaded data shape:", df.shape)
# print("Columns:", df.columns.tolist())

# Step 3 — Indicator calculation functions

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Compute MACD and MACD signal per ticker."""
    df = df.copy()
    df["MACD"] = df.groupby("ticker")[price_col].transform(lambda s: compute_ema(s, 12) - compute_ema(s, 26))
    df["MACD_signal"] = df.groupby("ticker")["MACD"].transform(lambda s: compute_ema(s, 9))
    return df


def compute_rsi(df: pd.DataFrame, price_col: str = "close", period: int = 14) -> pd.DataFrame:
    """Compute RSI per ticker."""
    df = df.copy()

    def rsi_for_series(close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    df["RSI"] = df.groupby("ticker")[price_col].transform(rsi_for_series)
    return df

# Step 4 — Generate signals

def generate_indicator_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["ticker", "date"])

    # MACD cross detection
    df["MACD_prev"] = df.groupby("ticker")["MACD"].shift(1)
    df["MACD_signal_prev"] = df.groupby("ticker")["MACD_signal"].shift(1)
# THIS IS TO DETECT THE CROSSOVERS:INTYERCETIONS WITH SIGNAL(THERSHOLD)
    def macd_signal_row(row):
        if pd.isna(row["MACD_prev"]) or pd.isna(row["MACD_signal_prev"]):
            return "Neutral"
        if (row["MACD_prev"] < row["MACD_signal_prev"]) and (row["MACD"] > row["MACD_signal"]):
            return "Buy"
        if (row["MACD_prev"] > row["MACD_signal_prev"]) and (row["MACD"] < row["MACD_signal"]):
            return "Sell"
        return "Neutral"

    df["MACD_flag"] = df.apply(macd_signal_row, axis=1)

    # RSI flag
    df["RSI_flag"] = df["RSI"].apply(lambda r: "Buy" if r < 30 else ("Sell" if r > 70 else "Neutral"))

    # Combined signal
    def combined_flag(row):
        if row["MACD_flag"] == "Buy" and row["RSI_flag"] == "Buy":
            return "Buy"
        if row["MACD_flag"] == "Sell" and row["RSI_flag"] == "Sell":
            return "Sell"
        return "Hold"

    df["signal"] = df.apply(combined_flag, axis=1)

    df.drop(columns=["MACD_prev", "MACD_signal_prev"], inplace=True)
    return df

# Step 3 — Compute MACD and -NOT REQUIRED
df = compute_macd(df)
df = compute_rsi(df)

# Step 4 — Generate Buy/Hold/Sell signals
df = generate_indicator_signals(df)

# Quick check
print("Columns after signal generation:", df.columns.tolist())
print(df[["ticker","date","MACD","MACD_signal","RSI","MACD_flag","RSI_flag","signal"]].head())

# Step 5 — Prepare features and labels

def prepare_ml_dataset(df: pd.DataFrame):
    df = df.copy().sort_values(["ticker", "date"])

    # Features to use (numeric columns)
    features = [c for c in ["open","high","low","close","volume","ma_7","ma_30","volatility_30","daily_return","MACD","MACD_signal","RSI"] if c in df.columns]
    df["ticker_id"] = df["ticker"].astype("category").cat.codes
    features.append("ticker_id")
    #ML models cannot work with strings, so we encode categories as numbers.

    df = df.dropna(subset=features + ["signal"]).copy()
    #features is a list of columns used as input for ML (e.g., "open", "close", "MACD", "RSI", "ticker_id").

#"signal" is the target column (Buy/Hold/Sell).

    # Map labels: Buy=1, Hold=0, Sell=2
    df["label"] = df["signal"].map({"Buy":1, "Hold":0, "Sell":2})

    X = df[features].copy()
    y = df["label"].copy()
    return X, y, df

# Step 5 — Prepare features and labels

X, y, df_full = prepare_ml_dataset(df)

print("ML dataset shape:", X.shape)
print("Label distribution:\n", y.value_counts())

# Step 6 — Split data for training/testing

# Prepare ML dataset
X, y, df_full = prepare_ml_dataset(df)
print("ML dataset shape:", X.shape)
print("Label distribution:\n", y.value_counts())

# Time-series safe split
# 70% train, 15% validation, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.17647, shuffle=False)
# 0.17647 * 0.85 ≈ 0.15 to match 15% validation

print("Shapes — Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# Step 7 — Train & evaluate Logistic Regression, Random Forest, SVM- Perform all models one by one and then pas them through
# the pipeline concept

# def train_and_evaluate(X_train, X_test, y_train, y_test):
#     trained_models = {}
#     results = []
#
#     # Models and hyperparameter grids
#     pipelines = {
#         "LogisticRegression": (Pipeline([
#             ("scaler", StandardScaler()),
#             ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
#         ]), {
#             "clf__C": [0.01, 0.1, 1.0]
#         }),
#         "RandomForest": (Pipeline([
#             ("scaler", StandardScaler()),
#             ("clf", RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42))
#         ]), {
#             "clf__n_estimators": [50, 100],
#             "clf__max_depth": [5, 10, None]
#         }),
#         "SVM": (Pipeline([
#             ("scaler", StandardScaler()),
#             ("clf", SVC(class_weight="balanced", probability=True))
#         ]), {
#             "clf__C": [0.1, 1],
#             "clf__kernel": ["rbf"]
#         })
#     }
#
#     skf = StratifiedKFold(n_splits=3, shuffle=False)
#
#     for name, (pipe, params) in pipelines.items():
#         print(f"\nTraining {name}...")
#         grid = GridSearchCV(pipe, params, cv=skf, scoring="f1_weighted", n_jobs=-1)
#         grid.fit(X_train, y_train)
#         best_model = grid.best_estimator_
#         trained_models[name] = best_model
#
#         y_pred = best_model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
#         rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
#         f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
#
#         print(f"{name} — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#         print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#
#         results.append({
#             "model": name,
#             "best_params": grid.best_params_,
#             "accuracy": acc,
#             "precision": prec,
#             "recall": rec,
#             "f1_weighted": f1
#         })
#
#     return trained_models, results
#

def train_and_evaluate(X_train, X_test, y_train, y_test):
    trained_models = {}
    results = []

    # Simplified models and hyperparameters
    pipelines = {
        "LogisticRegression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
            ]),
            {
                "clf__C": [0.1, 1.0]  # reduced grid
            }
        ),
        "RandomForest": (
            Pipeline([
                ("clf", RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=1,  # <- prevent RAM explosion
                    random_state=42
                ))
            ]),
            {
                "clf__n_estimators": [50],
                "clf__max_depth": [10, None]  # reduced grid
            }
        )
    }

    # Reduce folds to avoid huge memory usage
    skf = StratifiedKFold(n_splits=2, shuffle=False)

    for name, (pipe, params) in pipelines.items():
        print(f"\nTraining {name}...")

        # Only 1 job to avoid RAM explosion
        grid = GridSearchCV(
            pipe,
            params,
            cv=skf,
            scoring="f1_weighted",
            n_jobs=1,
            verbose=1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        trained_models[name] = best_model

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(f"{name} — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        results.append({
            "model": name,
            "best_params": grid.best_params_,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_weighted": f1
        })

    return trained_models, results

# Step 8 — Main workflow: train models, save models & predictions
trained_models, results = train_and_evaluate(X_train, X_test, y_train, y_test)

# Save models and predictions
for model_name, model in trained_models.items():
    model_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {model_name} to {model_path}")

    # Save predictions
    df_pred = df_full.loc[X.index, ["date", "ticker", "close", "signal"]].copy()
    df_pred[f"pred_{model_name}"] = model.predict(X)
    df_pred.to_csv(os.path.join(OUTPUT_DIR, f"predictions_{model_name}.csv"), index=False)
    print(f"Saved predictions for {model_name}")


# Step 9 — Evaluation summary
summary_df = pd.DataFrame(results)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_summary.csv"), index=False)
print("Saved evaluation summary to:", os.path.join(OUTPUT_DIR, "evaluation_summary.csv"))