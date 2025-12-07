import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, roc_auc_score,
                             precision_recall_curve, auc)


# -------------------------
# Reproducibility settings
# -------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# -------------------------
# 1) LOAD DATA
# -------------------------
df = pd.read_csv('titanic.csv')
print(f"Loaded data: {df.shape} rows, columns: {list(df.columns)}")

# -------------------------
# 2) EXPLORE DATA (brief)
# -------------------------
print("\nMissing values per column:\n", df.isnull().sum())
print("\nSurvived distribution:\n", df['Survived'].value_counts(normalize=True))

# Quick EDA plots saved
os.makedirs("figures", exist_ok=True)
plt.figure()
df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title("Survival Distribution")
plt.xticks([0,1], ['Died','Survived'])
plt.show()
plt.savefig("figures/survival_distribution_ass11.png", dpi=200)
plt.close()

# -------------------------
# 3) PREPROCESSING
# -------------------------
data = df.copy()

# Fill Age using median grouped by Pclass & Sex
data['Age'] = data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Fill Embarked with mode
if data['Embarked'].isnull().sum() > 0:
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Has_Cabin binary feature
data['Has_Cabin'] = data['Cabin'].notna().astype(int)

# Fill Fare with median by Pclass
data['Fare'] = data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'))

# Feature engineering
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
    'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
    'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
}
data['Title'] = data['Title'].map(title_mapping).fillna('Rare')

# AgeGroup and FareBin
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
data['FareBin'] = pd.qcut(data['Fare'].rank(method='first'), 4, labels=['Low', 'Medium', 'High', 'VeryHigh'])

# Encoding categorical features
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
le = LabelEncoder()
for col in ['Embarked', 'Title', 'AgeGroup', 'FareBin']:
    data[col] = le.fit_transform(data[col].astype(str))

# Drop unused columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data.drop(columns=[c for c in drop_cols if c in data.columns], inplace=True)

# Final feature matrix and label
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked', 'FamilySize', 'IsAlone', 'Has_Cabin',
                'Title', 'AgeGroup', 'FareBin']
X = data[feature_cols]
y = data['Survived']
print(f"\nFinal features: {X.shape[1]} columns. Example rows:\n", X.head())

# Save preprocessed snapshot
data.to_csv("titanic_preprocessed.csv", index=False)

# -------------------------
# 4) TRAIN/TEST SPLIT & SCALING
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Save scaler
joblib.dump(scaler, "scaler.joblib")

# -------------------------
# 5) MODEL SELECTION & TRAINING
# -------------------------
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'SVM': SVC(random_state=RANDOM_STATE, probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\nTraining {name}...")
    # use scaled inputs for linear models and distance-based ones
    if name in ['SVM', 'KNN', 'Logistic Regression']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # cross-val scores on training set
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    # ROC AUC (if probability available or decision_function)
    try:
        if y_proba is not None:
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            # fallback: use decision_function if present
            if hasattr(model, 'decision_function'):
                df_dec = model.decision_function(X_te)
                roc_auc = roc_auc_score(y_test, df_dec)
            else:
                roc_auc = np.nan
    except Exception:
        roc_auc = np.nan

    results.append({
        'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
        'CV_Mean': cv_mean, 'CV_Std': cv_std, 'ROC_AUC': roc_auc
    })

    print(f"{name} -> Acc: {acc:.4f}, F1: {f1:.4f}, ROC_AUC: {roc_auc if not np.isnan(roc_auc) else 'N/A'}")
    # save model
    joblib.dump(model, f"model_{name.replace(' ', '_')}.joblib")

# Save results
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
results_df.to_csv("model_results_summary.csv", index=False)
print("\nModel comparison:\n", results_df)

# -------------------------
# 6) HYPERPARAMETER TUNING (Random Forest example)
# -------------------------
print("\nHyperparameter tuning for Random Forest with GridSearchCV")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=RANDOM_STATE)
grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0)
# Use unscaled data for tree-based RF
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
print("Best RF params:", grid.best_params_)
joblib.dump(best_rf, "best_random_forest.joblib")

# Evaluate tuned RF on test
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
print("Tuned RF test accuracy:", accuracy_score(y_test, y_pred_rf))
print("Tuned RF test ROC AUC:", roc_auc_score(y_test, y_proba_rf))

# append tuned RF to results
new_row = {
    "Model": "Decision Tree",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1 Score": f1
}
results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)


# -------------------------
# 7) DETAILED EVALUATION FOR BEST MODEL
# -------------------------
best_row = results_df.sort_values('Accuracy', ascending=False).iloc[0]
best_name = best_row['Model']
print("\nBest model by accuracy:", best_name)

# Load best model file (if we saved), else use object
best_model = None
if best_name == 'RandomForest_Tuned':
    best_model = best_rf
else:
    model_file = f"model_{best_name.replace(' ', '_')}.joblib"
    if os.path.exists(model_file):
        best_model = joblib.load(model_file)
    else:
        # fallback: pick from models dict
        best_model = models.get(best_name.split('_Tuned')[0], best_rf)

# Select correct X_test usage (scaled or not)
if best_name.split('_')[0] in ['SVM', 'KNN', 'Logistic']:
    X_test_for_eval = X_test_scaled
    X_train_for_eval = X_train_scaled
else:
    X_test_for_eval = X_test
    X_train_for_eval = X_train

y_pred_best = best_model.predict(X_test_for_eval)
y_proba_best = best_model.predict_proba(X_test_for_eval)[:, 1] if hasattr(best_model, 'predict_proba') else None

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.savefig("figures/confusion_matrix_best_ass11.png", dpi=200)
plt.close()

# ROC and PR curves
plt.figure(figsize=(10,4))
if y_proba_best is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    roc_auc = roc_auc_score(y_test, y_proba_best)
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend()
    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba_best)
    pr_auc = auc(recall, precision)
    plt.subplot(1,2,2)
    plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    plt.show()
    plt.savefig("figures/roc_pr_best_ass11.png", dpi=200)
    plt.close()
else:
    print("Best model does not provide probabilities; ROC/PR not available.")

# Feature importance if available
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_}).sort_values('Importance', ascending=False)
    fi.to_csv("feature_importances.csv", index=False)
    plt.figure(figsize=(8,6))
    plt.barh(fi['Feature'][:10], fi['Importance'][:10])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()
    plt.savefig("figures/feature_importances_ass11.png", dpi=200)
    plt.close()
    print("\nTop features:\n", fi.head(10))

# -------------------------
# 8) SAVE RESULTS & REPORT SNIPPETS
# -------------------------
# Save models summary
results_df.to_csv("final_model_results.csv", index=False)

# Create a short auto-report text file (can be used in main report)
with open("auto_report_summary.txt", "w") as f:
    f.write("Titanic ML Pipeline - Auto Summary\n")
    f.write("Best model: " + str(best_name) + "\n")
    f.write(results_df.to_string(index=False))
print("\nSaved results and artifacts in working directory.")


print("Plots are in ./figures/")

