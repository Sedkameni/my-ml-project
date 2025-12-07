import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# 1️  Extract and Load Data
zip_path = "wine-data.zip"
extract_folder = "wine_data"

if not os.path.exists(extract_folder):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"Files extracted to: {extract_folder}")

print(" Extracted files:", os.listdir(extract_folder))

# Find dataset file (either .data or .csv)
for f in os.listdir(extract_folder):
    if f.endswith(".data") or f.endswith(".csv") or f.endswith(".txt"):
        file_name = os.path.join(extract_folder, f)
        break

# Define column names (UCI Wine dataset)
columns = [
    'Class', 'Alcohol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash',
    'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols',
    'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280_OD315', 'Proline'
]

# Load dataset
df = pd.read_csv(file_name, header=None, names=columns)
print("\n Wine Dataset Loaded Successfully!")
print(df.head())


# 2️  Exploratory Data Analysis
print("\n=== Exploratory Data Analysis ===")
print(df.describe())
print("\nMissing values per column:\n", df.isnull().sum())

plt.figure(figsize=(10, 5))
sns.countplot(x="Class", data=df, palette="viridis")
plt.title("Wine Class Distribution", fontsize=14, fontweight="bold")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 3️  Data Preprocessing
X = df.iloc[:, 1:].values  # all features
y = df.iloc[:, 0].values  # class labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n Data Standardized!")
print(f"Mean after scaling: {np.mean(X_scaled):.3f}, Std: {np.std(X_scaled):.3f}")


# 4️  Hierarchical Clustering
print("\n=== Hierarchical Clustering ===")
linkage_methods = ['ward', 'complete', 'average']

for method in linkage_methods:
    print(f"\n--- Method: {method.upper()} ---")
    Z = linkage(X_scaled, method=method)

    # Dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode='lastp', p=30)
    plt.title(f"Dendrogram - {method.capitalize()} Linkage")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.axhline(y=10, color='r', linestyle='--')
    plt.show()

    # Create 3 clusters
    clusters = fcluster(Z, t=3, criterion='maxclust')

    # Evaluation
    silhouette = silhouette_score(X_scaled, clusters)
    db_index = davies_bouldin_score(X_scaled, clusters)
    calinski = calinski_harabasz_score(X_scaled, clusters)
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Calinski-Harabasz Score: {calinski:.2f}")

# PCA visualization for Ward linkage
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
clusters = fcluster(linkage(X_scaled, method='ward'), 3, criterion='maxclust')

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=50)
plt.title("Ward Linkage Clusters (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# 5️  DBSCAN Clustering
print("\n=== DBSCAN Clustering ===")
eps_values = [0.5, 1.0, 1.5]

for eps in eps_values:
    print(f"\n--- eps={eps} ---")
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"Clusters found: {n_clusters}, Noise points: {n_noise}")

    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="plasma", s=50)
    plt.title(f"DBSCAN Clustering (eps={eps})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
