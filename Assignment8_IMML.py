"""
ANALYSE DE LA QUALITÉ DU VIN - Régression Linéaire Multiple
Dataset: Propriétés chimiques du vin rouge
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("STEP 1: EXPLORATION DU DATASET")
print("="*80)

# Charger les données
df = pd.read_excel('Sheet2_Dataset.xlsx')

print("\n1.1 - Aperçu des premières lignes:")
print(df.head(10))

print("\n1.2 - Informations sur le dataset:")
print(f"Nombre de lignes: {df.shape[0]}")
print(f"Nombre de colonnes: {df.shape[1]}")
print(f"\nNoms des colonnes:")
print(df.columns.tolist())

print("\n1.3 - Vérification des valeurs manquantes:")
missing_values = df.isnull().sum()
print(missing_values)
print(f"\nTotal de valeurs manquantes: {missing_values.sum()}")

print("\n1.4 - Types de données:")
print(df.dtypes)

print("\n1.5 - Statistiques descriptives:")
print(df.describe().round(3))

print("\n1.6 - Distribution de la variable cible (Quality):")
print(df['quality'].value_counts().sort_index())
print(f"\nQualité moyenne: {df['quality'].mean():.2f}")
print(f"Qualité médiane: {df['quality'].median():.0f}")

# VISUALISATIONS

# 1.7 - Distribution de la qualité
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogramme
axes[0].hist(df['quality'], bins=6, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Quality Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Distribution de la Qualité du Vin', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Boxplot
axes[1].boxplot(df['quality'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
axes[1].set_ylabel('Quality Score', fontsize=12, fontweight='bold')
axes[1].set_title('Boxplot de la Qualité', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('1_distribution_quality.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique sauvegardé: 1_distribution_quality.png")
plt.close()

# 1.8 - Matrice de corrélation
print("\n1.8 - Matrice de corrélation avec Quality:")
correlation_matrix = df.corr()
quality_corr = correlation_matrix['quality'].sort_values(ascending=False)
print(quality_corr.round(3))

# Heatmap de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation des Propriétés Chimiques',
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: 2_correlation_heatmap.png")
plt.close()

# 1.9 - Pairplot des variables les plus corrélées avec quality
top_features = quality_corr.abs().nlargest(6).index.tolist()
top_features.remove('quality')
top_features.append('quality')

print(f"\n1.9 - Top 5 variables corrélées avec quality: {top_features[:-1]}")

# Note: pairplot peut être lent avec beaucoup de données
# On peut le commenter si nécessaire
# sns.pairplot(df[top_features], hue='quality', palette='viridis', plot_kws={'alpha':0.6})
# plt.savefig('3_pairplot_top_features.png', dpi=300, bbox_inches='tight')
# print("✓ Graphique sauvegardé: 3_pairplot_top_features.png")
# plt.close()

# Boxplots des top features par qualité
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features[:-1]):
    df.boxplot(column=feature, by='quality', ax=axes[idx])
    axes[idx].set_title(f'{feature} vs Quality', fontweight='bold')
    axes[idx].set_xlabel('Quality Score', fontweight='bold')
    axes[idx].set_ylabel(feature, fontweight='bold')
    plt.sca(axes[idx])
    plt.xticks(rotation=0)

# Supprimer le subplot vide
axes[-1].axis('off')

plt.suptitle('Distribution des Top Features par Niveau de Qualité',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('3_boxplots_by_quality.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: 3_boxplots_by_quality.png")
plt.close()


print("\n" + "="*80)
print("STEP 2: PRÉTRAITEMENT DES DONNÉES")
print("="*80)

# 2.1 - Séparer features (X) et target (y)
X = df.drop('quality', axis=1)
y = df['quality']

print(f"\n2.1 - Dimensions:")
print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")

# 2.2 - Vérifier qu'il n'y a pas de valeurs manquantes
print(f"\n2.2 - Valeurs manquantes dans X: {X.isnull().sum().sum()}")
print(f"Valeurs manquantes dans y: {y.isnull().sum()}")

# 2.3 - Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n2.3 - Découpage des données (80/20):")
print(f"Training set: {X_train.shape[0]} échantillons")
print(f"Testing set: {X_test.shape[0]} échantillons")

# 2.4 - Standardisation (optionnel mais recommandé)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n2.4 - Standardisation appliquée")
print(f"Moyenne des features (avant): {X_train.mean().mean():.3f}")
print(f"Moyenne des features (après): {X_train_scaled.mean():.3f}")
print(f"Écart-type des features (avant): {X_train.std().mean():.3f}")
print(f"Écart-type des features (après): {X_train_scaled.std():.3f}")


print("\n" + "="*80)
print("STEP 3: CONSTRUCTION DU MODÈLE DE RÉGRESSION")
print("="*80)

# 3.1 - Modèle avec données non standardisées
model_original = LinearRegression()
model_original.fit(X_train, y_train)

print("\n3.1 - Modèle de Régression Linéaire Multiple (données originales)")
print(f"Équation: quality = β₀ + β₁*fixed_acidity + β₂*volatile_acidity + ... + β₁₁*alcohol")
print(f"\nIntercept (β₀): {model_original.intercept_:.4f}")

# 3.2 - Modèle avec données standardisées
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

print("\n3.2 - Modèle avec données standardisées")
print(f"Intercept (β₀): {model_scaled.intercept_:.4f}")

# 3.3 - Afficher les coefficients
coefficients_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient (Original)': model_original.coef_,
    'Coefficient (Scaled)': model_scaled.coef_,
    'Abs Coefficient (Scaled)': np.abs(model_scaled.coef_)
}).sort_values('Abs Coefficient (Scaled)', ascending=False)

print("\n3.3 - Coefficients du modèle:")
print(coefficients_df.to_string(index=False))


print("\n" + "="*80)
print("STEP 4: ÉVALUATION DU MODÈLE")
print("="*80)

# 4.1 - Prédictions
y_train_pred_original = model_original.predict(X_train)
y_test_pred_original = model_original.predict(X_test)

y_train_pred_scaled = model_scaled.predict(X_train_scaled)
y_test_pred_scaled = model_scaled.predict(X_test_scaled)

# 4.2 - Métriques pour le modèle original
print("\n4.1 - MÉTRIQUES - Modèle Original (données non standardisées)")
print("-" * 60)
print("TRAINING SET:")
r2_train_orig = r2_score(y_train, y_train_pred_original)
mae_train_orig = mean_absolute_error(y_train, y_train_pred_original)
mse_train_orig = mean_squared_error(y_train, y_train_pred_original)
rmse_train_orig = np.sqrt(mse_train_orig)

print(f"  R² Score: {r2_train_orig:.4f} ({r2_train_orig*100:.2f}% de variance expliquée)")
print(f"  MAE (Mean Absolute Error): {mae_train_orig:.4f}")
print(f"  MSE (Mean Squared Error): {mse_train_orig:.4f}")
print(f"  RMSE (Root Mean Squared Error): {rmse_train_orig:.4f}")

print("\nTESTING SET:")
r2_test_orig = r2_score(y_test, y_test_pred_original)
mae_test_orig = mean_absolute_error(y_test, y_test_pred_original)
mse_test_orig = mean_squared_error(y_test, y_test_pred_original)
rmse_test_orig = np.sqrt(mse_test_orig)

print(f"  R² Score: {r2_test_orig:.4f} ({r2_test_orig*100:.2f}% de variance expliquée)")
print(f"  MAE (Mean Absolute Error): {mae_test_orig:.4f}")
print(f"  MSE (Mean Squared Error): {mse_test_orig:.4f}")
print(f"  RMSE (Root Mean Squared Error): {rmse_test_orig:.4f}")

# 4.3 - Métriques pour le modèle standardisé
print("\n4.2 - MÉTRIQUES - Modèle Standardisé")
print("-" * 60)
print("TRAINING SET:")
r2_train_scaled = r2_score(y_train, y_train_pred_scaled)
mae_train_scaled = mean_absolute_error(y_train, y_train_pred_scaled)
mse_train_scaled = mean_squared_error(y_train, y_train_pred_scaled)
rmse_train_scaled = np.sqrt(mse_train_scaled)

print(f"  R² Score: {r2_train_scaled:.4f} ({r2_train_scaled*100:.2f}% de variance expliquée)")
print(f"  MAE: {mae_train_scaled:.4f}")
print(f"  MSE: {mse_train_scaled:.4f}")
print(f"  RMSE: {rmse_train_scaled:.4f}")

print("\nTESTING SET:")
r2_test_scaled = r2_score(y_test, y_test_pred_scaled)
mae_test_scaled = mean_absolute_error(y_test, y_test_pred_scaled)
mse_test_scaled = mean_squared_error(y_test, y_test_pred_scaled)
rmse_test_scaled = np.sqrt(mse_test_scaled)

print(f"  R² Score: {r2_test_scaled:.4f} ({r2_test_scaled*100:.2f}% de variance expliquée)")
print(f"  MAE: {mae_test_scaled:.4f}")
print(f"  MSE: {mse_test_scaled:.4f}")
print(f"  RMSE: {rmse_test_scaled:.4f}")

# 4.4 - Analyse des résidus
residuals_train = y_train - y_train_pred_original
residuals_test = y_test - y_test_pred_original

# Graphiques de résidus
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

# 3. Distribution des résidus (Training)
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

plt.suptitle('Analyse des Résidus', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('4_residual_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique sauvegardé: 4_residual_analysis.png")
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
plt.savefig('5_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: 5_actual_vs_predicted.png")
plt.close()


print("\n" + "="*80)
print("STEP 5: INTERPRÉTATION DES RÉSULTATS")
print("="*80)

# 5.1 - Visualisation des coefficients
fig, ax = plt.subplots(figsize=(10, 8))

# Trier par valeur absolue
coef_sorted = coefficients_df.sort_values('Coefficient (Scaled)', ascending=True)

colors = ['red' if x < 0 else 'green' for x in coef_sorted['Coefficient (Scaled)']]
ax.barh(coef_sorted['Feature'], coef_sorted['Coefficient (Scaled)'], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Coefficient (Standardized)', fontweight='bold', fontsize=12)
ax.set_ylabel('Chemical Property', fontweight='bold', fontsize=12)
ax.set_title('Impact des Propriétés Chimiques sur la Qualité du Vin\n(Régression Linéaire Multiple)',
             fontweight='bold', fontsize=14, pad=20)
ax.grid(axis='x', alpha=0.3)

# Ajouter les valeurs
for i, v in enumerate(coef_sorted['Coefficient (Scaled)']):
    ax.text(v, i, f' {v:.3f}', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('6_coefficients_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique sauvegardé: 6_coefficients_visualization.png")
plt.close()

# 5.2 - Identification des facteurs les plus influents
print("\n5.1 - Top 5 facteurs POSITIFS (augmentent la qualité):")
positive_coefs = coefficients_df[coefficients_df['Coefficient (Scaled)'] > 0].nlargest(5, 'Coefficient (Scaled)')
for idx, row in positive_coefs.iterrows():
    print(f"  • {row['Feature']:25s}: +{row['Coefficient (Scaled)']:.4f}")

print("\n5.2 - Top 5 facteurs NÉGATIFS (diminuent la qualité):")
negative_coefs = coefficients_df[coefficients_df['Coefficient (Scaled)'] < 0].nsmallest(5, 'Coefficient (Scaled)')
for idx, row in negative_coefs.iterrows():
    print(f"  • {row['Feature']:25s}: {row['Coefficient (Scaled)']:.4f}")

print("\n5.3 - Importance relative (valeur absolue):")
print(coefficients_df[['Feature', 'Abs Coefficient (Scaled)']].head(5).to_string(index=False))


print("\n" + "="*80)
print("STEP 6: RECOMMANDATIONS AUX VIGNERONS")
print("="*80)

print("\n📊 RÉSUMÉ DE L'ANALYSE:")
print("-" * 60)
print(f"✓ Modèle: Régression Linéaire Multiple")
print(f"✓ Nombre de features: {X.shape[1]}")
print(f"✓ Nombre d'échantillons: {len(df)}")
print(f"✓ R² Score (Test): {r2_test_orig:.4f} → {r2_test_orig*100:.2f}% de variance expliquée")
print(f"✓ Erreur moyenne (MAE): {mae_test_orig:.4f} points sur l'échelle de qualité")

print("\n🍷 RECOMMANDATIONS STRATÉGIQUES:")
print("-" * 60)

# Analyser les coefficients pour donner des recommandations
top_positive = coefficients_df.nlargest(3, 'Coefficient (Scaled)')
top_negative = coefficients_df.nsmallest(3, 'Coefficient (Scaled)')

print("\n1. PROPRIÉTÉS À AUGMENTER (impact positif sur la qualité):")
for idx, row in top_positive.iterrows():
    feature = row['Feature']
    coef = row['Coefficient (Scaled)']
    if coef > 0:
        print(f"\n   ✓ {feature.upper()}")
        print(f"     - Impact: +{coef:.4f} (coefficient standardisé)")
        print(f"     - Recommandation: Optimiser cette propriété pour améliorer la qualité")

print("\n2. PROPRIÉTÉS À CONTRÔLER/RÉDUIRE (impact négatif sur la qualité):")
for idx, row in top_negative.iterrows():
    feature = row['Feature']
    coef = row['Coefficient (Scaled)']
    if coef < 0:
        print(f"\n   ⚠ {feature.upper()}")
        print(f"     - Impact: {coef:.4f} (coefficient standardisé)")
        print(f"     - Recommandation: Maintenir à des niveaux bas pour préserver la qualité")

print("\n3. ACTIONS CONCRÈTES:")
print("""
   • Surveiller étroitement les propriétés à fort impact (top 3-5)
   • Établir des seuils de contrôle qualité basés sur les coefficients
   • Ajuster les processus de fermentation pour optimiser les propriétés clés
   • Utiliser ce modèle pour prédire la qualité avant embouteillage
   • Investir dans l'équipement pour mesurer précisément les top facteurs
""")

print("\n4. LIMITES DU MODÈLE:")
print(f"""
   • R² = {r2_test_orig:.4f} signifie que {(1-r2_test_orig)*100:.1f}% de la variance n'est pas expliquée
   • D'autres facteurs non mesurés influencent la qualité (terroir, climat, âge, etc.)
   • Le modèle suppose des relations linéaires (peut ne pas capturer toutes les interactions)
   • Erreur moyenne de ±{mae_test_orig:.2f} points sur l'échelle de qualité
""")

print("\n" + "="*80)
print("ANALYSE COMPLÈTE TERMINÉE!")
print("="*80)
print("\nFichiers générés:")
print("  1. 1_distribution_quality.png")
print("  2. 2_correlation_heatmap.png")
print("  3. 3_boxplots_by_quality.png")
print("  4. 4_residual_analysis.png")
print("  5. 5_actual_vs_predicted.png")
print("  6. 6_coefficients_visualization.png")
print("\n✓ Tous les graphiques sont sauvegardés et prêts pour votre rapport!")