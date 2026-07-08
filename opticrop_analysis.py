"""
OptiCrop: Smart Agricultural Production Optimization Engine
Data Analysis, Preprocessing, and Model Building Script

This script performs:
- Epic 2: Data Collection and Analysis (EDA)
- Epic 3: Data Pre-Processing
- Epic 4: Model Building (K-Means, Logistic Regression)
"""

# ============================================================
# STEP 1: Import Required Libraries
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import pickle
import warnings
import os
import sys

# Fix Windows console encoding for Unicode characters
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("  OptiCrop - Smart Agricultural Production Optimization")
print("=" * 60)
print("\n✅ All libraries imported successfully!\n")

# ============================================================
# STEP 2: Read and Explore the Dataset
# ============================================================
print("=" * 60)
print("  STEP 2: Reading the Dataset")
print("=" * 60)

df = pd.read_csv('Crop_recommendation.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Total Samples: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")

print("\n--- First 5 Records ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print(f"\n--- Unique Crops ({df['label'].nunique()}) ---")
print(df['label'].unique())

print(f"\n--- Samples per Crop ---")
print(df['label'].value_counts())

# ============================================================
# STEP 3: Univariate Analysis
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: Univariate Analysis")
print("=" * 60)

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Distribution of Agricultural Conditions', fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette('husl', 7)[i])
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

axes[7].axis('off')
plt.tight_layout()
plt.savefig('plots/univariate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Univariate analysis plots saved to plots/univariate_analysis.png")

# Crop count plot
fig, ax = plt.subplots(figsize=(14, 6))
sns.countplot(data=df, y='label', order=df['label'].value_counts().index, palette='viridis', ax=ax)
ax.set_title('Crop Distribution in Dataset', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
ax.set_ylabel('Crop')
plt.tight_layout()
plt.savefig('plots/crop_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Crop distribution plot saved")

# ============================================================
# STEP 4: Bivariate Analysis
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: Bivariate Analysis")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 8))
crop_list = df['label'].unique()
colors = sns.color_palette('husl', len(crop_list))
for i, crop in enumerate(crop_list):
    crop_data = df[df['label'] == crop]
    ax.scatter(crop_data['humidity'], crop_data['temperature'],
               label=crop, alpha=0.6, s=30, color=colors[i])
ax.set_title('Humidity vs Temperature (by Crop)', fontsize=14, fontweight='bold')
ax.set_xlabel('Humidity (%)')
ax.set_ylabel('Temperature (°C)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('plots/bivariate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Bivariate analysis plot saved")

# ============================================================
# STEP 5: Multivariate Analysis
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: Multivariate Analysis")
print("=" * 60)

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
correlation = df[features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap of Agricultural Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/multivariate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Multivariate analysis (correlation heatmap) saved")

# ============================================================
# STEP 6: Check for Null Values
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: Checking for Null Values")
print("=" * 60)

print("\n--- Null Value Count ---")
print(df.isnull().sum())
print(f"\nTotal null values: {df.isnull().sum().sum()}")
print(f"Dataset shape: {df.shape}")

# ============================================================
# STEP 7: Handling Outliers
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: Handling Outliers")
print("=" * 60)

# Boxplots before treatment
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Boxplots - Outlier Detection', fontsize=16, fontweight='bold')
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(data=df, y=col, ax=axes[i], color=sns.color_palette('pastel')[i])
    axes[i].set_title(f'{col}')
axes[7].axis('off')
plt.tight_layout()
plt.savefig('plots/outliers_before.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Outlier detection boxplots saved")

# Calculate IQR for Potassium
Q1 = df['K'].quantile(0.25)
Q3 = df['K'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"\nPotassium (K) - IQR Analysis:")
print(f"  Q1: {Q1:.2f}")
print(f"  Q3: {Q3:.2f}")
print(f"  IQR: {IQR:.2f}")
print(f"  Lower Bound: {lower_bound:.2f}")
print(f"  Upper Bound: {upper_bound:.2f}")

outliers_k = df[(df['K'] < lower_bound) | (df['K'] > upper_bound)]
print(f"  Outliers found in K: {len(outliers_k)}")

# Log transformation on K to handle outliers
df['K_log'] = np.log1p(df['K'])
print("✅ Log transformation applied to Potassium (K)")

# Boxplots after treatment
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Potassium (K) - Before vs After Log Transformation', fontsize=14, fontweight='bold')
sns.boxplot(data=df, y='K', ax=axes[0], color='salmon')
axes[0].set_title('Before (Original K)')
sns.boxplot(data=df, y='K_log', ax=axes[1], color='lightgreen')
axes[1].set_title('After (Log Transformed K)')
plt.tight_layout()
plt.savefig('plots/outliers_after.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Outlier treatment comparison plots saved")

# ============================================================
# STEP 8: Extracting Seasonal Crops
# ============================================================
print("\n" + "=" * 60)
print("  STEP 8: Extracting Seasonal Crops")
print("=" * 60)

# Group crops by seasonal conditions
crop_summary = df.groupby('label')[['temperature', 'humidity', 'rainfall']].mean()

summer_crops = crop_summary[crop_summary['temperature'] > 28].index.tolist()
winter_crops = crop_summary[crop_summary['temperature'] < 22].index.tolist()
rainy_crops = crop_summary[crop_summary['rainfall'] > 150].index.tolist()
all_season = [c for c in crop_summary.index if c not in summer_crops + winter_crops]

print(f"\n🌞 Summer Crops (Temp > 28°C): {summer_crops}")
print(f"❄️  Winter Crops (Temp < 22°C): {winter_crops}")
print(f"🌧️  Rainy Season Crops (Rainfall > 150mm): {rainy_crops}")

# Seasonal crop visualization
fig, ax = plt.subplots(figsize=(14, 6))
crop_summary_sorted = crop_summary.sort_values('temperature')
colors = ['#3498db' if c in winter_crops else '#e74c3c' if c in summer_crops else '#2ecc71'
          for c in crop_summary_sorted.index]
bars = ax.bar(range(len(crop_summary_sorted)), crop_summary_sorted['temperature'], color=colors)
ax.set_xticks(range(len(crop_summary_sorted)))
ax.set_xticklabels(crop_summary_sorted.index, rotation=45, ha='right')
ax.set_title('Average Temperature by Crop (Seasonal Classification)', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Temperature (°C)')
ax.axhline(y=28, color='red', linestyle='--', alpha=0.5, label='Summer threshold (28°C)')
ax.axhline(y=22, color='blue', linestyle='--', alpha=0.5, label='Winter threshold (22°C)')
ax.legend()
plt.tight_layout()
plt.savefig('plots/seasonal_crops.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Seasonal crop analysis plot saved")

# ============================================================
# STEP 9: Splitting Data into Train and Test Sets
# ============================================================
print("\n" + "=" * 60)
print("  STEP 9: Train-Test Split")
print("=" * 60)

X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nFeature columns: {features}")
print(f"Target column: label")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
print(f"Test size: 20%")
print(f"Random state: 42")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Feature scaling applied using StandardScaler")

# ============================================================
# STEP 10: K-Means Clustering
# ============================================================
print("\n" + "=" * 60)
print("  STEP 10: K-Means Clustering")
print("=" * 60)

# Elbow Method
wcss = []
K_range = range(1, 15)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)

# Elbow Graph
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
ax.set_title('Elbow Method - Optimal Number of Clusters', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/elbow_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Elbow graph saved")

# Train K-Means with optimal clusters
n_clusters = df['label'].nunique()
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans_model.fit_predict(X_train_scaled)
print(f"\nK-Means trained with {n_clusters} clusters")
print(f"Cluster labels: {np.unique(clusters)}")

# K-Means cluster visualization
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                     c=clusters, cmap='tab20', alpha=0.5, s=20)
ax.set_title('K-Means Clustering Results', fontsize=14, fontweight='bold')
ax.set_xlabel('Feature 1 (Nitrogen - Scaled)')
ax.set_ylabel('Feature 2 (Phosphorous - Scaled)')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('plots/kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ K-Means cluster visualization saved")

# ============================================================
# STEP 11: Logistic Regression
# ============================================================
print("\n" + "=" * 60)
print("  STEP 11: Logistic Regression")
print("=" * 60)

lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("✅ Logistic Regression model trained successfully")

y_pred = lr_model.predict(X_test_scaled)
print("✅ Predictions generated on test set")

# ============================================================
# STEP 12: Evaluating Model Performance
# ============================================================
print("\n" + "=" * 60)
print("  STEP 12: Model Evaluation")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 Model Performance Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(16, 14))
cm = confusion_matrix(y_test, y_pred, labels=lr_model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=lr_model.classes_, yticklabels=lr_model.classes_, ax=ax)
ax.set_title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Crop')
ax.set_ylabel('Actual Crop')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Confusion matrix saved")

# ============================================================
# STEP 13: Save the Best Model
# ============================================================
print("\n" + "=" * 60)
print("  STEP 13: Saving the Best Model")
print("=" * 60)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✅ Model saved as model.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved as scaler.pkl")

# ============================================================
# STEP 14: Predict the Best Crop (Demo)
# ============================================================
print("\n" + "=" * 60)
print("  STEP 14: Sample Prediction Demo")
print("=" * 60)

# Sample input: N=90, P=42, K=43, temp=20.8, humidity=82, pH=6.5, rainfall=202
sample_input = np.array([[90, 42, 43, 20.8, 82.0, 6.5, 202.96]])
sample_scaled = scaler.transform(sample_input)
prediction = lr_model.predict(sample_scaled)
print(f"\n🌾 Sample Input:")
print(f"  Nitrogen (N):    90")
print(f"  Phosphorous (P): 42")
print(f"  Potassium (K):   43")
print(f"  Temperature:     20.8°C")
print(f"  Humidity:        82.0%")
print(f"  pH Level:        6.5")
print(f"  Rainfall:        202.96 mm")
print(f"\n🎯 Predicted Crop: {prediction[0]}")

print("\n" + "=" * 60)
print("  ✅ Analysis Complete! All outputs saved.")
print("=" * 60)
print(f"\nFiles generated:")
print(f"  📁 model.pkl    - Trained Logistic Regression model")
print(f"  📁 scaler.pkl   - Feature scaler")
print(f"  📁 plots/       - All visualization plots")
