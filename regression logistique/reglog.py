import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from skimage.feature import hog
from skimage import exposure
import joblib
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Connexion à Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Tracking du temps total
total_start_time = time.time()

# Configuration des chemins
base_path = '/content/drive/MyDrive/chest_Xray'
test_dir = os.path.join(base_path, 'test')

def collect_test_images():
    """Collecte toutes les images du dossier test"""
    normal_dir = os.path.join(test_dir, 'NORMAL')
    pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
    
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(normal_dir) else []
    
    pneumonia_images = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(pneumonia_dir) else []
    
    return normal_images, pneumonia_images

# Collecte des images du test
print("📂 Collecte des images du dossier test...")
collection_start = time.time()
test_normal_all, test_pneumonia_all = collect_test_images()
collection_time = time.time() - collection_start

print(f"   Images test disponibles:")
print(f"   • Normal: {len(test_normal_all)} images")
print(f"   • Pneumonie: {len(test_pneumonia_all)} images")
print(f"   • Total: {len(test_normal_all) + len(test_pneumonia_all)} images")

normal_sample_size = 234
pneumonia_train_size = 234  
pneumonia_test_size = 390   
pneumonia_total_needed = pneumonia_train_size + pneumonia_test_size 

pneumonia_train_size = min(234, pneumonia_total_needed // 2)
pneumonia_test_size = pneumonia_total_needed - pneumonia_train_size

# Échantillonnage aléatoire avec seed fixe pour reproductibilité
np.random.seed(42)

selected_normal = np.random.choice(test_normal_all, size=normal_sample_size, replace=False)
selected_pneumonia_all = np.random.choice(test_pneumonia_all, size=pneumonia_total_needed, replace=False)

selected_pneumonia_train = selected_pneumonia_all[:pneumonia_train_size]
selected_pneumonia_test = selected_pneumonia_all[pneumonia_train_size:pneumonia_train_size + pneumonia_test_size]

print(f"\n Échantillonnage réalisé:")
print(f"   • Normal (train): {len(selected_normal)} images")
print(f"   • Pneumonie (train): {len(selected_pneumonia_train)} images")
print(f"   • Pneumonie (test): {len(selected_pneumonia_test)} images")
print(f"   • Ratio train: {len(selected_pneumonia_train)}/{len(selected_normal)} = {len(selected_pneumonia_train)/len(selected_normal):.2f}:1")

def optimized_preprocess_image(image_path, target_size=(96, 96)):
    """Prétraitement optimisé"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        image = cv2.resize(image, target_size)
        image = image / 255.0
        image = exposure.equalize_hist(image)
        
        return image
    except:
        return None

def extract_features(image):
    """Extraction de features"""
    if image is None:
        return None
    
    try:
        features_list = []
        
        # HOG Features
        hog_features, _ = hog(image, orientations=9, pixels_per_cell=(12, 12),
                             cells_per_block=(2, 2), visualize=True, 
                             block_norm='L2-Hys', feature_vector=True)
        features_list.extend(hog_features)
        
        # Statistiques d'intensité
        intensity_stats = [
            np.mean(image), np.std(image), 
            np.median(image), np.percentile(image, 25), np.percentile(image, 75)
        ]
        features_list.extend(intensity_stats)
        
        # Features de gradient
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        gradient_stats = [np.mean(gradient_magnitude), np.std(gradient_magnitude)]
        features_list.extend(gradient_stats)
        
        return np.array(features_list)
    except:
        return None

def load_dataset_from_paths(normal_paths, pneumonia_paths, name=""):
    """Chargement du dataset à partir des chemins"""
    X, y = [], []
    total_images = len(normal_paths) + len(pneumonia_paths)
    processed = 0
    
    print(f"⏳ Extraction de features {name} pour {total_images} images...")
    start_time = time.time()
    
    # Traitement des images normales
    for img_path in normal_paths:
        features = extract_features(optimized_preprocess_image(img_path))
        if features is not None:
            X.append(features)
            y.append(0) 
        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total_images - processed) / rate if rate > 0 else 0
            print(f"   Traité: {processed}/{total_images} ({rate:.1f} img/s) - Reste: {remaining:.1f}s")
    
    # Traitement des images pneumonie
    for img_path in pneumonia_paths:
        features = extract_features(optimized_preprocess_image(img_path))
        if features is not None:
            X.append(features)
            y.append(1) 
        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total_images - processed) / rate if rate > 0 else 0
            print(f"   Traité: {processed}/{total_images} ({rate:.1f} img/s) - Reste: {remaining:.1f}s")
    
    extraction_time = time.time() - start_time
    print(f" Extraction terminée en {extraction_time:.1f}s")
    
    return np.array(X), np.array(y), extraction_time

# Chargement des données d'entraînement
print("\n🔄 Chargement des données d'entraînement...")
X_train, y_train, train_extraction_time = load_dataset_from_paths(
    selected_normal, selected_pneumonia_train, "d'entraînement"
)

print(f"   Shape d'entraînement: {X_train.shape}")
print(f"   Distribution train: Normal={sum(y_train==0)}, Pneumonie={sum(y_train==1)}")

# Chargement des données de test
print("\n🔄 Chargement des données de test...")
X_test, y_test, test_extraction_time = load_dataset_from_paths(
    [], selected_pneumonia_test, "de test (pneumonie uniquement)"
)

print(f"   Shape de test: {X_test.shape}")
print(f"   Distribution test: Pneumonie={sum(y_test==1)} (test spécialisé)")


# ENTRAÎNEMENT AVEC ET SANS PCA

print(" ENTRAÎNEMENT DES MODÈLES (AVEC ET SANS PCA)")

# Standardisation pour le modèle sans PCA
print("\n Standardisation pour modèle SANS PCA...")
scaler_no_pca = StandardScaler()
X_train_scaled = scaler_no_pca.fit_transform(X_train)
X_test_scaled = scaler_no_pca.transform(X_test)

# Configuration commune
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
    'max_iter': [2000]
}

# MODÈLE SANS PCA
print("\n Entraînement SANS PCA...")
grid_search_no_pca = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
)

no_pca_start = time.time()
grid_search_no_pca.fit(X_train_scaled, y_train)
no_pca_training_time = time.time() - no_pca_start

print(f"   ✅ SANS PCA - Temps: {no_pca_training_time:.1f}s")
print(f"   🏆 Meilleurs paramètres: {grid_search_no_pca.best_params_}")
print(f"   🎯 Meilleur score F1: {grid_search_no_pca.best_score_:.3f}")

# Prédictions sans PCA (sur test pneumonie uniquement)
y_pred_no_pca = grid_search_no_pca.best_estimator_.predict(X_test_scaled)
y_prob_no_pca = grid_search_no_pca.best_estimator_.predict_proba(X_test_scaled)[:, 1]

# Métriques sans PCA (seulement recall pour pneumonie, puisque test = que pneumonie)
recall_no_pca = recall_score(y_test, y_pred_no_pca)
accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)

print(f"   📊 Test sur pneumonie: Accuracy={accuracy_no_pca:.3f}, Recall={recall_no_pca:.3f}")

# MODÈLE AVEC PCA
print("\n Application PCA et entraînement AVEC PCA...")
pca_start = time.time()

# PCA avec nombre optimal de composantes
n_components = min(50, X_train.shape[1]-1)
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"   📉 Réduction: {X_train.shape[1]} → {X_train_pca.shape[1]} features")
print(f"   📊 Variance expliquée: {pca.explained_variance_ratio_.sum():.3f}")

# Standardisation après PCA
scaler_pca = StandardScaler()
X_train_pca_scaled = scaler_pca.fit_transform(X_train_pca)
X_test_pca_scaled = scaler_pca.transform(X_test_pca)

# Entraînement avec PCA
grid_search_pca = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
)

grid_search_pca.fit(X_train_pca_scaled, y_train)
pca_training_time = time.time() - pca_start

print(f"   ✅ AVEC PCA - Temps: {pca_training_time:.1f}s")
print(f"   🏆 Meilleurs paramètres: {grid_search_pca.best_params_}")
print(f"   🎯 Meilleur score F1: {grid_search_pca.best_score_:.3f}")

# Prédictions avec PCA
y_pred_pca = grid_search_pca.best_estimator_.predict(X_test_pca_scaled)
y_prob_pca = grid_search_pca.best_estimator_.predict_proba(X_test_pca_scaled)[:, 1]

# Métriques avec PCA
recall_pca = recall_score(y_test, y_pred_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

print(f"   📊 Test sur pneumonie: Accuracy={accuracy_pca:.3f}, Recall={recall_pca:.3f}")

# VISUALISATION PCA 2D
pca_2d = PCA(n_components=2, random_state=42)
X_train_pca_2d = pca_2d.fit_transform(X_train)
variance_2d = pca_2d.explained_variance_ratio_

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Projection PCA 2D
normal_mask = (y_train == 0)
pneumonia_mask = (y_train == 1)

axes[0,0].scatter(X_train_pca_2d[normal_mask, 0], X_train_pca_2d[normal_mask, 1], 
                 c='purple', alpha=0.7, s=30, label='Normal', edgecolors='none')
axes[0,0].scatter(X_train_pca_2d[pneumonia_mask, 0], X_train_pca_2d[pneumonia_mask, 1], 
                 c='yellow', alpha=0.7, s=30, label='Pneumonie', edgecolors='none')

axes[0,0].set_xlabel(f'PC1 ({variance_2d[0]:.1%})')
axes[0,0].set_ylabel(f'PC2 ({variance_2d[1]:.1%})')
axes[0,0].set_title('Projection PCA 2D (données équilibrées)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_facecolor('#f8f8f8')

# Distribution des données
data_counts = [len(selected_normal), len(selected_pneumonia_train), len(selected_pneumonia_test)]
data_labels = ['Normal\n(train)', 'Pneumonie\n(train)', 'Pneumonie\n(test)']
colors = ['purple', 'yellow', 'orange']

bars = axes[0,1].bar(data_labels, data_counts, color=colors, alpha=0.8)
axes[0,1].set_ylabel('Nombre d\'images')
axes[0,1].set_title('Distribution des Données Utilisées')
axes[0,1].grid(axis='y', alpha=0.3)

for bar, count in zip(bars, data_counts):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   str(count), ha='center', va='bottom', fontweight='bold')

# Comparaison des performances CV
cv_scores_no_pca = grid_search_no_pca.cv_results_['mean_test_score']
cv_scores_pca = grid_search_pca.cv_results_['mean_test_score']

axes[0,2].boxplot([cv_scores_no_pca, cv_scores_pca], labels=['Sans PCA', 'Avec PCA'])
axes[0,2].set_ylabel('Score F1 (CV)')
axes[0,2].set_title('Comparaison des Performances (CV)')
axes[0,2].grid(axis='y', alpha=0.3)

# Variance expliquée par composantes PCA
n_show = min(20, len(pca.explained_variance_ratio_))
axes[1,0].bar(range(1, n_show+1), pca.explained_variance_ratio_[:n_show], alpha=0.8)
axes[1,0].set_xlabel('Composante')
axes[1,0].set_ylabel('Variance expliquée')
axes[1,0].set_title(f'Variance par Composante (Top {n_show})')
axes[1,0].grid(axis='y', alpha=0.3)

# Distribution des probabilités de prédiction
axes[1,1].hist(y_prob_no_pca, bins=20, alpha=0.7, label='Sans PCA', color='blue', density=True)
axes[1,1].hist(y_prob_pca, bins=20, alpha=0.7, label='Avec PCA', color='red', density=True)
axes[1,1].set_xlabel('Probabilité prédite (Pneumonie)')
axes[1,1].set_ylabel('Densité')
axes[1,1].set_title('Distribution des Probabilités\n(Test Pneumonie)')
axes[1,1].legend()
axes[1,1].grid(axis='y', alpha=0.3)

# Résumé des temps
time_data = ['Extraction\nTrain', 'Extraction\nTest', 'Training\nSans PCA', 'Training\nAvec PCA']
time_values = [train_extraction_time, test_extraction_time, no_pca_training_time, pca_training_time]

bars_time = axes[1,2].bar(time_data, time_values, color=['lightblue', 'lightgreen', 'orange', 'red'], alpha=0.8)
axes[1,2].set_ylabel('Temps (secondes)')
axes[1,2].set_title('Temps d\'Exécution par Étape')
axes[1,2].grid(axis='y', alpha=0.3)

for bar, time_val in zip(bars_time, time_values):
    axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Calcul du temps total
total_time = time.time() - total_start_time

# Résumé final
print(f"\n📊 RÉSUMÉ FINAL:")
print("=" * 50)
print(f"Données utilisées:")
print(f"  • Normal (train): {len(selected_normal)} images")
print(f"  • Pneumonie (train): {len(selected_pneumonia_train)} images") 
print(f"  • Pneumonie (test): {len(selected_pneumonia_test)} images")
print(f"  • Ratio train équilibré: 1:1")

print(f"\nPerformances (CV F1-Score):")
print(f"  • Sans PCA: {grid_search_no_pca.best_score_:.3f}")
print(f"  • Avec PCA: {grid_search_pca.best_score_:.3f}")

print(f"\nTest sur pneumonie (Recall):")
print(f"  • Sans PCA: {recall_no_pca:.3f}")
print(f"  • Avec PCA: {recall_pca:.3f}")

print(f"\nTemps d'exécution:")
print(f"  • Extraction: {train_extraction_time + test_extraction_time:.1f}s")
print(f"  • Training sans PCA: {no_pca_training_time:.1f}s")
print(f"  • Training avec PCA: {pca_training_time:.1f}s")
print(f"  • Total: {total_time:.1f}s ({total_time/60:.1f} min)")

# Sauvegarde
save_path = '/content/drive/MyDrive/chest_Xray/model_test_subset_234_390_234.joblib'
joblib.dump({
    'models': {
        'without_pca': grid_search_no_pca.best_estimator_,
        'with_pca': grid_search_pca.best_estimator_
    },
    'scalers': {
        'without_pca': scaler_no_pca,
        'with_pca': scaler_pca
    },
    'pca': pca,
    'pca_2d': pca_2d,
    'data_config': {
        'normal_train': len(selected_normal),
        'pneumonia_train': len(selected_pneumonia_train),
        'pneumonia_test': len(selected_pneumonia_test)
    },
    'results': {
        'cv_f1_no_pca': grid_search_no_pca.best_score_,
        'cv_f1_pca': grid_search_pca.best_score_,
        'test_recall_no_pca': recall_no_pca,
        'test_recall_pca': recall_pca
    },
    'times': {
        'extraction': train_extraction_time + test_extraction_time,
        'training_no_pca': no_pca_training_time,
        'training_pca': pca_training_time,
        'total': total_time
    }
}, save_path)

print(f"\n💾 Modèles sauvegardés: {save_path}")
print("✅ ENTRAÎNEMENT TERMINÉ!")