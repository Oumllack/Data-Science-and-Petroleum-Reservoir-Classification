import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def perform_advanced_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Effectue une analyse avancée sur les données de réservoirs pétroliers.
    """
    results = {}
    try:
        # Préparation des données
        X = prepare_data(df)
        # PCA
        results['pca'] = perform_pca_analysis(X)
        # t-SNE
        results['tsne'] = perform_tsne_analysis(X)
        # Clustering
        results['clustering'] = perform_clustering_analysis(X)
        # Analyse des motifs
        results['patterns'] = analyze_patterns(df)
        # Sauvegarde
        save_advanced_analysis_results(results, output_dir)
        logger.info("Analyse avancée terminée avec succès")
        return results
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse avancée : {str(e)}")
        raise

def prepare_data(df: pd.DataFrame) -> np.ndarray:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def perform_pca_analysis(X: np.ndarray) -> Dict:
    results = {}
    # PCA standard
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)
    results['standard'] = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'components': X_pca.tolist()
    }
    # Kernel PCA
    kpca = KernelPCA(n_components=2, kernel='rbf')
    X_kpca = kpca.fit_transform(X)
    results['kernel'] = {
        'components': X_kpca.tolist()
    }
    return results

def perform_tsne_analysis(X: np.ndarray) -> Dict:
    results = {}
    for perplexity in [5, 30, 50]:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        results[f'perp_{perplexity}'] = {
            'transformed_data': X_tsne.tolist()
        }
    return results

def perform_clustering_analysis(X: np.ndarray) -> Dict:
    results = {'kmeans': {}, 'dbscan': {}, 'gmm': {}}
    # K-means
    for k in [2, 3, 4, 5]:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X, labels)
        results['kmeans'][k] = {'labels': labels.tolist(), 'silhouette_score': float(score)}
    # DBSCAN
    dbscan = DBSCAN(eps=2, min_samples=5)
    labels = dbscan.fit_predict(X)
    results['dbscan'] = {'labels': labels.tolist()}
    # GMM
    for k in [2, 3, 4, 5]:
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(X)
        bic = gmm.bic(X)
        results['gmm'][k] = {'labels': labels.tolist(), 'bic': float(bic)}
    return results

def analyze_patterns(df: pd.DataFrame) -> Dict:
    patterns = {}
    # Séquences de types de réservoirs
    if 'Facies' in df.columns:
        facies_seq = df['Facies'].astype(str).tolist()
        from collections import Counter
        # Séquences de longueur 3
        seq3 = ["-".join(facies_seq[i:i+3]) for i in range(len(facies_seq)-2)]
        patterns['sequence_patterns'] = dict(Counter(seq3))
        # Transitions
        transitions = [f"{facies_seq[i]}->{facies_seq[i+1]}" for i in range(len(facies_seq)-1)]
        patterns['transitions'] = dict(Counter(transitions))
    return patterns

def save_advanced_analysis_results(results: Dict, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / 'advanced_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Résultats de l'analyse avancée sauvegardés dans {output_dir}") 