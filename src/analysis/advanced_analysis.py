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
    
    # Sauvegarde des résultats JSON
    with open(output_path / 'advanced_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # Génération des visualisations
    generate_advanced_visualizations(results, output_path)
    
    logger.info(f"Résultats de l'analyse avancée sauvegardés dans {output_dir}")

def generate_advanced_visualizations(results: Dict, output_path: Path) -> None:
    """
    Génère les visualisations pour l'analyse avancée.
    """
    # Création du répertoire pour les visualisations
    viz_path = output_path / 'visualizations'
    viz_path.mkdir(exist_ok=True)
    
    # Analyse PCA
    plot_pca_visualization(results['pca'], viz_path)
    
    # Analyse t-SNE
    plot_tsne_visualization(results['tsne'], viz_path)
    
    # Analyse de clustering
    plot_clustering_visualization(results['clustering'], viz_path)
    
    # Analyse des motifs
    plot_pattern_visualization(results['patterns'], viz_path)

def plot_pca_visualization(pca_results: Dict, output_path: Path) -> None:
    """
    Plot PCA visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Explained variance
    plt.figure(figsize=(12, 5))
    
    # Individual variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca_results['standard']['explained_variance_ratio']) + 1),
            pca_results['standard']['explained_variance_ratio'])
    plt.title('Explained Variance by Component')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance')
    
    # Cumulative variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca_results['standard']['cumulative_variance']) + 1),
             pca_results['standard']['cumulative_variance'], 'bo-')
    plt.title('Cumulative Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    
    plt.tight_layout()
    plt.savefig(output_path / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne_visualization(tsne_results: Dict, output_path: Path) -> None:
    """
    Plot t-SNE visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, len(tsne_results), figsize=(15, 5))
    if len(tsne_results) == 1:
        axes = [axes]
    
    for ax, (perplexity, results) in zip(axes, tsne_results.items()):
        data = np.array(results['transformed_data'])
        sns.scatterplot(x=data[:, 0], y=data[:, 1], ax=ax)
        ax.set_title(f't-SNE (Perplexity = {perplexity.split("_")[1]})')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig(output_path / 'tsne_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_clustering_visualization(clustering_results: Dict, output_path: Path) -> None:
    """
    Plot clustering visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # K-means
    plt.figure(figsize=(12, 5))
    
    # Silhouette scores
    plt.subplot(1, 2, 1)
    kmeans_scores = {
        k: v['silhouette_score']
        for k, v in clustering_results['kmeans'].items()
    }
    plt.bar(kmeans_scores.keys(), kmeans_scores.values())
    plt.title('Silhouette Scores - K-means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    # BIC for GMM
    plt.subplot(1, 2, 2)
    gmm_scores = {
        k: v['bic']
        for k, v in clustering_results['gmm'].items()
    }
    plt.plot(gmm_scores.keys(), gmm_scores.values(), 'bo-')
    plt.title('BIC Criterion - GMM')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    
    plt.tight_layout()
    plt.savefig(output_path / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pattern_visualization(patterns: Dict, output_path: Path) -> None:
    """
    Plot pattern visualizations.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(15, 6))
    
    # Top 10 sequences
    plt.subplot(1, 2, 1)
    top_sequences = dict(list(patterns['sequence_patterns'].items())[:10])
    plt.bar(top_sequences.keys(), top_sequences.values())
    plt.title('Top 10 Reservoir Type Sequences')
    plt.xlabel('Sequence')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # Top 10 transitions
    plt.subplot(1, 2, 2)
    top_transitions = dict(list(patterns['transitions'].items())[:10])
    plt.bar(top_transitions.keys(), top_transitions.values())
    plt.title('Top 10 Reservoir Type Transitions')
    plt.xlabel('Transition')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close() 