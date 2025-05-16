#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la génération de rapports d'analyse.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List
from jinja2 import Template
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

def generate_analysis_report(eda_results: Dict, advanced_results: Dict, output_dir: str) -> None:
    """
    Génère un rapport d'analyse complet au format HTML.
    
    Args:
        eda_results (Dict): Résultats de l'analyse exploratoire
        advanced_results (Dict): Résultats de l'analyse avancée
        output_dir (str): Répertoire de sortie pour le rapport
    """
    try:
        # Création du répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Génération des visualisations
        generate_visualizations(eda_results, advanced_results, output_path)
        
        # Génération du rapport HTML
        generate_html_report(eda_results, advanced_results, output_path)
        
        logger.info(f"Rapport d'analyse généré dans {output_dir}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport : {str(e)}")
        raise

def generate_visualizations(eda_results: Dict, advanced_results: Dict, output_path: Path) -> None:
    """
    Génère les visualisations pour le rapport.
    """
    # Création du répertoire pour les visualisations
    viz_path = output_path / 'visualizations'
    viz_path.mkdir(exist_ok=True)
    
    # Distribution des types de réservoirs
    plot_reservoir_distribution(eda_results, viz_path)
    
    # Matrice de corrélation
    plot_correlation_matrix(eda_results, viz_path)
    
    # Analyse PCA
    plot_pca_analysis(advanced_results, viz_path)
    
    # Analyse t-SNE
    plot_tsne_analysis(advanced_results, viz_path)
    
    # Résultats du clustering
    plot_clustering_results(advanced_results, viz_path)
    
    # Analyse des motifs
    plot_pattern_analysis(advanced_results, viz_path)

def plot_reservoir_distribution(eda_results: Dict, output_path: Path) -> None:
    """
    Trace la distribution des types de réservoirs.
    """
    distribution = eda_results['reservoir_analysis']['distribution']
    
    fig = px.bar(
        x=list(distribution.keys()),
        y=list(distribution.values()),
        title='Distribution des Types de Réservoirs',
        labels={'x': 'Type de Réservoir', 'y': 'Nombre d\'échantillons'}
    )
    
    fig.write_html(output_path / 'reservoir_distribution.html')
    fig.write_image(output_path / 'reservoir_distribution.png')

def plot_correlation_matrix(eda_results: Dict, output_path: Path) -> None:
    """
    Trace la matrice de corrélation.
    """
    corr_matrix = pd.DataFrame(eda_results['correlations']['pearson'])
    
    fig = px.imshow(
        corr_matrix,
        title='Matrice de Corrélation',
        color_continuous_scale='RdBu'
    )
    
    fig.write_html(output_path / 'correlation_matrix.html')
    fig.write_image(output_path / 'correlation_matrix.png')

def plot_pca_analysis(advanced_results: Dict, output_path: Path) -> None:
    """
    Trace l'analyse PCA.
    """
    pca_results = advanced_results['pca']['standard']
    
    # Variance expliquée
    fig = make_subplots(rows=1, cols=2)
    
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(pca_results['explained_variance_ratio']) + 1)),
            y=pca_results['explained_variance_ratio'],
            name='Variance Expliquée'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(pca_results['cumulative_variance']) + 1)),
            y=pca_results['cumulative_variance'],
            name='Variance Cumulée'
        ),
        row=1, col=2
    )
    
    fig.update_layout(title='Analyse en Composantes Principales')
    fig.write_html(output_path / 'pca_analysis.html')
    fig.write_image(output_path / 'pca_analysis.png')

def plot_tsne_analysis(advanced_results: Dict, output_path: Path) -> None:
    """
    Trace l'analyse t-SNE.
    """
    tsne_results = advanced_results['tsne']
    
    fig = make_subplots(
        rows=1, cols=len(tsne_results),
        subplot_titles=[f'Perplexité = {p.split("_")[1]}' for p in tsne_results.keys()]
    )
    
    for idx, (perplexity, results) in enumerate(tsne_results.items(), 1):
        data = np.array(results['transformed_data'])
        fig.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode='markers',
                name=f'Perplexité {perplexity.split("_")[1]}'
            ),
            row=1, col=idx
        )
    
    fig.update_layout(title='Analyse t-SNE avec Différentes Perplexités')
    fig.write_html(output_path / 'tsne_analysis.html')
    fig.write_image(output_path / 'tsne_analysis.png')

def plot_clustering_results(advanced_results: Dict, output_path: Path) -> None:
    """
    Trace les résultats du clustering.
    """
    clustering_results = advanced_results['clustering']
    
    # K-means
    kmeans_scores = {
        k: v['silhouette_score']
        for k, v in clustering_results['kmeans'].items()
    }
    
    fig = px.bar(
        x=list(kmeans_scores.keys()),
        y=list(kmeans_scores.values()),
        title='Scores de Silhouette pour K-means',
        labels={'x': 'Nombre de Clusters', 'y': 'Score de Silhouette'}
    )
    
    fig.write_html(output_path / 'kmeans_analysis.html')
    fig.write_image(output_path / 'kmeans_analysis.png')
    
    # GMM
    gmm_scores = {
        k: v['bic']
        for k, v in clustering_results['gmm'].items()
    }
    
    fig = px.line(
        x=list(gmm_scores.keys()),
        y=list(gmm_scores.values()),
        title='Critère BIC pour GMM',
        labels={'x': 'Nombre de Composantes', 'y': 'BIC'}
    )
    
    fig.write_html(output_path / 'gmm_analysis.html')
    fig.write_image(output_path / 'gmm_analysis.png')

def plot_pattern_analysis(advanced_results: Dict, output_path: Path) -> None:
    """
    Trace l'analyse des motifs.
    """
    patterns = advanced_results['patterns']
    
    # Top 10 des séquences
    top_sequences = dict(list(patterns['sequence_patterns'].items())[:10])
    
    fig = px.bar(
        x=list(top_sequences.keys()),
        y=list(top_sequences.values()),
        title='Top 10 des Séquences de Types de Réservoirs',
        labels={'x': 'Séquence', 'y': 'Fréquence'}
    )
    
    fig.write_html(output_path / 'sequence_patterns.html')
    fig.write_image(output_path / 'sequence_patterns.png')
    
    # Top 10 des transitions
    top_transitions = dict(list(patterns['transitions'].items())[:10])
    
    fig = px.bar(
        x=list(top_transitions.keys()),
        y=list(top_transitions.values()),
        title='Top 10 des Transitions entre Types de Réservoirs',
        labels={'x': 'Transition', 'y': 'Fréquence'}
    )
    
    fig.write_html(output_path / 'transitions.html')
    fig.write_image(output_path / 'transitions.png')

def generate_html_report(eda_results: Dict, advanced_results: Dict, output_path: Path) -> None:
    """
    Génère le rapport HTML final.
    """
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport d'Analyse des Réservoirs Pétroliers</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1, h2 { color: #2c3e50; }
            .section { margin: 20px 0; padding: 20px; background: #f8f9fa; }
            .visualization { margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Rapport d'Analyse des Réservoirs Pétroliers</h1>
        
        <div class="section">
            <h2>Résumé Exécutif</h2>
            <p>Ce rapport présente une analyse complète des données de réservoirs pétroliers, incluant :</p>
            <ul>
                <li>Analyse exploratoire des données (EDA)</li>
                <li>Analyse avancée avec techniques de data science</li>
                <li>Visualisations interactives</li>
                <li>Recommandations</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Analyse Exploratoire des Données</h2>
            <h3>Distribution des Types de Réservoirs</h3>
            <div class="visualization">
                <iframe src="visualizations/reservoir_distribution.html" width="100%" height="500px" frameborder="0"></iframe>
            </div>
            
            <h3>Matrice de Corrélation</h3>
            <div class="visualization">
                <iframe src="visualizations/correlation_matrix.html" width="100%" height="500px" frameborder="0"></iframe>
            </div>
        </div>
        
        <div class="section">
            <h2>Analyse Avancée</h2>
            <h3>Analyse en Composantes Principales (PCA)</h3>
            <div class="visualization">
                <iframe src="visualizations/pca_analysis.html" width="100%" height="500px" frameborder="0"></iframe>
            </div>
            
            <h3>Analyse t-SNE</h3>
            <div class="visualization">
                <iframe src="visualizations/tsne_analysis.html" width="100%" height="500px" frameborder="0"></iframe>
            </div>
            
            <h3>Analyse de Clustering</h3>
            <div class="visualization">
                <iframe src="visualizations/kmeans_analysis.html" width="100%" height="500px" frameborder="0"></iframe>
                <iframe src="visualizations/gmm_analysis.html" width="100%" height="500px" frameborder="0"></iframe>
            </div>
            
            <h3>Analyse des Motifs</h3>
            <div class="visualization">
                <iframe src="visualizations/sequence_patterns.html" width="100%" height="500px" frameborder="0"></iframe>
                <iframe src="visualizations/transitions.html" width="100%" height="500px" frameborder="0"></iframe>
            </div>
        </div>
        
        <div class="section">
            <h2>Recommandations</h2>
            <ul>
                <li>Points clés identifiés dans l'analyse</li>
                <li>Suggestions pour l'exploration future</li>
                <li>Recommandations pour la modélisation</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Génération du rapport
    template = Template(template)
    html_content = template.render()
    
    # Sauvegarde du rapport
    with open(output_path / 'analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("Rapport HTML généré avec succès") 