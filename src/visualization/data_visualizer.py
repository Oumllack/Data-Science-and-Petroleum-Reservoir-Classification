#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de visualisation des données pour l'analyse des réservoirs pétroliers.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from data.reservoir_types import ReservoirType, map_facies_to_reservoir_type

logger = logging.getLogger(__name__)

def visualize_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Génère et sauvegarde les visualisations des données.
    
    Args:
        df (pd.DataFrame): DataFrame à visualiser
        output_dir (str): Répertoire de sortie pour les visualisations
    """
    try:
        # Création du répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Distribution des types de réservoirs
        plot_reservoir_distribution(df, output_path / 'reservoir_distribution.png')
        
        # Distribution des formations
        plot_formation_distribution(df, output_path / 'formation_distribution.png')
        
        # Distribution des caractéristiques par type de réservoir
        plot_features_by_reservoir(df, output_path / 'features_by_reservoir.png')
        
        # Matrice de corrélation
        plot_correlation_matrix(df, output_path / 'correlation_matrix.png')
        
        # Visualisation PCA
        plot_pca_analysis(df, output_path / 'pca_analysis.png')
        
        # Visualisation des logs de puits
        plot_well_logs(df, output_path / 'well_logs')
        
        logger.info(f"Visualisations sauvegardées dans {output_dir}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des données : {str(e)}")
        raise

def plot_reservoir_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Trace la distribution des types de réservoirs.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        output_path (Path): Chemin de sauvegarde du graphique
    """
    # Conversion des faciès en types de réservoirs
    df['ReservoirType'] = df['Facies'].apply(map_facies_to_reservoir_type)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='ReservoirType')
    plt.title('Distribution des Types de Réservoirs')
    plt.xlabel('Type de Réservoir')
    plt.ylabel('Nombre d\'échantillons')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_formation_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Trace la distribution des formations.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        output_path (Path): Chemin de sauvegarde du graphique
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Formation')
    plt.title('Distribution des Formations')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_features_by_reservoir(df: pd.DataFrame, output_path: Path) -> None:
    """
    Trace la distribution des caractéristiques par type de réservoir.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        output_path (Path): Chemin de sauvegarde du graphique
    """
    # Conversion des faciès en types de réservoirs
    df['ReservoirType'] = df['Facies'].apply(map_facies_to_reservoir_type)
    
    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        sns.boxplot(data=df, x='ReservoirType', y=feature, ax=axes[idx])
        axes[idx].set_title(f'Distribution de {feature} par Type de Réservoir')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """
    Trace la matrice de corrélation.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        output_path (Path): Chemin de sauvegarde du graphique
    """
    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'Facies']
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
    plt.title('Matrice de Corrélation')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_pca_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """
    Trace l'analyse PCA des données.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        output_path (Path): Chemin de sauvegarde du graphique
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Préparation des données
    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X = df[features].copy()
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Création du DataFrame pour la visualisation
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=['PC1', 'PC2']
    )
    pca_df['ReservoirType'] = df['Facies'].apply(map_facies_to_reservoir_type)
    
    # Visualisation
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='ReservoirType',
        palette='deep'
    )
    plt.title('Analyse PCA des Types de Réservoirs')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} de variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} de variance)')
    plt.legend(title='Type de Réservoir', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_well_logs(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Trace les logs de puits pour chaque puits.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données
        output_dir (Path): Répertoire de sauvegarde des graphiques
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for well_name in df['Well Name'].unique():
        well_data = df[df['Well Name'] == well_name].copy()
        
        # Création de la figure
        fig, axes = plt.subplots(1, 5, figsize=(20, 8))
        fig.suptitle(f'Logs de puits - {well_name}', fontsize=16)
        
        # Tracé des logs
        logs = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
        for idx, log in enumerate(logs):
            ax = axes[idx]
            ax.plot(well_data[log], well_data['Depth'], 'b-')
            ax.set_title(log)
            ax.invert_yaxis()  # Inverser l'axe Y pour la profondeur
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{well_name}_logs.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_feature_importance(importance: np.ndarray, feature_names: List[str], output_path: Path) -> None:
    """
    Trace l'importance des caractéristiques.
    
    Args:
        importance (np.ndarray): Importance des caractéristiques
        feature_names (List[str]): Noms des caractéristiques
        output_path (Path): Chemin de sauvegarde du graphique
    """
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Importance des caractéristiques')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() 