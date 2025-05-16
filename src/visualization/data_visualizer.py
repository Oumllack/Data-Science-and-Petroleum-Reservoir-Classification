#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data visualization module for petroleum reservoir analysis.
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
    Plot the distribution of reservoir types.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_path (Path): Path to save the plot
    """
    # Convert facies to reservoir types
    df['ReservoirType'] = df['Facies'].apply(map_facies_to_reservoir_type)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='ReservoirType')
    plt.title('Reservoir Type Distribution')
    plt.xlabel('Reservoir Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_formation_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot the distribution of formations.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_path (Path): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Formation')
    plt.title('Formation Distribution')
    plt.xlabel('Formation')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_features_by_reservoir(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot feature distributions by reservoir type.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_path (Path): Path to save the plot
    """
    # Convert facies to reservoir types
    df['ReservoirType'] = df['Facies'].apply(map_facies_to_reservoir_type)
    
    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    feature_names = {
        'GR': 'Gamma Ray',
        'ILD_log10': 'Deep Resistivity (log10)',
        'DeltaPHI': 'Delta Porosity',
        'PHIND': 'Neutron Porosity',
        'PE': 'Photoelectric Effect'
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        sns.boxplot(data=df, x='ReservoirType', y=feature, ax=axes[idx])
        axes[idx].set_title(f'{feature_names[feature]} by Reservoir Type')
        axes[idx].set_xlabel('Reservoir Type')
        axes[idx].set_ylabel(feature_names[feature])
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot the correlation matrix.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_path (Path): Path to save the plot
    """
    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'Facies']
    feature_names = {
        'GR': 'Gamma Ray',
        'ILD_log10': 'Deep Resistivity',
        'DeltaPHI': 'Delta Porosity',
        'PHIND': 'Neutron Porosity',
        'PE': 'Photoelectric Effect',
        'Facies': 'Facies'
    }
    
    plt.figure(figsize=(12, 8))
    correlation = df[features].corr()
    correlation.index = [feature_names[f] for f in features]
    correlation.columns = [feature_names[f] for f in features]
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_pca_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot PCA analysis of the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_path (Path): Path to save the plot
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Data preparation
    features = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    X = df[features].copy()
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for visualization
    pca_df = pd.DataFrame(
        data=X_pca,
        columns=['PC1', 'PC2']
    )
    pca_df['ReservoirType'] = df['Facies'].apply(map_facies_to_reservoir_type)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='ReservoirType',
        palette='deep'
    )
    plt.title('PCA Analysis of Reservoir Types')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} of variance)')
    plt.legend(title='Reservoir Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_well_logs(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot well logs for each well.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        output_dir (Path): Directory to save the plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_names = {
        'GR': 'Gamma Ray',
        'ILD_log10': 'Deep Resistivity (log10)',
        'DeltaPHI': 'Delta Porosity',
        'PHIND': 'Neutron Porosity',
        'PE': 'Photoelectric Effect'
    }
    
    for well_name in df['Well Name'].unique():
        well_data = df[df['Well Name'] == well_name].copy()
        
        # Create figure
        fig, axes = plt.subplots(1, 5, figsize=(20, 8))
        fig.suptitle(f'Well Logs - {well_name}', fontsize=16)
        
        # Plot logs
        logs = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
        for idx, log in enumerate(logs):
            ax = axes[idx]
            ax.plot(well_data[log], well_data['Depth'], 'b-')
            ax.set_title(feature_names[log])
            ax.set_xlabel(feature_names[log])
            ax.set_ylabel('Depth')
            ax.invert_yaxis()  # Invert Y axis for depth
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{well_name}_logs.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_feature_importance(importance: np.ndarray, feature_names: List[str], output_path: Path) -> None:
    """
    Plot feature importance.
    
    Args:
        importance (np.ndarray): Feature importance values
        feature_names (List[str]): Feature names
        output_path (Path): Path to save the plot
    """
    feature_name_map = {
        'GR': 'Gamma Ray',
        'ILD_log10': 'Deep Resistivity',
        'DeltaPHI': 'Delta Porosity',
        'PHIND': 'Neutron Porosity',
        'PE': 'Photoelectric Effect',
        'NM_M': 'Nonmarine/Marine Indicator',
        'RELPOS': 'Relative Position'
    }
    
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': [feature_name_map.get(f, f) for f in feature_names],
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() 