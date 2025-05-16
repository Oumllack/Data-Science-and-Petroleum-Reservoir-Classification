#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la visualisation des résultats des modèles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def visualize_model_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Génère et sauvegarde les visualisations des résultats des modèles.
    
    Args:
        results (Dict[str, Any]): Résultats d'évaluation des modèles
        output_dir (str): Répertoire de sortie pour les visualisations
    """
    try:
        # Création du répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, result in results.items():
            # Création d'un sous-répertoire pour chaque modèle
            model_dir = output_path / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Matrice de confusion
            plot_confusion_matrix(
                result['metrics']['confusion_matrix'],
                model_name,
                model_dir / 'confusion_matrix.png'
            )
            
            # Courbes ROC
            plot_roc_curves(
                result['metrics']['roc_data'],
                model_name,
                model_dir / 'roc_curves.png'
            )
            
            # Si c'est un Random Forest, on trace l'importance des caractéristiques
            if model_name == 'Random Forest':
                plot_feature_importance(
                    result['model'].feature_importances_,
                    result['model'].feature_names_in_,
                    model_dir / 'feature_importance.png'
                )
        
        # Comparaison des modèles
        plot_model_comparison(results, output_path / 'model_comparison.png')
        
        logger.info(f"Visualisations des résultats sauvegardées dans {output_dir}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation des résultats : {str(e)}")
        raise

def plot_confusion_matrix(conf_matrix: np.ndarray, model_name: str, output_path: Path) -> None:
    """
    Trace la matrice de confusion.
    
    Args:
        conf_matrix (np.ndarray): Matrice de confusion
        model_name (str): Nom du modèle
        output_path (Path): Chemin de sauvegarde du graphique
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curves(roc_data: Dict[str, Dict[str, np.ndarray]], model_name: str, output_path: Path) -> None:
    """
    Trace les courbes ROC pour chaque classe.
    
    Args:
        roc_data (Dict[str, Dict[str, np.ndarray]]): Données ROC pour chaque classe
        model_name (str): Nom du modèle
        output_path (Path): Chemin de sauvegarde du graphique
    """
    plt.figure(figsize=(10, 8))
    
    for class_name, data in roc_data.items():
        plt.plot(
            data['fpr'],
            data['tpr'],
            label=f'Classe {class_name} (AUC = {data["auc"]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title(f'Courbes ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_importance(importance: np.ndarray, feature_names: np.ndarray, output_path: Path) -> None:
    """
    Trace l'importance des caractéristiques.
    
    Args:
        importance (np.ndarray): Importance des caractéristiques
        feature_names (np.ndarray): Noms des caractéristiques
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

def plot_model_comparison(results: Dict[str, Any], output_path: Path) -> None:
    """
    Trace la comparaison des performances des modèles.
    
    Args:
        results (Dict[str, Any]): Résultats d'évaluation des modèles
        output_path (Path): Chemin de sauvegarde du graphique
    """
    # Préparation des données
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())
    
    # Création du DataFrame pour la visualisation
    comparison_data = []
    for model_name in model_names:
        for metric in metrics:
            comparison_data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': results[model_name]['metrics'][metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x='Model', y='Value', hue='Metric')
    plt.title('Comparaison des performances des modèles')
    plt.xticks(rotation=45)
    plt.legend(title='Métrique', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() 