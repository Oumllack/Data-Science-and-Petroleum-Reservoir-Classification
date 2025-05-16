#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for model results visualization.
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
    Plot the confusion matrix.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix
        model_name (str): Model name
        output_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curves(roc_data: Dict[str, Dict[str, np.ndarray]], model_name: str, output_path: Path) -> None:
    """
    Plot ROC curves for each class.
    
    Args:
        roc_data (Dict[str, Dict[str, np.ndarray]]): ROC data for each class
        model_name (str): Model name
        output_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    for class_name, data in roc_data.items():
        plt.plot(
            data['fpr'],
            data['tpr'],
            label=f'Class {class_name} (AUC = {data["auc"]:.3f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_feature_importance(importance: np.ndarray, feature_names: np.ndarray, output_path: Path) -> None:
    """
    Plot feature importance.
    
    Args:
        importance (np.ndarray): Feature importance values
        feature_names (np.ndarray): Feature names
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
    
    plt.figure(figsize=(12, 6))
    importance_df = pd.DataFrame({
        'Feature': [feature_name_map.get(f, f) for f in feature_names],
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Relative Importance')
    plt.ylabel('Geological Features')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_model_comparison(results: Dict[str, Any], output_path: Path) -> None:
    """
    Plot model performance comparison.
    
    Args:
        results (Dict[str, Any]): Model evaluation results
        output_path (Path): Path to save the plot
    """
    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score'
    }
    model_names = list(results.keys())
    
    # Create DataFrame for visualization
    comparison_data = []
    for model_name in model_names:
        for metric in metrics:
            comparison_data.append({
                'Model': model_name,
                'Metric': metric_names[metric],
                'Value': results[model_name]['metrics'][metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x='Model', y='Value', hue='Metric')
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close() 