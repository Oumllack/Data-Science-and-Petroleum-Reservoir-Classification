#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour l'évaluation des modèles de classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import logging

logger = logging.getLogger(__name__)

def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Évalue les performances des modèles entraînés.
    
    Args:
        models (dict): Dictionnaire des modèles entraînés
        X_test (np.ndarray): Features de test
        y_test (np.ndarray): Target de test
        
    Returns:
        dict: Dictionnaire contenant les résultats d'évaluation
    """
    try:
        results = {}
        
        for name, model in models.items():
            logger.info(f"Évaluation du modèle {name}...")
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calcul des métriques
            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Stockage des résultats
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba,
                'metrics': metrics
            }
            
            # Affichage des résultats
            logger.info(f"Résultats pour {name}:")
            logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1-score: {metrics['f1']:.3f}")
            
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation des modèles : {str(e)}")
        raise

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Calcule les métriques d'évaluation pour un modèle.
    
    Args:
        y_true (np.ndarray): Valeurs réelles
        y_pred (np.ndarray): Prédictions
        y_pred_proba (np.ndarray): Probabilités de prédiction
        
    Returns:
        dict: Dictionnaire contenant les métriques calculées
    """
    # Métriques de base
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Rapport de classification détaillé
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Courbe ROC (pour chaque classe)
    roc_data = {}
    n_classes = y_pred_proba.shape[1]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[f'class_{i}'] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'roc_data': roc_data
    }

def compare_models(results: dict) -> None:
    """
    Compare les performances des différents modèles.
    
    Args:
        results (dict): Résultats d'évaluation des modèles
    """
    logger.info("Comparaison des modèles :")
    
    # Création d'un tableau comparatif
    comparison = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-score': []
    }
    
    for name, result in results.items():
        metrics = result['metrics']
        comparison['Model'].append(name)
        comparison['Accuracy'].append(metrics['accuracy'])
        comparison['Precision'].append(metrics['precision'])
        comparison['Recall'].append(metrics['recall'])
        comparison['F1-score'].append(metrics['f1'])
    
    # Affichage du tableau comparatif
    logger.info("\nTableau comparatif des modèles :")
    for i in range(len(comparison['Model'])):
        logger.info(f"\n{comparison['Model'][i]}:")
        logger.info(f"  Accuracy:  {comparison['Accuracy'][i]:.3f}")
        logger.info(f"  Precision: {comparison['Precision'][i]:.3f}")
        logger.info(f"  Recall:    {comparison['Recall'][i]:.3f}")
        logger.info(f"  F1-score:  {comparison['F1-score'][i]:.3f}") 