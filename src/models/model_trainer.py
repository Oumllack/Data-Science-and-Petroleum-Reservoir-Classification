#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour l'entraînement des modèles de classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)

def train_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Entraîne plusieurs modèles de classification.
    
    Args:
        X_train (np.ndarray): Features d'entraînement
        y_train (np.ndarray): Target d'entraînement
        
    Returns:
        dict: Dictionnaire contenant les modèles entraînés
    """
    try:
        models = {}
        
        # Random Forest (version rapide)
        logger.info("Entraînement du Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['Random Forest'] = rf_model
        
        # SVM (version rapide)
        logger.info("Entraînement du SVM...")
        svm_model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train, y_train)
        models['SVM'] = svm_model
        
        logger.info("Entraînement de tous les modèles terminé")
        return models
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement des modèles : {str(e)}")
        raise

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Entraîne un modèle Random Forest avec optimisation des hyperparamètres.
    
    Args:
        X_train (np.ndarray): Features d'entraînement
        y_train (np.ndarray): Target d'entraînement
        
    Returns:
        RandomForestClassifier: Modèle Random Forest entraîné
    """
    # Définition des hyperparamètres à tester
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Création du modèle de base
    base_model = RandomForestClassifier(random_state=42)
    
    # Grid Search avec validation croisée
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Entraînement
    grid_search.fit(X_train, y_train)
    
    # Affichage des meilleurs paramètres
    logger.info("Meilleurs paramètres pour Random Forest :")
    logger.info(grid_search.best_params_)
    
    return grid_search.best_estimator_

def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    Entraîne un modèle SVM avec optimisation des hyperparamètres.
    
    Args:
        X_train (np.ndarray): Features d'entraînement
        y_train (np.ndarray): Target d'entraînement
        
    Returns:
        SVC: Modèle SVM entraîné
    """
    # Définition des hyperparamètres à tester
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    
    # Création du modèle de base
    base_model = SVC(probability=True, random_state=42)
    
    # Grid Search avec validation croisée
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Entraînement
    grid_search.fit(X_train, y_train)
    
    # Affichage des meilleurs paramètres
    logger.info("Meilleurs paramètres pour SVM :")
    logger.info(grid_search.best_params_)
    
    return grid_search.best_estimator_ 