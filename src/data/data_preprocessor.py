#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de prétraitement des données pour l'analyse des réservoirs pétroliers.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any

from data.reservoir_types import map_facies_to_reservoir_type, ReservoirType

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame, n_components: int = 2) -> tuple:
    """
    Prétraite les données pour la modélisation.
    
    Args:
        df (pd.DataFrame): DataFrame brut
        n_components (int): Nombre de composantes pour la PCA
        
    Returns:
        tuple: (X, y, X_pca) où X sont les features, y est la target, et X_pca sont les features réduites
    """
    try:
        # Sélection des features
        features = [
            'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE',
            'NM_M', 'RELPOS'
        ]
        
        # Extraction des features et de la target
        X = df[features].copy()
        y_facies = df['Facies'].copy()
        
        # Conversion des faciès en types de réservoirs
        y = y_facies.apply(map_facies_to_reservoir_type)
        
        # Gestion des valeurs manquantes
        X = handle_missing_values(X)
        
        # Gestion des valeurs aberrantes
        X = handle_outliers(X)
        
        # Standardisation des features
        X = standardize_features(X)
        
        # Réduction de dimensionnalité avec PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Log des informations sur la PCA
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"Variance expliquée par les composantes PCA : {explained_variance}")
        logger.info(f"Variance totale expliquée : {sum(explained_variance):.2%}")
        
        logger.info("Prétraitement terminé avec succès")
        logger.info(f"Shape des données : X={X.shape}, y={y.shape}, X_pca={X_pca.shape}")
        
        return X, y, X_pca
        
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement : {str(e)}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame avec potentiellement des valeurs manquantes
        
    Returns:
        pd.DataFrame: DataFrame sans valeurs manquantes
    """
    # Vérification des valeurs manquantes
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning("Valeurs manquantes avant traitement :")
        logger.warning(missing_values[missing_values > 0])
        
        # Pour les colonnes numériques, on remplace par la médiane
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
            
        logger.info("Valeurs manquantes remplacées par la médiane")
    
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gère les valeurs aberrantes dans le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame avec potentiellement des valeurs aberrantes
        
    Returns:
        pd.DataFrame: DataFrame avec les valeurs aberrantes traitées
    """
    # Colonnes à traiter (uniquement les mesures physiques)
    columns_to_treat = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    
    for col in columns_to_treat:
        if col in df.columns:
            # Calcul des quartiles et de l'IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Définition des bornes
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remplacement des valeurs aberrantes par les bornes
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info(f"Valeurs aberrantes traitées pour la colonne {col}")
    
    return df

def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les features numériques.
    
    Args:
        df (pd.DataFrame): DataFrame avec les features à standardiser
        
    Returns:
        pd.DataFrame: DataFrame avec les features standardisées
    """
    # Colonnes à standardiser (uniquement les mesures physiques)
    columns_to_scale = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
    
    # Création d'une copie pour ne pas modifier l'original
    df_scaled = df.copy()
    
    # Standardisation
    scaler = StandardScaler()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    
    logger.info("Features standardisées avec succès")
    
    return df_scaled

def analyze_reservoir_characteristics(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Analyse les caractéristiques des différents types de réservoirs.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Types de réservoirs
    """
    try:
        # Calcul des statistiques par type de réservoir
        stats = pd.DataFrame(X)
        stats['ReservoirType'] = y
        
        # Statistiques descriptives par type
        for reservoir_type in ReservoirType:
            type_stats = stats[stats['ReservoirType'] == reservoir_type].describe()
            logger.info(f"\nStatistiques pour {reservoir_type.name}:")
            logger.info(type_stats)
            
        # Corrélations entre features par type
        for reservoir_type in ReservoirType:
            # On sélectionne uniquement les colonnes numériques pour la corrélation
            type_data = stats[stats['ReservoirType'] == reservoir_type]
            numeric_cols = X.columns if isinstance(X, pd.DataFrame) else [col for col in stats.columns if col != 'ReservoirType']
            type_corr = type_data[numeric_cols].corr()
            logger.info(f"\nCorrélations pour {reservoir_type.name}:")
            logger.info(type_corr)
            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse des caractéristiques : {str(e)}")
        raise 