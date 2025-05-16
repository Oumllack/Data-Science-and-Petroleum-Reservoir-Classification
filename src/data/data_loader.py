#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour le chargement et la validation des données.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV.
    
    Args:
        file_path (Path): Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: DataFrame contenant les données
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        pd.errors.EmptyDataError: Si le fichier est vide
        ValueError: Si les colonnes requises sont manquantes
    """
    try:
        # Vérification de l'existence du fichier
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
        
        # Chargement des données
        df = pd.read_csv(file_path)
        
        # Vérification des colonnes requises
        required_columns = [
            'Facies', 'Formation', 'Well Name', 'Depth',
            'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE',
            'NM_M', 'RELPOS'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes : {missing_columns}")
        
        # Vérification des valeurs manquantes
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning("Valeurs manquantes détectées :")
            logger.warning(missing_values[missing_values > 0])
        
        # Vérification des types de données
        logger.info("Types de données des colonnes :")
        logger.info(df.dtypes)
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error("Le fichier est vide")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {str(e)}")
        raise

def validate_data(df: pd.DataFrame) -> bool:
    """
    Valide les données chargées.
    
    Args:
        df (pd.DataFrame): DataFrame à valider
        
    Returns:
        bool: True si les données sont valides, False sinon
    """
    try:
        # Vérification des valeurs négatives pour les mesures physiques
        physical_columns = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
        for col in physical_columns:
            if (df[col] < 0).any():
                logger.warning(f"Valeurs négatives détectées dans la colonne {col}")
        
        # Vérification des valeurs aberrantes
        for col in physical_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
            if not outliers.empty:
                logger.warning(f"Valeurs aberrantes détectées dans la colonne {col}")
        
        # Vérification de la cohérence des faciès
        valid_facies = df['Facies'].unique()
        logger.info(f"Faciès uniques détectés : {valid_facies}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation des données : {str(e)}")
        return False 