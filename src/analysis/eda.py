#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyse exploratoire des données (EDA) pour l'analyse des réservoirs pétroliers.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json
from data.reservoir_types import ReservoirType, map_facies_to_reservoir_type

logger = logging.getLogger(__name__)

def perform_eda(df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Effectue une analyse exploratoire complète des données.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        output_dir (str): Répertoire de sortie pour les résultats
        
    Returns:
        Dict: Résultats de l'analyse
    """
    results = {}
    
    try:
        # Analyse statistique descriptive
        results['descriptive_stats'] = analyze_descriptive_stats(df)
        
        # Analyse de la qualité des données
        results['data_quality'] = analyze_data_quality(df)
        
        # Analyse des corrélations
        results['correlations'] = analyze_correlations(df)
        
        # Analyse des distributions
        results['distributions'] = analyze_distributions(df)
        
        # Analyse des relations entre variables
        results['relationships'] = analyze_relationships(df)
        
        # Analyse des caractéristiques par type de réservoir
        results['reservoir_analysis'] = analyze_reservoir_types(df)
        
        # Sauvegarde des résultats
        save_analysis_results(results, output_dir)
        
        logger.info("Analyse exploratoire terminée avec succès")
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse exploratoire : {str(e)}")
        raise

def analyze_descriptive_stats(df: pd.DataFrame) -> Dict:
    """
    Analyse statistique descriptive des données.
    """
    stats_dict = {}
    
    # Statistiques de base
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_dict['basic_stats'] = df[numeric_cols].describe().to_dict()
    
    # Statistiques par formation
    stats_dict['formation_stats'] = df.groupby('Formation')[numeric_cols].agg(['mean', 'std', 'min', 'max']).to_dict()
    
    # Statistiques par type de réservoir
    df['ReservoirType'] = df['Facies'].apply(lambda x: str(map_facies_to_reservoir_type(x)))
    stats_dict['reservoir_stats'] = df.groupby('ReservoirType')[numeric_cols].agg(['mean', 'std', 'min', 'max']).to_dict()
    
    return stats_dict

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Analyse de la qualité des données.
    """
    quality_dict = {}
    
    # Valeurs manquantes
    quality_dict['missing_values'] = df.isnull().sum().to_dict()
    
    # Valeurs uniques
    quality_dict['unique_values'] = df.nunique().to_dict()
    
    # Valeurs aberrantes (z-score > 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers[col] = int(np.sum(z_scores > 3))
    quality_dict['outliers'] = outliers
    
    return quality_dict

def analyze_correlations(df: pd.DataFrame) -> Dict:
    """
    Analyse des corrélations entre variables.
    """
    corr_dict = {}
    
    # Corrélations de Pearson
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_dict['pearson'] = df[numeric_cols].corr(method='pearson').to_dict()
    
    # Corrélations de Spearman
    corr_dict['spearman'] = df[numeric_cols].corr(method='spearman').to_dict()
    
    # Corrélations par type de réservoir
    df['ReservoirType'] = df['Facies'].apply(lambda x: str(map_facies_to_reservoir_type(x)))
    reservoir_corrs = {}
    for reservoir_type in df['ReservoirType'].unique():
        reservoir_data = df[df['ReservoirType'] == reservoir_type][numeric_cols]
        reservoir_corrs[reservoir_type] = reservoir_data.corr().to_dict()
    corr_dict['by_reservoir_type'] = reservoir_corrs
    
    return corr_dict

def analyze_distributions(df: pd.DataFrame) -> Dict:
    """
    Analyse des distributions des variables.
    """
    dist_dict = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        dist_dict[col] = {
            'skewness': float(stats.skew(df[col].dropna())),
            'kurtosis': float(stats.kurtosis(df[col].dropna())),
            'normality_test': {
                'statistic': float(stats.normaltest(df[col].dropna())[0]),
                'p_value': float(stats.normaltest(df[col].dropna())[1])
            }
        }
    
    return dist_dict

def analyze_relationships(df: pd.DataFrame) -> Dict:
    """
    Analyse des relations entre variables.
    """
    rel_dict = {}
    
    # ANOVA pour chaque variable numérique par type de réservoir
    df['ReservoirType'] = df['Facies'].apply(lambda x: str(map_facies_to_reservoir_type(x)))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        groups = [group for _, group in df.groupby('ReservoirType')[col]]
        f_stat, p_val = stats.f_oneway(*groups)
        rel_dict[col] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_val)
        }
    
    return rel_dict

def analyze_reservoir_types(df: pd.DataFrame) -> Dict:
    """
    Analyse spécifique des types de réservoirs.
    """
    reservoir_dict = {}
    
    # Distribution des types de réservoirs
    df['ReservoirType'] = df['Facies'].apply(lambda x: str(map_facies_to_reservoir_type(x)))
    reservoir_dict['distribution'] = df['ReservoirType'].value_counts().to_dict()
    
    # Caractéristiques moyennes par type
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    reservoir_dict['characteristics'] = df.groupby('ReservoirType')[numeric_cols].mean().to_dict()
    
    # Analyse des formations par type
    reservoir_dict['formations'] = df.groupby(['ReservoirType', 'Formation']).size().to_dict()
    
    return reservoir_dict

def dict_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): dict_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [dict_keys_to_str(i) for i in obj]
    else:
        return obj

def save_analysis_results(results: Dict, output_dir: str) -> None:
    """
    Sauvegarde les résultats de l'analyse dans des fichiers JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Correction des clés tuple pour JSON
    if 'reservoir_analysis' in results and 'formations' in results['reservoir_analysis']:
        formations = results['reservoir_analysis']['formations']
        results['reservoir_analysis']['formations'] = {str(k): v for k, v in formations.items()}

    # Sauvegarde des résultats détaillés
    with open(output_path / 'eda_results.json', 'w', encoding='utf-8') as f:
        json.dump(dict_keys_to_str(results), f, indent=4, ensure_ascii=False)
    
    # Génération d'un rapport sommaire
    summary = {
        'nombre_echantillons': len(results['descriptive_stats']['basic_stats']),
        'qualite_donnees': {
            'valeurs_manquantes': sum(results['data_quality']['missing_values'].values()),
            'valeurs_aberrantes': sum(results['data_quality']['outliers'].values())
        },
        'types_reservoirs': results['reservoir_analysis']['distribution']
    }
    
    with open(output_path / 'eda_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Résultats de l'analyse sauvegardés dans {output_dir}") 