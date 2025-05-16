#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script principal pour l'analyse des réservoirs pétroliers.
"""

import os
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.logger import setup_logger
from data.data_loader import load_data
from data.data_preprocessor import preprocess_data, analyze_reservoir_characteristics
from models.model_trainer import train_models
from models.model_evaluator import evaluate_models
from visualization.data_visualizer import visualize_data
from visualization.model_visualizer import visualize_model_results
from analysis.eda import perform_eda
from analysis.advanced_analysis import perform_advanced_analysis
from analysis.report_generator import generate_analysis_report

def main():
    # Configuration du logging
    logger = setup_logger()
    logger.info("Démarrage de l'analyse des réservoirs pétroliers")

    try:
        # Chargement des données
        logger.info("Chargement des données...")
        data_path = Path("data/raw/training_data.csv")
        df = load_data(data_path)
        logger.info(f"Données chargées : {len(df)} échantillons")

        # Analyse exploratoire des données (EDA)
        logger.info("Démarrage de l'analyse exploratoire des données...")
        eda_results = perform_eda(df, output_dir="results/eda")
        logger.info("Analyse exploratoire terminée")

        # Analyse avancée
        logger.info("Démarrage de l'analyse avancée...")
        advanced_results = perform_advanced_analysis(df, output_dir="results/advanced_analysis")
        logger.info("Analyse avancée terminée")

        # Génération du rapport d'analyse
        logger.info("Génération du rapport d'analyse...")
        generate_analysis_report(eda_results, advanced_results, output_dir="results/report")
        logger.info("Rapport d'analyse généré")

        # Prétraitement des données pour la modélisation
        logger.info("Prétraitement des données...")
        X, y, X_pca = preprocess_data(df, n_components=2)
        logger.info("Prétraitement terminé")

        # Analyse des caractéristiques des réservoirs
        logger.info("Analyse des caractéristiques des réservoirs...")
        analyze_reservoir_characteristics(X, y)
        logger.info("Analyse des caractéristiques terminée")

        # Split des données
        # Conversion de y en chaîne de caractères pour éviter l'erreur avec l'Enum
        y_str = y.astype(str)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_str, test_size=0.2, random_state=42, stratify=y_str
        )
        logger.info(f"Données divisées en train ({len(X_train)} échantillons) et test ({len(X_test)} échantillons)")

        # Visualisation des données
        logger.info("Génération des visualisations...")
        visualize_data(df, output_dir="results/visualizations")
        logger.info("Visualisations générées")

        # Entraînement des modèles
        logger.info("Entraînement des modèles...")
        models = train_models(X_train, y_train)
        logger.info("Entraînement terminé")

        # Évaluation des modèles
        logger.info("Évaluation des modèles...")
        results = evaluate_models(models, X_test, y_test)
        logger.info("Évaluation terminée")

        # Visualisation des résultats
        logger.info("Génération des visualisations des résultats...")
        visualize_model_results(results, output_dir="results/model_results")
        logger.info("Visualisations des résultats générées")

        logger.info("Analyse terminée avec succès!")

    except Exception as e:
        logger.error(f"Une erreur est survenue : {str(e)}")
        raise

if __name__ == "__main__":
    main() 