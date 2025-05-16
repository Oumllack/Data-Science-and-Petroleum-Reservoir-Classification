#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la configuration du logging.
"""

import logging
from pathlib import Path
import sys

def setup_logger() -> logging.Logger:
    """
    Configure le système de logging.
    
    Returns:
        logging.Logger: Logger configuré
    """
    # Création du répertoire pour les logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuration du logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Format du logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour le fichier
    file_handler = logging.FileHandler(
        log_dir / "pipeline.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger 