#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module définissant les types de réservoirs pétroliers et leurs caractéristiques.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple

class ReservoirType(Enum):
    """
    Types de réservoirs pétroliers basés sur leurs caractéristiques géologiques.
    """
    SANDSTONE = 1  # Réservoir sableux
    CARBONATE = 2  # Réservoir carbonaté
    SHALE = 3      # Réservoir de schiste
    TIGHT = 4      # Réservoir compact
    UNCONVENTIONAL = 5  # Réservoir non conventionnel

@dataclass
class ReservoirCharacteristics:
    """
    Caractéristiques typiques d'un type de réservoir.
    """
    porosity_range: Tuple[float, float]  # Plage de porosité typique
    permeability_range: Tuple[float, float]  # Plage de perméabilité typique
    gr_range: Tuple[float, float]  # Plage de Gamma Ray typique
    ild_range: Tuple[float, float]  # Plage de résistivité typique
    extraction_methods: List[str]  # Méthodes d'extraction recommandées

# Définition des caractéristiques pour chaque type de réservoir
RESERVOIR_CHARACTERISTICS: Dict[ReservoirType, ReservoirCharacteristics] = {
    ReservoirType.SANDSTONE: ReservoirCharacteristics(
        porosity_range=(0.15, 0.35),  # 15-35% de porosité
        permeability_range=(100, 1000),  # 100-1000 mD
        gr_range=(20, 60),  # Faible radioactivité
        ild_range=(0.1, 1.0),  # Résistivité modérée
        extraction_methods=["Production conventionnelle", "Injection d'eau"]
    ),
    ReservoirType.CARBONATE: ReservoirCharacteristics(
        porosity_range=(0.05, 0.25),  # 5-25% de porosité
        permeability_range=(1, 100),  # 1-100 mD
        gr_range=(10, 40),  # Très faible radioactivité
        ild_range=(0.5, 2.0),  # Haute résistivité
        extraction_methods=["Acidification", "Fracturation hydraulique"]
    ),
    ReservoirType.SHALE: ReservoirCharacteristics(
        porosity_range=(0.02, 0.10),  # 2-10% de porosité
        permeability_range=(0.001, 0.1),  # 0.001-0.1 mD
        gr_range=(60, 150),  # Haute radioactivité
        ild_range=(0.01, 0.1),  # Très basse résistivité
        extraction_methods=["Fracturation hydraulique", "Forage horizontal"]
    ),
    ReservoirType.TIGHT: ReservoirCharacteristics(
        porosity_range=(0.03, 0.12),  # 3-12% de porosité
        permeability_range=(0.01, 1.0),  # 0.01-1 mD
        gr_range=(30, 80),  # Radioactivité modérée
        ild_range=(0.1, 0.5),  # Résistivité basse
        extraction_methods=["Fracturation hydraulique", "Stimulation acide"]
    ),
    ReservoirType.UNCONVENTIONAL: ReservoirCharacteristics(
        porosity_range=(0.01, 0.08),  # 1-8% de porosité
        permeability_range=(0.0001, 0.01),  # 0.0001-0.01 mD
        gr_range=(40, 120),  # Radioactivité variable
        ild_range=(0.001, 0.1),  # Très basse résistivité
        extraction_methods=["Fracturation hydraulique", "Forage horizontal", "Injection de CO2"]
    )
}

def map_facies_to_reservoir_type(facies: int) -> ReservoirType:
    """
    Mappe un numéro de faciès vers un type de réservoir.
    
    Args:
        facies (int): Numéro de faciès (1-9)
        
    Returns:
        ReservoirType: Type de réservoir correspondant
    """
    # Cette fonction doit être adaptée en fonction de la correspondance réelle
    # entre les faciès et les types de réservoirs dans vos données
    mapping = {
        1: ReservoirType.SANDSTONE,
        2: ReservoirType.SANDSTONE,
        3: ReservoirType.SHALE,
        4: ReservoirType.CARBONATE,
        5: ReservoirType.CARBONATE,
        6: ReservoirType.TIGHT,
        7: ReservoirType.TIGHT,
        8: ReservoirType.UNCONVENTIONAL,
        9: ReservoirType.UNCONVENTIONAL
    }
    return mapping.get(facies, ReservoirType.UNCONVENTIONAL)

def get_extraction_methods(reservoir_type: ReservoirType) -> List[str]:
    """
    Retourne les méthodes d'extraction recommandées pour un type de réservoir.
    
    Args:
        reservoir_type (ReservoirType): Type de réservoir
        
    Returns:
        List[str]: Liste des méthodes d'extraction recommandées
    """
    return RESERVOIR_CHARACTERISTICS[reservoir_type].extraction_methods 