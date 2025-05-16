"""
Package d'analyse pour l'analyse des réservoirs pétroliers.
"""

from .eda import perform_eda
from .advanced_analysis import perform_advanced_analysis
from .report_generator import generate_analysis_report

__all__ = [
    'perform_eda',
    'perform_advanced_analysis',
    'generate_analysis_report'
] 