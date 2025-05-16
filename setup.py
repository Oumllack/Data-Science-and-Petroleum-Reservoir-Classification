"""
Configuration du package pour l'analyse des réservoirs pétroliers.
"""

from setuptools import setup, find_packages

setup(
    name="petroleum_reservoir_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "plotly>=5.3.1",
        "jinja2>=3.0.1",
        "scipy>=1.7.0",
        "jupyter>=1.0.0",
        "kaleido>=0.2.1",
        "statsmodels>=0.13.0",
    ],
    python_requires=">=3.8",
) 