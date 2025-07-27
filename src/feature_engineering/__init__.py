"""
Feature Engineering Module
Modular advanced feature engineering per Richter Predictor

Usage:
    from feature_engineering import AdvancedFeatureEngineer
    
    engineer = AdvancedFeatureEngineer()
    train_enhanced = engineer.fit_transform(train_df)
    test_enhanced = engineer.transform(test_df)
"""

# Main orchestrator (public API)
from .orchestrator import AdvancedFeatureEngineer

# Specialized engineers (for advanced usage)
from .seismic_features import SeismicFeatureEngineer
from .age_decay_models import AgeDecayModelEngineer
from .statistical_features import StatisticalFeatureEngineer
from .polynomial_features import PolynomialFeatureEngineer
from .encoding_features import EncodingFeatureEngineer
from .binning_features import BinningFeatureEngineer

# Base classes (for extensions)
from .base import BaseFeatureEngineer, ConfigurableFeatureEngineer, SeismicConstants


__all__ = [
    # Main API
    'AdvancedFeatureEngineer',
    
    # Specialized engineers
    'SeismicFeatureEngineer',
    'AgeDecayModelEngineer', 
    'StatisticalFeatureEngineer',
    'PolynomialFeatureEngineer',
    'EncodingFeatureEngineer',
    'BinningFeatureEngineer',
    
    # Base classes
    'BaseFeatureEngineer',
    'ConfigurableFeatureEngineer',
    'SeismicConstants',
    
]

# Version info
__version__ = '2.1.0'  # Bumped version after removing legacy code
__author__ = 'Claude Code'
__description__ = 'Modular advanced feature engineering for seismic damage prediction'
