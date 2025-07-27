"""
Base classes e interfaces per feature engineering modulare
Definisce contratti comuni e utilità condivise
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Set


class BaseFeatureEngineer(ABC):
    """
    Base class per tutti i feature engineers specializzati
    Definisce il contratto comune e metodi di utilità
    """
    
    def __init__(self, name: str):
        self.name = name
        self.fitted = False
        self.created_features: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Fit sui dati di training e applica trasformazioni"""
        pass
        
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica trasformazioni sui dati di test"""
        pass
        
    def _safe_column_check(self, df: pd.DataFrame, required_cols: list) -> list:
        """Verifica sicura della presenza di colonne richieste"""
        return [col for col in required_cols if col in df.columns]
    
    def _safe_numeric_fill(self, series: pd.Series, default_value: float = 0.0) -> pd.Series:
        """Fill sicuro per valori numerici con gestione robusta"""
        if series.isna().all():
            return pd.Series([default_value] * len(series), index=series.index)
        
        # Usa mediana se disponibile, altrimenti media, altrimenti default
        fill_value = series.median()
        if pd.isna(fill_value):
            fill_value = series.mean()
            if pd.isna(fill_value):
                fill_value = default_value
                
        return series.fillna(fill_value)
    
    def _handle_inf_and_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gestione robusta di valori infiniti e NaN"""
        # Sostituisci infiniti con NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN con 0 per consistency
        df = df.fillna(0.0)
        
        return df
    
    def _log_feature_creation(self, feature_names: list, description: str = ""):
        """Log della creazione di features per debugging"""
        self.created_features.update(feature_names)
        if description:
            print(f"      {self.name}: Created {len(feature_names)} features - {description}")
        else:
            print(f"      {self.name}: Created {len(feature_names)} features")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Ritorna summary delle features create"""
        return {
            'name': self.name,
            'fitted': self.fitted,
            'features_created': len(self.created_features),
            'feature_names': list(self.created_features),
            'metadata': self.metadata
        }


class SeismicConstants:
    """Costanti domain-specific per analisi sismica"""
    
    # Age decay time constants (anni)
    AGE_DECAY_FAST = 15      # Per vulnerabilità sismica
    AGE_DECAY_MEDIUM = 20    # Per deterioramento generale
    AGE_DECAY_SLOW = 25      # Per fattori ambientali
    
    # Material decay rates
    MATERIAL_DECAY_RATES = {
        'adobe_mud': 0.8,
        'mud_mortar': 0.6,
        'cement_mortar': 0.3,
        'rc_engineered': 0.15,
        'rc_non_engineered': 0.4,
        'timber': 0.5,
        'bamboo': 0.7
    }
    
    # Weight factors per vulnerability components
    VULNERABILITY_WEIGHTS = {
        'age': 0.25,
        'height': 0.20,
        'material': 0.30,
        'geometry': 0.15,
        'density': 0.10
    }
    
    # Critical age thresholds
    AGE_THRESHOLDS = {
        'pre_1980': 45,      # Pre-modern building codes
        'pre_2000': 25,      # Pre-updated seismic codes  
        'very_old': 60       # Very old buildings
    }
    
    # Risk category thresholds
    RISK_CATEGORIES = {
        'new': 5,         # Low risk
        'recent': 15,     # Medium-low risk
        'mature': 30,     # Medium risk
        'old': 50         # High risk (>50 = very high)
    }


class FeatureValidationMixin:
    """Mixin per validazione features"""
    
    def validate_features_created(self, df_before: pd.DataFrame, df_after: pd.DataFrame, 
                                expected_min_features: int = 1) -> bool:
        """Valida che le features siano state create correttamente"""
        features_added = len(df_after.columns) - len(df_before.columns)
        
        if features_added < expected_min_features:
            print(f"WARNING: Expected at least {expected_min_features} features, got {features_added}")
            return False
            
        # Verifica che tutte le nuove colonne siano numeriche
        new_columns = set(df_after.columns) - set(df_before.columns)
        for col in new_columns:
            if not pd.api.types.is_numeric_dtype(df_after[col]):
                print(f"WARNING: New feature {col} is not numeric: {df_after[col].dtype}")
                return False
                
        return True
    
    def validate_no_data_leakage(self, train_features: Set[str], test_features: Set[str]) -> bool:
        """Valida che non ci sia data leakage tra train e test"""
        missing_in_test = train_features - test_features
        extra_in_test = test_features - train_features
        
        # Filter out expected differences (target columns that should NOT be in test)
        expected_missing = {'damage_grade', 'target', 'label', 'y'}
        critical_missing = missing_in_test - expected_missing
        
        if critical_missing:
            print(f"WARNING: Critical features missing in test: {list(critical_missing)[:5]}...")
            
        if extra_in_test:
            print(f"WARNING: Extra features in test: {list(extra_in_test)[:5]}...")
        
        # Expected missing features (like target) are OK, but critical missing or extra features are not
        return len(critical_missing) == 0 and len(extra_in_test) == 0


class FeatureEngineeringError(Exception):
    """Custom exception per errori di feature engineering"""
    pass


class ConfigurableFeatureEngineer(BaseFeatureEngineer, FeatureValidationMixin):
    """
    Feature engineer base con configurabilità e validazione
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.config = config or {}
        self.verbose = self.config.get('verbose', True)
        self.validate_output = self.config.get('validate_output', True)
        
    def _print_if_verbose(self, message: str):
        """Print condizionale basato su verbose setting"""
        if self.verbose:
            print(message)
            
    def _validate_input(self, df: pd.DataFrame, required_cols: Optional[list] = None) -> None:
        """Validazione input comune"""
        if df.empty:
            raise FeatureEngineeringError(f"{self.name}: Input DataFrame is empty")
            
        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise FeatureEngineeringError(
                    f"{self.name}: Missing required columns: {missing_cols}"
                )