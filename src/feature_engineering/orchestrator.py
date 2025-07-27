"""
Feature Engineering Orchestrator
Coordina tutti i moduli di feature engineering in una pipeline unificata
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from .base import ConfigurableFeatureEngineer, FeatureValidationMixin
from .seismic_features import SeismicFeatureEngineer
from .age_decay_models import AgeDecayModelEngineer  
from .statistical_features import StatisticalFeatureEngineer
from .polynomial_features import PolynomialFeatureEngineer
from .encoding_features import EncodingFeatureEngineer
from .binning_features import BinningFeatureEngineer


class AdvancedFeatureEngineer(ConfigurableFeatureEngineer, FeatureValidationMixin):
    """
    Orchestrator principale per feature engineering avanzato
    Coordina tutti i moduli specializzati
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("AdvancedFeatureEngineer", config)
        
        # Initialize specialized engineers
        self.engineers = {
            'seismic': SeismicFeatureEngineer(config),
            'age_decay': AgeDecayModelEngineer(config),
            'statistical': StatisticalFeatureEngineer(config),
            'encoding': EncodingFeatureEngineer(config),
            'polynomial': PolynomialFeatureEngineer(config),
            'binning': BinningFeatureEngineer(config)
        }
        
        # Processing order (dependencies matter)
        self.processing_order = [
            'seismic',      # 1. Domain knowledge base
            'statistical',  # 2. Group stats e interactions  
            'age_decay',    # 3. Age models (dopo seismic per context)
            'encoding',     # 4. Target encoding (per altri moduli)
            'polynomial',   # 5. Polynomial (dopo statistical per base features)
            'binning'       # 6. Binning (finale)
        ]
        
        # Configuration
        self.validate_each_step = config.get('validate_each_step', False) if config else False
        self.enable_cross_module_sharing = config.get('enable_cross_module_sharing', True) if config else True
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Pipeline completa di feature engineering per training"""
        
        self._print_if_verbose("Advanced Feature Engineering - Training...")
        
        df_result = df.copy()
        original_features = len(df_result.columns)
        
        # Apply each engineer in order
        for step_name in self.processing_order:
            engineer = self.engineers[step_name]
            
            df_before = df_result.copy()
            
            try:
                df_result = engineer.fit_transform(df_result, target_col)
                
                # Cross-module data sharing
                if self.enable_cross_module_sharing:
                    self._share_cross_module_data(step_name)
                
                # Optional validation
                if self.validate_each_step:
                    features_added = len(df_result.columns) - len(df_before.columns)
                    self.validate_features_created(df_before, df_result, expected_min_features=0)
                    self._print_if_verbose(f"      {step_name}: +{features_added} features validated")
                
            except Exception as e:
                self._print_if_verbose(f"      ERROR: {step_name} failed: {e}")
                # Continue with other engineers on failure
                continue
        
        # Final processing
        df_result = self._final_data_cleaning(df_result)
        
        # Track all created features
        original_columns = set(df.columns)
        self.created_features = set(df_result.columns) - original_columns
        
        total_new_features = len(self.created_features)
        self._print_if_verbose(f"      TOTAL NEW FEATURES: +{total_new_features}")
        
        self.fitted = True
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline completa di feature engineering per test"""
        
        if not self.fitted:
            raise ValueError("AdvancedFeatureEngineer must be fitted before transform!")
        
        self._print_if_verbose("Advanced Feature Engineering - Test...")
        
        df_result = df.copy()
        
        # Apply each engineer in same order
        for step_name in self.processing_order:
            engineer = self.engineers[step_name]
            
            try:
                df_result = engineer.transform(df_result)
                
            except Exception as e:
                self._print_if_verbose(f"      ERROR: {step_name} transform failed: {e}")
                # Continue with other engineers on failure
                continue
        
        # Ensure all created features are present
        df_result = self._ensure_missing_features(df_result)
        
        # Final processing
        df_result = self._final_data_cleaning(df_result)
        
        self._print_if_verbose(f"   Test features final: {len(df_result.columns)} total (all numeric)")
        
        return df_result
    
    def _share_cross_module_data(self, current_step: str):
        """Condivide dati tra moduli per migliorare integration"""
        
        if not self.enable_cross_module_sharing:
            return
        
        # Geographic encodings to seismic module
        if current_step == 'encoding' and 'seismic' in self.engineers:
            encoding_engineer = self.engineers['encoding']
            seismic_engineer = self.engineers['seismic']
            
            geo_encodings = encoding_engineer.get_geographic_encodings()
            if geo_encodings:
                seismic_engineer.set_geographic_encodings(geo_encodings)
                self._print_if_verbose("      Cross-module: Shared geographic encodings with seismic module")
    
    def _ensure_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assicura che tutte le features create siano presenti nel test set"""
        
        missing_features = self.created_features - set(df.columns)
        
        if missing_features:
            self._print_if_verbose(f"      Adding {len(missing_features)} missing features with intelligent defaults")
            
            for missing_feature in missing_features:
                # Intelligent default values based on feature type
                if 'risk' in missing_feature:
                    df[missing_feature] = 2.0  # Neutral risk
                elif 'ratio' in missing_feature:
                    df[missing_feature] = 1.0  # Neutral ratio
                elif 'binned' in missing_feature:
                    df[missing_feature] = 2.0  # Middle bin
                elif 'percentile' in missing_feature or 'pctrank' in missing_feature:
                    df[missing_feature] = 0.5  # Median percentile
                elif 'zscore' in missing_feature:
                    df[missing_feature] = 0.0  # Mean z-score
                elif 'vulnerability' in missing_feature or 'decay' in missing_feature:
                    df[missing_feature] = 0.3  # Low-medium vulnerability/decay
                elif 'quality' in missing_feature:
                    df[missing_feature] = 0.5  # Medium quality
                else:
                    df[missing_feature] = 0.0  # Safe default
        
        return df
    
    def _final_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulizia finale dei dati per compatibilità modelli"""
        
        self._print_if_verbose("   Final data cleaning for model compatibility...")
        
        # Convert non-numeric columns
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = 0.0
        
        # Handle infinite and NaN values
        df = self._handle_inf_and_nan(df)
        
        # Ensure consistent data types
        for col in df.columns:
            if df[col].dtype not in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].astype('float64')
        
        return df
    
    def get_engineering_summary(self) -> Dict:
        """Ritorna summary completo del feature engineering"""
        
        summary = {
            'orchestrator': self.get_feature_summary(),
            'total_features_created': len(self.created_features),
            'engineers': {}
        }
        
        # Summary per ogni engineer
        for name, engineer in self.engineers.items():
            if engineer.fitted:
                summary['engineers'][name] = engineer.get_feature_summary()
        
        # Processing statistics
        summary['processing_order'] = self.processing_order
        summary['config'] = {
            'validate_each_step': self.validate_each_step,
            'cross_module_sharing': self.enable_cross_module_sharing
        }
        
        return summary
    
    def get_feature_importance_by_module(self) -> Dict[str, List[str]]:
        """Raggruppa features create per modulo"""
        
        module_features = {}
        
        for name, engineer in self.engineers.items():
            if engineer.fitted:
                module_features[name] = list(engineer.created_features)
        
        return module_features
    
    def validate_pipeline_integrity(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, bool]:
        """Valida l'integrità completa della pipeline"""
        
        validation_results = {}
        
        # 1. Check no data leakage
        train_features = set(train_df.columns)
        test_features = set(test_df.columns)
        validation_results['no_data_leakage'] = self.validate_no_data_leakage(train_features, test_features)
        
        # 2. Check all features are numeric
        non_numeric_train = [col for col in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[col])]
        non_numeric_test = [col for col in test_df.columns if not pd.api.types.is_numeric_dtype(test_df[col])]
        validation_results['all_numeric'] = len(non_numeric_train) == 0 and len(non_numeric_test) == 0
        
        # 3. Check no infinite/NaN values
        has_inf_train = np.any(np.isinf(train_df.select_dtypes(include=[np.number]).values))
        has_nan_train = train_df.select_dtypes(include=[np.number]).isna().any().any()
        has_inf_test = np.any(np.isinf(test_df.select_dtypes(include=[np.number]).values))
        has_nan_test = test_df.select_dtypes(include=[np.number]).isna().any().any()
        validation_results['no_inf_nan'] = not (has_inf_train or has_nan_train or has_inf_test or has_nan_test)
        
        # 4. Check reasonable feature count
        reasonable_feature_count = 50 <= len(train_df.columns) <= 500  # Reasonable range
        validation_results['reasonable_feature_count'] = reasonable_feature_count
        
        # Summary
        all_passed = all(validation_results.values())
        validation_results['overall_valid'] = all_passed
        
        if not all_passed:
            failed_checks = [check for check, passed in validation_results.items() if not passed and check != 'overall_valid']
            self._print_if_verbose(f"   Pipeline validation FAILED: {failed_checks}")
        else:
            self._print_if_verbose("   Pipeline validation PASSED")
        
        return validation_results


# Backwards compatibility alias
class AdvancedFeatureEngineerLegacy(AdvancedFeatureEngineer):
    """Alias per backward compatibility con il vecchio file"""
    pass