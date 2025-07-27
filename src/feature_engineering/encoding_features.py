"""
Encoding Features
Geographic e material encoding con target encoding
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .base import ConfigurableFeatureEngineer


class EncodingFeatureEngineer(ConfigurableFeatureEngineer):
    """
    Feature engineer per geographic e material encoding
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("EncodingFeatures", config)
        self.target_encoding_smoothing = config.get('target_encoding_smoothing', 100) if config else 100
        
        # State per test set
        self.geo_target_means: Dict[str, Dict] = {}
        self.material_risk_scores: Dict[str, Dict] = {}
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Crea encoding features durante training"""
        self._print_if_verbose("   Creating encoding features...")
        
        df_result = df.copy()
        original_count = len(df_result.columns)
        
        # 1. Geographic encoding
        df_result = self._create_geographic_encoding(df_result, target_col)
        
        # 2. Material risk scores
        df_result = self._create_material_risk_scores(df_result, target_col)
        
        # Track features create
        new_features = list(set(df_result.columns) - set(df.columns))
        self._log_feature_creation(new_features, "geographic and material encoding")
        
        self.fitted = True
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica encoding features su test set"""
        if not self.fitted:
            raise ValueError("EncodingFeatureEngineer must be fitted before transform")
            
        df_result = df.copy()
        
        # Apply saved encodings
        df_result = self._apply_geographic_encoding(df_result)
        df_result = self._apply_material_risk_scores(df_result)
        
        return df_result
    
    def _create_geographic_encoding(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Crea geographic encoding con target encoding"""
        
        self._print_if_verbose("   Creating unified geographic encoding...")
        
        geo_cols = [col for col in df.columns if col.startswith('geo_level_')]
        
        if not geo_cols or target_col not in df.columns:
            return df
        
        for geo_col in geo_cols:
            try:
                # Calculate target statistics by geographic group
                geo_stats = df.groupby(geo_col)[target_col].agg(['mean', 'count']).fillna(0)
                global_mean = df[target_col].mean()
                
                # Target encoding with smoothing
                mapping = {}
                for geo_value in geo_stats.index:
                    count = geo_stats.loc[geo_value, 'count']
                    geo_mean = geo_stats.loc[geo_value, 'mean']
                    
                    if count > 0:
                        # Smoothed target encoding
                        smoothed_mean = (geo_mean * count + global_mean * self.target_encoding_smoothing) / (count + self.target_encoding_smoothing)
                    else:
                        smoothed_mean = global_mean
                    
                    mapping[geo_value] = smoothed_mean
                
                # Save mapping for test set
                self.geo_target_means[geo_col] = mapping
                
                # Create encoded feature
                df[f'{geo_col}_risk'] = df[geo_col].map(mapping).fillna(global_mean)
                
                self._print_if_verbose(f"     {geo_col}: {len(mapping)} levels encoded")
                
            except Exception as e:
                self._print_if_verbose(f"     Warning: Geographic encoding failed for {geo_col}: {e}")
                continue
        
        return df
    
    def _create_material_risk_scores(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Crea material risk scores"""
        
        self._print_if_verbose("   Creating material risk scores...")
        
        material_cols = ['foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type']
        
        for col in material_cols:
            if col not in df.columns or target_col not in df.columns:
                continue
                
            try:
                # Target encoding for materials
                material_risk = df.groupby(col)[target_col].mean()
                
                # Z-score relative to global mean
                global_mean = df[target_col].mean()
                global_std = df[target_col].std()
                
                if global_std > 0:
                    material_risk_zscore = (material_risk - global_mean) / global_std
                else:
                    material_risk_zscore = material_risk * 0  # All zeros if no variance
                
                # Create encoded feature
                df[f'{col}_risk_zscore'] = df[col].map(material_risk_zscore).fillna(0)
                
                # Save for test set
                self.material_risk_scores[col] = material_risk_zscore.to_dict()
                
                self._print_if_verbose(f"     {col}: Risk encoding created")
                
            except Exception as e:
                self._print_if_verbose(f"     Warning: Material risk encoding failed for {col}: {e}")
                continue
        
        return df
    
    def _apply_geographic_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica geographic encoding salvato"""
        
        global_mean = 2.0  # Default fallback
        
        for geo_col, mapping in self.geo_target_means.items():
            if geo_col in df.columns:
                df[f'{geo_col}_risk'] = df[geo_col].map(mapping).fillna(global_mean)
        
        return df
    
    def _apply_material_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica material risk scores salvati"""
        
        for material_col, mapping in self.material_risk_scores.items():
            if material_col in df.columns:
                df[f'{material_col}_risk_zscore'] = df[material_col].map(mapping).fillna(0)
        
        return df
    
    def get_geographic_encodings(self) -> Dict[str, Dict]:
        """Getter per geographic encodings (per altri moduli)"""
        return self.geo_target_means
    
    def get_material_encodings(self) -> Dict[str, Dict]:
        """Getter per material encodings (per altri moduli)"""
        return self.material_risk_scores