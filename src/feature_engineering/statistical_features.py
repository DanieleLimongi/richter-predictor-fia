"""
Statistical Features
Genera features statistiche per gruppi geografici e materiali
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .base import ConfigurableFeatureEngineer


class StatisticalFeatureEngineer(ConfigurableFeatureEngineer):
    """
    Feature engineer per group statistics e aggregazioni
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("StatisticalFeatures", config)
        self.min_samples_per_group = config.get('min_samples_per_group', 20) if config else 20
        self.max_features_per_geo = config.get('max_features_per_geo', 3) if config else 3
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Crea statistical features durante training"""
        self._print_if_verbose("   Creating group statistical features...")
        
        df_result = df.copy()
        original_count = len(df_result.columns)
        
        # 1. Group statistical features
        df_result = self._create_group_statistics(df_result)
        
        # 2. Advanced interactions
        df_result = self._create_advanced_interactions(df_result)
        
        # 3. Aggregation features
        df_result = self._create_aggregation_features(df_result)
        
        # Track features create
        new_features = list(set(df_result.columns) - set(df.columns))
        self._log_feature_creation(new_features, "statistical and aggregation")
        
        self.fitted = True
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica statistical features su test set"""
        if not self.fitted:
            raise ValueError("StatisticalFeatureEngineer must be fitted before transform")
            
        df_result = df.copy()
        
        # Applica le stesse trasformazioni (safe per test)
        df_result = self._create_group_statistics(df_result)
        df_result = self._create_advanced_interactions(df_result)
        df_result = self._create_aggregation_features(df_result)
        
        return df_result
    
    def _create_group_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea group statistics features"""
        
        # Identify grouping columns
        geo_cols = [col for col in df.columns if col.startswith('geo_level_')]
        material_cols = ['foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Filter valid numeric columns
        valid_numeric_cols = []
        for col in numeric_cols:
            if col != 'damage_grade' and df[col].notna().sum() > self.min_samples_per_group:
                valid_numeric_cols.append(col)
        
        if len(valid_numeric_cols) < 2:
            self._print_if_verbose("      Warning: Not enough numeric columns for group statistics")
            return df
        
        self._print_if_verbose(f"      Computing statistics for {len(valid_numeric_cols)} numeric features")
        
        # Geographic group statistics
        df = self._create_geographic_group_stats(df, geo_cols, valid_numeric_cols)
        
        # Material group statistics
        df = self._create_material_group_stats(df, material_cols, valid_numeric_cols)
        
        # Cross-group interaction statistics
        df = self._create_cross_group_stats(df, geo_cols)
        
        # Outlier indicators
        df = self._create_outlier_indicators(df, geo_cols)
        
        return df
    
    def _create_geographic_group_stats(self, df: pd.DataFrame, geo_cols: List[str], 
                                     valid_numeric_cols: List[str]) -> pd.DataFrame:
        """Crea statistiche per gruppi geografici"""
        
        for geo_col in geo_cols:
            if geo_col not in df.columns:
                continue
                
            try:
                geo_groups = df.groupby(geo_col)
                
                # Focus on key features for efficiency
                key_features = ['age', 'count_floors_pre_eq', 'count_families'][:self.max_features_per_geo]
                available_key_features = [f for f in key_features if f in valid_numeric_cols]
                
                for feature in available_key_features:
                    if feature not in df.columns:
                        continue
                        
                    # Mean within geographic group
                    group_means = geo_groups[feature].transform('mean')
                    df[f'{feature}_{geo_col}_mean'] = group_means
                    
                    # Std within geographic group  
                    group_stds = geo_groups[feature].transform('std').fillna(0)
                    df[f'{feature}_{geo_col}_std'] = group_stds
                    
                    # Z-score within group
                    z_score = (df[feature] - group_means) / (group_stds + 1e-8)
                    df[f'{feature}_{geo_col}_zscore'] = z_score
                    
                    # Percentile rank within group
                    percentile_rank = geo_groups[feature].transform(lambda x: x.rank(pct=True))
                    df[f'{feature}_{geo_col}_pctrank'] = percentile_rank
                
                self._print_if_verbose(f"      {geo_col}: Group stats for {len(available_key_features)} features")
                
            except Exception as e:
                self._print_if_verbose(f"      Warning: Geographic group stats failed for {geo_col}: {e}")
                continue
        
        return df
    
    def _create_material_group_stats(self, df: pd.DataFrame, material_cols: List[str], 
                                   valid_numeric_cols: List[str]) -> pd.DataFrame:
        """Crea statistiche per gruppi materiali"""
        
        available_material_cols = [col for col in material_cols if col in df.columns]
        
        for material_col in available_material_cols:
            try:
                # Skip if too few unique values
                if df[material_col].nunique() < 3:
                    continue
                
                material_groups = df.groupby(material_col)
                
                # Focus on most important features for materials
                material_features = ['age', 'count_floors_pre_eq']
                available_material_features = [f for f in material_features if f in valid_numeric_cols]
                
                for feature in available_material_features:
                    if feature not in df.columns:
                        continue
                        
                    # Mean within material group
                    group_means = material_groups[feature].transform('mean')
                    df[f'{feature}_{material_col}_mean'] = group_means
                    
                    # Percentile rank within material group
                    percentile_rank = material_groups[feature].transform(lambda x: x.rank(pct=True))
                    df[f'{feature}_{material_col}_pctrank'] = percentile_rank
                
                self._print_if_verbose(f"      {material_col}: Group stats for {len(available_material_features)} features")
                
            except Exception as e:
                self._print_if_verbose(f"      Warning: Material group stats failed for {material_col}: {e}")
                continue
        
        return df
    
    def _create_cross_group_stats(self, df: pd.DataFrame, geo_cols: List[str]) -> pd.DataFrame:
        """Crea cross-group interaction statistics"""
        
        if len(geo_cols) < 2:
            return df
            
        try:
            # Use top 2 geographic levels
            geo1, geo2 = geo_cols[0], geo_cols[1]
            
            # Create combined grouping
            combined_group = df[geo1].astype(str) + '_' + df[geo2].astype(str)
            
            # Focus on age (most predictive feature)
            if 'age' in df.columns:
                cross_groups = df.groupby(combined_group)
                
                # Mean age in cross-group
                cross_group_mean = cross_groups['age'].transform('mean')
                df['age_cross_geo_mean'] = cross_group_mean
                
                # Deviation from cross-group average
                age_deviation = np.abs(df['age'] - cross_group_mean)
                df['age_cross_geo_deviation'] = age_deviation
                
                self._print_if_verbose(f"      Cross-group stats: {geo1} × {geo2}")
            
        except Exception as e:
            self._print_if_verbose(f"      Warning: Cross-group stats failed: {e}")
        
        return df
    
    def _create_outlier_indicators(self, df: pd.DataFrame, geo_cols: List[str]) -> pd.DataFrame:
        """Crea outlier indicators basati su distance da group centroids"""
        
        try:
            if not geo_cols or 'age' not in df.columns or 'count_floors_pre_eq' not in df.columns:
                return df
                
            primary_geo = geo_cols[0]
            geo_groups = df.groupby(primary_geo)
            
            # Calculate group centroids
            age_centroid = geo_groups['age'].transform('mean')
            floors_centroid = geo_groups['count_floors_pre_eq'].transform('mean')
            
            # Euclidean distance from group centroid
            age_dist = (df['age'] - age_centroid) ** 2
            floors_dist = (df['count_floors_pre_eq'] - floors_centroid) ** 2
            centroid_distance = np.sqrt(age_dist + floors_dist)
            
            df['group_centroid_distance'] = centroid_distance
            
            # Outlier flag (top 10% most distant)
            distance_threshold = centroid_distance.quantile(0.9)
            df['is_group_outlier'] = (centroid_distance > distance_threshold).astype(int)
            
            self._print_if_verbose(f"      Outlier indicators based on {primary_geo} groups")
            
        except Exception as e:
            self._print_if_verbose(f"      Warning: Outlier indicators failed: {e}")
        
        return df
    
    def _create_advanced_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea advanced interactions tra features"""
        
        self._print_if_verbose("   Creating advanced interactions...")
        
        # Age-based interactions
        if 'age' in df.columns:
            age_interaction_cols = ['area_percentage', 'height_percentage', 'count_floors_pre_eq', 'count_families']
            available_cols = self._safe_column_check(df, age_interaction_cols)
            
            for col in available_cols:
                # Linear interaction
                df[f'age_{col}_interaction'] = df['age'] * df[col]
                
                # Non-linear interaction
                df[f'age_sqrt_{col}'] = np.sqrt(df['age'] + 1) * df[col]
        
        # Geometric interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        important_pairs = [
            ('area_percentage', 'height_percentage'),
            ('count_floors_pre_eq', 'height_percentage'), 
            ('age', 'count_floors_pre_eq'),
            ('count_families', 'count_floors_pre_eq')
        ]
        
        for col1, col2 in important_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                # Ratio
                df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                
                # Product
                df[f'{col1}_{col2}_product'] = df[col1] * df[col2]
                
                # Difference
                df[f'{col1}_{col2}_diff'] = np.abs(df[col1] - df[col2])
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea aggregation features"""
        
        self._print_if_verbose("   Creating aggregation features...")
        
        # Binary features aggregation
        binary_cols = [col for col in df.columns if col.startswith('has_')]
        if binary_cols:
            df['total_binary_features'] = df[binary_cols].sum(axis=1)
            df['binary_diversity_score'] = (df[binary_cols] > 0).sum(axis=1)
            df['binary_density'] = df['total_binary_features'] / len(binary_cols)
        
        # Superstructure aggregations
        superstructure_cols = [col for col in df.columns if 'superstructure' in col]
        if superstructure_cols:
            df['superstructure_count'] = df[superstructure_cols].sum(axis=1)
            df['has_multiple_superstructures'] = (df['superstructure_count'] > 1).astype(int)
        
        # Secondary use aggregations
        secondary_cols = [col for col in df.columns if 'secondary_use' in col]
        if secondary_cols:
            df['secondary_use_count'] = df[secondary_cols].sum(axis=1)
            df['has_secondary_use_flag'] = (df['secondary_use_count'] > 0).astype(int)
        
        return df