"""
Seismic Domain Features
Genera features basate su domain knowledge sismico specializzato
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .base import ConfigurableFeatureEngineer, SeismicConstants


class SeismicFeatureEngineer(ConfigurableFeatureEngineer):
    """
    Feature engineer specializzato per domain knowledge sismico
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("SeismicDomain", config)
        self.vulnerability_components = []
        self.quality_indicators = []
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Crea features domain-specific per sismica durante training"""
        self._print_if_verbose("   Creating seismic domain features...")
        
        df_result = df.copy()
        original_count = len(df_result.columns)
        
        # 1. Composite Seismic Vulnerability Score
        df_result = self._create_vulnerability_score(df_result)
        
        # 2. Building Quality Index  
        df_result = self._create_quality_index(df_result)
        
        # 3. Location Risk Density (se geographic encoding è disponibile)
        if hasattr(self, 'geo_target_means') and self.geo_target_means:
            df_result = self._create_location_risk_density(df_result)
        
        # 4. Basic seismic domain features
        df_result = self._create_basic_seismic_features(df_result)
        
        # Track features create
        new_features = list(set(df_result.columns) - set(df.columns))
        self._log_feature_creation(new_features, "seismic domain knowledge")
        
        self.fitted = True
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica features seismiche su test set"""
        if not self.fitted:
            raise ValueError("SeismicFeatureEngineer must be fitted before transform")
            
        df_result = df.copy()
        
        # Applica le stesse trasformazioni (safe per test)
        df_result = self._create_vulnerability_score(df_result)
        df_result = self._create_quality_index(df_result)
        df_result = self._create_basic_seismic_features(df_result)
        
        return df_result
    
    def _create_vulnerability_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea Seismic Vulnerability Score multi-component"""
        
        vulnerability_components = []
        
        # Age vulnerability (deterioramento strutturale)
        if 'age' in df.columns:
            age_data = self._safe_numeric_fill(df['age'], 25)
            age_factor = 1 - np.exp(-age_data / SeismicConstants.AGE_DECAY_MEDIUM)
            vulnerability_components.append(('age_vulnerability', age_factor, SeismicConstants.VULNERABILITY_WEIGHTS['age']))
            df['age_vulnerability'] = age_factor
        
        # Height vulnerability (effetto altezza su stabilità)
        if 'count_floors_pre_eq' in df.columns:
            floors_data = self._safe_numeric_fill(df['count_floors_pre_eq'], 2)
            floor_factor = np.where(floors_data <= 3, 
                                  floors_data / 10,  # Linear per bassi
                                  0.3 + (floors_data - 3) * 0.15)  # Exponential per alti
            vulnerability_components.append(('height_vulnerability', floor_factor, SeismicConstants.VULNERABILITY_WEIGHTS['height']))
            df['height_vulnerability'] = floor_factor
        
        # Material vulnerability
        material_risk = self._calculate_material_vulnerability(df)
        if material_risk is not None:
            vulnerability_components.append(('material_vulnerability', material_risk, SeismicConstants.VULNERABILITY_WEIGHTS['material']))
            df['material_vulnerability'] = material_risk
        
        # Geometry vulnerability
        geometry_risk = self._calculate_geometry_vulnerability(df)
        if geometry_risk is not None:
            vulnerability_components.append(('geometry_vulnerability', geometry_risk, SeismicConstants.VULNERABILITY_WEIGHTS['geometry']))
            df['geometry_vulnerability'] = geometry_risk
        
        # Density vulnerability
        density_risk = self._calculate_density_vulnerability(df)
        if density_risk is not None:
            vulnerability_components.append(('density_vulnerability', density_risk, SeismicConstants.VULNERABILITY_WEIGHTS['density']))
            df['density_vulnerability'] = density_risk
        
        # Combina in score finale
        if vulnerability_components:
            weighted_sum = 0
            total_weight = 0
            
            for comp_name, comp_values, weight in vulnerability_components:
                weighted_sum += comp_values * weight
                total_weight += weight
            
            if total_weight > 0:
                df['seismic_vulnerability_score'] = weighted_sum / total_weight
                self._print_if_verbose(f"      Created seismic vulnerability score from {len(vulnerability_components)} components")
        
        return df
    
    def _calculate_material_vulnerability(self, df: pd.DataFrame) -> np.ndarray:
        """Calcola vulnerabilità materiali"""
        if not any(col in df.columns for col in ['foundation_type', 'roof_type']):
            return None
        
        material_risk = np.zeros(len(df))
        
        # Foundation weakness mapping
        weakness_mapping = {
            'has_superstructure_adobe_mud': 0.4,
            'has_superstructure_mud_mortar_stone': 0.3,
            'has_superstructure_mud_mortar_brick': 0.2
        }
        
        for col, risk_factor in weakness_mapping.items():
            if col in df.columns:
                material_risk += df[col] * risk_factor
        
        return material_risk
    
    def _calculate_geometry_vulnerability(self, df: pd.DataFrame) -> np.ndarray:
        """Calcola vulnerabilità geometrica"""
        required_cols = ['area_percentage', 'height_percentage']
        if not all(col in df.columns for col in required_cols):
            return None
        
        area_data = self._safe_numeric_fill(df['area_percentage'], 50)
        height_data = self._safe_numeric_fill(df['height_percentage'], 40)
        
        # Aspect ratio instability
        aspect_ratio = height_data / (area_data + 1e-8)
        geometry_risk = np.where(aspect_ratio > 2, (aspect_ratio - 2) * 0.1, 0)
        
        # Volume amplification (edifici grandi = più energia sismica)
        volume_proxy = area_data * height_data
        volume_risk = np.tanh(volume_proxy / 1000) * 0.2
        
        return geometry_risk + volume_risk
    
    def _calculate_density_vulnerability(self, df: pd.DataFrame) -> np.ndarray:
        """Calcola vulnerabilità densità popolazione"""
        required_cols = ['count_families', 'area_percentage']
        if not all(col in df.columns for col in required_cols):
            return None
        
        families_data = self._safe_numeric_fill(df['count_families'], 1)
        area_data = self._safe_numeric_fill(df['area_percentage'], 50)
        
        density = families_data / (area_data + 1e-8)
        return np.tanh(density / 5) * 0.1  # Saturating function
    
    def _create_quality_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea Building Quality Index (0=poor, 1=excellent)"""
        
        quality_indicators = []
        
        # Foundation quality
        foundation_quality = self._calculate_foundation_quality(df)
        quality_indicators.append(('foundation_quality', foundation_quality))
        df['foundation_quality'] = foundation_quality
        
        # Structural complexity quality
        structural_quality = self._calculate_structural_quality(df)
        if structural_quality is not None:
            quality_indicators.append(('structural_quality', structural_quality))
            df['structural_quality'] = structural_quality
        
        # Age quality
        if 'age' in df.columns:
            age_data = self._safe_numeric_fill(df['age'], 25)
            age_quality = np.maximum(0.1, 1 - age_data / 100)  # Linear decay, min 0.1
            quality_indicators.append(('age_quality', age_quality))
            df['age_quality'] = age_quality
        
        # Overall building quality
        if quality_indicators:
            quality_values = [values for _, values in quality_indicators]
            df['building_quality_index'] = np.mean(quality_values, axis=0)
            self._print_if_verbose(f"      Created building quality index from {len(quality_indicators)} indicators")
        
        return df
    
    def _calculate_foundation_quality(self, df: pd.DataFrame) -> np.ndarray:
        """Calcola qualità fondazioni"""
        foundation_quality = np.ones(len(df)) * 0.5  # Default medium
        
        # Better materials increase quality
        quality_bonuses = {
            'has_superstructure_rc_engineered': 0.4,
            'has_superstructure_rc_non_engineered': 0.2,
            'has_superstructure_cement_mortar_stone': 0.1
        }
        
        for col, bonus in quality_bonuses.items():
            if col in df.columns:
                foundation_quality += df[col] * bonus
        
        return np.clip(foundation_quality, 0.1, 1.0)
    
    def _calculate_structural_quality(self, df: pd.DataFrame) -> np.ndarray:
        """Calcola qualità strutturale"""
        structure_cols = [col for col in df.columns if 'superstructure' in col]
        if not structure_cols:
            return None
        
        complexity = df[structure_cols].sum(axis=1)
        
        # Optimal complexity = 1-2, penalty per 0 o >3
        complexity_quality = np.where(complexity == 0, 0.1,
                                    np.where(complexity <= 2, 0.8,
                                           np.maximum(0.2, 0.8 - (complexity - 2) * 0.1)))
        return complexity_quality
    
    def _create_location_risk_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea Location Risk Density da geographic encoding"""
        geo_cols = [col for col in df.columns if col.startswith('geo_level_')]
        
        if not geo_cols or not hasattr(self, 'geo_target_means'):
            return df
        
        location_risks = []
        for geo_col in geo_cols:
            if geo_col in self.geo_target_means:
                geo_risk = df[geo_col].map(self.geo_target_means[geo_col]).fillna(2.0)
                location_risks.append(geo_risk)
        
        if location_risks:
            # Media pesata (più dettagliato = peso maggiore)
            weights = [1.0, 1.5, 2.0][:len(location_risks)]
            weighted_location_risk = np.average(location_risks, axis=0, weights=weights)
            df['location_risk_density'] = weighted_location_risk
            self._print_if_verbose(f"      Created location risk density from {len(location_risks)} geo levels")
        
        return df
    
    def _create_basic_seismic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features seismiche base"""
        
        # 1. Building Vulnerability Index
        if 'age' in df.columns and 'count_floors_pre_eq' in df.columns:
            age_data = self._safe_numeric_fill(df['age'], 25)
            floors_data = self._safe_numeric_fill(df['count_floors_pre_eq'], 2)
            df['building_vulnerability_index'] = age_data * 0.4 + floors_data * 0.6
        
        # 2. Structural complexity features
        structure_cols = [col for col in df.columns if 'superstructure' in col]
        if structure_cols:
            df['structural_complexity'] = df[structure_cols].sum(axis=1)
            df['has_mixed_structure'] = (df['structural_complexity'] > 2).astype(int)
            
            # Dangerous material combinations
            if 'has_superstructure_adobe_mud' in df.columns and 'has_superstructure_mud_mortar_stone' in df.columns:
                df['has_weak_materials'] = (
                    df['has_superstructure_adobe_mud'] | 
                    df['has_superstructure_mud_mortar_stone']
                ).astype(int)
        
        # 3. Size-based risk factors
        if 'area_percentage' in df.columns and 'height_percentage' in df.columns:
            area_data = self._safe_numeric_fill(df['area_percentage'], 50)
            height_data = self._safe_numeric_fill(df['height_percentage'], 40)
            
            df['aspect_ratio'] = height_data / (area_data + 1e-8)
            df['building_volume_proxy'] = area_data * height_data
            
            # Size category binning
            df['size_category'] = pd.cut(
                area_data, 
                bins=[0, 33, 66, 100], 
                labels=[0, 1, 2]
            ).astype(float)
        
        # 4. Population density risk
        if 'count_families' in df.columns and 'area_percentage' in df.columns:
            families_data = self._safe_numeric_fill(df['count_families'], 1)
            area_data = self._safe_numeric_fill(df['area_percentage'], 50)
            
            df['family_density'] = families_data / (area_data + 1e-8)
            df['overcrowding_risk'] = (df['family_density'] > df['family_density'].quantile(0.8)).astype(int)
        
        return df
    
    def set_geographic_encodings(self, geo_target_means: Dict):
        """Setter per geographic encodings da altri moduli"""
        self.geo_target_means = geo_target_means