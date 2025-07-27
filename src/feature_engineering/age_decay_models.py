"""
Age Decay Models
Modelli avanzati per catturare il deterioramento strutturale nel tempo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base import ConfigurableFeatureEngineer, SeismicConstants


class AgeDecayModelEngineer(ConfigurableFeatureEngineer):
    """
    Feature engineer specializzato per modelli di deterioramento età
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("AgeDecayModels", config)
        self.material_decay_rates = SeismicConstants.MATERIAL_DECAY_RATES
        self.age_thresholds = SeismicConstants.AGE_THRESHOLDS
        self.risk_categories = SeismicConstants.RISK_CATEGORIES
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Crea modelli di decay dell'età durante training"""
        self._print_if_verbose("   Creating advanced age decay models...")
        
        if 'age' not in df.columns:
            self._print_if_verbose("      Warning: Age column not found, skipping age decay models")
            return df
        
        df_result = df.copy()
        age_data = self._safe_numeric_fill(df_result['age'], 25)
        
        # 1. Multi-phase decay model
        df_result = self._create_multiphase_decay(df_result, age_data)
        
        # 2. Material-specific decay rates
        df_result = self._create_material_specific_decay(df_result, age_data)
        
        # 3. Environmental exposure decay
        df_result = self._create_environmental_decay(df_result, age_data)
        
        # 4. Seismic vulnerability age amplification
        df_result = self._create_seismic_vulnerability_decay(df_result, age_data)
        
        # 5. Maintenance proxy decay
        df_result = self._create_maintenance_proxy_decay(df_result, age_data)
        
        # 6. Age interaction with building complexity
        df_result = self._create_complexity_interaction_decay(df_result, age_data)
        
        # 7. Age percentile features
        df_result = self._create_age_percentile_features(df_result, age_data)
        
        # Track features create
        new_features = [col for col in df_result.columns if col.startswith('age_') and col != 'age']
        new_features.extend([col for col in df_result.columns if 'maintenance' in col or 'structural_complexity' in col])
        created_features = list(set(new_features))
        
        self._log_feature_creation(created_features, f"{len(created_features)} age decay models")
        self._print_if_verbose(f"      Age decay feature types: multi-phase, material-specific, environmental, seismic, maintenance, complexity, percentile")
        
        self.fitted = True
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica age decay models su test set"""
        if not self.fitted:
            raise ValueError("AgeDecayModelEngineer must be fitted before transform")
            
        if 'age' not in df.columns:
            return df
            
        df_result = df.copy()
        age_data = self._safe_numeric_fill(df_result['age'], 25)
        
        # Applica gli stessi modelli
        df_result = self._create_multiphase_decay(df_result, age_data)
        df_result = self._create_material_specific_decay(df_result, age_data)
        df_result = self._create_environmental_decay(df_result, age_data)
        df_result = self._create_seismic_vulnerability_decay(df_result, age_data)
        df_result = self._create_maintenance_proxy_decay(df_result, age_data)
        df_result = self._create_complexity_interaction_decay(df_result, age_data)
        df_result = self._create_age_percentile_features(df_result, age_data)
        
        return df_result
    
    def _create_multiphase_decay(self, df: pd.DataFrame, age_data: pd.Series) -> pd.DataFrame:
        """
        Multi-phase decay model:
        - Fase 1: Deterioramento iniziale (0-10 anni) - lento
        - Fase 2: Deterioramento accelerato (10-30 anni) - rapido  
        - Fase 3: Deterioramento stabilizzato (30+ anni) - lento ma continuo
        """
        
        phase1_decay = np.where(age_data <= 10, 
                               age_data / 100,  # Linear decay molto lento
                               0.1)  # Max fase 1
        
        phase2_decay = np.where((age_data > 10) & (age_data <= 30),
                               0.1 + (age_data - 10) * 0.03,  # Accelerated decay
                               np.where(age_data > 30, 0.7, 0.1))  # Max o min
        
        phase3_decay = np.where(age_data > 30,
                               0.7 + (age_data - 30) * 0.01,  # Slow continued decay
                               0)
        
        df['age_phase1_decay'] = phase1_decay  
        df['age_phase2_decay'] = phase2_decay
        df['age_phase3_decay'] = phase3_decay
        df['age_total_decay'] = phase1_decay + phase2_decay + phase3_decay
        
        self._print_if_verbose(f"      Multi-phase decay: 4 models (phases 1-3 + total)")
        return df
    
    def _create_material_specific_decay(self, df: pd.DataFrame, age_data: pd.Series) -> pd.DataFrame:
        """Material-specific decay rates"""
        
        primary_material_decay = np.ones(len(df)) * 0.4  # Default decay rate
        
        for material, decay_rate in self.material_decay_rates.items():
            material_col = f'has_superstructure_{material}'
            if material_col in df.columns:
                material_mask = df[material_col] == 1
                if material_mask.any():
                    material_specific_decay = 1 - np.exp(-age_data * decay_rate / 30)
                    primary_material_decay = np.where(material_mask, 
                                                    material_specific_decay, 
                                                    primary_material_decay)
        
        df['age_material_specific_decay'] = primary_material_decay
        self._print_if_verbose(f"      Material-specific decay: 1 model")
        return df
    
    def _create_environmental_decay(self, df: pd.DataFrame, age_data: pd.Series) -> pd.DataFrame:
        """Environmental exposure decay factors"""
        
        environmental_factors = []
        
        # Height exposure
        if 'count_floors_pre_eq' in df.columns:
            floors_data = self._safe_numeric_fill(df['count_floors_pre_eq'], 2)
            height_exposure = np.tanh(floors_data / 3) * 0.3
            environmental_factors.append(('height_exposure', height_exposure))
        
        # Density stress
        if 'count_families' in df.columns:
            families_data = self._safe_numeric_fill(df['count_families'], 1)
            density_stress = np.tanh(families_data / 2) * 0.2
            environmental_factors.append(('density_stress', density_stress))
        
        # Area stress
        if 'area_percentage' in df.columns:
            area_data = self._safe_numeric_fill(df['area_percentage'], 50)
            area_stress = np.tanh(area_data / 50) * 0.15
            environmental_factors.append(('area_stress', area_stress))
        
        if environmental_factors:
            total_environmental_factor = sum(factor[1] for factor in environmental_factors)
            
            # Combined environmental decay
            environmental_decay = (1 - np.exp(-age_data / SeismicConstants.AGE_DECAY_SLOW)) * (1 + total_environmental_factor)
            df['age_environmental_decay'] = environmental_decay
            
            # Individual environmental components
            for name, factor in environmental_factors:
                age_component_decay = (1 - np.exp(-age_data / 30)) * (1 + factor)
                df[f'age_{name}_decay'] = age_component_decay
            
            self._print_if_verbose(f"      Environmental decay: {1 + len(environmental_factors)} models")
        
        return df
    
    def _create_seismic_vulnerability_decay(self, df: pd.DataFrame, age_data: pd.Series) -> pd.DataFrame:
        """Seismic vulnerability age amplification"""
        
        # Base seismic vulnerability increases exponentially with age
        seismic_age_vulnerability = 1 - np.exp(-age_data / SeismicConstants.AGE_DECAY_FAST)
        df['age_seismic_vulnerability'] = seismic_age_vulnerability
        
        # Critical age thresholds
        df['age_is_pre_1980'] = (age_data >= self.age_thresholds['pre_1980']).astype(int)
        df['age_is_pre_2000'] = (age_data >= self.age_thresholds['pre_2000']).astype(int)
        df['age_is_very_old'] = (age_data >= self.age_thresholds['very_old']).astype(int)
        
        # Age-based risk categories
        age_risk_category = np.where(age_data <= self.risk_categories['new'], 0,      # New
                           np.where(age_data <= self.risk_categories['recent'], 1,    # Recent  
                           np.where(age_data <= self.risk_categories['mature'], 2,    # Mature
                           np.where(age_data <= self.risk_categories['old'], 3, 4))))  # Old/Very old
        
        df['age_risk_category'] = age_risk_category
        
        self._print_if_verbose(f"      Seismic vulnerability: 5 models (vulnerability + 4 thresholds)")
        return df
    
    def _create_maintenance_proxy_decay(self, df: pd.DataFrame, age_data: pd.Series) -> pd.DataFrame:
        """Maintenance proxy decay based on materials"""
        
        maintenance_proxy = np.ones(len(df)) * 0.5  # Default maintenance level
        
        # Better materials = better maintenance
        maintenance_bonuses = {
            'has_superstructure_rc_engineered': 0.3,
            'has_superstructure_cement_mortar_stone': 0.2,
            'has_superstructure_cement_mortar_brick': 0.15
        }
        
        for col, bonus in maintenance_bonuses.items():
            if col in df.columns:
                maintenance_proxy += df[col] * bonus
        
        # Poor materials = poor maintenance
        maintenance_penalties = {
            'has_superstructure_adobe_mud': 0.3,
            'has_superstructure_mud_mortar_stone': 0.2
        }
        
        for col, penalty in maintenance_penalties.items():
            if col in df.columns:
                maintenance_proxy -= df[col] * penalty
        
        # Clip to reasonable range
        maintenance_proxy = np.clip(maintenance_proxy, 0.1, 1.0)
        
        # Decay modified by maintenance
        maintenance_adjusted_decay = (1 - np.exp(-age_data / SeismicConstants.AGE_DECAY_MEDIUM)) / maintenance_proxy
        df['age_maintenance_adjusted_decay'] = maintenance_adjusted_decay
        df['estimated_maintenance_level'] = maintenance_proxy
        
        self._print_if_verbose(f"      Maintenance proxy: 2 models (adjusted decay + maintenance level)")
        return df
    
    def _create_complexity_interaction_decay(self, df: pd.DataFrame, age_data: pd.Series) -> pd.DataFrame:
        """Age interaction with building complexity"""
        
        structure_cols = [col for col in df.columns if 'superstructure' in col]
        if not structure_cols:
            return df
        
        structural_complexity = df[structure_cols].sum(axis=1)
        
        # Complex buildings age worse (more failure points)
        complexity_factor = 1 + (structural_complexity * 0.1)  # 10% more decay per additional structure type
        
        complexity_adjusted_decay = (1 - np.exp(-age_data / SeismicConstants.AGE_DECAY_SLOW)) * complexity_factor
        df['age_complexity_adjusted_decay'] = complexity_adjusted_decay
        df['structural_complexity_factor'] = complexity_factor
        
        self._print_if_verbose(f"      Complexity interaction: 2 models")
        return df
    
    def _create_age_percentile_features(self, df: pd.DataFrame, age_data: pd.Series) -> pd.DataFrame:
        """Age percentile features (relative age in context)"""
        
        df['age_percentile_rank'] = age_data.rank(pct=True)
        df['age_is_top_quartile_old'] = (df['age_percentile_rank'] > 0.75).astype(int)
        df['age_is_bottom_quartile_new'] = (df['age_percentile_rank'] < 0.25).astype(int)
        
        self._print_if_verbose(f"      Age percentiles: 3 models")
        return df