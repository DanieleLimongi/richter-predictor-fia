"""
Polynomial Features
Genera polynomial interactions per massimizzare potere predittivo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Optional, Set
from .base import ConfigurableFeatureEngineer


class PolynomialFeatureEngineer(ConfigurableFeatureEngineer):
    """
    Feature engineer specializzato per polynomial features
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("PolynomialFeatures", config)
        self.max_features = config.get('max_features', 50) if config else 50
        self.top_n_features = config.get('top_n_features', 10) if config else 10
        self.degree = config.get('degree', 2) if config else 2
        
        # State per ricostruzione test
        self.polynomial_features_names: List[str] = []
        self.polynomial_feature_medians: Dict[str, float] = {}
        self.top_3_features: List[str] = []
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Crea polynomial features durante training"""
        self._print_if_verbose("   Creating expanded polynomial features...")
        
        try:
            if target_col not in df.columns:
                self._print_if_verbose("      Warning: Target column not found, using fallback polynomial features")
                self.fitted = True  # Mark as fitted to allow transform
                return self._create_fallback_polynomial_features(df)
            
            df_result = df.copy()
            
            # 1. Find top correlated features
            top_features = self._find_top_correlated_features(df_result, target_col)
            if not top_features:
                self._print_if_verbose("      Warning: No suitable features found, using fallback polynomial features")
                self.fitted = True  # Mark as fitted to allow transform
                return self._create_fallback_polynomial_features(df)
            
            # 2. Create polynomial features
            df_result = self._create_polynomial_interactions(df_result, top_features, target_col)
            
            # Track features created
            new_features = [col for col in df_result.columns if col.startswith('poly_')]
            if new_features:
                self._log_feature_creation(new_features, f"expanded polynomial (quadratic + cubic)")
                self._print_if_verbose(f"      Successfully created {len(new_features)} polynomial features")
            else:
                self._print_if_verbose("      Warning: No polynomial features created, using fallback")
                df_result = self._create_fallback_polynomial_features(df)
            
            self.fitted = True
            return df_result
            
        except Exception as e:
            self._print_if_verbose(f"      ERROR in polynomial fit_transform: {e}")
            self._print_if_verbose("      Using fallback polynomial features for consistency")
            self.fitted = True  # Mark as fitted to allow transform
            return self._create_fallback_polynomial_features(df)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ricostruisce polynomial features per test set"""
        if not self.fitted:
            self._print_if_verbose("      WARNING: PolynomialFeatureEngineer not fitted, using fallback")
            return self._create_fallback_polynomial_features(df)
            
        if not self.polynomial_features_names:
            self._print_if_verbose("      No polynomial features to recreate, using fallback")
            return self._create_fallback_polynomial_features(df)
            
        self._print_if_verbose("   Intelligently recreating polynomial features for test...")
        
        try:
            df_result = df.copy()
            
            # Strategia intelligente di ricostruzione
            df_result = self._reconstruct_polynomial_features(df_result)
            
            return df_result
            
        except Exception as e:
            self._print_if_verbose(f"      ERROR in polynomial transform: {e}")
            self._print_if_verbose("      Using fallback polynomial features for consistency")
            return self._create_fallback_polynomial_features(df)
    
    def _find_top_correlated_features(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """Trova top features correlate con target"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Filter valid numeric columns
        valid_numeric_cols = []
        for col in numeric_cols:
            if col != target_col and not df[col].isna().all() and df[col].notna().sum() > 10:
                valid_numeric_cols.append(col)
        
        if len(valid_numeric_cols) < 2:
            self._print_if_verbose("      Warning: Not enough valid numeric columns for polynomial features")
            return []
        
        # Calculate correlations
        correlations = df[valid_numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        correlations = correlations.dropna()
        
        if len(correlations) < 2:
            self._print_if_verbose("      Warning: Not enough correlated features for polynomial")
            return []
        
        # Get top N features
        top_features = correlations.head(self.top_n_features).index.tolist()
        self._print_if_verbose(f"      Top {self.top_n_features} features for polynomial: {top_features[:5]}... "
                             f"(correlation range: {correlations.iloc[0]:.3f}-{correlations.iloc[min(9, len(correlations)-1)]:.3f})")
        
        return top_features
    
    def _create_polynomial_interactions(self, df: pd.DataFrame, top_features: List[str], 
                                      target_col: str) -> pd.DataFrame:
        """Crea polynomial interactions"""
        
        if len(top_features) < 2:
            self._print_if_verbose("      Not enough features for polynomial interactions")
            return df
        
        try:
            # Prepare data with robust preprocessing
            poly_data = df[top_features].copy()
        
            # Save medians for test consistency
            for col in top_features:
                if col not in self.polynomial_feature_medians:
                    median_val = poly_data[col].median()
                    if pd.isna(median_val):
                        median_val = poly_data[col].mean()
                        if pd.isna(median_val):
                            median_val = 0.0
                    self.polynomial_feature_medians[col] = median_val
                
                # Fill NaN with saved median
                poly_data[col] = poly_data[col].fillna(self.polynomial_feature_medians[col])
            
            # Strategy: Quadratic for all + Cubic for top 3
            all_poly_features = []
            all_poly_names = []
            
            # 1. Quadratic interactions for all features
            quad_features, quad_names = self._create_quadratic_features(poly_data, top_features)
            if quad_features is not None:
                all_poly_features.append(quad_features)
                all_poly_names.extend(quad_names)
            
            # 2. Cubic interactions for top 3 features
            top_3_features = top_features[:3]
            if len(top_3_features) >= 3:
                cubic_features, cubic_names = self._create_cubic_features(poly_data, top_3_features)
                if cubic_features is not None:
                    all_poly_features.append(cubic_features)
                    all_poly_names.extend(cubic_names)
                    self.top_3_features = top_3_features
                    self._print_if_verbose(f"      Added {len(cubic_names)} cubic interactions for top 3 features")
            
            # Combine all polynomial features
            if all_poly_features:
                new_poly_features = np.concatenate(all_poly_features, axis=1)
                new_poly_names = all_poly_names
            else:
                return df
            
            # Feature selection if too many
            if len(new_poly_names) > self.max_features:
                new_poly_features, new_poly_names = self._select_best_polynomial_features(
                    new_poly_features, new_poly_names, df, target_col
                )
        
            # Add polynomial features to dataframe
            return self._add_polynomial_features_to_df(df, new_poly_features, new_poly_names)
            
        except Exception as e:
            self._print_if_verbose(f"      ERROR in polynomial interactions creation: {e}")
            return df
    
    def _create_quadratic_features(self, poly_data: pd.DataFrame, features: List[str]) -> tuple:
        """Crea quadratic features"""
        
        try:
            poly_quad = PolynomialFeatures(
                degree=2, 
                include_bias=False, 
                interaction_only=True
            )
            
            quad_features = poly_quad.fit_transform(poly_data)
            quad_names = poly_quad.get_feature_names_out(features)
            
            # Only new quadratic features (exclude originals)
            original_count = len(features)
            new_quad_features = quad_features[:, original_count:]
            new_quad_names = [f"poly_{name}" for name in quad_names[original_count:]]
            
            return new_quad_features, new_quad_names
            
        except Exception as e:
            self._print_if_verbose(f"      Warning: Quadratic feature creation failed: {e}")
            return None, []
    
    def _create_cubic_features(self, poly_data: pd.DataFrame, features: List[str]) -> tuple:
        """Crea cubic features"""
        
        try:
            poly_cubic = PolynomialFeatures(
                degree=3,
                include_bias=False,
                interaction_only=True
            )
            
            cubic_data = poly_data[features]
            cubic_features = poly_cubic.fit_transform(cubic_data)
            cubic_names = poly_cubic.get_feature_names_out(features)
            
            # Only cubic interactions (skip linear and quadratic)
            cubic_start_idx = len(features) + len(features) * (len(features) - 1) // 2
            new_cubic_features = cubic_features[:, cubic_start_idx:]
            new_cubic_names = [f"poly_{name}" for name in cubic_names[cubic_start_idx:]]
            
            return new_cubic_features, new_cubic_names
            
        except Exception as e:
            self._print_if_verbose(f"      Warning: Cubic feature creation failed: {e}")
            return None, []
    
    def _select_best_polynomial_features(self, new_poly_features: np.ndarray, new_poly_names: List[str], 
                                       df: pd.DataFrame, target_col: str) -> tuple:
        """Seleziona le migliori polynomial features"""
        
        self._print_if_verbose(f"      Selecting best {self.max_features} from {len(new_poly_names)} polynomial features...")
        
        try:
            poly_df_temp = pd.DataFrame(new_poly_features, columns=new_poly_names)
            
            # 1. Correlations with target
            poly_corr = poly_df_temp.corrwith(df[target_col]).abs()
            poly_corr = poly_corr.dropna()
            
            # 2. Variance (eliminate low-variance features)
            poly_var = poly_df_temp.var()
            poly_var = poly_var[poly_var > 1e-6]  # Minimum variance threshold
            
            # 3. Composite score: correlation * log(variance + 1)
            valid_features = set(poly_corr.index) & set(poly_var.index)
            
            if valid_features:
                composite_scores = {}
                for feat in valid_features:
                    corr_score = poly_corr[feat]
                    var_score = np.log1p(poly_var[feat])
                    composite_scores[feat] = corr_score * var_score
                
                # Sort by composite score
                sorted_features = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
                top_poly_features = [feat for feat, _ in sorted_features[:self.max_features]]
                
                # Select best features
                selected_indices = [new_poly_names.index(feat) for feat in top_poly_features 
                                  if feat in new_poly_names]
                
                if selected_indices:
                    new_poly_features = new_poly_features[:, selected_indices]
                    new_poly_names = [new_poly_names[i] for i in selected_indices]
                    self._print_if_verbose(f"      Selected features by composite score (corr × log_var)")
            
            else:
                # Fallback: correlation only
                if len(poly_corr) > 0:
                    n_select = min(self.max_features, len(poly_corr))
                    top_poly_features = poly_corr.nlargest(n_select).index
                    
                    selected_indices = [new_poly_names.index(feat) for feat in top_poly_features 
                                      if feat in new_poly_names]
                    
                    if selected_indices:
                        new_poly_features = new_poly_features[:, selected_indices]
                        new_poly_names = [new_poly_names[i] for i in selected_indices]
            
        except Exception as e:
            self._print_if_verbose(f"      Warning: Feature selection failed: {e}")
        
        return new_poly_features, new_poly_names
    
    def _add_polynomial_features_to_df(self, df: pd.DataFrame, new_poly_features: np.ndarray, 
                                     new_poly_names: List[str]) -> pd.DataFrame:
        """Aggiunge polynomial features al dataframe"""
        
        try:
            if len(new_poly_names) == 0:
                return df
                
            poly_df = pd.DataFrame(new_poly_features, columns=new_poly_names, index=df.index)
            
            # Robust cleaning
            poly_df = self._handle_inf_and_nan(poly_df)
            
            # Ensure all columns are numeric
            for col in poly_df.columns:
                if not pd.api.types.is_numeric_dtype(poly_df[col]):
                    poly_df[col] = pd.to_numeric(poly_df[col], errors='coerce').fillna(0.0)
            
            df_result = pd.concat([df, poly_df], axis=1)
            self.polynomial_features_names = new_poly_names
            
            self._print_if_verbose(f"      Created {len(new_poly_names)} expanded polynomial features (quadratic + cubic)")
            return df_result
            
        except Exception as e:
            self._print_if_verbose(f"      Warning: Polynomial feature addition failed: {e}")
            self.polynomial_features_names = []
            return df
    
    def _reconstruct_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ricostruisce polynomial features per test set"""
        
        # Find available base features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_base_features = []
        
        for poly_name in self.polynomial_features_names:
            if 'poly_' in poly_name:
                base_name = poly_name.replace('poly_', '')
                for col in numeric_cols:
                    if col in base_name:
                        available_base_features.append(col)
        
        available_base_features = list(set(available_base_features))
        self._print_if_verbose(f"      Found {len(available_base_features)} base features for polynomial reconstruction")
        
        if len(available_base_features) >= 2:
            try:
                # Reconstruct polynomial features when possible
                df = self._perform_polynomial_reconstruction(df, available_base_features)
            except Exception as e:
                self._print_if_verbose(f"      Warning: Polynomial reconstruction failed: {e}")
                # Fallback with intelligent values
                df = self._apply_intelligent_fallbacks(df)
        else:
            # Complete fallback with intelligent values
            df = self._apply_intelligent_fallbacks(df)
        
        return df
    
    def _perform_polynomial_reconstruction(self, df: pd.DataFrame, available_base_features: List[str]) -> pd.DataFrame:
        """Esegue ricostruzione reale delle polynomial features"""
        
        # Prepare data with saved medians
        poly_data = df[available_base_features].copy()
        for col in available_base_features:
            if col in self.polynomial_feature_medians:
                poly_data[col] = poly_data[col].fillna(self.polynomial_feature_medians[col])
            else:
                poly_data[col] = poly_data[col].fillna(poly_data[col].median())
        
        # Recreate quadratic features
        quad_mapping = self._recreate_quadratic_mapping(poly_data, available_base_features)
        
        # Recreate cubic features if available
        if self.top_3_features:
            cubic_mapping = self._recreate_cubic_mapping(poly_data, available_base_features)
            quad_mapping.update(cubic_mapping)
        
        # Apply reconstructed features
        reconstructed_count = 0
        for poly_name in self.polynomial_features_names:
            if poly_name in quad_mapping:
                df[poly_name] = quad_mapping[poly_name]
                reconstructed_count += 1
            elif poly_name not in df.columns:
                # Intelligent fallback for specific feature
                fallback_value = self._get_intelligent_fallback(poly_name, df)
                df[poly_name] = fallback_value
        
        self._print_if_verbose(f"      Reconstructed {reconstructed_count}/{len(self.polynomial_features_names)} polynomial features")
        return df
    
    def _recreate_quadratic_mapping(self, poly_data: pd.DataFrame, available_base_features: List[str]) -> Dict[str, np.ndarray]:
        """Ricrea mapping per quadratic features"""
        
        quad_mapping = {}
        
        try:
            poly_quad = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            quad_features = poly_quad.fit_transform(poly_data)
            quad_names = poly_quad.get_feature_names_out(available_base_features)
            
            for i, name in enumerate(quad_names):
                poly_name = f"poly_{name}"
                if poly_name in self.polynomial_features_names:
                    quad_mapping[poly_name] = quad_features[:, i]
        
        except Exception as e:
            self._print_if_verbose(f"      Warning: Quadratic reconstruction failed: {e}")
        
        return quad_mapping
    
    def _recreate_cubic_mapping(self, poly_data: pd.DataFrame, available_base_features: List[str]) -> Dict[str, np.ndarray]:
        """Ricrea mapping per cubic features"""
        
        cubic_mapping = {}
        
        try:
            top_3_available = [col for col in self.top_3_features if col in available_base_features]
            if len(top_3_available) >= 3:
                poly_cubic = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
                cubic_data = poly_data[top_3_available]
                cubic_features = poly_cubic.fit_transform(cubic_data)
                cubic_names = poly_cubic.get_feature_names_out(top_3_available)
                
                # Only cubic interactions
                cubic_start_idx = len(top_3_available) + len(top_3_available) * (len(top_3_available) - 1) // 2
                
                for i, name in enumerate(cubic_names[cubic_start_idx:], start=cubic_start_idx):
                    poly_name = f"poly_{name}"
                    if poly_name in self.polynomial_features_names:
                        cubic_mapping[poly_name] = cubic_features[:, i]
        
        except Exception as e:
            self._print_if_verbose(f"      Warning: Cubic reconstruction failed: {e}")
        
        return cubic_mapping
    
    def _apply_intelligent_fallbacks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica fallback intelligenti per features mancanti"""
        
        for poly_name in self.polynomial_features_names:
            if poly_name not in df.columns:
                fallback_value = self._get_intelligent_fallback(poly_name, df)
                df[poly_name] = fallback_value
        
        return df
    
    def _get_intelligent_fallback(self, poly_name: str, df: pd.DataFrame) -> pd.Series:
        """Genera valori fallback intelligenti per polynomial features"""
        
        # Pattern matching for different types of polynomial features
        if 'age' in poly_name:
            base_age = df.get('age', pd.Series([30] * len(df))).fillna(30)
            
            if 'area' in poly_name:
                base_area = df.get('area_percentage', pd.Series([50] * len(df))).fillna(50)
                return (base_age * base_area) / 1000
            elif 'height' in poly_name:
                base_height = df.get('height_percentage', pd.Series([40] * len(df))).fillna(40)
                return (base_age * base_height) / 1000
            elif 'floors' in poly_name:
                base_floors = df.get('count_floors_pre_eq', pd.Series([2] * len(df))).fillna(2)
                return (base_age * base_floors) / 10
            else:
                return base_age / 50
        
        elif 'area' in poly_name and 'height' in poly_name:
            base_area = df.get('area_percentage', pd.Series([50] * len(df))).fillna(50)
            base_height = df.get('height_percentage', pd.Series([40] * len(df))).fillna(40)
            return (base_area * base_height) / 1000
        
        elif 'floors' in poly_name:
            base_floors = df.get('count_floors_pre_eq', pd.Series([2] * len(df))).fillna(2)
            if 'height' in poly_name:
                base_height = df.get('height_percentage', pd.Series([40] * len(df))).fillna(40)
                return (base_floors * base_height) / 100
            else:
                return base_floors / 5
        
        elif 'families' in poly_name:
            base_families = df.get('count_families', pd.Series([1] * len(df))).fillna(1)
            if 'area' in poly_name:
                base_area = df.get('area_percentage', pd.Series([50] * len(df))).fillna(50)
                return (base_families * base_area) / 100
            else:
                return base_families / 3
        
        # Generic fallback
        return pd.Series([0.1] * len(df))  # Small but non-zero value
    
    def _create_fallback_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea polynomial features di fallback semplici ma consistenti"""
        
        df_result = df.copy()
        
        # Trova colonne numeriche comuni
        numeric_cols = df_result.select_dtypes(include=[np.number]).columns
        common_features = []
        
        # Priorità alle features più comuni
        priority_features = ['age', 'area_percentage', 'height_percentage', 'count_floors_pre_eq', 'count_families']
        
        for feat in priority_features:
            if feat in numeric_cols:
                common_features.append(feat)
        
        # Se non abbiamo abbastanza features prioritarie, aggiungi altre numeriche
        if len(common_features) < 3:
            for col in numeric_cols:
                if col not in common_features and len(common_features) < 5:
                    common_features.append(col)
        
        if len(common_features) >= 2:
            self._print_if_verbose(f"      Creating {len(common_features)} simple polynomial fallback features")
            
            # Crea interazioni semplici e robuste
            fallback_features = []
            
            # 1. Interazioni quadratiche semplici (age^2, area^2, ecc.)
            for feat in common_features[:3]:
                try:
                    values = df_result[feat].fillna(df_result[feat].median())
                    squared_values = values ** 2
                    # Normalizza per evitare valori troppo grandi
                    squared_values = squared_values / (squared_values.max() + 1e-6)
                    df_result[f'poly_fallback_{feat}_squared'] = squared_values
                    fallback_features.append(f'poly_fallback_{feat}_squared')
                except:
                    continue
            
            # 2. Interazioni a coppie semplici
            for i in range(min(3, len(common_features))):
                for j in range(i+1, min(3, len(common_features))):
                    try:
                        feat1, feat2 = common_features[i], common_features[j]
                        val1 = df_result[feat1].fillna(df_result[feat1].median())
                        val2 = df_result[feat2].fillna(df_result[feat2].median())
                        interaction = val1 * val2
                        # Normalizza
                        interaction = interaction / (interaction.max() + 1e-6)
                        df_result[f'poly_fallback_{feat1}_{feat2}'] = interaction
                        fallback_features.append(f'poly_fallback_{feat1}_{feat2}')
                    except:
                        continue
            
            # Salva i nomi delle features create per consistenza
            if fallback_features:
                self.polynomial_features_names = fallback_features
                self._print_if_verbose(f"      Created {len(fallback_features)} fallback polynomial features")
            
        else:
            # Fallback minimale se non ci sono abbastanza features numeriche
            self._print_if_verbose("      Creating minimal polynomial fallback features")
            df_result['poly_fallback_minimal'] = 0.1
            self.polynomial_features_names = ['poly_fallback_minimal']
        
        return df_result