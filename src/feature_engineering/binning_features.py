"""
Binning Features
Entropy-based binning che massimizza information gain
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .base import ConfigurableFeatureEngineer


class BinningFeatureEngineer(ConfigurableFeatureEngineer):
    """
    Feature engineer per entropy-based binning
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("BinningFeatures", config)
        self.min_samples_for_binning = config.get('min_samples_for_binning', 50) if config else 50
        self.min_unique_values = config.get('min_unique_values', 20) if config else 20
        self.max_splits = config.get('max_splits', 4) if config else 4
        
        # State per test set
        self.binning_info: Dict[str, Dict] = {}
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'damage_grade') -> pd.DataFrame:
        """Crea binning features durante training"""
        self._print_if_verbose("   Creating entropy-based target-aware binning features...")
        
        df_result = df.copy()
        
        # Create binning features
        df_result = self._create_binning_features(df_result, target_col)
        
        # Track features create
        new_features = [col for col in df_result.columns if col.endswith('_binned')]
        self._log_feature_creation(new_features, "entropy-based binning")
        
        self.fitted = True
        return df_result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica binning features su test set"""
        if not self.fitted:
            raise ValueError("BinningFeatureEngineer must be fitted before transform")
            
        df_result = df.copy()
        
        # Apply saved binning
        df_result = self._apply_test_binning(df_result)
        
        return df_result
    
    def _create_binning_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Crea binning features con entropy-based splitting"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == target_col or df[col].nunique() < self.min_unique_values:
                continue
                
            try:
                col_data = df[col].dropna()
                if len(col_data) < self.min_samples_for_binning:
                    continue
                
                # Entropy-based binning during training
                if target_col in df.columns:
                    bin_edges = self._create_entropy_based_bins(df, col, target_col)
                    method = 'entropy_based'
                    self._print_if_verbose(f"      {col}: Entropy-based binning with {len(bin_edges)-1} bins")
                else:
                    # Fallback to percentiles
                    bin_edges = np.percentile(col_data, [0, 25, 50, 75, 100])
                    method = 'percentile_fallback'
                
                # Apply binning
                df = self._apply_binning(df, col, bin_edges, method)
                
            except Exception as e:
                self._print_if_verbose(f"      Warning: Entropy binning failed for {col}: {e}")
                continue
        
        return df
    
    def _create_entropy_based_bins(self, df: pd.DataFrame, col: str, target_col: str) -> np.ndarray:
        """Crea bin edges usando entropy-based splitting"""
        
        # Get valid data
        valid_mask = df[col].notna() & df[target_col].notna()
        feature_values = df.loc[valid_mask, col].values
        target_values = df.loc[valid_mask, target_col].values
        
        if len(feature_values) < self.min_samples_for_binning or len(np.unique(target_values)) < 2:
            # Fallback to percentiles
            return np.percentile(feature_values, [0, 25, 50, 75, 100])
        
        # Use entropy-based splitting
        return self._find_best_split_entropy(feature_values, target_values)
    
    def _find_best_split_entropy(self, values: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Trova le migliori soglie usando information gain"""
        
        # Sort values and get unique candidate thresholds
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_target = target[sorted_idx]
        
        # Candidate thresholds: midpoints between unique values
        unique_values = np.unique(sorted_values)
        if len(unique_values) <= 3:
            return unique_values  # Too few values for splitting
        
        candidate_thresholds = []
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            candidate_thresholds.append(threshold)
        
        # Limit candidates for efficiency
        if len(candidate_thresholds) > 20:
            step = len(candidate_thresholds) // 20
            candidate_thresholds = candidate_thresholds[::step]
        
        # Calculate information gain for each threshold
        base_entropy = self._calculate_entropy(sorted_target)
        threshold_gains = []
        
        for threshold in candidate_thresholds:
            left_mask = sorted_values <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < 10 or np.sum(right_mask) < 10:
                continue  # Skip if split too unbalanced
            
            left_entropy = self._calculate_entropy(sorted_target[left_mask])
            right_entropy = self._calculate_entropy(sorted_target[right_mask])
            
            # Weighted average entropy after split
            left_weight = np.sum(left_mask) / len(sorted_target)
            right_weight = np.sum(right_mask) / len(sorted_target)
            
            weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
            information_gain = base_entropy - weighted_entropy
            
            threshold_gains.append((threshold, information_gain))
        
        if not threshold_gains:
            return unique_values[[0, -1]]  # Fallback: just min/max
        
        # Sort by information gain and select top splits
        threshold_gains.sort(key=lambda x: x[1], reverse=True)
        best_thresholds = [t for t, _ in threshold_gains[:self.max_splits]]
        
        # Add min/max to ensure complete coverage
        all_thresholds = [sorted_values[0] - 1e-6] + sorted(best_thresholds) + [sorted_values[-1] + 1e-6]
        
        return np.unique(all_thresholds)
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        """Calcola entropia di Shannon"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _apply_binning(self, df: pd.DataFrame, col: str, bin_edges: np.ndarray, method: str) -> pd.DataFrame:
        """Applica binning con gestione robusta"""
        
        # Remove duplicates and ensure minimum bins
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:  # Need at least 2 bins
            return df
        
        # Apply binning
        try:
            binned_values = pd.cut(
                df[col], 
                bins=bin_edges, 
                labels=range(len(bin_edges)-1),
                include_lowest=True,
                duplicates='drop'
            )
            
            df[f'{col}_binned'] = binned_values.astype(float)
            
            # Save binning info for test set
            if not self.fitted:  # Only during training
                # Extend range for robustness with future data
                extended_min = bin_edges[0] - abs(bin_edges[0]) * 0.1 - 1e-6
                extended_max = bin_edges[-1] + abs(bin_edges[-1]) * 0.1 + 1e-6
                
                extended_edges = np.concatenate([[extended_min], bin_edges[1:-1], [extended_max]])
                
                self.binning_info[col] = {
                    'bins': extended_edges,
                    'labels': list(range(len(extended_edges)-1)),
                    'n_bins': len(extended_edges)-1,
                    'method': method
                }
            
        except Exception as e:
            self._print_if_verbose(f"      Warning: Binning application failed for {col}: {e}")
        
        return df
    
    def _apply_test_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applica binning salvato per test set"""
        
        self._print_if_verbose("   Applying binning features...")
        
        for col, binning_config in self.binning_info.items():
            if col not in df.columns:
                continue
                
            try:
                bins = binning_config['bins']
                labels = binning_config['labels']
                
                # Robust binning with out-of-bounds handling
                binned_values = pd.cut(
                    df[col], 
                    bins=bins, 
                    labels=labels,
                    include_lowest=True
                )
                
                # Handle NaN (out-of-range values)
                binned_values = binned_values.astype(float)
                
                # Fill NaN with median bin
                median_bin = len(labels) // 2
                binned_values = binned_values.fillna(median_bin)
                
                df[f'{col}_binned'] = binned_values
                
            except Exception as e:
                self._print_if_verbose(f"      Warning: Test binning failed for {col}: {e}")
                # Safe fallback
                df[f'{col}_binned'] = 2.0
        
        return df
    
    def get_binning_info(self) -> Dict[str, Dict]:
        """Getter per binning info (per debugging)"""
        return self.binning_info