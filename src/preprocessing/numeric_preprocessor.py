#!/usr/bin/env python3
"""
Numeric Preprocessor - Richter Predictor
Gestisce le features numeriche con normalizzazione e outlier handling
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
import numpy as np
from .base_preprocessor import BasePreprocessor


class NumericPreprocessor(BasePreprocessor):
    """
    Preprocessore specializzato per features numeriche.
    
    Gestisce: age, area_percentage, height_percentage
    
    Applica:
    - Normalizzazione con tf.keras.layers.Normalization
    - Outlier detection e clipping (opzionale)
    - Trasformazioni specifiche per feature skewed
    """
    
    def __init__(self, feature_names: List[str], handle_outliers: bool = True):
        """
        Inizializza il preprocessore numerico.
        
        Args:
            feature_names: Lista con features numeriche
            handle_outliers: Se gestire outliers con clipping
        """
        super().__init__(feature_names, "NumericPreprocessor")
        
        self.handle_outliers = handle_outliers
        
        # Normalizzatori per ogni feature
        self.normalizers = {}
        
        # Statistiche per outlier detection
        self.outlier_bounds = {}
        
        # Metadati statistici
        self.stats = {}
    
    def fit(self, data_dict: Dict[str, tf.Tensor]) -> 'NumericPreprocessor':
        """
        Adatta normalizzatori e calcola bounds per outliers.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            self per method chaining
        """
        print(f" Fitting {self.name}...")
        
        available_features = self._filter_available_features(data_dict)
        
        for feature in available_features:
            print(f"    Processando {feature}...")
            
            feature_data = data_dict[feature]
            
            # Assicurati che sia float32
            feature_data = tf.cast(feature_data, tf.float32)
            
            # Espandi dimensioni se necessario per Normalization layer
            if len(feature_data.shape) == 1:
                feature_data = tf.expand_dims(feature_data, -1)
            
            # Crea e adatta normalizzatore
            normalizer = layers.Normalization(name=f'norm_{feature}')
            normalizer.adapt(feature_data)
            self.normalizers[feature] = normalizer
            
            # Salva statistiche
            mean_val = float(normalizer.mean.numpy()[0])
            var_val = float(normalizer.variance.numpy()[0])
            std_val = np.sqrt(var_val)
            
            self.stats[feature] = {
                'mean': mean_val,
                'variance': var_val,
                'std': std_val
            }
            
            # Calcola bounds per outliers (IQR method)
            if self.handle_outliers:
                self._calculate_outlier_bounds(feature, feature_data)
            
            # Analizza distribuzione
            self._analyze_distribution(feature, feature_data)
            
            print(f"       μ={mean_val:.3f}, σ={std_val:.3f}")
            
            if self.handle_outliers and feature in self.outlier_bounds:
                bounds = self.outlier_bounds[feature]
                print(f"       Outlier bounds: [{bounds['lower']:.2f}, {bounds['upper']:.2f}]")
        
        self.is_fitted = True
        print(f"    {self.name} fitted!")
        return self
    
    def _calculate_outlier_bounds(self, feature: str, data: tf.Tensor):
        """Calcola bounds per outlier detection usando IQR method"""
        # Flatten data
        flat_data = tf.reshape(data, [-1])
        
        # Calcola quartili
        q1 = tfp.stats.percentile(flat_data, 25.0)
        q3 = tfp.stats.percentile(flat_data, 75.0)
        iqr = q3 - q1
        
        # Bounds IQR: Q1 - 1.5*IQR, Q3 + 1.5*IQR
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        self.outlier_bounds[feature] = {
            'lower': float(lower_bound.numpy()),
            'upper': float(upper_bound.numpy()),
            'q1': float(q1.numpy()),
            'q3': float(q3.numpy()),
            'iqr': float(iqr.numpy())
        }
        
        # Conta outliers
        outliers = tf.logical_or(
            flat_data < lower_bound,
            flat_data > upper_bound
        )
        outlier_count = tf.reduce_sum(tf.cast(outliers, tf.int32))
        outlier_percent = (outlier_count / len(flat_data)) * 100
        
        self.metadata[f'{feature}_outlier_count'] = int(outlier_count.numpy())
        self.metadata[f'{feature}_outlier_percent'] = float(outlier_percent.numpy())
        
        if outlier_percent > 5:
            print(f"        {outlier_count} outliers ({outlier_percent:.1f}%)")
    
    def _analyze_distribution(self, feature: str, data: tf.Tensor):
        """Analizza distribuzione della feature per identificare skewness"""
        flat_data = tf.reshape(data, [-1])
        
        # Calcola skewness approssimata
        mean = tf.reduce_mean(flat_data)
        std = tf.math.reduce_std(flat_data)
        
        # Skewness usando momento terzo
        centered = flat_data - mean
        third_moment = tf.reduce_mean(tf.pow(centered / std, 3))
        
        skewness = float(third_moment.numpy())
        self.metadata[f'{feature}_skewness'] = skewness
        
        # Suggerisci trasformazioni se molto skewed
        if abs(skewness) > 2:
            if skewness > 0:
                suggestion = "log transform (right-skewed)"
            else:
                suggestion = "square transform (left-skewed)"
            print(f"       Skewness: {skewness:.2f} → {suggestion}")
            self.metadata[f'{feature}_transform_suggestion'] = suggestion
    
    def transform(self, data_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applica normalizzazione e outlier handling.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            Dict con features numeriche preprocessate
        """
        self._check_fitted()
        
        processed = {}
        
        for feature in self.feature_names:
            if feature in data_dict and feature in self.normalizers:
                feature_data = tf.cast(data_dict[feature], tf.float32)
                
                # Gestisci outliers se richiesto
                if self.handle_outliers and feature in self.outlier_bounds:
                    bounds = self.outlier_bounds[feature]
                    feature_data = tf.clip_by_value(
                        feature_data,
                        bounds['lower'],
                        bounds['upper']
                    )
                
                # Espandi dimensioni se necessario
                if len(feature_data.shape) == 1:
                    feature_data = tf.expand_dims(feature_data, -1)
                
                # Applica normalizzazione
                normalized = self.normalizers[feature](feature_data)
                processed[feature] = normalized
        
        return processed
    
    def get_output_specs(self) -> Dict[str, Tuple[int, str]]:
        """
        Specifiche output per features numeriche.
        
        Returns:
            Dict con {feature_name: (1, 'normalized_float')}
        """
        specs = {}
        
        for feature in self.feature_names:
            if feature in self.normalizers:
                specs[feature] = (1, 'normalized_float')
        
        return specs
    
    def get_keras_layers(self, inputs: Dict[str, layers.Input]) -> Dict[str, tf.Tensor]:
        """
        Crea layer Keras per preprocessing numerico.
        
        Args:
            inputs: Dict con {feature_name: Input layer}
            
        Returns:
            Dict con {feature_name: processed tensor}
        """
        outputs = {}
        
        for feature in self.feature_names:
            if feature in inputs and feature in self.normalizers:
                input_layer = inputs[feature]
                
                # Applica outlier clipping se necessario
                if self.handle_outliers and feature in self.outlier_bounds:
                    bounds = self.outlier_bounds[feature]
                    clipped = layers.Lambda(
                        lambda x: tf.clip_by_value(x, bounds['lower'], bounds['upper']),
                        name=f'clip_{feature}'
                    )(input_layer)
                else:
                    clipped = input_layer
                
                # Applica normalizzazione
                normalized = self.normalizers[feature](clipped)
                outputs[feature] = normalized
        
        return outputs
    
    def get_feature_statistics(self) -> Dict[str, Dict]:
        """
        Restituisce statistiche complete delle features.
        
        Returns:
            Dict con statistiche per ogni feature
        """
        return {
            'stats': self.stats,
            'outlier_bounds': self.outlier_bounds,
            'metadata': self.metadata
        }
    
    def suggest_transformations(self) -> Dict[str, str]:
        """
        Suggerisce trasformazioni basate su analisi distribuzione.
        
        Returns:
            Dict con {feature_name: transformation_suggestion}
        """
        suggestions = {}
        
        for feature in self.feature_names:
            if f'{feature}_transform_suggestion' in self.metadata:
                suggestions[feature] = self.metadata[f'{feature}_transform_suggestion']
        
        return suggestions


# Importa tensorflow_probability se disponibile
try:
    import tensorflow_probability as tfp
except ImportError:
    print("  tensorflow_probability non disponibile. Outlier detection limitata.")
    
    # Fallback per percentile
    def percentile_fallback(data, q):
        """Fallback per calcolo percentili senza tfp"""
        sorted_data = tf.sort(data)
        n = tf.cast(tf.shape(sorted_data)[0], tf.float32)
        index = (q / 100.0) * (n - 1)
        lower_idx = tf.cast(tf.floor(index), tf.int32)
        upper_idx = tf.minimum(lower_idx + 1, tf.cast(n - 1, tf.int32))
        
        lower_val = sorted_data[lower_idx]
        upper_val = sorted_data[upper_idx]
        weight = index - tf.cast(lower_idx, tf.float32)
        
        return lower_val + weight * (upper_val - lower_val)
    
    # Crea oggetto mock per tfp.stats
    class MockTFP:
        class stats:
            @staticmethod
            def percentile(data, q):
                return percentile_fallback(data, q)
    
    tfp = MockTFP()
