#!/usr/bin/env python3
"""
Binary Preprocessor - Richter Predictor
Gestisce features binarie (has_superstructure_*)
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
import numpy as np
from .base_preprocessor import BasePreprocessor


class BinaryPreprocessor(BasePreprocessor):
    """
    Preprocessore specializzato per features binarie.
    
    Gestisce tutte le features has_superstructure_*.
    
    Strategie:
    - Standard: Normalizzazione [0,1] e cast a float32
    - Grouped: Raggruppa features binarie correlate
    - Count: Aggiunge feature di conteggio (quante features = 1)
    """
    
    def __init__(self, feature_names: List[str], 
                 add_count_feature: bool = True,
                 group_correlated: bool = True):
        """
        Inizializza il preprocessore binario.
        
        Args:
            feature_names: Lista con features binarie
            add_count_feature: Se aggiungere feature di conteggio totale
            group_correlated: Se raggruppare features correlate
        """
        super().__init__(feature_names, "BinaryPreprocessor")
        
        self.add_count_feature = add_count_feature
        self.group_correlated = group_correlated
        
        # Analisi correlazioni
        self.correlation_matrix = None
        self.correlation_groups = []
        
        # Statistiche
        self.feature_means = {}
        self.feature_correlations = {}
        
        # Feature derivate
        self.derived_features = []
    
    def fit(self, data_dict: Dict[str, tf.Tensor]) -> 'BinaryPreprocessor':
        """
        Analizza features binarie e calcola statistiche.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            self per method chaining
        """
        print(f" Fitting {self.name}...")
        
        available_features = self._filter_available_features(data_dict)
        
        if not available_features:
            print("     Nessuna feature binaria trovata!")
            self.is_fitted = True
            return self
        
        # Analizza singole features
        for feature in available_features:
            print(f"    Analizzando {feature}...")
            
            feature_data = tf.cast(data_dict[feature], tf.float32)
            
            # Verifica che sia davvero binaria
            unique_vals = tf.unique(tf.reshape(feature_data, [-1]))[0]
            unique_count = len(unique_vals.numpy())
            
            if unique_count > 2:
                print(f"        {feature} ha {unique_count} valori unici (non binaria)")
            
            # Calcola statistiche base
            mean_val = tf.reduce_mean(feature_data)
            self.feature_means[feature] = float(mean_val.numpy())
            
            # Salva metadati
            self.metadata[f'{feature}_mean'] = float(mean_val.numpy())
            self.metadata[f'{feature}_unique_count'] = unique_count
            
            print(f"       Media: {mean_val:.3f}")
        
        # Analizza correlazioni se richiesto
        if self.group_correlated and len(available_features) > 1:
            self._analyze_correlations(data_dict, available_features)
        
        # Prepara feature derivate
        if self.add_count_feature:
            self.derived_features.append('binary_total_count')
            print(f"    Aggiunta feature: binary_total_count")
        
        self.is_fitted = True
        print(f"    {self.name} fitted!")
        return self
    
    def _analyze_correlations(self, data_dict: Dict[str, tf.Tensor], 
                             features: List[str]) -> None:
        """Analizza correlazioni tra features binarie"""
        print(f"    Analizzando correlazioni...")
        
        # Crea matrice features
        feature_tensors = []
        for feature in features:
            feature_data = tf.cast(data_dict[feature], tf.float32)
            feature_tensors.append(tf.reshape(feature_data, [-1, 1]))
        
        # Stack features
        features_matrix = tf.concat(feature_tensors, axis=1)
        
        # Calcola correlazioni
        corr_matrix = tf.linalg.matmul(
            features_matrix, features_matrix, transpose_a=True
        )
        
        # Normalizza per ottenere correlazioni di Pearson
        n_samples = tf.cast(tf.shape(features_matrix)[0], tf.float32)
        means = tf.reduce_mean(features_matrix, axis=0, keepdims=True)
        centered = features_matrix - means
        
        std_devs = tf.math.reduce_std(centered, axis=0, keepdims=True)
        normalized = centered / (std_devs + 1e-8)
        
        correlation_matrix = tf.linalg.matmul(
            normalized, normalized, transpose_a=True
        ) / n_samples
        
        self.correlation_matrix = correlation_matrix.numpy()
        
        # Trova gruppi di features correlate (|corr| > 0.5)
        self._find_correlation_groups(features, threshold=0.5)
        
        # Salva correlazioni più forti
        self._save_top_correlations(features)
    
    def _find_correlation_groups(self, features: List[str], threshold: float) -> None:
        """Trova gruppi di features altamente correlate"""
        if self.correlation_matrix is None:
            return
        
        n_features = len(features)
        visited = [False] * n_features
        groups = []
        
        for i in range(n_features):
            if visited[i]:
                continue
            
            # Trova features correlate con la i-esima
            current_group = [features[i]]
            visited[i] = True
            
            for j in range(i + 1, n_features):
                if not visited[j]:
                    corr_val = abs(self.correlation_matrix[i, j])
                    if corr_val > threshold:
                        current_group.append(features[j])
                        visited[j] = True
            
            if len(current_group) > 1:
                groups.append(current_group)
                print(f"       Gruppo correlato: {current_group}")
        
        self.correlation_groups = groups
    
    def _save_top_correlations(self, features: List[str]) -> None:
        """Salva le correlazioni più forti"""
        if self.correlation_matrix is None:
            return
        
        n_features = len(features)
        top_correlations = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_val = self.correlation_matrix[i, j]
                top_correlations.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'correlation': float(corr_val)
                })
        
        # Ordina per correlazione assoluta
        top_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Salva top 10
        self.feature_correlations = top_correlations[:10]
        
        print(f"       Top correlazioni:")
        for corr in self.feature_correlations[:5]:
            f1, f2, val = corr['feature1'], corr['feature2'], corr['correlation']
            print(f"         {f1} ↔ {f2}: {val:.3f}")
    
    def transform(self, data_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applica preprocessing alle features binarie.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            Dict con features binarie preprocessate
        """
        self._check_fitted()
        
        processed = {}
        available_features = []
        
        # Processa features binarie standard
        for feature in self.feature_names:
            if feature in data_dict:
                # Cast a float32 e normalizza [0,1]
                feature_data = tf.cast(data_dict[feature], tf.float32)
                
                # Assicura che sia in range [0,1]
                feature_data = tf.clip_by_value(feature_data, 0.0, 1.0)
                
                processed[feature] = feature_data
                available_features.append(feature)
        
        # Aggiungi feature derivate
        if self.add_count_feature and available_features:
            # Conta quante features binarie sono attive (= 1)
            binary_tensors = [processed[f] for f in available_features]
            stacked = tf.stack(binary_tensors, axis=-1)
            total_count = tf.reduce_sum(stacked, axis=-1, keepdims=True)
            
            # Normalizza per numero di features
            normalized_count = total_count / len(available_features)
            
            processed['binary_total_count'] = normalized_count
        
        # Aggiungi features per gruppi correlati
        for i, group in enumerate(self.correlation_groups):
            group_features = [f for f in group if f in processed]
            
            if len(group_features) > 1:
                # Media del gruppo
                group_tensors = [processed[f] for f in group_features]
                group_mean = tf.reduce_mean(tf.stack(group_tensors, axis=-1), 
                                          axis=-1, keepdims=True)
                
                processed[f'binary_group_{i}_mean'] = group_mean
        
        return processed
    
    def get_output_specs(self) -> Dict[str, Tuple[int, str]]:
        """
        Specifiche output per features binarie.
        
        Returns:
            Dict con {feature_name: (output_dim, encoding_type)}
        """
        specs = {}
        
        # Features originali
        for feature in self.feature_names:
            specs[feature] = (1, 'binary')
        
        # Features derivate
        for derived in self.derived_features:
            specs[derived] = (1, 'binary_derived')
        
        # Features di gruppo
        for i, _ in enumerate(self.correlation_groups):
            specs[f'binary_group_{i}_mean'] = (1, 'binary_group')
        
        return specs
    
    def get_keras_layers(self, inputs: Dict[str, layers.Input]) -> Dict[str, tf.Tensor]:
        """
        Crea layer Keras per preprocessing binario.
        
        Args:
            inputs: Dict con {feature_name: Input layer}
            
        Returns:
            Dict con {feature_name: processed tensor}
        """
        outputs = {}
        
        # Processa features binarie base
        for feature in self.feature_names:
            if feature in inputs:
                # Cast e normalizzazione
                binary_layer = layers.Lambda(
                    lambda x: tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0),
                    name=f'binary_norm_{feature}'
                )(inputs[feature])
                
                outputs[feature] = binary_layer
        
        return outputs
    
    def analyze_binary_patterns(self) -> Dict[str, any]:
        """
        Analizza pattern nelle features binarie.
        
        Returns:
            Dict con analisi dettagliata
        """
        analysis = {
            'feature_means': self.feature_means,
            'top_correlations': self.feature_correlations,
            'correlation_groups': self.correlation_groups,
            'derived_features': self.derived_features
        }
        
        return analysis
    
    def get_feature_importance_by_prevalence(self) -> Dict[str, float]:
        """
        Calcola importance basata su prevalenza (quanto spesso = 1).
        
        Features più rare potrebbero essere più informative.
        
        Returns:
            Dict con {feature_name: importance_score}
        """
        importance = {}
        
        for feature, mean_val in self.feature_means.items():
            # Score basato su distanza da 0.5 (massima entropia)
            # Features vicine a 0 o 1 sono potenzialmente più informative
            distance_from_balanced = abs(mean_val - 0.5)
            importance_score = distance_from_balanced * 2  # Normalizza [0,1]
            
            importance[feature] = importance_score
        
        return importance
    
    def suggest_feature_engineering(self) -> List[str]:
        """
        Suggerisce tecniche di feature engineering per binarie.
        
        Returns:
            Lista di suggerimenti
        """
        suggestions = []
        
        # Analizza correlazioni
        if self.correlation_groups:
            suggestions.append(
                f" Trovati {len(self.correlation_groups)} gruppi di features "
                "correlate - considera l'aggregazione"
            )
        
        # Analizza sparsità
        sparse_features = [
            f for f, mean in self.feature_means.items() 
            if mean < 0.1 or mean > 0.9
        ]
        
        if sparse_features:
            suggestions.append(
                f" {len(sparse_features)} features molto sbilanciate - "
                "considera tecniche di bilanciamento"
            )
        
        # Suggerimenti per interazioni
        if len(self.feature_names) > 5:
            suggestions.append(
                " Con molte features binarie, considera interazioni "
                "polynomial o count aggregations"
            )
        
        return suggestions
