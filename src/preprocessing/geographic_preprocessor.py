#!/usr/bin/env python3
"""
Geographic Preprocessor - Richter Predictor
Gestisce le features geografiche con strategie specifiche per gerarchia
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
import numpy as np
from .base_preprocessor import BasePreprocessor


class GeographicPreprocessor(BasePreprocessor):
    """
    Preprocessore specializzato per features geografiche.
    
    Gestisce geo_level_1_id, geo_level_2_id, geo_level_3_id con:
    - geo_level_1 (31 regioni): One-hot encoding
    - geo_level_2 (1414 regioni): Embedding dimensione ridotta  
    - geo_level_3 (11595 regioni): Embedding o Drop (troppo frammentato)
    
    Considera la struttura gerarchica e le correlazioni con target.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Inizializza il preprocessore geografico.
        
        Args:
            feature_names: Lista con ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
        """
        super().__init__(feature_names, "GeographicPreprocessor")
        
        # Strategie specifiche per livello geografico
        self.geo_strategies = {
            'geo_level_1_id': 'one_hot',     # 31 regioni → One-hot
            'geo_level_2_id': 'embedding',   # 1414 regioni → Embedding
            'geo_level_3_id': 'embedding'    # 11595 regioni → Embedding (o drop)
        }
        
        # Lookup tables per ogni livello
        self.geo_lookups = {}
        self.geo_vocabs = {}
        self.geo_vocab_sizes = {}
        
        # Dimensioni embedding ottimali
        self.embedding_dims = {
            'geo_level_1_id': None,  # One-hot
            'geo_level_2_id': 16,    # Embedding 16D
            'geo_level_3_id': 32     # Embedding 32D (se non droppato)
        }
    
    def fit(self, data_dict: Dict[str, tf.Tensor]) -> 'GeographicPreprocessor':
        """
        Adatta il preprocessore sulle geo features.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            self per method chaining
        """
        print(f"Fitting {self.name}...")
        
        available_features = self._filter_available_features(data_dict)
        
        for feature in available_features:
            print(f"    Processando {feature}...")
            
            # Ottieni vocabolario unico
            feature_data = data_dict[feature]
            unique_values = tf.unique(tf.reshape(feature_data, [-1]))[0]
            vocab = sorted(unique_values.numpy().tolist())
            
            # Crea IntegerLookup
            lookup = layers.IntegerLookup(
                vocabulary=vocab,
                mask_token=None,
                oov_token=0,  # Out-of-vocabulary → 0
                name=f'geo_lookup_{feature}'
            )
            
            self.geo_lookups[feature] = lookup
            self.geo_vocabs[feature] = vocab
            self.geo_vocab_sizes[feature] = len(vocab) + 1  # +1 per OOV
            
            # Analizza distribuzione e ottimizza strategia
            coverage_info = self._analyze_coverage(feature_data, vocab)
            self._optimize_strategy(feature, coverage_info)
            
            # Salva metadati
            self.metadata[f'{feature}_vocab_size'] = len(vocab)
            self.metadata[f'{feature}_strategy'] = self.geo_strategies[feature]
            self.metadata[f'{feature}_coverage_top10'] = coverage_info['top_10_coverage']
            
            print(f"       {len(vocab)} regioni uniche")
            print(f"       Strategia: {self.geo_strategies[feature]}")
            print(f"       Top 10 coprono {coverage_info['top_10_coverage']:.1f}%")
        
        self.is_fitted = True
        print(f"    {self.name} fitted!")
        return self
    
    def _analyze_coverage(self, feature_data: tf.Tensor, vocab: List) -> Dict:
        """Analizza copertura delle categorie geografiche"""
        # Conta frequenze
        values, _, counts = tf.unique_with_counts(tf.reshape(feature_data, [-1]))
        
        # Ordina per frequenza decrescente
        sorted_indices = tf.argsort(counts, direction='DESCENDING')
        sorted_counts = tf.gather(counts, sorted_indices)
        
        # Calcola copertura top 10
        top_10_counts = sorted_counts[:min(10, len(sorted_counts))]
        total_samples = tf.reduce_sum(counts)
        top_10_coverage = (tf.reduce_sum(top_10_counts) / total_samples) * 100
        
        return {
            'top_10_coverage': float(top_10_coverage.numpy()),
            'total_categories': len(vocab),
            'total_samples': int(total_samples.numpy())
        }
    
    def _optimize_strategy(self, feature: str, coverage_info: Dict):
        """Ottimizza strategia basata su copertura e cardinalità"""
        vocab_size = coverage_info['total_categories']
        coverage = coverage_info['top_10_coverage']
        
        if feature == 'geo_level_1_id':
            # Sempre one-hot per geo_level_1 (bassa cardinalità)
            self.geo_strategies[feature] = 'one_hot'
            
        elif feature == 'geo_level_2_id':
            # Embedding per geo_level_2
            self.geo_strategies[feature] = 'embedding'
            # Dimensione embedding basata su cardinalità
            self.embedding_dims[feature] = min(32, max(8, vocab_size // 50))
            
        elif feature == 'geo_level_3_id':
            # Decisione basata su frammentazione
            if coverage < 5:  # Troppo frammentato
                print(f"        {feature}: molto frammentato (top 10: {coverage:.1f}%)")
                print(f"       Suggerimento: considera di droppare questa feature")
                # Mantieni embedding ma con dimensione ridotta
                self.embedding_dims[feature] = 16
            else:
                self.embedding_dims[feature] = min(32, max(8, vocab_size // 100))
    
    def transform(self, data_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applica preprocessing alle geo features.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            Dict con features geografiche preprocessate
        """
        self._check_fitted()
        
        processed = {}
        
        for feature in self.feature_names:
            if feature in data_dict and feature in self.geo_lookups:
                # Converti a indices usando lookup
                indices = self.geo_lookups[feature](data_dict[feature])
                processed[f'{feature}_indices'] = indices
                
                # Applica strategia specifica
                strategy = self.geo_strategies[feature]
                
                if strategy == 'one_hot':
                    # One-hot encoding
                    vocab_size = self.geo_vocab_sizes[feature]
                    one_hot = tf.one_hot(
                        tf.cast(indices, tf.int32), 
                        depth=vocab_size,
                        name=f'{feature}_onehot'
                    )
                    processed[f'{feature}_encoded'] = one_hot
                    
                elif strategy == 'embedding':
                    # Gli embeddings verranno gestiti nel modello Keras
                    # Per ora salviamo solo gli indices
                    processed[f'{feature}_for_embedding'] = indices
        
        return processed
    
    def get_output_specs(self) -> Dict[str, Tuple[int, str]]:
        """
        Specifiche output per costruzione modello.
        
        Returns:
            Dict con {feature_name: (output_dim, encoding_type)}
        """
        specs = {}
        
        for feature in self.feature_names:
            if feature in self.geo_vocab_sizes:
                strategy = self.geo_strategies[feature]
                
                if strategy == 'one_hot':
                    # One-hot: dimensione = vocab_size
                    specs[feature] = (self.geo_vocab_sizes[feature], 'one_hot')
                    
                elif strategy == 'embedding':
                    # Embedding: dimensione personalizzata
                    embedding_dim = self.embedding_dims[feature]
                    specs[feature] = (embedding_dim, 'embedding')
        
        return specs
    
    def get_keras_layers(self, inputs: Dict[str, layers.Input]) -> Dict[str, tf.Tensor]:
        """
        Crea layer Keras per preprocessing geografico.
        
        Args:
            inputs: Dict con {feature_name: Input layer}
            
        Returns:
            Dict con {feature_name: processed tensor}
        """
        outputs = {}
        
        for feature in self.feature_names:
            if feature in inputs and feature in self.geo_lookups:
                input_layer = inputs[feature]
                
                # Applica lookup
                indices = self.geo_lookups[feature](input_layer)
                
                strategy = self.geo_strategies[feature]
                
                if strategy == 'one_hot':
                    # One-hot encoding
                    vocab_size = self.geo_vocab_sizes[feature]
                    encoded = layers.CategoryEncoding(
                        num_tokens=vocab_size,
                        output_mode='one_hot',
                        name=f'geo_onehot_{feature}'
                    )(indices)
                    outputs[feature] = encoded
                    
                elif strategy == 'embedding':
                    # Embedding layer
                    vocab_size = self.geo_vocab_sizes[feature]
                    embedding_dim = self.embedding_dims[feature]
                    
                    embedded = layers.Embedding(
                        input_dim=vocab_size,
                        output_dim=embedding_dim,
                        name=f'geo_embed_{feature}'
                    )(indices)
                    
                    # Flatten per concatenazione
                    flattened = layers.Flatten(name=f'geo_flat_{feature}')(embedded)
                    outputs[feature] = flattened
        
        return outputs
    
    def get_hierarchical_features(self) -> Dict[str, tf.Tensor]:
        """
        Crea features gerarchiche combinate (opzionale).
        Es: geo_level_1 + geo_level_2 come feature composita
        
        Returns:
            Dict con features gerarchiche
        """
        # Implementazione futura per sfruttare gerarchia geografica
        return {}
    
    def analyze_geographic_correlation(self, data_dict: Dict[str, tf.Tensor], 
                                     target: tf.Tensor) -> Dict[str, float]:
        """
        Analizza correlazione di ogni livello geografico con target.
        
        Args:
            data_dict: Dict con geo features
            target: Tensor target
            
        Returns:
            Dict con correlazioni
        """
        correlations = {}
        
        for feature in self.feature_names:
            if feature in data_dict:
                # Calcola correlazione Pearson (approssimata)
                geo_data = tf.cast(data_dict[feature], tf.float32)
                target_data = tf.cast(target, tf.float32)
                
                corr = tf.corrcoef(
                    tf.stack([geo_data, target_data], axis=0)
                )[0, 1]
                
                correlations[feature] = float(corr.numpy())
        
        return correlations
