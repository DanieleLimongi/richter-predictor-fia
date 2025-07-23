#!/usr/bin/env python3
"""
Categorical Preprocessor - Richter Predictor
Gestisce features categoriche standard (non geografiche)
"""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, List, Tuple, Optional
import numpy as np
from .base_preprocessor import BasePreprocessor


class CategoricalPreprocessor(BasePreprocessor):
    """
    Preprocessore specializzato per features categoriche standard.
    
    Gestisce: foundation_type, roof_type, ground_floor_type, etc.
    
    Strategia adattiva:
    - Bassa cardinalità (≤ 10): One-hot encoding
    - Media cardinalità (11-50): One-hot o Embedding basato su entropia
    - Alta cardinalità (> 50): Embedding obbligatorio
    """
    
    def __init__(self, feature_names: List[str], force_embedding: bool = False):
        """
        Inizializza il preprocessore categorico.
        
        Args:
            feature_names: Lista con features categoriche
            force_embedding: Se forzare embedding per tutte le features
        """
        super().__init__(feature_names, "CategoricalPreprocessor")
        
        self.force_embedding = force_embedding
        
        # Lookup tables
        self.categorical_lookups = {}
        self.categorical_vocabs = {}
        self.vocab_sizes = {}
        
        # Strategie per ogni feature
        self.encoding_strategies = {}
        self.embedding_dims = {}
        
        # Analisi entropia
        self.entropy_scores = {}
    
    def fit(self, data_dict: Dict[str, tf.Tensor]) -> 'CategoricalPreprocessor':
        """
        Adatta lookup tables e determina strategie di encoding.
        
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
            
            # Crea vocabolario
            unique_values = tf.unique(tf.reshape(feature_data, [-1]))[0]
            vocab = sorted(unique_values.numpy().tolist())
            vocab_size = len(vocab) + 1  # +1 per OOV token
            
            # Crea StringLookup (non IntegerLookup per valori stringa)
            lookup = layers.StringLookup(
                vocabulary=vocab,
                mask_token=None,
                oov_token="[UNK]",  # OOV diventa token speciale
                name=f'cat_lookup_{feature}'
            )
            
            self.categorical_lookups[feature] = lookup
            self.categorical_vocabs[feature] = vocab
            self.vocab_sizes[feature] = vocab_size
            
            # Analizza distribuzione e calcola entropia
            entropy = self._calculate_entropy(feature_data, vocab)
            self.entropy_scores[feature] = entropy
            
            # Determina strategia di encoding
            strategy = self._determine_encoding_strategy(feature, vocab_size, entropy)
            self.encoding_strategies[feature] = strategy
            
            # Calcola dimensione embedding se necessaria
            if strategy == 'embedding':
                self.embedding_dims[feature] = self._calculate_embedding_dim(vocab_size)
            
            # Salva metadati
            self.metadata[f'{feature}_vocab_size'] = len(vocab)
            self.metadata[f'{feature}_strategy'] = strategy
            self.metadata[f'{feature}_entropy'] = entropy
            
            print(f"       {len(vocab)} categorie, entropia={entropy:.2f}")
            print(f"       Strategia: {strategy}")
            
            if strategy == 'embedding':
                embed_dim = self.embedding_dims[feature]
                print(f"        Embedding dim: {embed_dim}")

        self.is_fitted = True
        print(f"    {self.name} fitted!")
        return self
    
    def _calculate_entropy(self, data: tf.Tensor, vocab: List) -> float:
        """Calcola entropia di Shannon per la feature"""
        # Conta frequenze
        values, _, counts = tf.unique_with_counts(tf.reshape(data, [-1]))
        
        # Calcola probabilità
        total_count = tf.reduce_sum(counts)
        probs = tf.cast(counts, tf.float32) / tf.cast(total_count, tf.float32)
        
        # Calcola entropia: H = -Σ(p * log2(p))
        log_probs = tf.math.log(probs + 1e-10) / tf.math.log(2.0)  # log2
        entropy = -tf.reduce_sum(probs * log_probs)
        
        return float(entropy.numpy())
    
    def _determine_encoding_strategy(self, feature: str, vocab_size: int, entropy: float) -> str:
        """
        Determina strategia di encoding basata su cardinalità ed entropia.
        
        Args:
            feature: Nome feature
            vocab_size: Dimensione vocabolario (+1 per OOV)
            entropy: Score di entropia
            
        Returns:
            'one_hot' o 'embedding'
        """
        if self.force_embedding:
            return 'embedding'
        
        # Regole adattive
        if vocab_size <= 10:
            # Bassa cardinalità → sempre one-hot
            return 'one_hot'
        
        elif vocab_size <= 50:
            # Media cardinalità → basato su entropia
            if entropy < 2.0:  # Distribuzione molto sbilanciata
                return 'embedding'  # One-hot sprecherebbe spazio
            else:
                return 'one_hot'  # Distribuzione equilibrata
        
        else:
            # Alta cardinalità → sempre embedding
            return 'embedding'
    
    def _calculate_embedding_dim(self, vocab_size: int) -> int:
        """
        Calcola dimensione ottimale per embedding.
        
        Rule of thumb: min(50, vocab_size^0.25 * 6)
        """
        base_dim = int(np.power(vocab_size, 0.25) * 6)
        return min(50, max(4, base_dim))
    
    def transform(self, data_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applica preprocessing alle features categoriche.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            Dict con features categoriche preprocessate
        """
        self._check_fitted()
        
        processed = {}
        
        for feature in self.feature_names:
            if feature in data_dict and feature in self.categorical_lookups:
                # Converti a indices
                indices = self.categorical_lookups[feature](data_dict[feature])
                
                strategy = self.encoding_strategies[feature]
                
                if strategy == 'one_hot':
                    # One-hot encoding immediato
                    vocab_size = self.vocab_sizes[feature]
                    one_hot = tf.one_hot(
                        tf.cast(indices, tf.int32),
                        depth=vocab_size,
                        name=f'{feature}_onehot'
                    )
                    processed[feature] = one_hot
                    
                elif strategy == 'embedding':
                    # Salva indices per embedding nel modello Keras
                    processed[f'{feature}_indices'] = indices
        
        return processed
    
    def get_output_specs(self) -> Dict[str, Tuple[int, str]]:
        """
        Specifiche output per features categoriche.
        
        Returns:
            Dict con {feature_name: (output_dim, encoding_type)}
        """
        specs = {}
        
        for feature in self.feature_names:
            if feature in self.encoding_strategies:
                strategy = self.encoding_strategies[feature]
                
                if strategy == 'one_hot':
                    vocab_size = self.vocab_sizes[feature]
                    specs[feature] = (vocab_size, 'one_hot')
                    
                elif strategy == 'embedding':
                    embed_dim = self.embedding_dims[feature]
                    specs[feature] = (embed_dim, 'embedding')
        
        return specs
    
    def get_keras_layers(self, inputs: Dict[str, layers.Input]) -> Dict[str, tf.Tensor]:
        """
        Crea layer Keras per preprocessing categorico.
        
        Args:
            inputs: Dict con {feature_name: Input layer}
            
        Returns:
            Dict con {feature_name: processed tensor}
        """
        outputs = {}
        
        for feature in self.feature_names:
            if feature in inputs and feature in self.categorical_lookups:
                input_layer = inputs[feature]
                
                # Applica lookup
                indices = self.categorical_lookups[feature](input_layer)
                
                strategy = self.encoding_strategies[feature]
                
                if strategy == 'one_hot':
                    # One-hot encoding
                    vocab_size = self.vocab_sizes[feature]
                    encoded = layers.CategoryEncoding(
                        num_tokens=vocab_size,
                        output_mode='one_hot',
                        name=f'cat_onehot_{feature}'
                    )(indices)
                    outputs[feature] = encoded
                    
                elif strategy == 'embedding':
                    # Embedding layer
                    vocab_size = self.vocab_sizes[feature]
                    embed_dim = self.embedding_dims[feature]
                    
                    embedded = layers.Embedding(
                        input_dim=vocab_size,
                        output_dim=embed_dim,
                        name=f'cat_embed_{feature}'
                    )(indices)
                    
                    # Flatten per concatenazione
                    flattened = layers.Flatten(name=f'cat_flat_{feature}')(embedded)
                    outputs[feature] = flattened
        
        return outputs
    
    def analyze_categorical_distributions(self) -> Dict[str, Dict]:
        """
        Analizza distribuzioni delle features categoriche.
        
        Returns:
            Dict con analisi dettagliata per feature
        """
        analysis = {}
        
        for feature in self.feature_names:
            if feature in self.categorical_vocabs:
                vocab = self.categorical_vocabs[feature]
                entropy = self.entropy_scores[feature]
                strategy = self.encoding_strategies[feature]
                
                analysis[feature] = {
                    'vocab_size': len(vocab),
                    'entropy': entropy,
                    'strategy': strategy,
                    'vocab_sample': vocab[:10],  # Prime 10 categorie
                    'embedding_dim': self.embedding_dims.get(feature, None)
                }
        
        return analysis
    
    def get_rare_categories(self, data_dict: Dict[str, tf.Tensor], 
                           threshold: float = 0.01) -> Dict[str, List]:
        """
        Identifica categorie rare (< threshold % dei dati).
        
        Args:
            data_dict: Dict con dati originali
            threshold: Soglia percentuale per categoria rara
            
        Returns:
            Dict con {feature_name: [rare_categories]}
        """
        rare_categories = {}
        
        for feature in self.feature_names:
            if feature in data_dict:
                # Conta frequenze
                values, _, counts = tf.unique_with_counts(
                    tf.reshape(data_dict[feature], [-1])
                )
                
                total_count = tf.reduce_sum(counts)
                threshold_count = tf.cast(total_count, tf.float32) * threshold
                
                # Trova categorie rare
                rare_mask = tf.cast(counts, tf.float32) < threshold_count
                rare_vals = tf.boolean_mask(values, rare_mask)
                
                rare_categories[feature] = rare_vals.numpy().tolist()
        
        return rare_categories
