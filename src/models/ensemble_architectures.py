"""
Architetture ottimizzate per ensemble learning
Diverse architetture per massimizzare diversità e performance
"""

import tensorflow as tf
import numpy as np

class EnsembleArchitectures:
    """
    Classe per gestire diverse architetture ensemble con massima diversità
    """
    
    def __init__(self, input_dim, n_classes=3):
        # Validazione parametri input
        if input_dim <= 0:
            raise ValueError(f"input_dim deve essere positivo, ricevuto: {input_dim}")
        if n_classes <= 0:
            raise ValueError(f"n_classes deve essere positivo, ricevuto: {n_classes}")
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # Registry delle architetture disponibili
        self.architecture_registry = {
            'deep_narrow': self.create_deep_narrow_architecture,
            'wide_shallow': self.create_wide_shallow_architecture,
            'residual_like': self.create_residual_like_architecture,
            'regularized': self.create_regularized_architecture,
            'swish_activation': self.create_swish_activation_architecture,
            'attention_like': self.create_attention_like_architecture
        }
    
    def create_deep_narrow_architecture(self):
        """Architettura profonda e stretta - OTTIMIZZATA"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            # CORREZIONE: Dropout meno aggressivo
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.35),  # Ridotto da 0.4
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.25),  # Ridotto da 0.3
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.15),  # Ridotto da 0.2
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),   # OK
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_wide_shallow_architecture(self):
        """Architettura larga e poco profonda - OTTIMIZZATA"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            # CORREZIONE: Dimensioni più ragionevoli per stabilità
            tf.keras.layers.Dense(800, activation='relu'),  # Ridotto da 1024
            tf.keras.layers.Dropout(0.4),   # Ridotto da 0.5
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(400, activation='relu'),  # Ridotto da 512
            tf.keras.layers.Dropout(0.3),   # Ridotto da 0.4
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_residual_like_architecture(self):
        """Architettura con skip connections - CORRETTA per stabilità"""
        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        
        # CORREZIONE: Dimensioni calcolate dinamicamente
        # Usa una dimensione che sia compatibile con diverse input_dim
        hidden_dim = min(512, max(128, self.input_dim // 2))
        
        # First block con proiezione se necessario
        x = tf.keras.layers.BatchNormalization()(input_layer)
        
        # Proiezione iniziale per compatibilità dimensioni
        x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)  # Ridotto da 0.3
        
        # Residual block con dimensioni compatibili
        residual = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
        residual = tf.keras.layers.Dropout(0.15)(residual)  # Ridotto da 0.2
        
        # Add funziona perché x e residual hanno stessa dimensione
        x = tf.keras.layers.Add()([x, residual])
        
        # Second block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(hidden_dim // 2, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.15)(x)  # Ridotto da 0.2
        
        # Output
        output = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs=input_layer, outputs=output)

    def create_regularized_architecture(self):
        """Architettura con regularization BILANCIATA"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            # CORREZIONE: Regularization meno aggressiva
            tf.keras.layers.Dense(512, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-6, l2=1e-4)),  # Ridotto l1
            tf.keras.layers.Dropout(0.4),   # Ridotto da 0.5
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-6, l2=1e-4)),  # Ridotto l1
            tf.keras.layers.Dropout(0.3),   # Ridotto da 0.4
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),   # Ridotto da 0.3
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_swish_activation_architecture(self):
        """Architettura con attivazione Swish - OTTIMIZZATA"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            # CORREZIONE: Dropout più conservativo per Swish
            tf.keras.layers.Dense(512, activation='swish'),
            tf.keras.layers.Dropout(0.3),   # Ridotto da 0.4
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(256, activation='swish'),
            tf.keras.layers.Dropout(0.25),  # Ridotto da 0.3
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(128, activation='relu'),  # Mix con ReLU per stabilità
            tf.keras.layers.Dropout(0.15),  # Ridotto da 0.2
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_attention_like_architecture(self):
        """Architettura con meccanismo attention-like - ROBUSTA"""
        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        
        # Feature attention weights con controllo di stabilità
        attention_weights = tf.keras.layers.Dense(
            self.input_dim, 
            activation='sigmoid', 
            name='feature_attention',
            kernel_initializer='glorot_uniform'  # Inizializzazione più stabile
        )(input_layer)
        
        # CORREZIONE: Aggiungi un piccolo epsilon per evitare zero gradients
        attention_weights = tf.keras.layers.Lambda(
            lambda x: x + 1e-7, 
            name='attention_stabilization'
        )(attention_weights)
        
        attended_features = tf.keras.layers.Multiply(name='apply_attention')([input_layer, attention_weights])
        
        # Standard layers con dropout ottimizzato
        x = tf.keras.layers.BatchNormalization()(attended_features)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)  # Ridotto da 0.3
        
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)   # OK
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)   # OK
        
        output = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs=input_layer, outputs=output)
    
    def get_available_architectures(self):
        """Restituisce lista delle architetture disponibili"""
        return list(self.architecture_registry.keys())
    
    def create_architecture(self, arch_name):
        """Crea una specifica architettura"""
        if arch_name not in self.architecture_registry:
            raise ValueError(f"Architettura '{arch_name}' non disponibile. Disponibili: {self.get_available_architectures()}")
        
        return self.architecture_registry[arch_name]()
    
    def create_ensemble_models(self, n_models=6):
        """Crea n modelli diversi per ensemble"""
        arch_names = self.get_available_architectures()[:n_models]
        
        models = []
        for arch_name in arch_names:
            model = self.create_architecture(arch_name)
            models.append((arch_name, model))
        
        return models

# Manteniamo compatibilità per codice esistente
def create_deep_narrow_architecture(input_dim):
    """Wrapper per compatibilità - deprecato, usa EnsembleArchitectures"""
    arch = EnsembleArchitectures(input_dim)
    return arch.create_deep_narrow_architecture()

def create_wide_shallow_architecture(input_dim):
    """Wrapper per compatibilità - deprecato, usa EnsembleArchitectures"""
    arch = EnsembleArchitectures(input_dim)
    return arch.create_wide_shallow_architecture()

def get_ensemble_architectures(input_dim, n_models=6):
    """Wrapper per compatibilità - deprecato, usa EnsembleArchitectures"""
    arch = EnsembleArchitectures(input_dim)
    return arch.create_ensemble_models(n_models)
