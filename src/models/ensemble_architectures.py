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
        """Architettura profonda e stretta - buona per pattern complessi"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_wide_shallow_architecture(self):
        """Architettura larga e poco profonda - cattura interazioni ampie"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_residual_like_architecture(self):
        """Architettura con skip connections - mitiga vanishing gradient"""
        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        
        # First block
        x = tf.keras.layers.BatchNormalization()(input_layer)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Residual block
        residual = tf.keras.layers.Dense(512, activation='relu')(x)
        residual = tf.keras.layers.Dropout(0.2)(residual)
        x = tf.keras.layers.Add()([x, residual])
        
        # Second block
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output
        output = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs=input_layer, outputs=output)

    def create_regularized_architecture(self):
        """Architettura con heavy regularization - previene overfitting"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(512, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_swish_activation_architecture(self):
        """Architettura con attivazione Swish - performance superiore a ReLU"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(512, activation='swish'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(256, activation='swish'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        
        return model

    def create_attention_like_architecture(self):
        """Architettura con meccanismo attention-like per feature importance"""
        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        
        # Feature attention weights - impara l'importanza di ogni feature
        attention_weights = tf.keras.layers.Dense(self.input_dim, activation='sigmoid', name='feature_attention')(input_layer)
        attended_features = tf.keras.layers.Multiply(name='apply_attention')([input_layer, attention_weights])
        
        # Standard layers con features ponderate
        x = tf.keras.layers.BatchNormalization()(attended_features)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
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
    
    @staticmethod
    def focal_loss(gamma=2.0, alpha=0.25):
        """Focal Loss per gestire hard examples e dataset sbilanciati"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # Convert to one-hot if needed
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.one_hot(y_true, depth=3)
            
            # Focal loss computation
            ce = -y_true * tf.math.log(y_pred)
            weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
            fl = weight * ce
            
            return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
        
        return focal_loss_fixed

    @staticmethod
    def f1_score_metric(y_true, y_pred):
        """Metrica F1-score per TensorFlow"""
        def f1_score_fn(y_true, y_pred):
            y_pred = tf.argmax(y_pred, axis=1)
            y_true = tf.cast(y_true, tf.int64)
            
            # Micro F1 approximation (accuracy for multi-class)
            tp = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))
            total = tf.cast(tf.shape(y_true)[0], tf.float32)
            
            return tp / total
        
        return f1_score_fn(y_true, y_pred)
    
    @staticmethod
    def get_diverse_optimizers():
        """Ottimizzatori diversi per massimizzare diversità ensemble"""
        return [
            tf.keras.optimizers.AdamW(learning_rate=0.002, weight_decay=1e-4),
            tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            tf.keras.optimizers.RMSprop(learning_rate=0.0007, momentum=0.9, centered=True),
            tf.keras.optimizers.Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999),
            tf.keras.optimizers.AdamW(learning_rate=0.0003, weight_decay=1e-3),
            tf.keras.optimizers.Adam(learning_rate=0.0008, beta_1=0.95, beta_2=0.999)
        ]

    @staticmethod
    def get_diverse_loss_functions():
        """Loss functions diverse per ensemble ottimizzate per il dataset"""
        return [
            'sparse_categorical_crossentropy',  # Standard, stabile
            'sparse_categorical_crossentropy',  # Standard (duplicato per stabilità)
            EnsembleArchitectures.focal_loss(gamma=1.0, alpha=0.75),  # Focal più leggero
            'sparse_categorical_crossentropy',  # Standard per compatibilità
            'sparse_categorical_crossentropy',   # Standard finale
            EnsembleArchitectures.focal_loss(gamma=0.5, alpha=0.85)   # Focal molto leggero
        ]


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
