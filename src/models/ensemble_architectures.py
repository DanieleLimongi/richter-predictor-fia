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
            
            # ✅ CORREZIONE: Dropout meno aggressivo
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
            
            # ✅ CORREZIONE: Dimensioni più ragionevoli per stabilità
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
        
        # ✅ CORREZIONE: Dimensioni calcolate dinamicamente
        # Usa una dimensione che sia compatibile con diverse input_dim
        hidden_dim = min(512, max(128, self.input_dim // 2))
        
        # First block con proiezione se necessario
        x = tf.keras.layers.BatchNormalization()(input_layer)
        
        # ✅ Proiezione iniziale per compatibilità dimensioni
        x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)  # Ridotto da 0.3
        
        # Residual block con dimensioni compatibili
        residual = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
        residual = tf.keras.layers.Dropout(0.15)(residual)  # Ridotto da 0.2
        
        # ✅ Add funziona perché x e residual hanno stessa dimensione
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
            
            # ✅ CORREZIONE: Regularization meno aggressiva
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
            
            # ✅ CORREZIONE: Dropout più conservativo per Swish
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
        
        # ✅ Feature attention weights con controllo di stabilità
        attention_weights = tf.keras.layers.Dense(
            self.input_dim, 
            activation='sigmoid', 
            name='feature_attention',
            kernel_initializer='glorot_uniform'  # Inizializzazione più stabile
        )(input_layer)
        
        # ✅ CORREZIONE: Aggiungi un piccolo epsilon per evitare zero gradients
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
    
    @staticmethod
    def focal_loss(gamma=2.0, alpha=0.25):
        """Focal Loss CORRETTA per gestire hard examples"""
        def focal_loss_fixed(y_true, y_pred):
            # ✅ CORREZIONE: Forza tutti i tensori a float32 da subito
            y_pred = tf.cast(y_pred, tf.float32)
            
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            # ✅ CORREZIONE: Rileva automaticamente n_classes
            n_classes = tf.shape(y_pred)[-1]
            
            # Convert to one-hot if needed e garantisci dtype float32
            y_true = tf.cast(y_true, tf.int32)
            y_true = tf.one_hot(y_true, depth=n_classes)  # ✅ Dinamico!
            y_true = tf.cast(y_true, tf.float32)  # ✅ CORREZIONE: Forza float32
            
            # ✅ CORREZIONE: Garantisci che alpha e gamma siano float32
            alpha_f32 = tf.cast(alpha, tf.float32)
            gamma_f32 = tf.cast(gamma, tf.float32)
            
            # Focal loss computation con tipi consistenti
            ce = -y_true * tf.math.log(y_pred)
            weight = alpha_f32 * y_true * tf.pow((1 - y_pred), gamma_f32)
            fl = weight * ce
            
            return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
        
        return focal_loss_fixed

    @staticmethod
    def f1_score_metric(y_true, y_pred):
        """Metrica F1-score ROBUSTA per TensorFlow"""
        def f1_score_fn(y_true, y_pred):
            # ✅ CORREZIONE: Gestione più robusta dei tipi
            y_pred_classes = tf.argmax(y_pred, axis=1)
            y_true = tf.cast(tf.squeeze(y_true), tf.int64)  # Squeeze per sicurezza
            y_pred_classes = tf.cast(y_pred_classes, tf.int64)
            
            # Micro F1 approximation (accuracy for multi-class)
            correct_predictions = tf.equal(y_true, y_pred_classes)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            
            return accuracy
        
        return f1_score_fn(y_true, y_pred)
    
    @staticmethod
    def get_diverse_optimizers():
        """Ottimizzatori BILANCIATI per massimizzare diversità"""
        return [
            # ✅ CORREZIONE: Learning rates più conservative
            tf.keras.optimizers.AdamW(learning_rate=0.0015, weight_decay=1e-4),  # Ridotto
            tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),  # Standard
            tf.keras.optimizers.RMSprop(learning_rate=0.0008, momentum=0.9, centered=True),  # Aumentato
            tf.keras.optimizers.Nadam(learning_rate=0.0007, beta_1=0.9, beta_2=0.999),  # Aumentato
            tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=1e-3),  # Aumentato
            tf.keras.optimizers.Adam(learning_rate=0.0012, beta_1=0.95, beta_2=0.999)  # Aumentato
        ]

    @staticmethod
    def get_diverse_loss_functions():
        """Loss functions OTTIMIZZATE per stabilità"""
        return [
            'sparse_categorical_crossentropy',  # Standard, stabile
            'sparse_categorical_crossentropy',  # Standard per diversità
            # ✅ CORREZIONE: Focal loss meno aggressiva
            EnsembleArchitectures.focal_loss(gamma=0.8, alpha=0.6),   # Molto più leggero
            'sparse_categorical_crossentropy',  # Standard per compatibilità
            'sparse_categorical_crossentropy',  # Standard per robustezza
            EnsembleArchitectures.focal_loss(gamma=0.5, alpha=0.7)    # Leggerissimo
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
