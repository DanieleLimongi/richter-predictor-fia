#!/usr/bin/env python3
"""
Training MLP semplice per Richter Predictor
Utilizza pipeline di preprocessing completa + GPU ottimizzata
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Aggiungi path per import preprocessing
sys.path.append('/home/claudio/richter-predictor-fia/src')
from preprocessing.main_pipeline import RichterPreprocessingPipeline

# Configurazione GPU/CPU ottimizzata
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.optimizer.set_jit(True)  # XLA compilation

# Usa tutte le GPU disponibili
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" GPU trovate: {len(gpus)}")
    except RuntimeError as e:
        print(f"Errore GPU: {e}")

# Thread optimization
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

def load_and_preprocess_data():
    """Carica dati raw e applica pipeline di preprocessing completa"""
    print(" Caricamento dati raw...")
    
    # Carica dati raw
    train_values = pd.read_csv('/home/claudio/richter-predictor-fia/data/raw/train_values.csv')
    train_labels = pd.read_csv('/home/claudio/richter-predictor-fia/data/raw/train_labels.csv')
    
    # Merge training data
    train_df = train_values.merge(train_labels, on='building_id', how='inner')
    
    print(f" Dati raw caricati: {train_df.shape[0]} samples, {train_df.shape[1]} features")
    
    # Separa features e target
    feature_cols = [col for col in train_df.columns if col not in ['building_id', 'damage_grade']]
    X_df = train_df[feature_cols]
    y = train_df['damage_grade'] - 1  # Convert to 0-2 for model
    
    print(" Usando configurazione AdaptiveStrategy (validata da Nested K-Fold)")
    print(" Inizializzazione pipeline di preprocessing...")
    
    # Inizializza pipeline senza config path (creeremo manualmente)
    pipeline = RichterPreprocessingPipeline()
    
    # Setup con configurazione AdaptiveStrategy validata
    pipeline.setup_preprocessors(
        force_embedding_categorical=False,  # AdaptiveStrategy
        add_binary_count=True,             # AdaptiveStrategy  
        group_binary_correlated=True,      # AdaptiveStrategy
        outlier_detection=True             # AdaptiveStrategy
    )
    
    # Converti DataFrame in dict di tensori per la pipeline
    data_dict = {}
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            # Categorical features
            data_dict[col] = tf.constant(X_df[col].astype(str).values)
        else:
            # Numeric features
            data_dict[col] = tf.constant(X_df[col].astype(np.float32).values)
    
    print(" Fitting pipeline sui dati di training...")
    
    # Fit della pipeline
    pipeline.fit(data_dict)
    
    print(" Trasformazione dati...")
    
    # Transform dei dati
    X_transformed = pipeline.transform(data_dict)
    
    # Converti il risultato in array numpy per sklearn
    # Concatena tutte le features trasformate
    feature_arrays = []
    for key, tensor in X_transformed.items():
        if len(tensor.shape) == 1:
            feature_arrays.append(tensor.numpy().reshape(-1, 1))
        else:
            feature_arrays.append(tensor.numpy())
    
    X = np.concatenate(feature_arrays, axis=1).astype(np.float32)
    y = y.astype(np.int32)
    
    print(f" Preprocessing completato!")
    print(f" Shape finale: {X.shape}")
    print(f" Target distribution: {np.bincount(y)}")
    
    return X, y, pipeline

def create_optimized_mlp(input_dim, num_classes=3):
    """Crea MLP ottimizzata per GPU"""
    
    model = tf.keras.Sequential([
        # Input layer con normalizzazione
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        
        # Hidden layers con dropout
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer ottimizzato
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Training principale"""
    print(" RICHTER PREDICTOR - MLP TRAINING CON PREPROCESSING COMPLETO")
    print("=" * 60)
    
    # Carica e preprocessa dati
    X, y, preprocessing_pipeline = load_and_preprocess_data()
    
    # HOLDOUT: Prima separa il test set finale (mai visto dal modello)
    X_work, X_holdout, y_work, y_holdout = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Train/Val split sui dati rimanenti
    X_train, X_val, y_train, y_val = train_test_split(
        X_work, y_work, test_size=0.2, random_state=42, stratify=y_work
    )
    
    print(f" Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Holdout: {X_holdout.shape[0]}")
    print(f" Split: {X_train.shape[0]/len(X)*100:.1f}% train | {X_val.shape[0]/len(X)*100:.1f}% val | {X_holdout.shape[0]/len(X)*100:.1f}% holdout")
    
    # Crea modello
    model = create_optimized_mlp(X_train.shape[1])
    print(f" Modello creato: {model.count_params():,} parametri")
    
    # Callbacks ULTRA-PAZIENTI per training esteso
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=50,  #  Doppia pazienza
            restore_best_weights=True,
            min_delta=0.00005  #  Soglia ancora più fine
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',  # Monitora accuracy
            factor=0.8,  #  Riduzione ancora più graduale (80% del precedente)
            patience=35,  #  Più pazienza prima di ridurre
            min_lr=1e-9,  #  LR minimo ultra-basso
            mode='max',  # Massimizza accuracy
            verbose=1
        )
    ]
    
    # Training
    print("\n Inizio training...")
    history = model.fit(
        X_train, y_train,
        batch_size=1024,  # Batch size grande per GPU
        epochs=300,  #  Più epoche per training esteso
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1  # Mostra progress nativamente
    )
    
    # Valutazione finale su holdout (mai visto dal modello)
    print("\n Valutazione finale su holdout...")
    y_pred_holdout = model.predict(X_holdout, batch_size=2048, verbose=0)
    y_pred_holdout_classes = np.argmax(y_pred_holdout, axis=1)
    
    holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout_classes)
    holdout_f1 = f1_score(y_holdout, y_pred_holdout_classes, average='weighted')
    
    print(f" Holdout Accuracy (stima non distorta): {holdout_accuracy:.4f}")
    print(f" Holdout F1-Score: {holdout_f1:.4f}")
    
    # Report dettagliato
    print("\n Classification Report (Holdout):")
    print(classification_report(y_holdout, y_pred_holdout_classes, 
                              target_names=['Grade 1', 'Grade 2', 'Grade 3']))
    
    # Salva modello e pipeline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'/home/claudio/richter-predictor-fia/models/mlp_model_full_preprocessing_{timestamp}.keras'
    pipeline_path = f'/home/claudio/richter-predictor-fia/models/preprocessing_pipeline_{timestamp}.pkl'
    
    # Salva modello
    model.save(model_path)
    
    # Salva pipeline di preprocessing
    with open(pipeline_path, 'wb') as f:
        pickle.dump(preprocessing_pipeline, f)
    
    # Salva risultati
    results = {
        'timestamp': timestamp,
        'final_accuracy': float(holdout_accuracy),
        'holdout_f1': float(holdout_f1),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'holdout_samples': int(X_holdout.shape[0]),
        'features_count': int(X_train.shape[1]),
        'model_path': model_path,
        'preprocessing_pipeline_path': pipeline_path,
        'used_full_preprocessing': True,
        'holdout_split': True,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    results_path = f'/home/claudio/richter-predictor-fia/reports/mlp_results/full_preprocessing_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f" Modello salvato: {model_path}")
    print(f" Pipeline salvata: {pipeline_path}")
    print(f" Risultati salvati: {results_path}")
    print("\n Training con preprocessing completo completato!")
    
    return model, preprocessing_pipeline, results

if __name__ == "__main__":
    model, pipeline, results = main()
