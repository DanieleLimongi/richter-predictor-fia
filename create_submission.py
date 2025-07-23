#!/usr/bin/env python3
"""
Script per creare submission finale per DrivenData
Ricostruisce la pipeline di preprocessing e carica solo i weights del modello
"""

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Aggiungi path per import preprocessing
sys.path.append('/home/claudio/richter-predictor-fia/src')
from preprocessing.main_pipeline import RichterPreprocessingPipeline

def find_best_model():
    """Trova il modello con la migliore accuracy dai risultati salvati"""
    results_dir = Path('/home/claudio/richter-predictor-fia/reports/mlp_results')
    models_base_dir = Path('/home/claudio/richter-predictor-fia/models')
    
    best_accuracy = 0
    best_model_info = None
    
    print(" Ricerca del modello migliore...")
    
    # Cerca tutti i file di risultati JSON
    for result_file in results_dir.glob('full_preprocessing_results_*.json'):
        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            accuracy = results.get('final_accuracy', 0)
            timestamp = results.get('timestamp', '')
            
            print(f" {timestamp}: Accuracy = {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_info = results
                
        except Exception as e:
            print(f" Errore leggendo {result_file}: {e}")
            continue
    
    if best_model_info is None:
        raise ValueError("Nessun modello trovato!")
    
    print(f"\n Modello migliore trovato:")
    print(f"   Timestamp: {best_model_info['timestamp']}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   F1-Score: {best_model_info.get('holdout_f1', 'N/A')}")
    
    # Trova il file .keras corrispondente
    timestamp = best_model_info['timestamp']
    model_path = None
    
    # CASO 1: train_simple_holdout.py - modello diretto in models/
    if 'model_path' in best_model_info:
        # Il JSON contiene già il path del modello
        model_path = best_model_info['model_path']
        if Path(model_path).exists():
            print(f"   Model path (holdout): {Path(model_path).name}")
        else:
            print(f"   Model path nel JSON non esiste: {model_path}")
            model_path = None
    
    # CASO 2: train_final_nested_cv.py - modello in nested_cv_final_*/
    if model_path is None:
        nested_cv_dir = models_base_dir / f"nested_cv_final_{timestamp}"
        if nested_cv_dir.exists():
            keras_files = list(nested_cv_dir.glob('*.keras'))
            if keras_files:
                model_path = str(keras_files[0])
                print(f"   Model path (nested): {keras_files[0].name}")
            else:
                h5_files = list(nested_cv_dir.glob('*.h5'))
                if h5_files:
                    model_path = str(h5_files[0])
                    print(f"   Model path (nested h5): {h5_files[0].name}")
    
    # FALLBACK: cerca tutti i modelli con timestamp corrispondente
    if model_path is None:
        print("   Ricerca fallback per timestamp...")
        # Cerca in models/ direttamente
        direct_keras = list(models_base_dir.glob(f'*{timestamp}*.keras'))
        if direct_keras:
            model_path = str(direct_keras[0])
            print(f"   Model path (direct): {direct_keras[0].name}")
        else:
            # Cerca in tutte le directory nested_cv_final_*
            all_nested_dirs = list(models_base_dir.glob('nested_cv_final_*'))
            if all_nested_dirs:
                print(f"   Directory nested_cv disponibili: {len(all_nested_dirs)}")
                latest_dir = max(all_nested_dirs, key=lambda x: x.stat().st_mtime)
                keras_files = list(latest_dir.glob('*.keras'))
                h5_files = list(latest_dir.glob('*.h5'))
                if keras_files:
                    model_path = str(keras_files[0])
                    print(f"   Model path (latest dir): {keras_files[0].name}")
                elif h5_files:
                    model_path = str(h5_files[0])
                    print(f"   Model path (latest h5): {h5_files[0].name}")
    
    if model_path is None:
        raise ValueError(f"Nessun modello .keras/.h5 trovato per timestamp {timestamp}")
    
    # Aggiorna il model_path nel dizionario
    best_model_info['model_path'] = model_path
    print(f"   Path finale: {model_path}")
    
    return best_model_info

def load_train_and_test_data():
    """Carica sia i dati di training che di test"""
    print(" Caricamento dati di training e test...")
    
    # Carica training data per fit della pipeline
    train_values = pd.read_csv('/home/claudio/richter-predictor-fia/data/raw/train_values.csv')
    train_labels = pd.read_csv('/home/claudio/richter-predictor-fia/data/raw/train_labels.csv')
    train_df = train_values.merge(train_labels, on='building_id', how='inner')
    
    # Carica test data
    test_df = pd.read_csv('/home/claudio/richter-predictor-fia/data/raw/test_values.csv')
    
    print(f" Training data: {train_df.shape[0]} samples")
    print(f" Test data: {test_df.shape[0]} samples")
    
    return train_df, test_df

def recreate_and_fit_pipeline(train_df):
    """Ricrea e fitta la pipeline di preprocessing con gli stessi parametri"""
    print(" Ricreazione pipeline di preprocessing...")
    
    # Separa features e target dai training data
    feature_cols = [col for col in train_df.columns if col not in ['building_id', 'damage_grade']]
    X_train_df = train_df[feature_cols]
    
    # Inizializza pipeline con la stessa configurazione AdaptiveStrategy
    pipeline = RichterPreprocessingPipeline()
    
    # Setup con configurazione AdaptiveStrategy validata
    pipeline.setup_preprocessors(
        force_embedding_categorical=False,  # AdaptiveStrategy
        add_binary_count=True,             # AdaptiveStrategy  
        group_binary_correlated=True,      # AdaptiveStrategy
        outlier_detection=True             # AdaptiveStrategy
    )
    
    # Converti DataFrame in dict di tensori per la pipeline
    train_data_dict = {}
    for col in X_train_df.columns:
        if X_train_df[col].dtype == 'object':
            # Categorical features
            train_data_dict[col] = tf.constant(X_train_df[col].astype(str).values)
        else:
            # Numeric features
            train_data_dict[col] = tf.constant(X_train_df[col].astype(np.float32).values)
    
    print(" Fitting pipeline sui dati di training...")
    
    # Fit della pipeline
    pipeline.fit(train_data_dict)
    
    print(" Pipeline ricreata e fitted!")
    
    return pipeline

def preprocess_test_data(test_df, pipeline):
    """Applica il preprocessing ai dati di test"""
    print(" Applicazione preprocessing ai test data...")
    
    # Estrai building_id per submission
    building_ids = test_df['building_id'].copy()
    
    # Rimuovi building_id per preprocessing
    feature_cols = [col for col in test_df.columns if col != 'building_id']
    X_test_df = test_df[feature_cols]
    
    # Converti DataFrame in dict di tensori per la pipeline
    test_data_dict = {}
    for col in X_test_df.columns:
        if X_test_df[col].dtype == 'object':
            # Categorical features
            test_data_dict[col] = tf.constant(X_test_df[col].astype(str).values)
        else:
            # Numeric features
            test_data_dict[col] = tf.constant(X_test_df[col].astype(np.float32).values)
    
    # Transform dei dati usando la pipeline fitted
    X_transformed = pipeline.transform(test_data_dict)
    
    # Converti il risultato in array numpy
    feature_arrays = []
    for key, tensor in X_transformed.items():
        if len(tensor.shape) == 1:
            feature_arrays.append(tensor.numpy().reshape(-1, 1))
        else:
            feature_arrays.append(tensor.numpy())
    
    X_test = np.concatenate(feature_arrays, axis=1).astype(np.float32)
    
    print(f" Preprocessing completato! Shape finale: {X_test.shape}")
    
    return X_test, building_ids

def create_model_architecture(input_dim):
    """Ricrea l'architettura del modello"""
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
        tf.keras.layers.Dense(3, activation='softmax')
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

def load_model_weights_and_predict(model_path, X_test):
    """Carica i pesi del modello e genera predizioni"""
    print(" Caricamento modello...")
    
    # Configurazione GPU (se disponibile)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" GPU trovate: {len(gpus)}")
        except RuntimeError as e:
            print(f" Errore GPU: {e}")
    
    # Ricrea l'architettura del modello
    model = create_model_architecture(X_test.shape[1])
    print(f" Architettura ricreata: {model.count_params():,} parametri")
    
    # Carica i pesi salvati
    try:
        model.load_weights(model_path.replace('.keras', '.weights.h5'))
        print(" Pesi caricati da file .weights.h5")
    except:
        try:
            # Prova a caricare il modello completo e estrarre i pesi
            saved_model = tf.keras.models.load_model(model_path)
            model.set_weights(saved_model.get_weights())
            print(" Pesi estratti dal modello completo")
            del saved_model
        except Exception as e:
            print(f" Errore caricamento pesi: {e}")
            # Se tutto fallisce, carica il modello completo saltando i layer problematici
            print(" Tentativo caricamento modello con custom_objects...")
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print(" Modello caricato con compile=False")
    
    print(" Generazione predizioni...")
    
    # Genera predizioni
    predictions_prob = model.predict(X_test, batch_size=2048, verbose=1)
    
    # Converti probabilità in classi (0, 1, 2) e poi in damage_grade (1, 2, 3)
    predictions_classes = np.argmax(predictions_prob, axis=1)
    damage_grades = predictions_classes + 1  # Converti da 0-2 a 1-3
    
    print(f" Predizioni generate!")
    print(f" Distribuzione predizioni:")
    unique, counts = np.unique(damage_grades, return_counts=True)
    for grade, count in zip(unique, counts):
        percentage = count / len(damage_grades) * 100
        print(f"   Grade {grade}: {count:,} samples ({percentage:.1f}%)")
    
    return damage_grades, predictions_prob

def create_submission_file(building_ids, damage_grades, model_info):
    """Crea il file di submission per DrivenData"""
    print(" Creazione file submission...")
    
    # Crea DataFrame per submission
    submission_df = pd.DataFrame({
        'building_id': building_ids,
        'damage_grade': damage_grades
    })
    
    # Verifica formato
    print(f" Submission format check:")
    print(f"   Righe: {len(submission_df):,}")
    print(f"   Colonne: {list(submission_df.columns)}")
    print(f"   Valori damage_grade unici: {sorted(submission_df['damage_grade'].unique())}")
    print(f"   Valori mancanti: {submission_df.isnull().sum().sum()}")
    
    # Nome file con timestamp del modello
    timestamp = model_info['timestamp']
    accuracy = model_info['final_accuracy']
    submission_filename = f"submission_acc{accuracy:.4f}_{timestamp}.csv"
    submission_path = f"/home/claudio/richter-predictor-fia/{submission_filename}"
    
    # Salva file
    submission_df.to_csv(submission_path, index=False)
    
    print(f" Submission salvata: {submission_path}")
    
    # Mostra sample
    print(f"\n Sample submission (primi 10 righi):")
    print(submission_df.head(10).to_string(index=False))
    
    return submission_path

def main():
    """Funzione principale"""
    print(" RICHTER PREDICTOR - CREAZIONE SUBMISSION FINALE")
    print("=" * 60)
    
    try:
        # 1. Trova il modello migliore
        best_model_info = find_best_model()
        
        # 2. Carica dati di training e test
        train_df, test_df = load_train_and_test_data()
        
        # 3. Ricrea e fitta la pipeline
        pipeline = recreate_and_fit_pipeline(train_df)
        
        # 4. Applica preprocessing ai test data
        X_test, building_ids = preprocess_test_data(test_df, pipeline)
        
        # 5. Carica modello e genera predizioni
        model_path = best_model_info['model_path']
        damage_grades, predictions_prob = load_model_weights_and_predict(model_path, X_test)
        
        # 6. Crea file submission
        submission_path = create_submission_file(building_ids, damage_grades, best_model_info)
        
        print(f"\n SUBMISSION CREATA CON SUCCESSO!")
        print(f" File: {submission_path}")
        print(f" Modello: Accuracy {best_model_info['final_accuracy']:.4f}")
        print(f" Samples: {len(building_ids):,}")
        print(f"\n Pronto per il submit su DrivenData!")
        
        return submission_path
        
    except Exception as e:
        print(f" Errore durante la creazione submission: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    submission_path = main()
