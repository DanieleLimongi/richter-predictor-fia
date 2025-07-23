#!/usr/bin/env python3
"""
Final Training with Nested Cross-Validation - Richter Predictor
Training finale con Nested K-Fold Cross Validation per risultati robusti
150 epoche, early stopping 15, configurazione AdaptiveStrategy validata
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
from datetime import datetime
import time
import joblib
from tqdm import tqdm
import multiprocessing

# Ottimizzazioni per Intel Ultra 9 185H (16 cores, 22 threads)
def configure_ultra9_185h():
    """Ottimizza per Intel Ultra 9 185H con 6 P-cores + 8 E-cores + 2 LP-E"""
    
    # Threading ottimale per Ultra 9 185H
    os.environ['OMP_NUM_THREADS'] = '22'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '22'
    os.environ['TF_NUM_INTEROP_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '22'
    
    # Cache optimization per L3 24MB
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    
    # Memory allocation ottimizzata
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Intel specific optimizations
    if hasattr(tf.config, 'threading'):
        tf.config.threading.set_intra_op_parallelism_threads(22)
        tf.config.threading.set_inter_op_parallelism_threads(4)
    
    # Pandas ottimizzazioni
    pd.set_option('mode.chained_assignment', None)
    
    print(f" OTTIMIZZATO PER INTEL ULTRA 9 185H:")
    print(f"    P-cores: 6 (fino a 5.1 GHz)")
    print(f"    E-cores: 8 + 2 LP-E")
    print(f"    Threads: 22 totali")
    print(f"    Cache L3: 24MB")
    print(f"    Parallel training attivo")

# Configurazione Ultra 9 185H
configure_ultra9_185h()

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.main_pipeline import RichterPreprocessingPipeline

# Configurazione GPU/CPU
print("Configurazione TensorFlow...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.enable_op_determinism()
        tf.config.optimizer.set_jit(True)
        print(f"GPU trovate: {len(gpus)}")
    else:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
        print("Utilizzo CPU ottimizzato")
    
    # Set seed per determinismo
    tf.random.set_seed(42)
    np.random.seed(42)
    
except Exception as e:
    print(f"Configurazione hardware: {e}")

def load_and_preprocess_data():
    """Carica e preprocessa i dati usando la configurazione AdaptiveStrategy validata"""
    print("Caricamento dati raw...")
    
    # Carica dati senza usare building_id come index
    train_values = pd.read_csv('data/raw/train_values.csv')
    train_labels = pd.read_csv('data/raw/train_labels.csv')
    
    # Merge sui building_id
    data = train_values.merge(train_labels, on='building_id', how='inner')
    
    # Separa features e target
    feature_cols = [col for col in data.columns if col not in ['building_id', 'damage_grade']]
    X = data[feature_cols]
    y = data['damage_grade'] - 1  # 0, 1, 2 per TensorFlow
    
    print(f"Dati caricati: {len(data):,} samples, {len(X.columns)} features")
    
    # Inizializza pipeline con configurazione AdaptiveStrategy validata
    print("Inizializzazione pipeline...")
    pipeline = RichterPreprocessingPipeline()
    
    # Configurazione validata da Nested K-Fold
    pipeline.setup_preprocessors(
        force_embedding_categorical=False,
        add_binary_count=True,
        group_binary_correlated=True,
        outlier_detection=True
    )
    
    print("Fitting pipeline...")
    pipeline.fit(X)
    X_processed = pipeline.transform(X)
    
    # La pipeline restituisce tensori TensorFlow, convertiamo in array numpy
    if isinstance(X_processed, dict):
        feature_arrays = []
        
        # Seleziona solo le features che dovremmo usare per il training
        # (escludi gli indici che sono per embedding layers)
        features_to_use = [
            'geo_level_1_id_encoded',  # One-hot encoded
            'count_families', 'count_floors_pre_eq', 'age',  # Numeric
            'foundation_type', 'roof_type', 'ground_floor_type', 
            'other_floor_type', 'position', 'land_surface_condition', 
            'legal_ownership_status',  # One-hot categorical
            'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone',
            'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick',
            'has_superstructure_timber', 'has_superstructure_bamboo',
            'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
            'has_superstructure_other', 'binary_total_count'  # Binary features
        ]
        
        for feature_name in features_to_use:
            if feature_name in X_processed:
                tensor = X_processed[feature_name]
                # Converti tensor in numpy
                array = tensor.numpy()
                # Assicurati che sia 2D
                if len(array.shape) == 1:
                    array = array.reshape(-1, 1)
                feature_arrays.append(array)
                print(f"Aggiunto {feature_name}: shape {array.shape}")
        
        # Concatena tutti gli array
        X_processed = np.concatenate(feature_arrays, axis=1)
    
    print(f"Preprocessing completato: {X_processed.shape}")
    
    return X_processed, y, pipeline

def create_model_architecture(input_dim, fold=0, num_classes=3):
    """
    Crea architetture ottimizzate per Ultra 9 185H con training veloce
    """
    
    # Architetture più snelle e veloci per Ultra 9 185H
    architectures = [
        [128, 64],            # Fold 1: Velocissima
        [192, 96],            # Fold 2: Piccola
        [160, 80],            # Fold 3: Compatta
        [256, 128],           # Fold 4: Media
        [224, 112]            # Fold 5: Bilanciata
    ]
    
    # Learning rates più alti per convergenza veloce
    learning_rates = [0.003, 0.004, 0.005, 0.0035, 0.0045]
    
    # Dropout ridotto per velocità
    dropout_rates = [0.1, 0.15, 0.2, 0.12, 0.18]
    
    arch = architectures[fold % len(architectures)]
    lr = learning_rates[fold % len(learning_rates)]
    dropout = dropout_rates[fold % len(dropout_rates)]
    
    print(f"Fold {fold+1}: Arch {arch}, LR {lr}, Dropout {dropout}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(arch[0], activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(arch[1], activation='relu'),
        tf.keras.layers.Dropout(dropout/2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Ottimizzatore con parametri per velocità
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks():
    """Callbacks ottimizzati per training bilanciato"""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Patience bilanciata per training efficiente
            restore_best_weights=True,
            mode='max',
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Riduzione più graduale
            patience=6,   # Patience ridotta
            min_lr=1e-7,
            verbose=0
        )
    ]

def nested_cross_validation(X, y, outer_folds=5, inner_folds=3):
    """
    Nested Cross-Validation per training finale robusto
    
    Args:
        X: Features preprocessate
        y: Target
        outer_folds: Folds esterni per valutazione finale
        inner_folds: Folds interni per model selection
    """
    print("NESTED CROSS-VALIDATION TRAINING")
    print("=" * 60)
    print(f"Outer folds: {outer_folds} | Inner folds: {inner_folds}")
    print(f"Epoche max: 100 | Early stopping: 15")
    print(f"Batch size: 2048 | Training bilanciato")
    print("=" * 60)
    
    # Conversione a numpy per cross-validation
    X = np.array(X)
    y = np.array(y)
    
    # Setup CV stratificato
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=123)
    
    outer_scores = []
    all_models = []
    fold_results = []
    
    start_time = time.time()
    total_iterations = outer_folds * inner_folds
    current_iteration = 0
    
    # Progress bar principale
    main_pbar = tqdm(total=outer_folds, desc="Outer Folds", position=0)
    
    # OUTER LOOP - Valutazione finale
    for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X, y)):
        main_pbar.set_description(f"Outer Fold {outer_fold+1}/{outer_folds}")
        
        # Split outer
        X_train_outer = X[train_outer_idx]
        X_test_outer = X[test_outer_idx]
        y_train_outer = y[train_outer_idx]
        y_test_outer = y[test_outer_idx]
        
        tqdm.write(f"\nOUTER FOLD {outer_fold+1}/{outer_folds}")
        tqdm.write(f"Train: {len(X_train_outer):,} | Test: {len(X_test_outer):,}")
        
        # INNER LOOP - Model Selection
        best_inner_score = 0
        best_model = None
        inner_scores = []
        
        # Progress bar inner folds
        inner_pbar = tqdm(total=inner_folds, desc="Inner Folds", position=1, leave=False)
        
        for inner_fold, (train_inner_idx, val_inner_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
            current_iteration += 1
            inner_pbar.set_description(f"Inner {inner_fold+1}/{inner_folds}")
            
            # Tempo stimato
            elapsed = time.time() - start_time
            if current_iteration > 0:
                eta = elapsed / current_iteration * (total_iterations - current_iteration)
                inner_pbar.set_postfix(ETA=f"{eta/60:.1f}min")
            
            # Split inner
            X_train_inner = X_train_outer[train_inner_idx]
            X_val_inner = X_train_outer[val_inner_idx]
            y_train_inner = y_train_outer[train_inner_idx]
            y_val_inner = y_train_outer[val_inner_idx]
            
            # Crea modello per questo inner fold
            model = create_model_architecture(X.shape[1], fold=inner_fold)
            callbacks = create_callbacks()
            
            # Training ottimizzato per bilanciamento velocità/robustezza
            batch_size = 2048  # Batch size aumentato per velocità
            
            callbacks = create_callbacks()
            
            # Progress callback personalizzato per inner training
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, outer_fold, inner_fold, total_epochs):
                    self.outer_fold = outer_fold
                    self.inner_fold = inner_fold
                    self.total_epochs = total_epochs
                    self.pbar = None
                
                def on_train_begin(self, logs=None):
                    self.pbar = tqdm(total=self.total_epochs, 
                                   desc=f"Outer {self.outer_fold+1}/5 Inner {self.inner_fold+1}/3",
                                   position=2, leave=False, 
                                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Val_Acc:{postfix}')
                
                def on_epoch_end(self, epoch, logs=None):
                    if self.pbar and logs:
                        val_acc = logs.get('val_accuracy', 0)
                        # Fix: usa un dizionario invece di una stringa
                        self.pbar.set_postfix({'val_acc': f"{val_acc:.4f}"})
                        self.pbar.update(1)
                
                def on_train_end(self, logs=None):
                    if self.pbar:
                        self.pbar.close()
            
            progress_callback = ProgressCallback(outer_fold, inner_fold, 100)
            callbacks.append(progress_callback)

            tqdm.write(f"   Training Outer {outer_fold+1}/5 Inner {inner_fold+1}/3...")

            history = model.fit(
                X_train_inner, y_train_inner,
                batch_size=batch_size,
                epochs=100,  # Epoche ridotte per training più veloce
                validation_data=(X_val_inner, y_val_inner),
                callbacks=callbacks,
                verbose=0  # Silenzioso per evitare conflitti con progress bar
            )
            
            # Valutazione
            y_pred = model.predict(X_val_inner, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            accuracy = accuracy_score(y_val_inner, y_pred_classes)
            f1 = f1_score(y_val_inner, y_pred_classes, average='weighted')
            
            inner_scores.append(accuracy)
            
            # Salva il migliore
            if accuracy > best_inner_score:
                best_inner_score = accuracy
                # Libera memoria del modello precedente
                if best_model is not None:
                    del best_model
                    tf.keras.backend.clear_session()
                best_model = model
            else:
                # Libera memoria del modello non necessario
                del model
                tf.keras.backend.clear_session()
            
            print(f"       Completed: Accuracy {accuracy:.4f} | F1 {f1:.4f} | Best: {best_inner_score:.4f}")
            
            inner_pbar.update(1)
            inner_pbar.set_postfix(Best=f"{best_inner_score:.4f}", Current=f"{accuracy:.4f}", F1=f"{f1:.4f}")
        
        inner_pbar.close()
        
        # Forza garbage collection
        tf.keras.backend.clear_session()
        
        tqdm.write(f"Best inner score: {best_inner_score:.4f}")
        
        # Test finale su outer test set con progress bar
        tqdm.write(" Final test on outer fold...")
        
        # Progress bar per predizione finale
        test_pbar = tqdm(total=1, desc=f"Testing Fold {outer_fold+1}", position=2, leave=False)
        y_pred_outer = best_model.predict(X_test_outer, verbose=0)
        test_pbar.update(1)
        test_pbar.close()
        
        y_pred_outer_classes = np.argmax(y_pred_outer, axis=1)
        
        outer_accuracy = accuracy_score(y_test_outer, y_pred_outer_classes)
        outer_f1 = f1_score(y_test_outer, y_pred_outer_classes, average='weighted')
        
        outer_scores.append(outer_accuracy)
        all_models.append(best_model)
        
        fold_results.append({
            'fold': outer_fold + 1,
            'inner_scores': inner_scores,
            'best_inner_score': best_inner_score,
            'outer_accuracy': outer_accuracy,
            'outer_f1': outer_f1
        })
        
        tqdm.write(f" Outer accuracy: {outer_accuracy:.4f} | F1: {outer_f1:.4f}")
        
        main_pbar.update(1)
        main_pbar.set_postfix(
            Accuracy=f"{np.mean(outer_scores):.4f}",
            Current=f"{outer_accuracy:.4f}",
            Folds=f"{len(outer_scores)}/{outer_folds}"
        )
    
    main_pbar.close()
    
    # RISULTATI FINALI
    total_time = time.time() - start_time
    final_mean = np.mean(outer_scores)
    final_std = np.std(outer_scores)
    
    print(f"\n" + "=" * 60)
    print(f" NESTED CV RISULTATI FINALI")
    print(f"=" * 60)
    print(f"  Tempo totale: {total_time/60:.1f} minuti")
    print(f" Accuracy finale: {final_mean:.4f} ± {final_std:.4f}")
    print(f" Outer scores: {[f'{s:.4f}' for s in outer_scores]}")
    print(f"  Configurazione: 100 epoche, early stopping 15, batch 2048")
    
    return {
        'models': all_models,
        'outer_scores': outer_scores,
        'fold_results': fold_results,
        'final_accuracy': final_mean,
        'final_std': final_std,
        'total_time_minutes': total_time / 60
    }

def create_ensemble_predictions(models, X_test):
    """Crea predizioni ensemble da tutti i modelli del nested CV"""
    print(" Creazione ensemble finale...")
    
    all_predictions = []
    
    # Progress bar dettagliata per ensemble
    for i, model in enumerate(tqdm(models, desc="Ensemble Models", 
                                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} models')):
        pred = model.predict(X_test, verbose=0)
        all_predictions.append(pred)
    
    # Media ensemble
    ensemble_pred = np.mean(all_predictions, axis=0)
    ensemble_classes = np.argmax(ensemble_pred, axis=1)
    
    return ensemble_pred, ensemble_classes

def save_nested_cv_results(results, preprocessing_pipeline, output_dir):
    """Salva tutti i risultati del Nested CV"""
    print(f" Salvataggio risultati...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva modelli
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Progress bar dettagliata per salvataggio modelli
    for i, model in enumerate(tqdm(results['models'], desc="Saving Models",
                                 bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} models')):
        model_path = models_dir / f"model_fold_{i+1}.keras"
        model.save(model_path)
    
    # Salva pipeline
    pipeline_path = output_dir / "preprocessing_pipeline.pkl"
    joblib.dump(preprocessing_pipeline, pipeline_path)
    print(f"Pipeline salvata: {pipeline_path}")
    
    # Salva risultati
    results_clean = {
        'final_accuracy': results['final_accuracy'],
        'final_std': results['final_std'],
        'outer_scores': results['outer_scores'],
        'fold_results': results['fold_results'],
        'total_time_minutes': results['total_time_minutes'],
        'training_config': {
            'epochs': 100,
            'early_stopping_patience': 15,
            'batch_size': 2048,
            'outer_folds': 5,
            'inner_folds': 3,
            'preprocessing': 'AdaptiveStrategy'
        }
    }
    
    results_path = output_dir / "nested_cv_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    # Summary report
    summary_path = output_dir / "nested_cv_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("NESTED CROSS-VALIDATION TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Epochs: 100\n")
        f.write(f"- Early Stopping Patience: 15\n")
        f.write(f"- Batch Size: 2048\n")
        f.write(f"- Outer Folds: 5\n")
        f.write(f"- Inner Folds: 3\n")
        f.write(f"- Preprocessing: AdaptiveStrategy\n\n")
        
        f.write(f"Results:\n")
        f.write(f"- Final Accuracy: {results['final_accuracy']:.4f} ± {results['final_std']:.4f}\n")
        f.write(f"- Total Time: {results['total_time_minutes']:.1f} minutes\n\n")
        
        f.write("Fold Details:\n")
        for fold_result in results['fold_results']:
            f.write(f"- Fold {fold_result['fold']}: {fold_result['outer_accuracy']:.4f}\n")
    
    print(f"Risultati salvati in: {output_dir}")
    return output_dir

def main():
    """Funzione principale per Nested CV training"""
    print("RICHTER PREDICTOR - NESTED CROSS-VALIDATION TRAINING")
    print("=" * 60)
    
    # Carica e preprocessa dati
    X, y, preprocessing_pipeline = load_and_preprocess_data()
    
    # Holdout finale per test finale
    X_main, X_final_test, y_main, y_final_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    print(f"Main training: {len(X_main):,} | Final test: {len(X_final_test):,}")
    
    # Nested Cross-Validation
    results = nested_cross_validation(X_main, y_main)
    
    # Test ensemble finale
    print(f"\nTest ensemble su holdout finale...")
    ensemble_pred, ensemble_classes = create_ensemble_predictions(results['models'], X_final_test)
    
    final_accuracy = accuracy_score(y_final_test, ensemble_classes)
    final_f1 = f1_score(y_final_test, ensemble_classes, average='weighted')
    
    print(f"Ensemble finale - Accuracy: {final_accuracy:.4f} | F1: {final_f1:.4f}")
    
    # Report dettagliato
    print(f"\nClassification Report:")
    print(classification_report(y_final_test, ensemble_classes, 
                              target_names=['Grade 1', 'Grade 2', 'Grade 3']))
    
    # Aggiungi risultati ensemble
    results['ensemble_final_accuracy'] = final_accuracy
    results['ensemble_final_f1'] = final_f1
    
    # Salvataggio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"models/nested_cv_final_{timestamp}"
    
    save_nested_cv_results(results, preprocessing_pipeline, output_dir)
    
    print(f"\n" + "=" * 60)
    print(" NESTED CV TRAINING COMPLETATO!")
    print(f" CV Accuracy: {results['final_accuracy']:.4f} ± {results['final_std']:.4f}")
    print(f" Ensemble Final Test: {final_accuracy:.4f}")
    print(f" Modelli salvati in: {output_dir}")
    print(f"  Config: 100 epoche, early stopping 15, batch 2048")
    
    return results

if __name__ == "__main__":
    main()
