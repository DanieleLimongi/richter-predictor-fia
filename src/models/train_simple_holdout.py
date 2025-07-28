#!/usr/bin/env python3
"""
Training MLP semplice per Richter Predictor - REFACTORED
Utilizza componenti modulari esistenti eliminando duplicazioni
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import json
import pickle
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup path dinamico
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

# Import componenti modulari esistenti
from feature_engineering import AdvancedFeatureEngineer
from data.data_analysis import DataAnalyzer
from models.ensemble_architectures import EnsembleArchitectures

# Configurazione TensorFlow ottimizzata
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_and_preprocess_data():
    """Carica dati usando DataAnalyzer e applica feature engineering modulare"""
    print("📊 Caricamento dati tramite DataAnalyzer...")
    
    # Usa DataAnalyzer esistente invece di duplicare logica
    analyzer = DataAnalyzer()
    train_df = analyzer.load_data()
    
    print(f"   Dati caricati: {train_df.shape[0]} samples, {train_df.shape[1]} features")
    
    # Separazione features e target tramite DataAnalyzer
    feature_cols = [col for col in train_df.columns if col not in ['building_id', 'damage_grade']]
    X_df = train_df[feature_cols]
    y = (train_df['damage_grade'] - 1).astype(np.int32)  # Convert to 0-2
    
    print("🔧 Applicazione feature engineering modulare...")
    
    # Feature engineering con architettura modulare
    engineer = AdvancedFeatureEngineer()
    X_enhanced = engineer.fit_transform(X_df)
    
    # Conversione ottimizzata
    X = X_enhanced.values.astype(np.float32)
    
    print(f"✅ Feature engineering completato!")
    print(f"   Shape finale: {X.shape}")
    print(f"   Features create: +{len(X_enhanced.columns) - len(X_df.columns)}")
    print(f"   Target distribution: {np.bincount(y)}")
    
    return X, y, engineer

def create_model_from_ensemble_architectures(input_dim, architecture='regularized', num_classes=3):
    """Crea modello usando EnsembleArchitectures esistente"""
    print(f"🏗️  Creazione modello '{architecture}' tramite EnsembleArchitectures...")
    
    # Usa architettura esistente invece di duplicare
    ensemble = EnsembleArchitectures(input_dim, num_classes)
    model = ensemble.create_architecture(architecture)
    
    # Compilazione ottimizzata
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Architettura: {architecture}")
    print(f"   Parametri: {model.count_params():,}")
    
    return model

def main():
    """Training principale con componenti modulari"""
    print("🚀 RICHTER PREDICTOR - MLP TRAINING REFACTORED")
    print("=" * 60)
    print("   ✅ Usa DataAnalyzer per data loading")
    print("   ✅ Usa EnsembleArchitectures per modello")
    print("   ✅ Path dinamici e configurazione centralizzata")
    print()
    
    # Carica e preprocessa dati con componenti esistenti
    X, y, feature_engineer = load_and_preprocess_data()
    
    # HOLDOUT: Prima separa il test set finale (mai visto dal modello)
    X_work, X_holdout, y_work, y_holdout = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Train/Val split sui dati rimanenti
    X_train, X_val, y_train, y_val = train_test_split(
        X_work, y_work, test_size=0.2, random_state=42, stratify=y_work
    )
    
    print(f"📊 Split dati:")
    print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Holdout: {X_holdout.shape[0]}")
    print(f"   Percentuali: {X_train.shape[0]/len(X)*100:.1f}% train | {X_val.shape[0]/len(X)*100:.1f}% val | {X_holdout.shape[0]/len(X)*100:.1f}% holdout")
    print()
    
    # Crea modello usando EnsembleArchitectures esistente
    model = create_model_from_ensemble_architectures(X_train.shape[1], 'regularized')
    
    # Callbacks ottimizzati
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=50,
            restore_best_weights=True,
            min_delta=0.00005
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.8,
            patience=35,
            min_lr=1e-9,
            mode='max',
            verbose=1
        )
    ]
    
    # Training
    print("🏃 Inizio training...")
    history = model.fit(
        X_train, y_train,
        batch_size=1024,  # Batch size grande per GPU
        epochs=300,  #  Più epoche per training esteso
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1  # Mostra progress nativamente
    )
    
    # Valutazione finale su holdout (mai visto dal modello)
    print("\n🎯 Valutazione finale su holdout...")
    y_pred_holdout = model.predict(X_holdout, batch_size=2048, verbose=0)
    y_pred_holdout_classes = np.argmax(y_pred_holdout, axis=1)
    
    holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout_classes)
    holdout_f1 = f1_score(y_holdout, y_pred_holdout_classes, average='weighted')
    
    print(f"📈 Risultati finali:")
    print(f"   Holdout Accuracy: {holdout_accuracy:.4f}")
    print(f"   Holdout F1-Score: {holdout_f1:.4f}")
    
    # Report dettagliato
    print("\n📊 Classification Report (Holdout):")
    print(classification_report(y_holdout, y_pred_holdout_classes, 
                              target_names=['Grade 1', 'Grade 2', 'Grade 3']))
    
    # Salvataggio con path dinamici
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Path dinamici basati su project_root
    models_dir = project_root / 'models' / 'simple_models'
    reports_dir = project_root / 'reports' / 'mlp_results'
    
    # Assicura che le directory esistano
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Path dei file
    model_path = models_dir / f'mlp_model_refactored_{timestamp}.keras'
    engineer_path = models_dir / f'feature_engineer_{timestamp}.pkl'
    
    print(f"\n💾 Salvataggio modello e pipeline...")
    
    # Salva modello
    model.save(str(model_path))
    
    # Salva feature engineer
    with open(engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    
    # Salva risultati
    results = {
        'timestamp': timestamp,
        'final_accuracy': float(holdout_accuracy),
        'holdout_f1': float(holdout_f1),
        'train_samples': int(X_train.shape[0]),
        'val_samples': int(X_val.shape[0]),
        'holdout_samples': int(X_holdout.shape[0]),
        'features_count': int(X_train.shape[1]),
        'model_path': str(model_path),
        'feature_engineer_path': str(engineer_path),
        'used_modular_components': True,
        'refactored_version': True,
        'holdout_split': True,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }
    }
    
    results_path = reports_dir / f'refactored_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Modello salvato: {model_path}")
    print(f"   ✅ Feature Engineer salvato: {engineer_path}")
    print(f"   ✅ Risultati salvati: {results_path}")
    print("\n🎉 Training refactored completato con successo!")
    print("   🔄 Eliminazione duplicazioni completata")
    print("   📦 Riutilizzo componenti modulari attivato")
    
    return model, feature_engineer, results

if __name__ == "__main__":
    model, engineer, results = main()
