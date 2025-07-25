"""
Training ensemble avanzato con feature engineering e architetture diverse
Obiettivo: Massimizzare F1-micro score da 0.70 a 0.80+
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import classi esistenti e nuove
from data.data_analysis import DataAnalyzer
from feature_engineering.advanced_features import AdvancedFeatureEngineer
from models.ensemble_architectures import (
    get_ensemble_architectures, 
    get_diverse_optimizers, 
    get_diverse_loss_functions,
    f1_score_metric
)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnsembleTrainer:
    """Trainer per ensemble avanzato con feature engineering"""
    
    def __init__(self, n_models=6, random_state=42):
        self.n_models = n_models
        self.random_state = random_state
        self.models = []
        self.oof_predictions = None
        self.feature_engineer = None
        self.label_encoders = {}
        self.scaler = None
        
    def setup_gpu(self):
        """Configurazione GPU se disponibile"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   GPU configuration: {len(gpus)} GPU(s) found")
                return True
            except RuntimeError as e:
                print(f"   GPU error: {e}")
                return False
        else:
            print("   No GPU found, using CPU")
            return False
    
    def load_and_engineer_features(self):
        """Carica dati e applica feature engineering avanzato"""
        print("LOADING DATA AND FEATURE ENGINEERING")
        print("=" * 60)
        
        # 1. Usa DataAnalyzer esistente per caricare dati
        analyzer = DataAnalyzer()
        df = analyzer.load_data()
        
        print(f"Original data: {df.shape}")
        
        # 2. Feature engineering avanzato
        self.feature_engineer = AdvancedFeatureEngineer(target_encoding_smoothing=100)
        df_enhanced = self.feature_engineer.fit_transform(df, 'damage_grade')
        
        print(f"After feature engineering: {df_enhanced.shape}")
        
        return df_enhanced
    
    def preprocess_data(self, df):
        """Preprocessing semplice e robusto"""
        print("\nPREPROCESSING DATA")
        print("=" * 60)
        
        df_processed = df.copy()
        
        # Rimuovi building_id se presente
        if 'building_id' in df_processed.columns:
            df_processed = df_processed.drop('building_id', axis=1)
        
        # Separa features e target
        if 'damage_grade' in df_processed.columns:
            X_df = df_processed.drop('damage_grade', axis=1)
            y = df_processed['damage_grade'].values - 1  # Convert to 0-2
        else:
            X_df = df_processed
            y = None
        
        print(f"Features before preprocessing: {X_df.shape}")
        
        # Encode categorical features
        categorical_cols = X_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if X_df[col].dtype == 'object':
                le = LabelEncoder()
                # Gestisci NaN convertendo a stringa
                X_df[col] = X_df[col].astype(str)
                X_df[col] = le.fit_transform(X_df[col])
                self.label_encoders[col] = le
        
        # Gestisci valori infiniti e NaN
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median())
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_df)
        
        print(f"Final features shape: {X_scaled.shape}")
        print(f"Categorical columns encoded: {len(categorical_cols)}")
        
        return X_scaled, y
    
    def train_ensemble(self, X, y, epochs=150, batch_size=512, validation_split=0.1):
        """Training ensemble con diverse architetture"""
        print("\nENSEMBLE TRAINING")
        print("=" * 60)
        
        # Setup GPU
        self.setup_gpu()
        
        # Stratified K-Fold per robustezza
        skf = StratifiedKFold(n_splits=self.n_models, shuffle=True, random_state=self.random_state)
        self.oof_predictions = np.zeros((len(X), 3))
        
        # Ottieni architetture, ottimizzatori e loss diverse
        architectures = get_ensemble_architectures(X.shape[1], self.n_models)
        optimizers = get_diverse_optimizers()
        loss_functions = get_diverse_loss_functions()
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFOLD {fold+1}/{self.n_models}")
            print("-" * 40)
            
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Usa architettura e configurazione specifica
            arch_name, model = architectures[fold]
            optimizer = optimizers[fold]
            loss_fn = loss_functions[fold]
            
            print(f"Architecture: {arch_name}")
            print(f"Optimizer: {type(optimizer).__name__}")
            print(f"Loss: {type(loss_fn).__name__ if callable(loss_fn) else loss_fn}")
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy', f1_score_metric]
            )
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_f1_score_metric',
                    patience=25,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_f1_score_metric',
                    factor=0.5,
                    patience=15,
                    min_lr=1e-7,
                    mode='max',
                    verbose=1
                )
            ]
            
            # Training
            print(f"Training with {len(X_train_fold):,} samples...")
            
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Out-of-fold predictions
            val_pred = model.predict(X_val_fold, verbose=0)
            self.oof_predictions[val_idx] = val_pred
            
            # Evaluation
            y_pred_classes = np.argmax(val_pred, axis=1)
            fold_f1 = f1_score(y_val_fold, y_pred_classes, average='micro')
            fold_scores.append(fold_f1)
            
            print(f"Fold {fold+1} F1-micro: {fold_f1:.4f}")
            print(f"Best val F1 from training: {max(history.history.get('val_f1_score_metric', [0])):.4f}")
            
            # Salva modello
            self.models.append(model)
        
        # Overall ensemble performance
        oof_pred_classes = np.argmax(self.oof_predictions, axis=1)
        ensemble_f1 = f1_score(y, oof_pred_classes, average='micro')
        
        print(f"\nENSEMBLE RESULTS")
        print("=" * 60)
        print(f"Individual fold F1 scores: {[f'{f:.4f}' for f in fold_scores]}")
        print(f"Mean fold F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"Ensemble F1-micro: {ensemble_f1:.4f}")
        
        # Classification report dettagliato
        print(f"\nDetailed Classification Report:")
        print(classification_report(y, oof_pred_classes, target_names=['Grade 1', 'Grade 2', 'Grade 3']))
        
        return {
            'fold_scores': fold_scores,
            'ensemble_score': ensemble_f1,
            'mean_fold_score': np.mean(fold_scores),
            'std_fold_score': np.std(fold_scores),
            'oof_predictions': self.oof_predictions,
            'architectures_used': [arch[0] for arch in architectures]
        }
    
    def save_ensemble(self, results, output_dir):
        """Salva ensemble completo"""
        print(f"\nSAVING ENSEMBLE")
        print("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva modelli individuali
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = models_dir / f"ensemble_model_{i+1}.keras"
            model.save(model_path)
            print(f"   Model {i+1} saved: {model_path.name}")
        
        # Salva preprocessing artifacts
        joblib.dump(self.feature_engineer, output_dir / "feature_engineer.pkl")
        joblib.dump(self.label_encoders, output_dir / "label_encoders.pkl")
        joblib.dump(self.scaler, output_dir / "scaler.pkl")
        
        print(f"   Preprocessing artifacts saved")
        
        # Salva risultati
        results_clean = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                        for k, v in results.items() if k != 'oof_predictions'}
        
        with open(output_dir / "training_results.json", 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        # Salva configurazione
        config = {
            'n_models': self.n_models,
            'random_state': self.random_state,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'tensorflow_version': tf.__version__,
            'final_f1_score': results['ensemble_score']
        }
        
        with open(output_dir / "ensemble_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   Results and config saved")
        print(f"\nAll files saved to: {output_dir}")
        
        return str(output_dir)

def main():
    """Training principale"""
    print("ADVANCED ENSEMBLE TRAINING - RICHTER PREDICTOR")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize trainer
    trainer = EnsembleTrainer(n_models=6, random_state=42)
    
    try:
        # 1. Load data and feature engineering
        df_enhanced = trainer.load_and_engineer_features()
        
        # 2. Preprocessing
        X, y = trainer.preprocess_data(df_enhanced)
        
        # 3. Ensemble training
        results = trainer.train_ensemble(X, y, epochs=150, batch_size=512)
        
        # 4. Save everything
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/advanced_ensemble_{timestamp}"
        saved_path = trainer.save_ensemble(results, output_dir)
        
        # 5. Final summary
        print(f"\nTRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Final F1-micro score: {results['ensemble_score']:.4f}")
        print(f"Improvement over baseline (0.70): +{(results['ensemble_score'] - 0.70)*100:.1f} percentage points")
        print(f"Models saved to: {saved_path}")
        print(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Suggerimenti prossimi passi
        print(f"\nNEXT STEPS:")
        print(f"1. Use this ensemble for submission with create_ensemble_submission.py")
        print(f"2. If F1 < 0.78, consider hyperparameter tuning")
        print(f"3. If F1 > 0.78, ready for competition submission!")
        
        return results['ensemble_score']
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    final_score = main()
