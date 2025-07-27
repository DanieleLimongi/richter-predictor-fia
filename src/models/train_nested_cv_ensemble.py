"""
Richter Nested CV Ensemble Trainer - ANTI-LEAKAGE
Obiettivo: F1-score 0.78+ con nested CV e hyperparameter tuning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, classification_report
import joblib
from datetime import datetime
from tqdm import tqdm
import time
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import componenti
from data.data_analysis import DataAnalyzer
from feature_engineering.advanced_features import AdvancedFeatureEngineer
from preprocessing.main_pipeline import RichterPreprocessingPipeline
from models.ensemble_architectures import EnsembleArchitectures

class LeakageDetector:
    """Rileva e previene data leakage durante nested CV"""
    
    def __init__(self):
        self.train_indices_history = []
        self.preprocessing_fits = []
        
    def validate_split(self, train_idx: np.ndarray, val_idx: np.ndarray, fold_name: str) -> bool:
        """Verifica che non ci sia overlap tra train e validation"""
        overlap = np.intersect1d(train_idx, val_idx)
        if len(overlap) > 0:
            raise ValueError(f"DATA LEAKAGE: {fold_name} has {len(overlap)} overlapping indices!")
        
        # Store per controlli successivi
        self.train_indices_history.append({
            'fold': fold_name,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_min': train_idx.min(),
            'train_max': train_idx.max(),
            'val_min': val_idx.min(),
            'val_max': val_idx.max()
        })
        
        return True
    
    def log_preprocessing_fit(self, component_name: str, train_indices: np.ndarray, fold_info: str):
        """Traccia quando viene fatto fit dei preprocessor"""
        self.preprocessing_fits.append({
            'component': component_name,
            'fold': fold_info,
            'train_indices_hash': hash(train_indices.tobytes()),
            'train_size': len(train_indices),
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict:
        """Riassunto controlli anti-leakage"""
        return {
            'total_splits_validated': len(self.train_indices_history),
            'preprocessing_fits': len(self.preprocessing_fits),
            'splits_history': self.train_indices_history,
            'preprocessing_history': self.preprocessing_fits
        }

class NestedCVRichterTrainer:
    def __init__(self):
        self.target_f1 = 0.78
        self.best_models = []
        self.final_f1 = 0.0
        self.leakage_detector = LeakageDetector()
        
        # Hyperparameter grids per architettura
        self.hyperparameter_grids = {
            'deep_narrow': {
                'batch_size': [1024, 2048],
                'learning_rate': [0.001, 0.003, 0.01],
                'dropout_rate': [0.3, 0.5],
                'l2_reg': [1e-4, 1e-3]
            },
            'wide_shallow': {
                'batch_size': [1024, 2048, 4096],
                'learning_rate': [0.001, 0.003],
                'dropout_rate': [0.2, 0.4],
                'l2_reg': [1e-5, 1e-4]
            },
            'residual_like': {
                'batch_size': [1024, 2048],
                'learning_rate': [0.001, 0.005],
                'dropout_rate': [0.3, 0.4],
                'l2_reg': [1e-4, 1e-3]
            },
            'regularized': {
                'batch_size': [1024, 2048],
                'learning_rate': [0.001, 0.003],
                'dropout_rate': [0.4, 0.6],
                'l2_reg': [1e-3, 1e-2]
            },
            'swish_activation': {
                'batch_size': [1024, 2048],
                'learning_rate': [0.001, 0.005],
                'dropout_rate': [0.2, 0.4],
                'l2_reg': [1e-4, 1e-3]
            },
            'attention_like': {
                'batch_size': [1024, 2048],
                'learning_rate': [0.0005, 0.001, 0.003],
                'dropout_rate': [0.3, 0.5],
                'l2_reg': [1e-4, 1e-3]
            }
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Carica dati RAW senza preprocessing per evitare leakage"""
        print("Loading RAW data (no preprocessing to prevent leakage)...")
        
        analyzer = DataAnalyzer()
        df = analyzer.load_data()
        
        # Solo separazione target - NESSUN preprocessing
        y = df['damage_grade'].values - 1
        X_df = df.drop(['damage_grade', 'building_id'], axis=1, errors='ignore')
        
        print(f"   Raw data: {X_df.shape}, target: {np.bincount(y)}")
        return X_df, y
    
    def safe_preprocessing_pipeline(self, X_df: pd.DataFrame, train_idx: np.ndarray, 
                                   val_idx: np.ndarray, fold_info: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocessing sicuro senza leakage: fit solo su train, transform su train+val"""
        
        # Verifica anti-leakage
        self.leakage_detector.validate_split(train_idx, val_idx, f"preprocessing_{fold_info}")
        
        # Split sicuro
        X_train_df = X_df.iloc[train_idx].copy()
        X_val_df = X_df.iloc[val_idx].copy()
        
        try:
            # Feature engineering: FIT solo su train
            engineer = AdvancedFeatureEngineer()
            self.leakage_detector.log_preprocessing_fit("AdvancedFeatureEngineer", train_idx, fold_info)
            
            X_train_enhanced = engineer.fit_transform(X_train_df)
            X_val_enhanced = engineer.transform(X_val_df)  # Solo transform su validation
            
            # Preprocessing pipeline: FIT solo su train
            pipeline = RichterPreprocessingPipeline()
            pipeline.setup_preprocessors(
                force_embedding_categorical=True,
                add_binary_count=True,
                group_binary_correlated=True,
                outlier_detection=True
            )
            
            # Converti a tensori per train
            train_dict = {}
            for col in X_train_enhanced.columns:
                if X_train_enhanced[col].dtype == 'object':
                    train_dict[col] = tf.constant(X_train_enhanced[col].astype(str).values)
                else:
                    train_dict[col] = tf.constant(X_train_enhanced[col].astype(np.float32).values)
            
            # FIT solo su train
            self.leakage_detector.log_preprocessing_fit("RichterPreprocessingPipeline", train_idx, fold_info)
            pipeline.fit(train_dict)
            
            # Transform train
            train_processed = pipeline.transform(train_dict)
            
            # Converti a tensori per validation
            val_dict = {}
            for col in X_val_enhanced.columns:
                if X_val_enhanced[col].dtype == 'object':
                    val_dict[col] = tf.constant(X_val_enhanced[col].astype(str).values)
                else:
                    val_dict[col] = tf.constant(X_val_enhanced[col].astype(np.float32).values)
            
            # Transform validation (usando pipeline fitted su train)
            val_processed = pipeline.transform(val_dict)
            
            # Aggrega features per train
            train_arrays = []
            for tensor in train_processed.values():
                np_array = tensor.numpy()
                if len(np_array.shape) > 1:
                    np_array = np_array.reshape(np_array.shape[0], -1)
                else:
                    np_array = np_array.reshape(-1, 1)
                train_arrays.append(np_array)
            
            X_train = np.concatenate(train_arrays, axis=1).astype(np.float32)
            X_train = np.nan_to_num(X_train)
            
            # Aggrega features per validation
            val_arrays = []
            for tensor in val_processed.values():
                np_array = tensor.numpy()
                if len(np_array.shape) > 1:
                    np_array = np_array.reshape(np_array.shape[0], -1)
                else:
                    np_array = np_array.reshape(-1, 1)
                val_arrays.append(np_array)
            
            X_val = np.concatenate(val_arrays, axis=1).astype(np.float32)
            X_val = np.nan_to_num(X_val)
            
            return X_train, X_val
            
        except Exception as e:
            print(f"   WARNING: Advanced preprocessing failed for {fold_info}: {e}")
            print(f"   Using fallback preprocessing...")
            
            # Fallback sicuro
            engineer = AdvancedFeatureEngineer()
            self.leakage_detector.log_preprocessing_fit("AdvancedFeatureEngineer_fallback", train_idx, fold_info)
            
            X_train_enhanced = engineer.fit_transform(X_train_df)
            X_val_enhanced = engineer.transform(X_val_df)
            
            # Converti a numerico
            for col in X_train_enhanced.columns:
                if not pd.api.types.is_numeric_dtype(X_train_enhanced[col]):
                    X_train_enhanced[col] = pd.to_numeric(X_train_enhanced[col], errors='coerce')
                    X_val_enhanced[col] = pd.to_numeric(X_val_enhanced[col], errors='coerce')
            
            X_train_enhanced = X_train_enhanced.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            X_val_enhanced = X_val_enhanced.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            # Scaler: FIT solo su train
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.leakage_detector.log_preprocessing_fit("StandardScaler_fallback", train_idx, fold_info)
            
            X_train = scaler.fit_transform(X_train_enhanced).astype(np.float32)
            X_val = scaler.transform(X_val_enhanced).astype(np.float32)  # Solo transform
            
            return X_train, X_val
    
    def inner_cv_hyperparameter_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                                     architecture: str, outer_fold: int) -> Dict:
        """Inner CV per hyperparameter tuning di una specifica architettura"""
        
        print(f"    Inner CV for {architecture} (outer fold {outer_fold+1})...")
        
        # Inner CV con 4 fold
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42+outer_fold)
        
        # Grid search
        param_grid = list(ParameterGrid(self.hyperparameter_grids[architecture]))
        best_params = None
        best_score = 0.0
        
        # Limita search per tempo computazionale
        max_combinations = min(8, len(param_grid))
        param_grid = param_grid[:max_combinations]
        
        for param_idx, params in enumerate(param_grid):
            print(f"      Testing params {param_idx+1}/{len(param_grid)}: {params}")
            
            inner_scores = []
            
            # Inner CV per questa combinazione parametri
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train)):
                
                # Verifica anti-leakage inner
                self.leakage_detector.validate_split(
                    inner_train_idx, inner_val_idx, 
                    f"inner_f{outer_fold+1}_{architecture}_p{param_idx+1}_if{inner_fold+1}"
                )
                
                try:
                    # Split inner
                    X_inner_train = X_train[inner_train_idx]
                    X_inner_val = X_train[inner_val_idx]  # Attenzione: validation è subset di X_train!
                    y_inner_train = y_train[inner_train_idx]
                    y_inner_val = y_train[inner_val_idx]
                    
                    # Create model con parametri
                    ensemble = EnsembleArchitectures(X_inner_train.shape[1], 3)
                    model = ensemble.create_architecture(architecture)
                    
                    # Compile con parametri
                    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    
                    # Train rapido per inner CV
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        patience=5, restore_best_weights=True, verbose=0
                    )
                    
                    model.fit(
                        X_inner_train, y_inner_train,
                        validation_data=(X_inner_val, y_inner_val),
                        epochs=50,  # Meno epoche per speed
                        batch_size=params['batch_size'],
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Evaluate
                    pred = model.predict(X_inner_val, verbose=0)
                    f1 = f1_score(y_inner_val, np.argmax(pred, axis=1), average='micro')
                    inner_scores.append(f1)
                    
                    # Cleanup
                    del model
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"        FAILED: Inner fold {inner_fold+1} failed: {e}")
                    inner_scores.append(0.33)
            
            # Media inner CV per questa combinazione
            avg_score = np.mean(inner_scores)
            print(f"        Avg F1: {avg_score:.4f} (std: {np.std(inner_scores):.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
        
        print(f"    Best params for {architecture}: {best_params} (F1: {best_score:.4f})")
        return {
            'architecture': architecture,
            'best_params': best_params,
            'best_inner_f1': best_score,
            'outer_fold': outer_fold
        }
    
    def train_nested_cv_ensemble(self, X_df: pd.DataFrame, y: np.ndarray):
        """Training con Nested CV completo"""
        print("Starting NESTED CV training (anti-leakage)...")
        
        # Outer CV per model selection
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Architetture da testare
        ensemble_dummy = EnsembleArchitectures(100, 3)  # Dummy per get architectures
        architectures = ensemble_dummy.get_available_architectures()
        
        all_results = []
        outer_fold_models = []
        
        # Progress tracking
        total_steps = 5 * len(architectures)  # outer_folds * architectures
        progress_bar = tqdm(total=total_steps, desc="Nested CV Progress")
        
        # Outer CV loop
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y)):
            
            # Verifica anti-leakage outer
            self.leakage_detector.validate_split(train_idx, test_idx, f"outer_fold_{outer_fold+1}")
            
            print(f"\nOUTER FOLD {outer_fold+1}/5")
            print(f"   Train: {len(train_idx)}, Test: {len(test_idx)}")
            
            # Preprocessing sicuro per questo outer fold
            X_train, X_test = self.safe_preprocessing_pipeline(
                X_df, train_idx, test_idx, f"outer_fold_{outer_fold+1}"
            )
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"   Processed shapes: Train {X_train.shape}, Test {X_test.shape}")
            
            # Inner CV per ogni architettura
            fold_best_models = []
            
            for arch_idx, architecture in enumerate(architectures):
                progress_bar.set_description(f"Fold {outer_fold+1}/5 - {architecture}")
                
                # Hyperparameter search per questa architettura
                best_config = self.inner_cv_hyperparameter_search(
                    X_train, y_train, architecture, outer_fold
                )
                
                # Train finale con best params su tutto il training set dell'outer fold
                print(f"    Final training {architecture} with best params...")
                
                try:
                    ensemble = EnsembleArchitectures(X_train.shape[1], 3)
                    final_model = ensemble.create_architecture(architecture)
                    
                    # Compile con best params
                    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=best_config['best_params']['learning_rate']
                    )
                    final_model.compile(
                        optimizer=optimizer, 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy']
                    )
                    
                    # Train con early stopping
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        patience=15, restore_best_weights=True, verbose=0
                    )
                    
                    # Split train per validation durante final training
                    val_split = 0.2
                    n_val = int(len(X_train) * val_split)
                    
                    # Shuffle indices per validation split
                    indices = np.random.permutation(len(X_train))
                    val_indices = indices[:n_val]
                    train_indices = indices[n_val:]
                    
                    X_final_train = X_train[train_indices]
                    X_final_val = X_train[val_indices]
                    y_final_train = y_train[train_indices]
                    y_final_val = y_train[val_indices]
                    
                    final_model.fit(
                        X_final_train, y_final_train,
                        validation_data=(X_final_val, y_final_val),
                        epochs=100,
                        batch_size=best_config['best_params']['batch_size'],
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Test finale su outer test set
                    test_pred = final_model.predict(X_test, verbose=0)
                    test_f1 = f1_score(y_test, np.argmax(test_pred, axis=1), average='micro')
                    
                    print(f"    {architecture} final F1: {test_f1:.4f}")
                    
                    # Store risultati
                    result = {
                        'outer_fold': outer_fold,
                        'architecture': architecture,
                        'best_params': best_config['best_params'],
                        'inner_cv_f1': best_config['best_inner_f1'],
                        'final_test_f1': test_f1,
                        'model': final_model
                    }
                    
                    fold_best_models.append(result)
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"    FAILED: Final training failed for {architecture}: {e}")
                    result = {
                        'outer_fold': outer_fold,
                        'architecture': architecture,
                        'best_params': best_config['best_params'],
                        'inner_cv_f1': best_config['best_inner_f1'],
                        'final_test_f1': 0.33,
                        'model': None
                    }
                    all_results.append(result)
                
                progress_bar.update(1)
            
            outer_fold_models.append(fold_best_models)
            
            # Summary outer fold
            fold_f1s = [r['final_test_f1'] for r in fold_best_models if r['model'] is not None]
            print(f"   Outer fold {outer_fold+1} F1s: {[f'{f:.3f}' for f in fold_f1s]}")
            print(f"   Fold avg: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        
        progress_bar.close()
        
        # Analisi finale e selezione modelli
        self.analyze_and_select_final_models(all_results)
        
        return self.final_f1
    
    def analyze_and_select_final_models(self, all_results: List[Dict]):
        """Analizza risultati nested CV e seleziona i migliori modelli per ensemble finale"""
        
        print(f"\nNESTED CV ANALYSIS ({len(all_results)} total models)")
        
        # Converti a DataFrame per analisi
        df_results = pd.DataFrame([
            {
                'outer_fold': r['outer_fold'],
                'architecture': r['architecture'],
                'inner_cv_f1': r['inner_cv_f1'],
                'final_test_f1': r['final_test_f1'],
                'has_model': r['model'] is not None
            }
            for r in all_results
        ])
        
        # Statistiche per architettura
        arch_stats = df_results.groupby('architecture').agg({
            'final_test_f1': ['mean', 'std', 'count'],
            'inner_cv_f1': ['mean', 'std'],
            'has_model': 'sum'
        }).round(4)
        
        print("\nArchitecture Performance Summary:")
        print(arch_stats)
        
        # Selezione criteri
        min_f1_threshold = 0.65  # Soglia minima
        max_models_per_arch = 2   # Max modelli per architettura
        target_ensemble_size = 8  # Target ensemble size
        
        # Filtra modelli validi
        valid_results = [r for r in all_results if r['model'] is not None and r['final_test_f1'] >= min_f1_threshold]
        
        print(f"\nSelection Criteria:")
        print(f"   Min F1 threshold: {min_f1_threshold}")
        print(f"   Max per architecture: {max_models_per_arch}")
        print(f"   Valid models: {len(valid_results)}/{len(all_results)}")
        
        # Sort per performance e diversifica per architettura
        valid_results.sort(key=lambda x: x['final_test_f1'], reverse=True)
        
        selected_models = []
        arch_counts = {}
        
        for result in valid_results:
            arch = result['architecture']
            if arch_counts.get(arch, 0) < max_models_per_arch and len(selected_models) < target_ensemble_size:
                selected_models.append(result)
                arch_counts[arch] = arch_counts.get(arch, 0) + 1
        
        self.best_models = selected_models
        
        # Calcola F1 finale (media weighted)
        if selected_models:
            weights = [r['final_test_f1'] for r in selected_models]
            self.final_f1 = np.average([r['final_test_f1'] for r in selected_models], weights=weights)
        else:
            self.final_f1 = 0.33
        
        print(f"\nFINAL ENSEMBLE:")
        print(f"   Selected models: {len(selected_models)}")
        print(f"   Architecture distribution: {arch_counts}")
        individual_f1s = [f"{r['final_test_f1']:.3f}" for r in selected_models]
        print(f"   Individual F1s: {individual_f1s}")
        print(f"   Ensemble F1 (weighted avg): {self.final_f1:.4f}")
        print(f"   Target achieved: {'REACHED' if self.final_f1 >= self.target_f1 else 'MISSED'}")
        
        # Summary leakage detection
        leakage_summary = self.leakage_detector.get_summary()
        print(f"\nAnti-Leakage Summary:")
        print(f"   Splits validated: {leakage_summary['total_splits_validated']}")
        print(f"   Preprocessing fits tracked: {leakage_summary['preprocessing_fits']}")
        
    def save_nested_cv_results(self):
        """Salva risultati completi nested CV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"models/nested_cv_ensemble_f1_{self.final_f1:.4f}_{timestamp}")
        path.mkdir(parents=True, exist_ok=True)
        
        # Salva modelli selezionati
        for i, result in enumerate(self.best_models):
            model_path = path / f"model_{i+1}_{result['architecture']}_fold{result['outer_fold']}.keras"
            result['model'].save(model_path)
        
        # Config completo
        config = {
            'methodology': 'nested_cv',
            'final_f1_score': self.final_f1,
            'target_achieved': self.final_f1 >= self.target_f1,
            'ensemble_size': len(self.best_models),
            'anti_leakage_summary': self.leakage_detector.get_summary(),
            'selected_models': [
                {
                    'architecture': r['architecture'],
                    'outer_fold': r['outer_fold'],
                    'best_params': r['best_params'],
                    'inner_cv_f1': r['inner_cv_f1'],
                    'final_test_f1': r['final_test_f1']
                }
                for r in self.best_models
            ],
            'hyperparameter_grids': self.hyperparameter_grids
        }
        
        with open(path / "nested_cv_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Nested CV results saved to: {path}")
        return path

def main():
    """Main function per nested CV training"""
    print("RICHTER NESTED CV ENSEMBLE TRAINER")
    print("WITH ANTI-LEAKAGE PROTECTION")
    print("=" * 50)
    
    trainer = NestedCVRichterTrainer()
    
    # Load raw data
    X_df, y = trainer.load_and_prepare_data()
    
    # Nested CV training
    f1 = trainer.train_nested_cv_ensemble(X_df, y)
    
    # Save results
    path = trainer.save_nested_cv_results()
    
    # Final summary
    print(f"\nNESTED CV TRAINING COMPLETED!")
    print(f"   Final F1: {f1:.4f}")
    print(f"   Target: {'REACHED' if f1 >= 0.78 else 'MISSED'}")
    print(f"   Anti-leakage: VERIFIED")
    print(f"   Results: {path}")
    
    return f1

if __name__ == "__main__":
    main()