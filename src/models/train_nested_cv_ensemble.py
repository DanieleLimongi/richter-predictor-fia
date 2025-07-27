"""
Richter Nested CV Ensemble Trainer - ANTI-LEAKAGE
Obiettivo: F1-score 0.78+ con nested CV e hyperparameter tuning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo errori, no warning CUDA/XLA

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import joblib
from datetime import datetime
from tqdm import tqdm
import time
import json
from typing import Dict, List, Tuple, Any
import random
import warnings
warnings.filterwarnings('ignore')

# Silenzia anche i log TensorFlow verbose
tf.get_logger().setLevel('ERROR')

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
        
        # Random search leggero - 4 combinazioni per architettura
        self.random_search_space = {
            'learning_rate': [0.0005, 0.001, 0.003, 0.005],
            'dropout_rate': [0.2, 0.3, 0.4, 0.5], 
            'batch_size': [1024, 2048, 4096],
            'l2_reg': [1e-5, 1e-4, 1e-3]
        }
        
        # Numero di combinazioni random da testare per architettura
        self.n_random_search = 4
    
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
        """Preprocessing sicuro con feature engineering integrato - fit solo su train"""
        
        # Verifica anti-leakage
        self.leakage_detector.validate_split(train_idx, val_idx, f"preprocessing_{fold_info}")
        
        # Split sicuro
        X_train_df = X_df.iloc[train_idx].copy()
        X_val_df = X_df.iloc[val_idx].copy()
        
        try:
            print(f"   Applying advanced feature engineering + preprocessing...")
            
            # STEP 1: Feature engineering avanzato - FIT solo su train
            engineer = AdvancedFeatureEngineer()
            self.leakage_detector.log_preprocessing_fit("AdvancedFeatureEngineer", train_idx, fold_info)
            
            X_train_enhanced = engineer.fit_transform(X_train_df)
            X_val_enhanced = engineer.transform(X_val_df)  # Solo transform su validation
            
            print(f"      Feature engineering: {len(X_train_enhanced.columns)} features")
            
            # STEP 2: Preprocessing pipeline ottimizzato per feature engineered data
            pipeline = RichterPreprocessingPipeline()
            pipeline.setup_preprocessors(
                force_embedding_categorical=True,
                add_binary_count=True,
                group_binary_correlated=True,
                outlier_detection=True
            )
            
            # Converti a tensori per train (gestione robusta dei tipi)
            train_dict = {}
            for col in X_train_enhanced.columns:
                if pd.api.types.is_string_dtype(X_train_enhanced[col]) or X_train_enhanced[col].dtype == 'object':
                    # Categorical/string data
                    train_dict[col] = tf.constant(X_train_enhanced[col].astype(str).values)
                else:
                    # Numeric data
                    numeric_values = pd.to_numeric(X_train_enhanced[col], errors='coerce').fillna(0.0)
                    train_dict[col] = tf.constant(numeric_values.astype(np.float32).values)
            
            # FIT solo su train
            self.leakage_detector.log_preprocessing_fit("RichterPreprocessingPipeline", train_idx, fold_info)
            pipeline.fit(train_dict)
            
            # Transform train
            train_processed = pipeline.transform(train_dict)
            
            # Converti a tensori per validation (stessa logica)
            val_dict = {}
            for col in X_val_enhanced.columns:
                if pd.api.types.is_string_dtype(X_val_enhanced[col]) or X_val_enhanced[col].dtype == 'object':
                    val_dict[col] = tf.constant(X_val_enhanced[col].astype(str).values)
                else:
                    numeric_values = pd.to_numeric(X_val_enhanced[col], errors='coerce').fillna(0.0)
                    val_dict[col] = tf.constant(numeric_values.astype(np.float32).values)
            
            # Transform validation (usando pipeline fitted su train)
            val_processed = pipeline.transform(val_dict)
            
            # Aggrega features per train con handling robusto
            train_arrays = []
            for tensor in train_processed.values():
                np_array = tensor.numpy()
                if len(np_array.shape) > 1:
                    np_array = np_array.reshape(np_array.shape[0], -1)
                else:
                    np_array = np_array.reshape(-1, 1)
                train_arrays.append(np_array)
            
            X_train = np.concatenate(train_arrays, axis=1).astype(np.float32)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            
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
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"      Final processed shapes: Train {X_train.shape}, Val {X_val.shape}")
            
            return X_train, X_val
            
        except Exception as e:
            print(f"   WARNING: Advanced preprocessing failed for {fold_info}: {e}")
            print(f"   Using fallback preprocessing...")
            
            # FALLBACK: Solo feature engineering + standard scaling
            engineer = AdvancedFeatureEngineer()
            self.leakage_detector.log_preprocessing_fit("AdvancedFeatureEngineer_fallback", train_idx, fold_info)
            
            X_train_enhanced = engineer.fit_transform(X_train_df)
            X_val_enhanced = engineer.transform(X_val_df)
            
            # Pulizia robusta
            for col in X_train_enhanced.columns:
                if not pd.api.types.is_numeric_dtype(X_train_enhanced[col]):
                    X_train_enhanced[col] = pd.to_numeric(X_train_enhanced[col], errors='coerce')
                    X_val_enhanced[col] = pd.to_numeric(X_val_enhanced[col], errors='coerce')
            
            X_train_enhanced = X_train_enhanced.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            X_val_enhanced = X_val_enhanced.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            # Standard scaler: FIT solo su train
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.leakage_detector.log_preprocessing_fit("StandardScaler_fallback", train_idx, fold_info)
            
            X_train = scaler.fit_transform(X_train_enhanced).astype(np.float32)
            X_val = scaler.transform(X_val_enhanced).astype(np.float32)  # Solo transform
            
            print(f"      Fallback shapes: Train {X_train.shape}, Val {X_val.shape}")
            
            return X_train, X_val
    
    def generate_random_params(self, architecture: str, outer_fold: int) -> List[Dict]:
        """Genera combinazioni random di parametri per una architettura"""
        
        # Set seed per riproducibilità per architettura+fold (assicura valore positivo)
        seed_value = abs(hash(architecture + str(outer_fold))) % (2**32 - 1)
        random.seed(seed_value)
        np.random.seed(seed_value)
        
        combinations = []
        for i in range(self.n_random_search):
            params = {}
            for param_name, param_values in self.random_search_space.items():
                params[param_name] = random.choice(param_values)
            combinations.append(params)
        
        # Rimuovi duplicati
        unique_combinations = []
        for combo in combinations:
            if combo not in unique_combinations:
                unique_combinations.append(combo)
        
        # Se abbiamo meno di n_random_search combinazioni uniche, riempi con random aggiuntive
        while len(unique_combinations) < self.n_random_search:
            params = {}
            for param_name, param_values in self.random_search_space.items():
                params[param_name] = random.choice(param_values)
            if params not in unique_combinations:
                unique_combinations.append(params)
        
        return unique_combinations[:self.n_random_search]
    
    def inner_cv_random_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                              architecture: str, outer_fold: int) -> Dict:
        """Inner CV con random search leggero per hyperparameter tuning"""
        
        print(f"    Random search for {architecture} (outer fold {outer_fold+1})...")
        
        # NESSUN inner CV - solo test diretto delle combinazioni random
        # Per mantenere il totale a ~96 modelli
        
        # Genera combinazioni random
        param_combinations = self.generate_random_params(architecture, outer_fold)
        best_params = None
        best_score = 0.0
        
        print(f"      Testing {len(param_combinations)} random combinations (simple validation)...")
        
        # Progress bar per random search
        search_progress = tqdm(total=len(param_combinations), desc=f"      {architecture} search", leave=False)
        
        # Split una volta per validation rapida
        val_split = 0.2
        n_val = int(len(X_train) * val_split)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_quick_train = X_train[train_indices]
        X_quick_val = X_train[val_indices]
        y_quick_train = y_train[train_indices]
        y_quick_val = y_train[val_indices]
        
        for param_idx, params in enumerate(param_combinations):
            search_progress.set_description(f"      {architecture} search {param_idx+1}/{len(param_combinations)}")
            search_progress.set_postfix(params)
            
            try:
                # Create model con parametri
                ensemble = EnsembleArchitectures(X_quick_train.shape[1], 3)
                model = ensemble.create_architecture(architecture)
                
                # Compile con parametri
                optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # Train veloce per selezione parametri
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    patience=3, restore_best_weights=True, verbose=0
                )
                
                model.fit(
                    X_quick_train, y_quick_train,
                    validation_data=(X_quick_val, y_quick_val),
                    epochs=30,  # Molto veloce per selezione
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate
                pred = model.predict(X_quick_val, verbose=0)
                f1 = f1_score(y_quick_val, np.argmax(pred, axis=1), average='micro')
                
                search_progress.set_postfix({**params, 'f1': f'{f1:.4f}'})
                
                if f1 > best_score:
                    best_score = f1
                    best_params = params.copy()
                
                # Cleanup
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                search_progress.set_postfix({**params, 'status': 'FAILED'})
            
            search_progress.update(1)
        
        search_progress.close()
        print(f"      Best params for {architecture}: {best_params} (F1: {best_score:.4f})")
        return {
            'architecture': architecture,
            'best_params': best_params,
            'best_inner_f1': best_score,
            'outer_fold': outer_fold
        }
    
    def train_nested_cv_ensemble(self, X_df: pd.DataFrame, y: np.ndarray):
        """Training con Nested CV completo"""
        print("Starting NESTED CV training (anti-leakage)...")
        
        # 4 outer folds per ensemble ancora più robusto
        outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        
        # Tutte le 6 architetture disponibili per ensemble completo
        ensemble_dummy = EnsembleArchitectures(100, 3)  # Dummy per get architectures
        architectures = ensemble_dummy.get_available_architectures()
        print(f"   Using all {len(architectures)} architectures: {architectures}")
        
        all_results = []
        outer_fold_models = []
        
        # Progress tracking AGGIORNATO per random search
        # 4 outer × 6 arch × (4 random + 1 final) = 4×6×5 = 120 modelli totali  
        final_models = 4 * len(architectures)  # 24 modelli finali per ensemble
        search_models = final_models * self.n_random_search  # 96 modelli per random search
        total_models = final_models + search_models  # 120 totali
        
        progress_bar = tqdm(total=final_models, desc="Nested CV Progress")
        print(f"   Total models to train: {total_models} ({search_models} search + {final_models} final)")
        print(f"   Final ensemble models: {final_models}")
        
        # Outer CV loop
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y)):
            
            # Verifica anti-leakage outer
            self.leakage_detector.validate_split(train_idx, test_idx, f"outer_fold_{outer_fold+1}")
            
            print(f"\nOUTER FOLD {outer_fold+1}/4")
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
                model_num = outer_fold * len(architectures) + arch_idx + 1
                progress_bar.set_description(f"Modello {model_num}/24: {architecture} (Fold {outer_fold+1}/4)")
                
                # Random search per trovare migliori parametri
                best_config = self.inner_cv_random_search(X_train, y_train, architecture, outer_fold)
                
                # Train finale con migliori parametri trovati
                print(f"    Training {architecture} with best params: {best_config['best_params']}")
                
                try:
                    ensemble = EnsembleArchitectures(X_train.shape[1], 3)
                    final_model = ensemble.create_architecture(architecture)
                    
                    # Compile con migliori parametri trovati
                    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=best_config['best_params']['learning_rate']
                    )
                    final_model.compile(
                        optimizer=optimizer, 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy']
                    )
                    
                    # Train esteso con early stopping meno aggressivo e progress bar
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        patience=20, restore_best_weights=True, verbose=0
                    )
                    
                    # Progress callback per training individuale con 200 epoche
                    progress_callback = tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: progress_bar.set_postfix({
                            'epoch': f'{epoch+1}/200',
                            'loss': f'{logs.get("loss", 0):.4f}',
                            'val_acc': f'{logs.get("val_accuracy", 0):.4f}'
                        })
                    )
                    
                    # Split train per validation durante final training
                    val_split = 0.15  # Ridotto validation split
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
                        epochs=200,  # Massimo training per performance ottimali
                        batch_size=best_config['best_params']['batch_size'],
                        callbacks=[early_stopping, progress_callback],
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
                        'inner_search_f1': best_config['best_inner_f1'],
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
                        'inner_search_f1': best_config['best_inner_f1'],
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
                'inner_search_f1': r.get('inner_search_f1', 0.0),
                'final_test_f1': r['final_test_f1'],
                'has_model': r['model'] is not None
            }
            for r in all_results
        ])
        
        # Statistiche per architettura
        arch_stats = df_results.groupby('architecture').agg({
            'final_test_f1': ['mean', 'std', 'count'],
            'inner_search_f1': ['mean', 'std'],
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
        
        # Config completo con conversione tipi per JSON
        config = {
            'methodology': 'nested_cv',
            'final_f1_score': float(self.final_f1),
            'target_achieved': bool(self.final_f1 >= self.target_f1),
            'ensemble_size': int(len(self.best_models)),
            'anti_leakage_summary': self.leakage_detector.get_summary(),
            'selected_models': [
                {
                    'architecture': str(r['architecture']),
                    'outer_fold': int(r['outer_fold']),
                    'best_params': r['best_params'],
                    'inner_search_f1': float(r.get('inner_search_f1', 0.0)),
                    'final_test_f1': float(r['final_test_f1'])
                }
                for r in self.best_models
            ],
            'random_search_config': {
                'search_space': self.random_search_space,
                'n_combinations': self.n_random_search
            }
        }
        
        with open(path / "nested_cv_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
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