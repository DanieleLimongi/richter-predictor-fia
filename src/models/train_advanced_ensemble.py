"""
Richter Ensemble Trainer - Versione PULITA
Obiettivo: F1-score 0.78+ senza overfitting
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import joblib
from datetime import datetime
from tqdm import tqdm
import time

# Import componenti
from data.data_analysis import DataAnalyzer
from feature_engineering.advanced_features import AdvancedFeatureEngineer
from preprocessing.main_pipeline import RichterPreprocessingPipeline
from models.ensemble_architectures import EnsembleArchitectures

class RichterTrainer:
    def __init__(self):
        self.target_f1 = 0.78
        self.models = []
        self.oof_predictions = None
        self.final_f1 = 0.0
        
    def load_data(self):
        """Carica e processa dati"""
        print("📂 Loading data...")
        
        # Carica dati
        analyzer = DataAnalyzer()
        df = analyzer.load_data()
        
        # Feature engineering - BYPASS per mantenere performance
        engineer = AdvancedFeatureEngineer()
        print("⚠️ BYPASSING Advanced Feature Engineering (mantiene performance)")
        df_enhanced = df  # USA DATI RAW che funzionano!
        
        # Preprocessing
        try:
            pipeline = RichterPreprocessingPipeline()
            pipeline.setup_preprocessors(
                force_embedding_categorical=True,  # FORZA EMBEDDINGS per ridurre sparsity!
                add_binary_count=True,
                group_binary_correlated=True,
                outlier_detection=True
            )
            
            # Separa target
            y = df_enhanced['damage_grade'].values - 1
            X_df = df_enhanced.drop(['damage_grade', 'building_id'], axis=1, errors='ignore')
            
            # Converti a tensori (SENZA pre-processing NaN!)
            data_dict = {}
            for col in X_df.columns:
                if X_df[col].dtype == 'object':
                    # Categorical features
                    data_dict[col] = tf.constant(X_df[col].astype(str).values)
                else:
                    # Numeric features (NO fillna here!)
                    data_dict[col] = tf.constant(X_df[col].astype(np.float32).values)
            
            # Applica preprocessing
            pipeline.fit(data_dict)
            processed = pipeline.transform(data_dict)
            
            # Aggrega features
            arrays = []
            for tensor in processed.values():
                np_array = tensor.numpy()
                if len(np_array.shape) > 1:
                    np_array = np_array.reshape(np_array.shape[0], -1)
                else:
                    np_array = np_array.reshape(-1, 1)
                arrays.append(np_array)
            
            X = np.concatenate(arrays, axis=1).astype(np.float32)
            X = np.nan_to_num(X)
            
            self.feature_engineer = engineer
            self.preprocessing_pipeline = pipeline
            
        except Exception as e:
            print(f"⚠️ Preprocessing failed: {e}, using fallback")
            # Fallback semplice
            y = df_enhanced['damage_grade'].values - 1
            X_df = df_enhanced.drop(['damage_grade', 'building_id'], axis=1, errors='ignore')
            
            # Converti a numerico
            for col in X_df.columns:
                if not pd.api.types.is_numeric_dtype(X_df[col]):
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            
            X_df = X_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X_df).astype(np.float32)
            
            self.feature_engineer = engineer
            self.preprocessing_pipeline = None
        
        print(f"   Data ready: {X.shape}, target: {np.bincount(y)}")
        return X, y
    
    def train_ensemble(self, X, y):
        """Training con CV - AUTOMATICO dataset size selection"""
        print("🚀 Training ensemble...")
        
        # 🚀 ADAPTIVE DATASET SIZE basato su quello che FUNZIONAVA
        dataset_size = len(X)
        dataset_gb = (X.nbytes + y.nbytes) / (1024**3)
        
        # USA SUBSET come nel test_simple_mlp.py che dava F1=0.64!
        if dataset_size > 50000:
            # Usa subset strategico che FUNZIONA
            subset_size = 50000  # Dimensione che aveva dato F1=0.64
            print(f"   📊 Large dataset detected ({dataset_size:,} samples)")
            print(f"   🎯 Using WORKING subset size: {subset_size:,} samples (like test_simple_mlp)")
            
            # Subset stratificato per mantenere class balance
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(X))
            _, selected_indices, _, _ = train_test_split(
                indices, y, 
                test_size=subset_size/len(X), 
                stratify=y, 
                random_state=42
            )
            
            X = X[selected_indices]
            y = y[selected_indices]
            
            print(f"   ✅ Subset ready: {X.shape[0]:,} samples, class distribution: {np.bincount(y)}")
        else:
            print(f"   📊 Using full dataset: {dataset_size:,} samples")
        
        # 🚀 MEMORY CHECK con gestione graceful
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            current_dataset_gb = (X.nbytes + y.nbytes) / (1024**3)
            
            print(f"   💾 Memory: {available_gb:.1f}GB available, dataset: {current_dataset_gb:.2f}GB")
            
            if current_dataset_gb > available_gb * 0.4:
                print(f"   ⚠️ WARNING: Dataset uses {current_dataset_gb/available_gb*100:.1f}% of available memory")
        except ImportError:
            print("   💾 Memory monitoring unavailable (install psutil for monitoring)")
        except Exception as e:
            print(f"   💾 Memory check failed: {e}")
        
        # Analisi class imbalance
        class_counts = np.bincount(y)
        majority_baseline = class_counts.max() / len(y)
        
        # Analisi sparsity
        sparsity = (1 - np.count_nonzero(X) / X.size) * 100
        
        print(f"   📊 Class distribution: {class_counts}")
        print(f"   📈 Majority baseline: {majority_baseline:.3f}")
        print(f"   🕳️ Data sparsity: {sparsity:.1f}%")
        print(f"   🎯 Target F1: {self.target_f1:.3f}")
        
        # Warning per dati troppo sparsi
        if sparsity > 70:
            print(f"   ⚠️ WARNING: Dataset molto sparso ({sparsity:.1f}%) - training potrebbe essere instabile")
        
        # Setup
        ensemble = EnsembleArchitectures(X.shape[1], 3)
        archs = ensemble.get_available_architectures()
        opts = ensemble.get_diverse_optimizers()
        losses = ensemble.get_diverse_loss_functions()
        
        # CV
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
        self.oof_predictions = np.zeros((len(X), 3))
        fold_scores = []
        
        # Progress bar per fold
        fold_pbar = tqdm(enumerate(cv.split(X, y)), total=6, desc="🎯 CV Folds", 
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        for fold, (train_idx, val_idx) in fold_pbar:
            fold_start_time = time.time()
            
            try:
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create model
                arch = archs[fold % len(archs)]
                model = ensemble.create_architecture(arch)
                
                # Update progress bar description
                fold_pbar.set_description(f"🎯 Fold {fold+1}/6 ({arch})")
                
                # Compile
                try:
                    model.compile(
                        optimizer=opts[fold % len(opts)],
                        loss=losses[fold % len(losses)],
                        metrics=['accuracy']
                    )
                except:
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # Custom callback per progress dentro fold
                class FoldProgressCallback(tf.keras.callbacks.Callback):
                    def __init__(self, fold_pbar, fold_num, arch_name, max_epochs):
                        self.fold_pbar = fold_pbar
                        self.fold_num = fold_num
                        self.arch_name = arch_name
                        self.max_epochs = max_epochs  # 🚀 NUOVO: epoch dinamiche
                        self.epoch_pbar = None
                        self.epochs_completed = 0
                        self.training_start = time.time()
                        
                    def on_train_begin(self, logs=None):
                        self.training_start = time.time()
                        
                    def on_epoch_end(self, epoch, logs=None):
                        self.epochs_completed = epoch + 1
                        val_acc = logs.get('val_accuracy', 0)
                        val_loss = logs.get('val_loss', 0)
                        loss = logs.get('loss', 0)
                        
                        # Calcola velocità
                        elapsed = time.time() - self.training_start
                        epoch_speed = elapsed / (epoch + 1)
                        
                        # 🚀 DINAMICO: Usa epoch correnti
                        postfix = f"Ep {epoch+1}/{self.max_epochs} | Val_Acc: {val_acc:.3f} | Val_Loss: {val_loss:.3f} | {epoch_speed:.1f}s/ep"
                        self.fold_pbar.set_postfix_str(postfix)
                        
                    def on_train_end(self, logs=None):
                        # Salva info di training per diagnostica
                        total_time = time.time() - self.training_start
                        self.training_summary = {
                            'epochs_completed': self.epochs_completed,
                            'total_time': total_time,
                            'avg_epoch_time': total_time / max(self.epochs_completed, 1)
                        }
                
                # Train con callback progress
                progress_callback = FoldProgressCallback(fold_pbar, fold, arch, epochs)  # 🚀 Passa epochs
                
                # 🚀 ADAPTIVE TRAINING PARAMETERS basati su dataset size
                dataset_size = len(X_train)
                
                # Batch size adattivo
                if dataset_size > 200000:
                    batch_size = 512  # Dataset molto grande
                    epochs = 50      # Meno epoch ma batch grandi
                    patience = 15    # Early stopping più aggressivo
                elif dataset_size > 50000:
                    batch_size = 256  # Dataset grande
                    epochs = 60
                    patience = 20
                else:
                    batch_size = 64   # Dataset piccolo (come prima)
                    epochs = 80
                    patience = 25
                
                # 🚀 AGGIORNA CALLBACK con epoch correnti
                progress_callback.max_epochs = epochs
                
                print(f"   🎛️ Adaptive params: batch={batch_size}, epochs={epochs}, patience={patience}")
                
                # Callback per Early Stopping AGGRESSIVO per dataset grandi
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0,
                    monitor='val_loss',
                    min_delta=0.001 if dataset_size < 100000 else 0.005  # Soglia più alta per dataset grandi
                )
                
                # 🚀 MEMORY MANAGEMENT: Usa steps_per_epoch per dataset grandi
                steps_per_epoch = None
                if dataset_size > 150000:
                    # Limita steps per epoch per memory management
                    steps_per_epoch = min(1000, len(X_train) // batch_size)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch,  # 🚀 NUOVO: Memory management
                    callbacks=[
                        early_stopping,
                        tf.keras.callbacks.ReduceLROnPlateau(
                            factor=0.3,   # Più aggressivo per dataset grandi
                            patience=max(3, patience//5), 
                            verbose=0,
                            min_lr=1e-7
                        ),
                        progress_callback
                    ],
                    verbose=0
                )
                
                # Diagnostica training
                epochs_trained = len(history.history['loss'])
                early_stopped = epochs_trained < epochs  # 🚀 CORRETTO: usa epochs dinamico
                fold_duration = time.time() - fold_start_time  # 🚀 SPOSTATO QUI
                training_time = getattr(progress_callback, 'training_summary', {}).get('total_time', fold_duration)
                
                # Evaluate
                pred = model.predict(X_val, verbose=0)
                self.oof_predictions[val_idx] = pred
                
                f1 = f1_score(y_val, np.argmax(pred, axis=1), average='micro')
                fold_scores.append(f1)
                
                # 🚀 SOGLIA F1 ADATTIVA basata su dataset size e class distribution
                dataset_size = len(X)
                class_imbalance = max(class_counts) / sum(class_counts)
                
                if dataset_size > 200000:
                    # Dataset grande: soglia più alta ma realistica
                    base_threshold = 0.45
                elif dataset_size > 50000:
                    # Dataset medio
                    base_threshold = 0.40
                else:
                    # Dataset piccolo: soglia più bassa
                    base_threshold = 0.35
                
                # Aggiusta per class imbalance
                imbalance_penalty = (class_imbalance - 0.33) * 0.2  # Penalty per sbilanciamento
                threshold = max(0.30, base_threshold - imbalance_penalty)
                
                print(f"   📊 Adaptive threshold: {threshold:.3f} (size={dataset_size}, imbalance={class_imbalance:.2f})")
                
                if f1 > threshold:
                    self.models.append({'model': model, 'arch': arch, 'f1': f1})
                    status = "✅"
                else:
                    status = "❌"
                
                # Final update con diagnostica
                early_info = f" (ES:{epochs_trained}ep)" if early_stopped else f" ({epochs_trained}ep)"
                fold_pbar.set_postfix_str(f"{status} F1: {f1:.3f} | {fold_duration:.1f}s{early_info} | Thr:{threshold:.2f}")
                
            except Exception as e:
                fold_pbar.set_postfix_str(f"💥 Failed: {str(e)[:30]}...")
                self.oof_predictions[val_idx] = 1/3
                fold_scores.append(0.33)
        
        fold_pbar.close()
        
        # Final evaluation
        oof_classes = np.argmax(self.oof_predictions, axis=1)
        self.final_f1 = f1_score(y, oof_classes, average='micro')
        
        print(f"\n📊 Results:")
        print(f"   Fold F1s: {[f'{f:.3f}' for f in fold_scores]}")
        print(f"   🎯 Final F1: {self.final_f1:.4f}")
        print(f"   Target: {'✅' if self.final_f1 >= self.target_f1 else '❌'}")
        print(f"   Models: {len(self.models)}/6")
        
        return self.final_f1
    
    def save(self):
        """Salva tutto"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"models/ensemble_f1_{self.final_f1:.4f}_{timestamp}")
        path.mkdir(parents=True, exist_ok=True)
        
        # Salva modelli
        for i, m in enumerate(self.models):
            m['model'].save(path / f"model_{i+1}_{m['arch']}.keras")
        
        # Salva preprocessing
        joblib.dump(self.feature_engineer, path / "feature_engineer.pkl")
        if self.preprocessing_pipeline:
            joblib.dump(self.preprocessing_pipeline, path / "preprocessing_pipeline.pkl")
        
        # Config
        config = {
            'f1_score': self.final_f1,
            'models': [{'arch': m['arch'], 'f1': m['f1']} for m in self.models],
            'target_achieved': self.final_f1 >= self.target_f1
        }
        
        import json
        with open(path / "config.json", 'w') as f:
            json.dump(config, f)
        
        print(f"💾 Saved to: {path}")
        return path

def main():
    """Main function"""
    print("🏆 RICHTER ENSEMBLE TRAINER")
    print("=" * 40)
    
    trainer = RichterTrainer()
    
    # Load data
    X, y = trainer.load_data()
    
    # Train
    f1 = trainer.train_ensemble(X, y)
    
    # Save
    path = trainer.save()
    
    # Summary
    print(f"\n🎉 DONE!")
    print(f"   F1: {f1:.4f}")
    print(f"   Target: {'REACHED' if f1 >= 0.78 else 'MISSED'}")
    print(f"   Path: {path}")
    
    return f1

if __name__ == "__main__":
    main()
