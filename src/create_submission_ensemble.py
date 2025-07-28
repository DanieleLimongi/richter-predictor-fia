#!/usr/bin/env python3
"""
Richter Predictor - Ensemble Model Submission Generator
Genera submission.csv per DrivenData competition usando modelli da train_nested_cv_ensemble.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import json
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Setup path dinamico
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Import componenti
from feature_engineering import AdvancedFeatureEngineer

# Configurazione TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class EnsembleSubmissionGenerator:
    """Generatore di submission per modelli Nested CV Ensemble"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / 'data' / 'raw'
        self.nested_models_dir = project_root / 'models' / 'nested_models'
        self.submissions_dir = project_root / 'submissions'
        
        # Assicura che directory submissions esista
        self.submissions_dir.mkdir(exist_ok=True)
        
        print("RICHTER PREDICTOR - ENSEMBLE MODEL SUBMISSION GENERATOR")
        print("=" * 75)
        print(f"Models directory: {self.nested_models_dir}")
        
    def load_test_data(self):
        """Carica test set"""
        print("Loading test dataset...")
        
        test_path = self.data_dir / 'test_values.csv'
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
            
        test_df = pd.read_csv(test_path)
        print(f"   Test set loaded: {test_df.shape}")
        print(f"   Building IDs range: {test_df['building_id'].min()} - {test_df['building_id'].max()}")
        
        return test_df
    
    def find_best_ensemble(self, specific_ensemble=None):
        """Trova il miglior ensemble nested CV"""
        print("Finding best nested CV ensemble...")
        
        if specific_ensemble:
            ensemble_path = self.nested_models_dir / specific_ensemble
            if not ensemble_path.exists():
                raise FileNotFoundError(f"Specific ensemble not found: {ensemble_path}")
            print(f"   Using specific ensemble: {specific_ensemble}")
            return ensemble_path
        
        # Cerca directory ensemble nella directory nested_models
        ensemble_dirs = [d for d in self.nested_models_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('nested_cv_ensemble_f1_')]
        
        if not ensemble_dirs:
            raise FileNotFoundError(f"No ensemble directories found in {self.nested_models_dir}")
        
        # Ordina per F1 score nel nome (più alto prima)
        best_ensemble = max(ensemble_dirs, 
                           key=lambda x: float(x.name.split('_f1_')[1].split('_')[0]))
        
        print(f"   Best ensemble found: {best_ensemble.name}")
        
        # Carica configurazione
        config_path = best_ensemble / 'nested_cv_config.json'
        config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"   Ensemble F1 score: {config.get('final_f1', 'N/A')}")
            print(f"   Ensemble models: {len([f for f in best_ensemble.iterdir() if f.suffix == '.keras'])}")
        
        return best_ensemble, config
    
    def find_matching_feature_engineer(self, ensemble_path):
        """Trova il feature engineer corrispondente all'ensemble"""
        print("Finding matching feature engineer...")
        
        # Estrai timestamp dal nome dell'ensemble
        ensemble_name = ensemble_path.name
        
        # Cerca pattern di timestamp nel nome dell'ensemble
        timestamp_patterns = []
        parts = ensemble_name.split('_')
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # YYYYMMDD
                if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():  # HHMMSS
                    timestamp_patterns.append(f"{part}_{parts[i + 1]}")
        
        print(f"   Looking for feature engineer with timestamps: {timestamp_patterns}")
        
        # 1. Cerca feature engineer con stesso timestamp nella directory dell'ensemble
        for timestamp in timestamp_patterns:
            engineer_path = ensemble_path / f"feature_engineer_{timestamp}.pkl"
            if engineer_path.exists():
                print(f"   Found matching feature engineer: {engineer_path.name}")
                return engineer_path
        
        # 2. Cerca nella directory nested_models
        for timestamp in timestamp_patterns:
            engineer_path = self.nested_models_dir / f"feature_engineer_{timestamp}.pkl"
            if engineer_path.exists():
                print(f"   Found feature engineer in nested_models: {engineer_path.name}")
                return engineer_path
        
        # 3. Cerca nella directory models principale
        for timestamp in timestamp_patterns:
            engineer_path = self.project_root / 'models' / f"feature_engineer_{timestamp}.pkl"
            if engineer_path.exists():
                print(f"   Found feature engineer in main models dir: {engineer_path.name}")
                return engineer_path
        
        # 4. Fallback: cerca il feature engineer più recente nella directory dell'ensemble
        pkl_files = list(ensemble_path.glob('feature_engineer_*.pkl'))
        if pkl_files:
            latest_engineer = max(pkl_files, key=lambda x: x.stat().st_mtime)
            print(f"   Using latest feature engineer from ensemble dir: {latest_engineer.name}")
            return latest_engineer
        
        # 5. Fallback: cerca nella directory nested_models
        pkl_files = list(self.nested_models_dir.glob('feature_engineer_*.pkl'))
        if pkl_files:
            latest_engineer = max(pkl_files, key=lambda x: x.stat().st_mtime)
            print(f"   Using latest feature engineer from nested_models: {latest_engineer.name}")
            return latest_engineer
        
        # 6. Fallback: cerca nella directory models principale
        pkl_files = list(self.project_root.glob('models/**/*.pkl'))
        if pkl_files:
            latest_engineer = max(pkl_files, key=lambda x: x.stat().st_mtime)
            print(f"   Using feature engineer from models: {latest_engineer.name}")
            return latest_engineer
        
        # 7. Fallback finale: cerca in tutta la directory del progetto
        pkl_files = list(self.project_root.glob('**/*.pkl'))
        if pkl_files:
            latest_engineer = max(pkl_files, key=lambda x: x.stat().st_mtime)
            print(f"   Using feature engineer from project: {latest_engineer.name}")
            return latest_engineer
        
        # 8. Ultimo fallback: errore se non trovato
        raise FileNotFoundError(
            "No compatible feature engineer found! "
            "The nested CV training must save a feature engineer. "
            "Please re-run training with: ./docker-helper.sh train-nested"
        )
    
    def load_feature_engineer(self, engineer_path):
        """Carica feature engineer"""
        print(f"Loading feature engineer from: {engineer_path}")
        
        with open(engineer_path, 'rb') as f:
            engineer = pickle.load(f)
            
        print(f"   Feature engineer loaded successfully")
        print(f"   Fitted status: {engineer.fitted}")
        
        if not engineer.fitted:
            raise ValueError("Feature engineer is not fitted!")
        
        return engineer
    
    def apply_feature_engineering(self, test_df, engineer):
        """Applica feature engineering al test set"""
        print("Applying feature engineering to test set...")
        
        # Prepara features (rimuovi building_id se presente)
        feature_cols = [col for col in test_df.columns if col != 'building_id']
        X_test_df = test_df[feature_cols]
        
        print(f"   Original features: {len(feature_cols)}")
        
        # Il feature engineer DEVE essere fitted - no exceptions
        if not engineer.fitted:
            raise ValueError(
                "Feature engineer is not fitted! "
                "The nested CV training must save a fitted feature engineer. "
                "Please re-run training with: ./docker-helper.sh train-nested"
            )
        
        # Applica feature engineering (solo transform)
        X_test_enhanced = engineer.transform(X_test_df)
        
        print(f"   Features after engineering: {len(X_test_enhanced.columns)}")
        print(f"   Features added: +{len(X_test_enhanced.columns) - len(X_test_df.columns)}")
        
        # Conversione a numpy
        X_test = X_test_enhanced.values.astype(np.float32)
        
        # Verifica qualità dati
        nan_count = np.isnan(X_test).sum()
        inf_count = np.isinf(X_test).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"   Cleaning data: NaN={nan_count}, Inf={inf_count}")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"   Test set processed: {X_test.shape}")
        
        return X_test
    
    def load_ensemble_models(self, ensemble_path):
        """Carica tutti i modelli dell'ensemble"""
        print(f"Loading ensemble models from: {ensemble_path.name}")
        
        # Trova tutti i modelli .keras
        model_files = list(ensemble_path.glob('*.keras'))
        if not model_files:
            raise FileNotFoundError(f"No .keras models found in {ensemble_path}")
        
        # Ordina per nome per consistenza
        model_files = sorted(model_files)
        print(f"   Found {len(model_files)} models to load")
        
        models = []
        model_info = []
        
        for model_path in model_files:
            try:
                print(f"   Loading: {model_path.name}")
                model = tf.keras.models.load_model(model_path)
                
                models.append(model)
                model_info.append({
                    'name': model_path.name,
                    'path': str(model_path),
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape
                })
                
            except Exception as e:
                print(f"   Warning: Failed to load {model_path.name}: {e}")
                continue
        
        if not models:
            raise RuntimeError("No models could be loaded successfully")
        
        print(f"   Successfully loaded {len(models)} models")
        
        # Verifica consistenza dimensioni
        input_shapes = [model.input_shape for model in models]
        if not all(shape == input_shapes[0] for shape in input_shapes):
            raise ValueError("Models have inconsistent input shapes")
        
        return models, model_info
    
    def generate_ensemble_predictions(self, X_test, models, model_info):
        """Genera predizioni usando ensemble di modelli"""
        print("Generating ensemble predictions...")
        
        # Verifica compatibilità dimensioni
        expected_features = models[0].input_shape[1]
        if X_test.shape[1] != expected_features:
            raise ValueError(f"Feature mismatch: models expect {expected_features}, got {X_test.shape[1]}")
        
        ensemble_predictions = []
        successful_models = []
        
        for i, (model, info) in enumerate(zip(models, model_info), 1):
            try:
                print(f"   Predicting with model {i}/{len(models)}: {info['name']}")
                
                # Predizione
                pred = model.predict(X_test, batch_size=1024, verbose=0)
                ensemble_predictions.append(pred)
                successful_models.append(info)
                
                # Cleanup memoria
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"   Warning: Prediction failed for {info['name']}: {e}")
                continue
        
        if not ensemble_predictions:
            raise RuntimeError("No successful predictions generated")
        
        # Calcola media ensemble
        print(f"   Computing ensemble average from {len(ensemble_predictions)} predictions...")
        final_predictions = np.mean(ensemble_predictions, axis=0)
        
        # Converti a classi (0,1,2 -> 1,2,3 per submission)
        predicted_classes = np.argmax(final_predictions, axis=1) + 1
        
        print(f"   Ensemble predictions completed")
        print(f"   Distribution: {np.bincount(predicted_classes, minlength=4)[1:]}")  # Skip class 0
        
        return predicted_classes, final_predictions, successful_models
    
    def create_submission(self, building_ids, predictions, ensemble_info, output_name=None):
        """Crea file submission.csv"""
        print("Creating submission.csv...")
        
        # Crea DataFrame submission
        submission_df = pd.DataFrame({
            'building_id': building_ids,
            'damage_grade': predictions
        })
        
        # Verifica formato
        assert len(submission_df) == len(building_ids), "Prediction length mismatch"
        assert submission_df['damage_grade'].isin([1, 2, 3]).all(), "Damage grade must be 1, 2, or 3"
        assert submission_df['building_id'].is_unique, "Building IDs must be unique"
        
        # Nome file output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_name:
            filename = f"submission_ensemble_{output_name}_{timestamp}.csv"
        else:
            filename = f"submission_ensemble_{timestamp}.csv"
            
        output_path = self.submissions_dir / filename
        
        # Salva
        submission_df.to_csv(output_path, index=False)
        
        print(f"   Submission saved: {output_path}")
        print(f"   Samples: {len(submission_df)}")
        print(f"   Damage distribution: {submission_df['damage_grade'].value_counts().sort_index().to_dict()}")
        
        # Salva metadati
        metadata = {
            'timestamp': timestamp,
            'model_type': 'nested_cv_ensemble',
            'ensemble_info': ensemble_info,
            'submission_file': str(output_path),
            'num_predictions': len(submission_df),
            'damage_distribution': submission_df['damage_grade'].value_counts().sort_index().to_dict(),
            'building_id_range': [int(submission_df['building_id'].min()), 
                                 int(submission_df['building_id'].max())]
        }
        
        metadata_path = self.submissions_dir / f"metadata_ensemble_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"   Metadata saved: {metadata_path}")
        
        return output_path, metadata
    
    def generate_submission(self, output_name=None, specific_ensemble=None):
        """Pipeline completa per generare submission da ensemble"""
        try:
            # 1. Carica test data
            test_df = self.load_test_data()
            building_ids = test_df['building_id'].values
            
            # 2. Trova miglior ensemble
            if specific_ensemble:
                ensemble_path = self.nested_models_dir / specific_ensemble
                if not ensemble_path.exists():
                    raise FileNotFoundError(f"Specific ensemble not found: {ensemble_path}")
                ensemble_config = None
            else:
                ensemble_path, ensemble_config = self.find_best_ensemble()
            
            # 3. Trova feature engineer corrispondente
            engineer_path = self.find_matching_feature_engineer(ensemble_path)
            
            # 4. Carica feature engineer
            engineer = self.load_feature_engineer(engineer_path)
            
            # 5. Applica feature engineering
            X_test = self.apply_feature_engineering(test_df, engineer)
            
            # 6. Carica modelli ensemble
            models, model_info = self.load_ensemble_models(ensemble_path)
            
            # 7. Genera predizioni ensemble
            predictions, raw_predictions, successful_models = self.generate_ensemble_predictions(
                X_test, models, model_info)
            
            # 8. Informazioni ensemble
            ensemble_info = {
                'ensemble_path': str(ensemble_path),
                'engineer_path': str(engineer_path),
                'ensemble_config': ensemble_config,
                'successful_models': successful_models,
                'total_models': len(model_info),
                'successful_count': len(successful_models),
                'input_features': X_test.shape[1]
            }
            
            # 9. Crea submission
            submission_path, metadata = self.create_submission(
                building_ids, predictions, ensemble_info, output_name
            )
            
            print("\nENSEMBLE SUBMISSION GENERATED SUCCESSFULLY!")
            print(f"   File: {submission_path}")
            print(f"   Ensemble: {ensemble_path.name}")
            print(f"   Models used: {len(successful_models)}/{len(model_info)}")
            print(f"   Predictions: {len(predictions)}")
            
            return submission_path, metadata
            
        except Exception as e:
            print(f"\nERROR: {e}")
            raise


def main():
    """Main function con supporto argomenti command line"""
    parser = argparse.ArgumentParser(description='Generate submission for DrivenData using nested CV ensemble models')
    parser.add_argument('--output-name', type=str, 
                       help='Custom name for output file')
    parser.add_argument('--list-ensembles', action='store_true',
                       help='List available ensembles without generating submission')
    parser.add_argument('--ensemble-dir', type=str,
                       help='Specific ensemble directory name to use')
    
    args = parser.parse_args()
    
    # Inizializza generator
    generator = EnsembleSubmissionGenerator(project_root)
    
    if args.list_ensembles:
        print("AVAILABLE ENSEMBLE MODELS:")
        print("-" * 60)
        
        # Lista ensemble
        ensemble_dirs = [d for d in generator.nested_models_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('nested_cv_ensemble_f1_')]
        
        if ensemble_dirs:
            for ensemble_dir in sorted(ensemble_dirs, 
                                     key=lambda x: float(x.name.split('_f1_')[1].split('_')[0]), 
                                     reverse=True):
                f1_score = ensemble_dir.name.split('_f1_')[1].split('_')[0]
                num_models = len(list(ensemble_dir.glob('*.keras')))
                mtime = datetime.fromtimestamp(ensemble_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                
                # Carica info aggiuntive
                config_path = ensemble_dir / 'nested_cv_config.json'
                extra_info = ""
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        extra_info = f" (CV: {config.get('cv_mean_f1', 'N/A')}±{config.get('cv_std_f1', 'N/A')})"
                    except:
                        pass
                
                print(f"  - {ensemble_dir.name}")
                print(f"    F1: {f1_score}{extra_info}, Models: {num_models}, Modified: {mtime}")
        else:
            print("  No ensemble models found")
        
        return
    
    # Genera submission
    try:
        submission_path, metadata = generator.generate_submission(
            output_name=args.output_name,
            specific_ensemble=args.ensemble_dir
        )
        
        print(f"\nSubmission ready for upload to DrivenData!")
        print(f"   File: {submission_path}")
        
    except Exception as e:
        print(f"\nError generating submission: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()