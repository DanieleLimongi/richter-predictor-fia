#!/usr/bin/env python3
"""
Richter Predictor - Simple Model Submission Generator
Genera submission.csv per DrivenData competition usando modelli da train_simple_holdout.py
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


class SimpleModelSubmissionGenerator:
    """Generatore di submission per modelli Simple Holdout"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / 'data' / 'raw'
        self.simple_models_dir = project_root / 'models' / 'simple_models'
        self.submissions_dir = project_root / 'submissions'
        
        # Assicura che directory submissions esista
        self.submissions_dir.mkdir(exist_ok=True)
        
        print("RICHTER PREDICTOR - SIMPLE MODEL SUBMISSION GENERATOR")
        print("=" * 70)
        print(f"Models directory: {self.simple_models_dir}")
        
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
    
    def find_best_simple_model(self, specific_model=None):
        """Trova il miglior modello simple holdout"""
        print("Finding best simple holdout model...")
        
        if specific_model:
            model_path = self.simple_models_dir / specific_model
            if not model_path.exists():
                raise FileNotFoundError(f"Specific model not found: {model_path}")
            print(f"   Using specific model: {specific_model}")
            return model_path
        
        # Cerca modelli .keras nella directory simple_models
        keras_models = list(self.simple_models_dir.glob('*.keras'))
        
        if not keras_models:
            raise FileNotFoundError(f"No .keras models found in {self.simple_models_dir}")
        
        # Ordina per data di modifica (più recente prima)
        best_model = max(keras_models, key=lambda x: x.stat().st_mtime)
        
        print(f"   Best simple model found: {best_model.name}")
        
        # Cerca metadati associati se disponibili
        metadata_path = best_model.with_suffix('.json')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   Model metadata loaded: F1={metadata.get('test_f1', 'N/A')}")
        
        return best_model, metadata
    
    def find_matching_feature_engineer(self, model_path):
        """Trova il feature engineer corrispondente al modello"""
        print("Finding matching feature engineer...")
        
        # Estrai timestamp dal nome del modello
        model_name = model_path.stem
        
        # Cerca pattern di timestamp nel nome del modello
        timestamp_patterns = []
        parts = model_name.split('_')
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # YYYYMMDD
                if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():  # HHMMSS
                    timestamp_patterns.append(f"{part}_{parts[i + 1]}")
        
        # Cerca feature engineer con stesso timestamp
        for timestamp in timestamp_patterns:
            engineer_path = self.simple_models_dir / f"feature_engineer_{timestamp}.pkl"
            if engineer_path.exists():
                print(f"   Found matching feature engineer: {engineer_path.name}")
                return engineer_path
        
        # Fallback: cerca il feature engineer più recente nella directory simple_models
        pkl_files = list(self.simple_models_dir.glob('feature_engineer_*.pkl'))
        if pkl_files:
            latest_engineer = max(pkl_files, key=lambda x: x.stat().st_mtime)
            print(f"   Using latest feature engineer: {latest_engineer.name}")
            return latest_engineer
        
        # Fallback finale: cerca nella directory models principale
        pkl_files = list(self.project_root.glob('models/feature_engineer_*.pkl'))
        if pkl_files:
            latest_engineer = max(pkl_files, key=lambda x: x.stat().st_mtime)
            print(f"   Using feature engineer from main models dir: {latest_engineer.name}")
            return latest_engineer
        
        raise FileNotFoundError("No compatible feature engineer found")
    
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
        
        # Applica feature engineering (solo transform, non fit!)
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
    
    def generate_predictions(self, X_test, model_path):
        """Genera predizioni usando singolo modello"""
        print(f"Generating predictions with: {model_path.name}")
        
        # Carica modello
        model = tf.keras.models.load_model(model_path)
        print(f"   Model loaded successfully")
        print(f"   Input shape expected: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Verifica compatibilità dimensioni
        if model.input_shape[1] != X_test.shape[1]:
            raise ValueError(f"Dimension mismatch: model expects {model.input_shape[1]}, got {X_test.shape[1]}")
        
        # Predizione
        predictions = model.predict(X_test, batch_size=1024, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1) + 1  # 0,1,2 -> 1,2,3
        
        print(f"   Predictions completed")
        print(f"   Distribution: {np.bincount(predicted_classes, minlength=4)[1:]}")  # Skip class 0
        
        # Cleanup memoria
        del model
        tf.keras.backend.clear_session()
        
        return predicted_classes, predictions
    
    def create_submission(self, building_ids, predictions, model_info, output_name=None):
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
            filename = f"submission_simple_{output_name}_{timestamp}.csv"
        else:
            filename = f"submission_simple_{timestamp}.csv"
            
        output_path = self.submissions_dir / filename
        
        # Salva
        submission_df.to_csv(output_path, index=False)
        
        print(f"   Submission saved: {output_path}")
        print(f"   Samples: {len(submission_df)}")
        print(f"   Damage distribution: {submission_df['damage_grade'].value_counts().sort_index().to_dict()}")
        
        # Salva metadati
        metadata = {
            'timestamp': timestamp,
            'model_type': 'simple_holdout',
            'model_info': model_info,
            'submission_file': str(output_path),
            'num_predictions': len(submission_df),
            'damage_distribution': submission_df['damage_grade'].value_counts().sort_index().to_dict(),
            'building_id_range': [int(submission_df['building_id'].min()), 
                                 int(submission_df['building_id'].max())]
        }
        
        metadata_path = self.submissions_dir / f"metadata_simple_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"   Metadata saved: {metadata_path}")
        
        return output_path, metadata
    
    def generate_submission(self, output_name=None, specific_model=None):
        """Pipeline completa per generare submission da modello simple"""
        try:
            # 1. Carica test data
            test_df = self.load_test_data()
            building_ids = test_df['building_id'].values
            
            # 2. Trova miglior modello simple
            if specific_model:
                model_path = self.simple_models_dir / specific_model
                if not model_path.exists():
                    raise FileNotFoundError(f"Specific model not found: {model_path}")
                model_metadata = None
            else:
                model_path, model_metadata = self.find_best_simple_model()
            
            # 3. Trova feature engineer corrispondente
            engineer_path = self.find_matching_feature_engineer(model_path)
            
            # 4. Carica feature engineer
            engineer = self.load_feature_engineer(engineer_path)
            
            # 5. Applica feature engineering
            X_test = self.apply_feature_engineering(test_df, engineer)
            
            # 6. Genera predizioni
            predictions, raw_predictions = self.generate_predictions(X_test, model_path)
            
            # 7. Informazioni modello
            model_info = {
                'model_path': str(model_path),
                'engineer_path': str(engineer_path),
                'model_metadata': model_metadata,
                'input_features': X_test.shape[1]
            }
            
            # 8. Crea submission
            submission_path, metadata = self.create_submission(
                building_ids, predictions, model_info, output_name
            )
            
            print("\nSIMPLE MODEL SUBMISSION GENERATED SUCCESSFULLY!")
            print(f"   File: {submission_path}")
            print(f"   Model: {model_path.name}")
            print(f"   Predictions: {len(predictions)}")
            
            return submission_path, metadata
            
        except Exception as e:
            print(f"\nERROR: {e}")
            raise


def main():
    """Main function con supporto argomenti command line"""
    parser = argparse.ArgumentParser(description='Generate submission for DrivenData using simple holdout models')
    parser.add_argument('--output-name', type=str, 
                       help='Custom name for output file')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models without generating submission')
    parser.add_argument('--model-file', type=str,
                       help='Specific model file name to use')
    
    args = parser.parse_args()
    
    # Inizializza generator
    generator = SimpleModelSubmissionGenerator(project_root)
    
    if args.list_models:
        print("AVAILABLE SIMPLE MODELS:")
        print("-" * 50)
        
        # Lista modelli simple
        simple_models = list(generator.simple_models_dir.glob('*.keras'))
        if simple_models:
            for model in sorted(simple_models, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = datetime.fromtimestamp(model.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                
                # Cerca metadati
                metadata_path = model.with_suffix('.json')
                f1_info = ""
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        f1_info = f" (F1: {metadata.get('test_f1', 'N/A')})"
                    except:
                        pass
                
                print(f"  - {model.name}{f1_info} (Modified: {mtime})")
        else:
            print("  No simple models found")
        
        # Lista feature engineers
        engineers = list(generator.simple_models_dir.glob('feature_engineer_*.pkl'))
        if engineers:
            print("\nAVAILABLE FEATURE ENGINEERS:")
            for engineer in sorted(engineers, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = datetime.fromtimestamp(engineer.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  - {engineer.name} (Modified: {mtime})")
        
        return
    
    # Genera submission
    try:
        submission_path, metadata = generator.generate_submission(
            output_name=args.output_name,
            specific_model=args.model_file
        )
        
        print(f"\nSubmission ready for upload to DrivenData!")
        print(f"   File: {submission_path}")
        
    except Exception as e:
        print(f"\nError generating submission: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()