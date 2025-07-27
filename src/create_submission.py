#!/usr/bin/env python3
"""
Richter Predictor - Submission Generator
Genera submission.csv per DrivenData competition usando modelli addestrati
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
from data.data_analysis import DataAnalyzer
from models.ensemble_architectures import EnsembleArchitectures

# Configurazione TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class SubmissionGenerator:
    """Generatore di submission per DrivenData competition"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_dir = project_root / 'data' / 'raw'
        self.models_dir = project_root / 'models'
        self.submissions_dir = project_root / 'submissions'
        
        # Assicura che directory submissions esista
        self.submissions_dir.mkdir(exist_ok=True)
        
        print("RICHTER PREDICTOR - SUBMISSION GENERATOR")
        print("=" * 60)
        
    def load_test_data(self):
        """Carica test set"""
        print("Caricamento test set...")
        
        test_path = self.data_dir / 'test_values.csv'
        if not test_path.exists():
            raise FileNotFoundError(f"Test file non trovato: {test_path}")
            
        test_df = pd.read_csv(test_path)
        print(f"   Test set caricato: {test_df.shape}")
        print(f"   Building IDs: {test_df['building_id'].min()} - {test_df['building_id'].max()}")
        
        return test_df
    
    def find_best_model(self, model_type='single'):
        """Trova il miglior modello disponibile"""
        print(f"Ricerca migliore modello ({model_type})...")
        
        if model_type == 'single':
            # FIXED: Use latest refactored single model (F1 0.736) instead of ensemble (F1 0.702)
            best_model_path = self.models_dir / "mlp_model_refactored_20250727_203029.keras"
            if best_model_path.exists():
                print(f"   Migliore modello singolo: {best_model_path.name}")
                print(f"      Expected F1 Score: 0.736 (refactored)")
                print(f"      Features: 265")
                return best_model_path, None
            else:
                # Fallback: cerca il più recente
                keras_models = list(self.models_dir.glob('*.keras'))
                if keras_models:
                    best_single = max(keras_models, key=lambda x: x.stat().st_mtime)
                    print(f"   Fallback modello singolo: {best_single.name}")
                    return best_single, None
        
        elif model_type == 'ensemble':
            # Cerca migliore ensemble nested CV (ora opzione secondaria)
            ensemble_dirs = [d for d in self.models_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('nested_cv_ensemble_f1_')]
            
            if ensemble_dirs:
                # Ordina per F1 score nel nome
                best_ensemble = max(ensemble_dirs, 
                                  key=lambda x: float(x.name.split('_f1_')[1].split('_')[0]))
                
                config_path = best_ensemble / 'nested_cv_config.json'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    print(f"   Migliore ensemble trovato: {best_ensemble.name}")
                    print(f"      F1 Score: {config.get('final_f1', 'N/A')}")
                    print(f"      Modelli: {len([f for f in best_ensemble.iterdir() if f.suffix == '.keras'])}")
                    
                    return best_ensemble, config
        
        raise FileNotFoundError(f"Nessun modello {model_type} trovato in {self.models_dir}")
    
    def load_feature_engineer(self, specific_file=None):
        """Carica feature engineer salvato"""
        print("Caricamento feature engineer...")
        
        if specific_file:
            # Usa file specifico
            engineer_path = self.models_dir / specific_file
            if not engineer_path.exists():
                raise FileNotFoundError(f"Feature engineer specifico non trovato: {engineer_path}")
        else:
            # FIXED: Use matching feature engineer for refactored model
            engineer_path = self.models_dir / "feature_engineer_20250727_203029.pkl"
            if not engineer_path.exists():
                # Fallback: cerca il feature engineer più recente
                pkl_files = list(self.models_dir.glob('feature_engineer_*.pkl'))
                if not pkl_files:
                    raise FileNotFoundError("Nessun feature engineer trovato")
                engineer_path = max(pkl_files, key=lambda x: x.stat().st_mtime)
        
        with open(engineer_path, 'rb') as f:
            engineer = pickle.load(f)
            
        print(f"   Feature engineer caricato: {engineer_path.name}")
        print(f"      Fitted: {engineer.fitted}")
        
        return engineer
    
    def apply_feature_engineering(self, test_df, engineer):
        """Applica feature engineering al test set"""
        print("Applicazione feature engineering al test set...")
        
        # Prepara features (rimuovi building_id se presente)
        feature_cols = [col for col in test_df.columns if col != 'building_id']
        X_test_df = test_df[feature_cols]
        
        print(f"   Features originali: {len(feature_cols)}")
        
        # Applica feature engineering (solo transform, non fit!)
        X_test_enhanced = engineer.transform(X_test_df)
        
        print(f"   Features dopo engineering: {len(X_test_enhanced.columns)}")
        print(f"   Features aggiunte: +{len(X_test_enhanced.columns) - len(X_test_df.columns)}")
        
        # Conversione a numpy
        X_test = X_test_enhanced.values.astype(np.float32)
        
        # Verifica qualità dati
        nan_count = np.isnan(X_test).sum()
        inf_count = np.isinf(X_test).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"   Dati puliti: NaN={nan_count}, Inf={inf_count}")
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"   Test set processato: {X_test.shape}")
        
        return X_test
    
    def predict_ensemble(self, X_test, ensemble_dir):
        """Genera predizioni usando ensemble di modelli"""
        print("Generazione predizioni ensemble...")
        
        # Trova tutti i modelli .keras
        model_files = list(ensemble_dir.glob('*.keras'))
        if not model_files:
            raise FileNotFoundError(f"Nessun modello trovato in {ensemble_dir}")
            
        print(f"   Caricamento {len(model_files)} modelli...")
        
        ensemble_predictions = []
        
        for i, model_path in enumerate(sorted(model_files), 1):
            print(f"   Modello {i}/{len(model_files)}: {model_path.name}")
            
            try:
                # Carica modello
                model = tf.keras.models.load_model(model_path)
                
                # Predizione
                pred = model.predict(X_test, batch_size=1024, verbose=0)
                ensemble_predictions.append(pred)
                
                # Cleanup memoria
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"      Errore caricamento: {e}")
                continue
        
        if not ensemble_predictions:
            raise RuntimeError("Nessuna predizione valida generata")
            
        # Media ensemble
        print(f"   Calcolo media ensemble di {len(ensemble_predictions)} predizioni...")
        final_predictions = np.mean(ensemble_predictions, axis=0)
        
        # Converti a classi (0,1,2 -> 1,2,3 per submission)
        predicted_classes = np.argmax(final_predictions, axis=1) + 1
        
        print(f"   Predizioni completate")
        print(f"      Distribuzione: {np.bincount(predicted_classes)}")
        
        return predicted_classes, final_predictions
    
    def predict_single_model(self, X_test, model_path):
        """Genera predizioni usando singolo modello"""
        print("Generazione predizioni modello singolo...")
        
        # Carica modello
        model = tf.keras.models.load_model(model_path)
        print(f"   Modello caricato: {model_path.name}")
        
        # Predizione
        predictions = model.predict(X_test, batch_size=1024, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1) + 1  # 0,1,2 -> 1,2,3
        
        print(f"   Predizioni completate")
        print(f"      Distribuzione: {np.bincount(predicted_classes)}")
        
        return predicted_classes, predictions
    
    def create_submission(self, building_ids, predictions, model_info, output_name=None):
        """Crea file submission.csv"""
        print("Creazione submission.csv...")
        
        # Crea DataFrame submission
        submission_df = pd.DataFrame({
            'building_id': building_ids,
            'damage_grade': predictions
        })
        
        # Verifica formato
        assert len(submission_df) == len(building_ids), "Lunghezza predizioni non corretta"
        assert submission_df['damage_grade'].isin([1, 2, 3]).all(), "Damage grade deve essere 1, 2, o 3"
        assert submission_df['building_id'].is_unique, "Building IDs devono essere unici"
        
        # Nome file output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_name:
            filename = f"submission_{output_name}_{timestamp}.csv"
        else:
            filename = f"submission_{timestamp}.csv"
            
        output_path = self.submissions_dir / filename
        
        # Salva
        submission_df.to_csv(output_path, index=False)
        
        print(f"   Submission salvata: {output_path}")
        print(f"      Samples: {len(submission_df)}")
        print(f"      Damage distribution: {submission_df['damage_grade'].value_counts().sort_index().to_dict()}")
        
        # Salva metadati
        metadata = {
            'timestamp': timestamp,
            'model_info': model_info,
            'submission_file': str(output_path),
            'num_predictions': len(submission_df),
            'damage_distribution': submission_df['damage_grade'].value_counts().sort_index().to_dict(),
            'building_id_range': [int(submission_df['building_id'].min()), 
                                 int(submission_df['building_id'].max())]
        }
        
        metadata_path = self.submissions_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"   Metadati salvati: {metadata_path}")
        
        return output_path, metadata
    
    def generate_submission(self, model_type='ensemble', output_name=None, specific_model=None, specific_engineer=None):
        """Pipeline completa per generare submission"""
        try:
            # 1. Carica test data
            test_df = self.load_test_data()
            building_ids = test_df['building_id'].values
            
            # 2. Carica feature engineer
            engineer = self.load_feature_engineer(specific_engineer)
            
            # 3. Applica feature engineering
            X_test = self.apply_feature_engineering(test_df, engineer)
            
            # 4. Trova e carica modello migliore
            if model_type == 'ensemble':
                model_path, config = self.find_best_model('ensemble')
                predictions, raw_predictions = self.predict_ensemble(X_test, model_path)
                model_info = {
                    'type': 'ensemble',
                    'path': str(model_path),
                    'config': config,
                    'num_models': len(list(model_path.glob('*.keras')))
                }
            else:
                if specific_model:
                    model_path = self.models_dir / specific_model
                    if not model_path.exists():
                        raise FileNotFoundError(f"Modello specifico non trovato: {model_path}")
                    print(f"Usando modello specifico: {specific_model}")
                else:
                    model_path, _ = self.find_best_model('single')
                    
                predictions, raw_predictions = self.predict_single_model(X_test, model_path)
                model_info = {
                    'type': 'single',
                    'path': str(model_path)
                }
            
            # 5. Crea submission
            submission_path, metadata = self.create_submission(
                building_ids, predictions, model_info, output_name
            )
            
            print("\nSUBMISSION GENERATA CON SUCCESSO!")
            print(f"   File: {submission_path}")
            print(f"   Tipo modello: {model_type}")
            print(f"   Predizioni: {len(predictions)}")
            
            return submission_path, metadata
            
        except Exception as e:
            print(f"\nERRORE: {e}")
            raise


def main():
    """Main function con supporto argomenti command line"""
    parser = argparse.ArgumentParser(description='Genera submission per DrivenData')
    parser.add_argument('--model-type', choices=['ensemble', 'single'], 
                       default='single', help='Tipo di modello da usare')
    parser.add_argument('--output-name', type=str, 
                       help='Nome personalizzato per il file output')
    parser.add_argument('--list-models', action='store_true',
                       help='Lista modelli disponibili senza generare submission')
    parser.add_argument('--model-file', type=str,
                       help='Nome specifico del file modello da usare')
    parser.add_argument('--engineer-file', type=str,
                       help='Nome specifico del file feature engineer da usare')
    
    args = parser.parse_args()
    
    # Inizializza generator
    generator = SubmissionGenerator(project_root)
    
    if args.list_models:
        print("MODELLI DISPONIBILI:")
        print("-" * 40)
        
        # Lista ensemble
        ensemble_dirs = [d for d in generator.models_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('nested_cv_ensemble_f1_')]
        if ensemble_dirs:
            print("ENSEMBLE MODELS:")
            for ensemble_dir in sorted(ensemble_dirs):
                f1_score = ensemble_dir.name.split('_f1_')[1].split('_')[0]
                num_models = len(list(ensemble_dir.glob('*.keras')))
                print(f"  - {ensemble_dir.name} (F1: {f1_score}, {num_models} modelli)")
        
        # Lista singoli
        single_models = list(generator.models_dir.glob('*.keras'))
        if single_models:
            print("\nSINGLE MODELS:")
            for model in sorted(single_models, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = datetime.fromtimestamp(model.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  - {model.name} (Modified: {mtime})")
        
        # Lista feature engineers
        engineers = list(generator.models_dir.glob('feature_engineer_*.pkl'))
        if engineers:
            print("\nFEATURE ENGINEERS:")
            for engineer in sorted(engineers, key=lambda x: x.stat().st_mtime, reverse=True):
                mtime = datetime.fromtimestamp(engineer.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  - {engineer.name} (Modified: {mtime})")
        
        return
    
    # Genera submission
    try:
            submission_path, metadata = generator.generate_submission(
            model_type=args.model_type,
            output_name=args.output_name,
            specific_model=args.model_file,
            specific_engineer=args.engineer_file
        )
        
        print(f"\nSubmission pronta per upload su DrivenData!")
        print(f"   File: {submission_path}")
        
    except Exception as e:
        print(f"\nErrore generazione submission: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()