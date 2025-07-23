#!/usr/bin/env python3
"""
Dataset Loader per Richter Predictor - Preprocessing Pipeline

Questo script si occupa SOLO di:
1. Caricare i dati raw da data/raw/
2. Fare merge train_values + train_lab        """
        print("Salvataggio dataset training...")
         self.save_train_dataset(train_df, quality_info, feature_mapping)
        
        print("\nDataset training creato con successo!")
        print(f"Output directory: {self.interim_dir}")
        print("\nIMPORTANTE:")
        print("   • test_values.csv NON è stato toccato")
        print("   • Sarà usato SOLO per valutazione finale")
        print("\nProssimi passi:")
        print("1. Usa i dati di training per sviluppare la pipeline")
        print("2. Fai train/validation split sui dati di training")      # Salva SOLO training dataset
        train_path = self.interim_dir / "train_dataset.parquet"
        train_df.to_parquet(train_path, index=False)
        
        print(f"   Train dataset: {train_path}")
        print(f"   Test dataset: NON creato (test_values.csv protetto)")
        
        # Salva metadaticare feature_mapping.json da data_analysis.py
4. Salvare dataset pulito per la pipeline di preprocessing

NON fa preprocessing (quello è delegato alla pipeline modulare).
Mantiene i dati raw per permettere alla pipeline di decidere le trasformazioni.

Usage:
    python src/data/make_dataset_tf.py
    
Outputs:
    - data/interim/train_dataset.parquet (SOLO dati training)
    - data/interim/dataset_info.json (metadati)
    
NOTA: test_values.csv è PROTETTO e NON viene caricato!
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DatasetLoader:
    """
    Carica e prepara i dataset per la pipeline di preprocessing.
    
    Si occupa solo di operazioni 'safe' sui dati raw:
    - Merge train_values + train_labels
    - Validazione consistenza dati
    - Salvataggio in formato efficiente
    """
    
    def __init__(self, raw_dir: str = "data/raw", interim_dir: str = "data/interim"):
        """
        Inizializza il loader.
        
        Args:
            raw_dir: Directory con i dati raw
            interim_dir: Directory output per dati interim
        """
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        
        # Verifica presenza file necessari
        self._validate_raw_files()
        
    def _validate_raw_files(self) -> None:
        """Verifica che i file raw necessari esistano."""
        required_files = [
            "train_values.csv",
            "train_labels.csv"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.raw_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"File mancanti in {self.raw_dir}: {missing_files}")
        
        print(f"File training trovati in {self.raw_dir}")
        print(f"test_values.csv viene IGNORATO (riservato per valutazione finale)")
    
    def load_feature_mapping(self) -> Optional[Dict]:
        """
        Carica il feature mapping da data_analysis.py se disponibile.
        
        Returns:
            Dict con feature mapping o None se non trovato
        """
        mapping_path = Path("reports/eda/feature_mapping.json")
        
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            print(f"Feature mapping caricato da {mapping_path}")
            return mapping
        else:
            print("Feature mapping non trovato - verrà usata classificazione automatica")
            return None
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Carica e fa merge dei dati di training.
        
        Returns:
            DataFrame con features + target
        """
        print("Caricamento dati di training...")
        
        # Carica files
        train_values = pd.read_csv(self.raw_dir / "train_values.csv")
        train_labels = pd.read_csv(self.raw_dir / "train_labels.csv")
        
        print(f"   train_values: {train_values.shape}")
        print(f"   train_labels: {train_labels.shape}")
        
        # Merge
        train_df = train_values.merge(
            train_labels, 
            on="building_id", 
            how="inner",
            validate="one_to_one"
        )
        
        print(f"   Merged shape: {train_df.shape}")
        
        # Validazione merge
        if len(train_df) != len(train_values):
            print(f"Warning: Merge ha perso {len(train_values) - len(train_df)} righe")
        
        return train_df
    
    def load_test_data(self) -> None:
        """
        METODO DISABILITATO - test_values.csv è protetto!
        
        Il test set finale NON deve essere caricato durante lo sviluppo.
        Sarà usato SOLO per la valutazione finale del modello.
        """
        raise NotImplementedError(
            "ACCESSO NEGATO: test_values.csv è riservato per valutazione finale!"
        )
    
    def analyze_train_quality(self, train_df: pd.DataFrame) -> Dict:
        """
        Analizza la qualità SOLO dei dati di training.
        
        Args:
            train_df: DataFrame training
            
        Returns:
            Dict con statistiche qualità
        """
        print("Analisi qualità dati di training...")
        
        quality_info = {
            "train_samples": len(train_df),
            "total_features": len(train_df.columns) - 2,  # -building_id -damage_grade
            "missing_values": {},
            "duplicate_building_ids": 0,
            "target_distribution": {}
        }
        
        # Missing values
        train_missing = train_df.isnull().sum()
        quality_info["missing_values"] = {
            col: int(count) for col, count in train_missing.items() if count > 0
        }
        
        # Duplicati building_id
        train_dups = train_df['building_id'].duplicated().sum()
        quality_info["duplicate_building_ids"] = int(train_dups)
        
        # Distribuzione target
        target_dist = train_df['damage_grade'].value_counts().sort_index()
        quality_info["target_distribution"] = {
            str(grade): int(count) for grade, count in target_dist.items()
        }
        
        # Report
        print(f"   Train samples: {quality_info['train_samples']:,}")
        print(f"   Features: {quality_info['total_features']}")
        
        if quality_info["missing_values"]:
            print(f"   Missing values: {len(quality_info['missing_values'])} columns")
            
        if train_dups > 0:
            print(f"   Duplicate building_ids: {train_dups}")
        
        return quality_info
    
    def save_train_dataset(
        self, 
        train_df: pd.DataFrame, 
        quality_info: Dict,
        feature_mapping: Optional[Dict] = None
    ) -> None:
        """
        Salva SOLO il dataset di training.
        
        Args:
            train_df: DataFrame training
            quality_info: Info qualità dati
            feature_mapping: Mapping features (opzionale)
        """
        print(" Salvataggio dataset training...")
        
        # Salva SOLO training dataset
        train_path = self.interim_dir / "train_dataset.parquet"
        train_df.to_parquet(train_path, index=False)
        
        print(f"    Train dataset: {train_path}")
        print(f"    Test dataset: NON creato (test_values.csv protetto)")
        
        # Salva metadati
        metadata = {
            "created_by": "DatasetLoader",
            "data_quality": quality_info,
            "feature_mapping": feature_mapping,
            "file_paths": {
                "train": str(train_path),
                "test": "PROTECTED - not loaded"
            },
            "notes": [
                "Solo dati di training caricati",
                "test_values.csv NON è stato toccato", 
                "Usa train/validation split per sviluppo",
                "test_values.csv sarà usato SOLO per valutazione finale"
            ]
        }
        
        metadata_path = self.interim_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   Metadati: {metadata_path}")
    
    def create_datasets(self) -> pd.DataFrame:
        """
        Workflow completo: carica, analizza e salva SOLO i dati di training.
        
        Returns:
            DataFrame di training
        """
        print("CREAZIONE DATASET INTERIM (SOLO TRAINING)")
        print("=" * 50)
        
        # 1. Carica feature mapping
        feature_mapping = self.load_feature_mapping()
        
        # 2. Carica SOLO dati di training
        train_df = self.load_train_data()
        
        # 3. Analizza qualità (solo training)
        quality_info = self.analyze_train_quality(train_df)
        
        # 4. Salva SOLO training dataset
        self.save_train_dataset(train_df, quality_info, feature_mapping)
        
        print("\n Dataset training creato con successo!")
        print(f" Output directory: {self.interim_dir}")
        print("\n IMPORTANTE:")
        print("   • test_values.csv NON è stato toccato")
        print("   • Sarà usato SOLO per valutazione finale")
        print("\n Prossimi passi:")
        print("1. Usa i dati di training per sviluppare la pipeline")
        print("2. Fai train/validation split sui dati di training")
        
        return train_df


def main():
    """Funzione principale."""
    try:
        loader = DatasetLoader()
        train_df = loader.create_datasets()
        return train_df
        
    except Exception as e:
        print(f"Errore durante creazione dataset: {e}")
        raise


if __name__ == "__main__":
    train_df = main()