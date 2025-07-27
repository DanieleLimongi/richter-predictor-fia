#!/usr/bin/env python3
"""
Synthetic Data Factory - Factory centralizzato per la creazione di dati di test
Elimina la duplicazione di codice per la generazione di dataset sintetici
utilizzati in tutti i test del Richter Predictor
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import tempfile
import os


class SyntheticDataFactory:
    """Factory per la creazione standardizzata di dati di test"""
    
    # Configurazioni realistiche basate sul dataset Nepal
    REALISTIC_CONFIGS = {
        'geo_levels': {
            'geo_level_1_id': range(1, 32),  # 31 province/zone
            'geo_level_2_id': range(1, 201),  # 200 distretti
            'geo_level_3_id': range(1, 1001)  # 1000+ comuni
        },
        'categorical_features': {
            'foundation_type': ['r', 'w', 'i', 'u', 'h'],
            'roof_type': ['n', 'q', 'x'],
            'ground_floor_type': ['f', 'm', 'v', 'x', 'z'],
            'other_floor_type': ['j', 'q', 's', 'x'],
            'position': ['j', 'o', 's', 't'],
            'plan_configuration': ['a', 'c', 'd', 'f', 'm'],
            'land_surface_condition': ['n', 'o', 't'],
            'legal_ownership_status': ['a', 'r', 'v', 'w']
        },
        'superstructure_features': [
            'has_superstructure_adobe_mud',
            'has_superstructure_mud_mortar_stone',
            'has_superstructure_stone_flag',
            'has_superstructure_cement_mortar_stone',
            'has_superstructure_mud_mortar_brick',
            'has_superstructure_cement_mortar_brick',
            'has_superstructure_timber',
            'has_superstructure_bamboo',
            'has_superstructure_rc_non_engineered',
            'has_superstructure_rc_engineered',
            'has_superstructure_other'
        ],
        'damage_distribution': [0.12, 0.57, 0.31]  # Grade 1, 2, 3 realistic distribution
    }
    
    @classmethod
    def create_building_dataset(
        cls, 
        n_samples: int = 500, 
        seed: int = 42,
        damage_distribution: Optional[list] = None,
        include_percentages: bool = True,
        include_all_features: bool = True
    ) -> pd.DataFrame:
        """
        Crea dataset sintetico standardizzato per tutti i test
        
        Args:
            n_samples: Numero di campioni da generare
            seed: Seed per riproducibilità
            damage_distribution: Distribuzione danni personalizzata [grade1, grade2, grade3]
            include_percentages: Include area_percentage e height_percentage
            include_all_features: Include tutte le features o solo essenziali
            
        Returns:
            DataFrame con dati sintetici realistici
        """
        np.random.seed(seed)
        
        if damage_distribution is None:
            damage_distribution = cls.REALISTIC_CONFIGS['damage_distribution']
        
        # Dataset base
        data = {
            'building_id': range(1, n_samples + 1),
            'age': np.random.randint(1, 100, n_samples),
            'count_floors_pre_eq': np.random.randint(1, 8, n_samples),
            'count_families': np.random.randint(1, 15, n_samples)
        }
        
        # Aggiungi features geografiche
        geo_config = cls.REALISTIC_CONFIGS['geo_levels']
        for geo_level, values in geo_config.items():
            data[geo_level] = np.random.choice(list(values), n_samples)
        
        # Aggiungi percentuali se richieste
        if include_percentages:
            data.update({
                'area_percentage': np.random.uniform(10, 90, n_samples),
                'height_percentage': np.random.uniform(10, 90, n_samples)
            })
        
        # Aggiungi tutte le features categoriche se richieste
        if include_all_features:
            cat_config = cls.REALISTIC_CONFIGS['categorical_features']
            for feature, values in cat_config.items():
                data[feature] = np.random.choice(values, n_samples)
            
            # Aggiungi features di superstruttura (binarie)
            for feature in cls.REALISTIC_CONFIGS['superstructure_features']:
                data[feature] = np.random.choice([0, 1], n_samples)
        
        # Target variable
        data['damage_grade'] = np.random.choice([1, 2, 3], n_samples, p=damage_distribution)
        
        return pd.DataFrame(data)
    
    @classmethod
    def create_minimal_dataset(cls, seed: int = 42) -> pd.DataFrame:
        """
        Crea dataset minimale per test di robustezza e edge cases
        
        Returns:
            DataFrame minimale con 3 sample
        """
        np.random.seed(seed)
        
        return pd.DataFrame({
            'building_id': [1, 2, 3],
            'age': [10, 20, 30],
            'count_families': [1, 2, 3],
            'count_floors_pre_eq': [1, 2, 3],
            'geo_level_1_id': [1, 2, 3],
            'damage_grade': [1, 2, 3]
        })
    
    @classmethod
    def create_train_test_split(
        cls, 
        n_samples: int = 1000, 
        test_size: float = 0.25,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Crea dataset e lo splitta in train/test
        
        Returns:
            Tuple (train_df, test_df)
        """
        full_dataset = cls.create_building_dataset(n_samples=n_samples, seed=seed)
        
        # Split stratificato per damage_grade
        train_size = int(n_samples * (1 - test_size))
        
        # Shuffle mantenendo stratificazione approssimativa
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        
        train_df = full_dataset.iloc[indices[:train_size]].copy().reset_index(drop=True)
        test_df = full_dataset.iloc[indices[train_size:]].copy().reset_index(drop=True)
        
        return train_df, test_df
    
    @classmethod
    def create_csv_files(
        cls,
        temp_dir: str,
        n_samples: int = 100,
        seed: int = 42
    ) -> Tuple[str, str]:
        """
        Crea file CSV temporanei per test di I/O
        
        Returns:
            Tuple (train_values_path, train_labels_path)
        """
        dataset = cls.create_building_dataset(n_samples=n_samples, seed=seed)
        
        # Separa values e labels
        labels_cols = ['building_id', 'damage_grade']
        values_cols = [col for col in dataset.columns if col not in ['damage_grade']]
        
        train_values = dataset[values_cols]
        train_labels = dataset[labels_cols]
        
        # Percorsi file
        train_values_path = os.path.join(temp_dir, 'train_values.csv')
        train_labels_path = os.path.join(temp_dir, 'train_labels.csv')
        
        # Salva CSV
        train_values.to_csv(train_values_path, index=False)
        train_labels.to_csv(train_labels_path, index=False)
        
        return train_values_path, train_labels_path
    
    @classmethod
    def create_with_missing_values(
        cls,
        n_samples: int = 100,
        missing_rate: float = 0.1,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Crea dataset con valori mancanti per testare robustezza
        
        Args:
            missing_rate: Percentuale di valori da rendere NaN
            
        Returns:
            DataFrame con valori NaN
        """
        dataset = cls.create_building_dataset(n_samples=n_samples, seed=seed)
        
        np.random.seed(seed)
        
        # Colonne numeriche dove introdurre NaN
        numeric_cols = ['age', 'count_families', 'count_floors_pre_eq']
        
        for col in numeric_cols:
            if col in dataset.columns:
                n_missing = int(len(dataset) * missing_rate)
                missing_indices = np.random.choice(len(dataset), n_missing, replace=False)
                dataset.loc[missing_indices, col] = np.nan
        
        return dataset
    
    @classmethod
    def create_imbalanced_dataset(
        cls,
        n_samples: int = 500,
        imbalance_ratio: Tuple[float, float, float] = (0.05, 0.15, 0.8),
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Crea dataset sbilanciato per testare gestione classi sbilanciate
        
        Args:
            imbalance_ratio: Ratio per damage_grade (1, 2, 3)
            
        Returns:
            DataFrame con classi sbilanciate
        """
        return cls.create_building_dataset(
            n_samples=n_samples,
            seed=seed,
            damage_distribution=list(imbalance_ratio)
        )
    
    @classmethod
    def get_feature_info(cls) -> Dict[str, Any]:
        """
        Restituisce informazioni sui tipi di features
        
        Returns:
            Dizionario con info sui tipi di features
        """
        return {
            'geographic_features': list(cls.REALISTIC_CONFIGS['geo_levels'].keys()),
            'categorical_features': list(cls.REALISTIC_CONFIGS['categorical_features'].keys()),
            'binary_features': cls.REALISTIC_CONFIGS['superstructure_features'],
            'numeric_features': ['age', 'count_families', 'count_floors_pre_eq', 
                               'area_percentage', 'height_percentage'],
            'target_feature': 'damage_grade',
            'id_feature': 'building_id'
        }


class TestDataValidator:
    """Validatore per verificare la qualità dei dati di test generati"""
    
    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Valida la qualità del dataset generato
        
        Returns:
            Dizionario con risultati validazione
        """
        results = {}
        
        # Verifica presenza colonne essenziali
        essential_cols = ['building_id', 'age', 'damage_grade']
        results['has_essential_columns'] = all(col in df.columns for col in essential_cols)
        
        # Verifica unicità building_id (solo se la colonna esiste)
        if 'building_id' in df.columns:
            results['unique_building_ids'] = df['building_id'].nunique() == len(df)
        else:
            results['unique_building_ids'] = False
        
        # Verifica valori damage_grade (solo se la colonna esiste)
        if 'damage_grade' in df.columns:
            damage_values = df['damage_grade'].unique()
            results['valid_damage_grades'] = all(grade in [1, 2, 3] for grade in damage_values)
        else:
            results['valid_damage_grades'] = False
        
        # Verifica assenza di valori completamente nulli
        results['no_all_null_columns'] = not df.isnull().all().any() if len(df) > 0 else True
        
        # Verifica range valori numerici (solo se la colonna esiste)
        if 'age' in df.columns and len(df) > 0:
            results['valid_age_range'] = (df['age'].min() >= 0) and (df['age'].max() <= 100)
        else:
            results['valid_age_range'] = True  # Skip se non esiste
        
        # Verifica consistenza dimensioni
        results['consistent_dimensions'] = len(df) > 0 and len(df.columns) > 0
        
        return results
    
    @staticmethod
    def print_validation_report(df: pd.DataFrame) -> None:
        """Stampa report di validazione"""
        results = TestDataValidator.validate_dataset(df)
        
        print("DATA VALIDATION REPORT")
        print("=" * 40)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Rows: {len(df)}")
        print("\nValidation Results:")
        
        for check, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status} {check}: {passed}")
        
        # Summary
        passed_checks = sum(results.values())
        total_checks = len(results)
        print(f"\nSummary: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print("All validations passed!")
        else:
            print("WARNING: Some validations failed!")


# Funzioni di utilità per backward compatibility
def create_synthetic_data(n_samples=500, seed=42):
    """Wrapper per backward compatibility"""
    return SyntheticDataFactory.create_building_dataset(n_samples, seed)


def create_test_files(temp_dir, n_samples=100):
    """Wrapper per backward compatibility"""
    return SyntheticDataFactory.create_csv_files(temp_dir, n_samples)


if __name__ == "__main__":
    # Test della factory
    print("Testing SyntheticDataFactory...")
    
    # Test dataset base
    df = SyntheticDataFactory.create_building_dataset(n_samples=100)
    TestDataValidator.print_validation_report(df)
    
    print("\n" + "=" * 50)
    
    # Test dataset minimale
    minimal_df = SyntheticDataFactory.create_minimal_dataset()
    print(f"Minimal dataset: {minimal_df.shape}")
    print(minimal_df.head())
    
    print("\n" + "=" * 50)
    
    # Test train/test split
    train_df, test_df = SyntheticDataFactory.create_train_test_split(n_samples=200)
    print(f"Train/Test split: {train_df.shape} / {test_df.shape}")
    
    print("\nSyntheticDataFactory test completed!")
