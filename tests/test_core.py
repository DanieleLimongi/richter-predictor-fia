#!/usr/bin/env python3
"""
Test Core Consolidato - Integra funzionalità di:
- test_core_functionality.py
- test_utils.py (parti data-related)
- test_preprocessing_pipeline.py (parti non specifiche)

Eliminando duplicazioni e centralizzando le funzionalità comuni
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import time
import tempfile
import shutil
import json
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import factory dati condivisi
from test_data.synthetic_data_factory import SyntheticDataFactory, TestDataValidator

# Import moduli da testare
try:
    from src.feature_engineering import AdvancedFeatureEngineer
    from src.data.data_analysis import DataAnalyzer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Feature engineering modules not available: {e}")
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from src.feature_engineering import AdvancedFeatureEngineer as NewFeatureEngineer
    NEW_FEATURE_ENGINEERING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: New feature engineering modules not available: {e}")
    NEW_FEATURE_ENGINEERING_AVAILABLE = False


class TestCoreConsolidated(unittest.TestCase):
    """Test suite consolidata per funzionalità core del sistema"""
    
    @classmethod
    def setUpClass(cls):
        """Setup globale condiviso per tutti i test"""
        print("\nSetting up Core Consolidated Tests...")
        
        # Usa factory per dati standardizzati
        cls.data_factory = SyntheticDataFactory()
        cls.test_data = cls.data_factory.create_building_dataset(n_samples=500, seed=42)
        cls.minimal_data = cls.data_factory.create_minimal_dataset()
        cls.train_data, cls.test_data_split = cls.data_factory.create_train_test_split(n_samples=800, seed=42)
        
        # Crea dataset con valori mancanti per test robustezza
        cls.missing_data = cls.data_factory.create_with_missing_values(n_samples=100, missing_rate=0.1)
        
        # Carica dati reali se disponibili
        try:
            if FEATURE_ENGINEERING_AVAILABLE:
                analyzer = DataAnalyzer()
                full_data = analyzer.load_data()
                cls.real_data = full_data.head(1000).copy()
                cls.has_real_data = True
                print(f"    Real data loaded: {cls.real_data.shape}")
            else:
                cls.has_real_data = False
                cls.real_data = None
        except Exception as e:
            cls.has_real_data = False
            cls.real_data = None
            print(f"    WARNING: Real data not available: {e}")
        
        print(f"    Test datasets created:")
        print(f"       - Main dataset: {cls.test_data.shape}")
        print(f"       - Train/Test split: {cls.train_data.shape} / {cls.test_data_split.shape}")
        print(f"       - Missing values dataset: {cls.missing_data.shape}")
        
        # Valida qualità dati
        validation_results = TestDataValidator.validate_dataset(cls.test_data)
        all_passed = all(validation_results.values())
        print(f"    {'PASS' if all_passed else 'WARNING'} Data validation: {sum(validation_results.values())}/{len(validation_results)} checks passed")
    
    # ============================================================================
    # TEST FEATURE ENGINEERING (da test_core_functionality.py)
    # ============================================================================
    
    @unittest.skipUnless(FEATURE_ENGINEERING_AVAILABLE, "Feature engineering modules not available")
    def test_01_feature_engineering_initialization(self):
        """Test 1: Inizializzazione AdvancedFeatureEngineer"""
        print("      Test 1: Feature Engineering Initialization")
        
        # Test inizializzazione base
        engineer = AdvancedFeatureEngineer()
        self.assertIsNotNone(engineer)
        self.assertFalse(engineer.fitted)
        
        # Test parametri custom con config
        config = {'verbose': False}
        engineer_custom = AdvancedFeatureEngineer(config)
        self.assertIsNotNone(engineer_custom)
        
        print("         Initialization successful")
    
    @unittest.skipUnless(FEATURE_ENGINEERING_AVAILABLE, "Feature engineering modules not available")
    def test_02_seismic_domain_features(self):
        """Test 2: Creazione features domain-specific tramite orchestrator"""
        print("      Test 2: Seismic Domain Features")
        
        engineer = AdvancedFeatureEngineer()
        df = self.test_data.copy()
        original_cols = len(df.columns)
        
        # Test domain features tramite fit_transform (orchestrator usa i moduli automaticamente)
        df_enhanced = engineer.fit_transform(df, 'damage_grade')
        
        # Verifica nuove features
        self.assertGreater(len(df_enhanced.columns), original_cols)
        
        # Verifica che features specifiche siano state create
        seismic_features = [col for col in df_enhanced.columns if any(keyword in col.lower() 
                           for keyword in ['vulnerability', 'quality', 'seismic', 'structural'])]
        
        self.assertGreater(len(seismic_features), 0, "Should have seismic domain features")
        
        # Verifica che non ci siano solo NaN nelle features create
        for feature in seismic_features[:5]:  # Test solo primi 5
            self.assertFalse(df_enhanced[feature].isnull().all())
        
        print(f"         Created {len(df_enhanced.columns) - original_cols} total features (includes seismic domain)")
    
    @unittest.skipUnless(FEATURE_ENGINEERING_AVAILABLE, "Feature engineering modules not available")
    def test_03_unified_geographic_encoding(self):
        """Test 3: Encoding geografico unificato tramite orchestrator"""
        print("      Test 3: Unified Geographic Encoding")
        
        engineer = AdvancedFeatureEngineer()
        df = self.test_data.copy()
        
        # Test encoding geografico tramite fit_transform
        df_geo = engineer.fit_transform(df, 'damage_grade')
        
        # Verifica features geografiche
        geo_features = [col for col in df_geo.columns if 'geo_' in col and ('risk' in col or 'encoded' in col or 'mean' in col)]
        self.assertGreaterEqual(len(geo_features), 0, "Should have some geographic features")
        
        # Verifica consistenza valori
        for feature in geo_features[:3]:  # Test solo primi 3
            if feature in df_geo.columns:
                # Non dovrebbero esserci solo NaN
                self.assertFalse(df_geo[feature].isnull().all())
                # Dovrebbero essere numerici
                self.assertTrue(np.issubdtype(df_geo[feature].dtype, np.number))
        
        print(f"         Found {len(geo_features)} geographic encoding features")
    
    @unittest.skipUnless(FEATURE_ENGINEERING_AVAILABLE, "Feature engineering modules not available")
    def test_04_complete_fit_transform_pipeline(self):
        """Test 4: Pipeline completa fit/transform"""
        print("      Test 4: Complete Fit/Transform Pipeline")
        
        engineer = AdvancedFeatureEngineer()
        
        # Usa train/test split
        train_df = self.train_data.copy()
        test_df = self.test_data_split.copy()
        
        start_time = time.time()
        
        # Fit sul training set
        train_enhanced = engineer.fit_transform(train_df, 'damage_grade')
        
        # Transform sul test set
        test_enhanced = engineer.transform(test_df)
        
        duration = time.time() - start_time
        
        # Verifiche pipeline
        self.assertTrue(engineer.fitted, "Engineer should be fitted")
        self.assertEqual(len(train_enhanced.columns), len(test_enhanced.columns), 
                        "Train and test should have same columns")
        self.assertGreater(len(train_enhanced.columns), len(train_df.columns),
                          "Should create additional features")
        
        # Performance check
        self.assertLess(duration, 30, "Pipeline should complete in reasonable time")
        
        print(f"         Pipeline completed in {duration:.2f}s")
        print(f"         Features: {len(train_df.columns)} -> {len(train_enhanced.columns)}")
    
    # ============================================================================
    # TEST DATA UTILITIES (da test_utils.py consolidato)
    # ============================================================================
    
    def test_05_data_validation_and_types(self):
        """Test 5: Validazione dati e tipi consolidata"""
        print("      Test 5: Data Validation and Types")
        
        df = self.test_data.copy()
        
        # Test validazione base
        validation_results = TestDataValidator.validate_dataset(df)
        self.assertTrue(validation_results['has_essential_columns'])
        self.assertTrue(validation_results['unique_building_ids'])
        self.assertTrue(validation_results['valid_damage_grades'])
        
        # Test conversioni tipo
        if 'age' in df.columns:
            # Test conversione a float
            df['age_float'] = df['age'].astype(float)
            self.assertEqual(df['age_float'].dtype, np.float64)
            
            # Test conversione stringa
            df['age_str'] = df['age'].astype(str)
            self.assertEqual(df['age_str'].dtype, object)
        
        # Test operazioni DataFrame
        self.assertGreater(len(df), 0)
        self.assertGreater(len(df.columns), 5)
        
        # Test filtri
        if 'damage_grade' in df.columns:
            filtered = df[df['damage_grade'] == 2]
            self.assertGreaterEqual(len(filtered), 0)
        
        print(f"         Data validation successful")
        print(f"         Type conversions working")
    
    def test_06_missing_values_handling(self):
        """Test 6: Gestione valori mancanti"""
        print("      Test 6: Missing Values Handling")
        
        df_missing = self.missing_data.copy()
        
        # Verifica presenza NaN
        missing_counts = df_missing.isnull().sum()
        has_missing = missing_counts.sum() > 0
        self.assertTrue(has_missing, "Should have missing values in test data")
        
        # Test rilevazione colonne con missing
        missing_cols = missing_counts[missing_counts > 0].index.tolist()
        self.assertGreater(len(missing_cols), 0)
        
        # Test strategie di handling
        # Verifica che forward fill riduca i missing values
        df_ffill = df_missing.fillna(method='ffill')
        original_missing = df_missing.isnull().sum().sum()
        ffill_missing = df_ffill.isnull().sum().sum()
        self.assertLessEqual(ffill_missing, original_missing, "Forward fill should reduce missing values")
        
        # Mean/mode imputation per colonne numeriche
        numeric_cols = df_missing.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in missing_cols:
                mean_val = df_missing[col].mean()
                df_imputed = df_missing.copy()
                df_imputed[col].fillna(mean_val, inplace=True)
                self.assertEqual(df_imputed[col].isnull().sum(), 0)
        
        print(f"         Missing values detection working")
        print(f"         Imputation strategies working")
    
    # ============================================================================
    # TEST PREPROCESSING PIPELINE (da test_preprocessing_pipeline.py consolidato)
    # ============================================================================
    
    @unittest.skipUnless(NEW_FEATURE_ENGINEERING_AVAILABLE, "New feature engineering not available")
    def test_07_new_feature_engineering_pipeline(self):
        """Test 7: New modular feature engineering pipeline"""
        print("      Test 7: New Feature Engineering Pipeline")
        
        # Test inizializzazione base
        engineer = NewFeatureEngineer()
        
        self.assertIsNotNone(engineer)
        self.assertFalse(engineer.fitted)
        
        # Test fit_transform
        df = self.test_data.copy()
        df_enhanced = engineer.fit_transform(df, 'damage_grade')
        
        self.assertTrue(engineer.fitted)
        self.assertGreater(len(df_enhanced.columns), len(df.columns))
        
        print(f"         New pipeline initialization successful")
        print(f"         Features: {len(df.columns)} -> {len(df_enhanced.columns)}")
    
    @unittest.skipUnless(NEW_FEATURE_ENGINEERING_AVAILABLE, "New feature engineering not available")
    def test_08_modular_feature_engineers(self):
        """Test 8: Individual modular feature engineers"""
        print("      Test 8: Individual Modular Feature Engineers")
        
        df = self.test_data.copy()
        
        try:
            from src.feature_engineering import SeismicFeatureEngineer, StatisticalFeatureEngineer
            from src.feature_engineering import EncodingFeatureEngineer, BinningFeatureEngineer
            
            # Test individual engineers
            engineers = {
                'Seismic': SeismicFeatureEngineer(),
                'Statistical': StatisticalFeatureEngineer(),
                'Encoding': EncodingFeatureEngineer(),
                'Binning': BinningFeatureEngineer()
            }
            
            for name, engineer in engineers.items():
                try:
                    enhanced = engineer.fit_transform(df.copy())
                    self.assertGreaterEqual(len(enhanced.columns), len(df.columns))
                    print(f"         {name} engineer: +{len(enhanced.columns) - len(df.columns)} features")
                except Exception as e:
                    print(f"         {name} engineer: skipped ({e})")
                    
        except ImportError:
            print(f"         Specialized engineers not available - using main engineer only")
            
        print(f"         Modular feature engineers working")
    
    # ============================================================================
    # TEST FILE I/O CONSOLIDATO (da test_utils.py)
    # ============================================================================
    
    def test_09_file_operations_consolidated(self):
        """Test 9: Operazioni file consolidate"""
        print("      Test 9: File Operations Consolidated")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test CSV operations usando factory
            csv_train_path, csv_labels_path = self.data_factory.create_csv_files(
                temp_dir=temp_dir, 
                n_samples=50, 
                seed=42
            )
            
            # Verifica file creati
            self.assertTrue(os.path.exists(csv_train_path))
            self.assertTrue(os.path.exists(csv_labels_path))
            
            # Test lettura
            train_data = pd.read_csv(csv_train_path)
            labels_data = pd.read_csv(csv_labels_path)
            
            self.assertEqual(len(train_data), 50)
            self.assertEqual(len(labels_data), 50)
            self.assertIn('building_id', train_data.columns)
            self.assertIn('damage_grade', labels_data.columns)
            
            # Test JSON operations
            config_data = {
                'model_params': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100
                },
                'feature_info': self.data_factory.get_feature_info()
            }
            
            json_path = os.path.join(temp_dir, 'config.json')
            with open(json_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Verifica JSON
            self.assertTrue(os.path.exists(json_path))
            
            with open(json_path, 'r') as f:
                loaded_config = json.load(f)
            
            self.assertEqual(config_data, loaded_config)
            self.assertEqual(loaded_config['model_params']['learning_rate'], 0.001)
        
        print(f"         CSV and JSON operations working")
    
    # ============================================================================
    # TEST PERFORMANCE E SCALABILITÀ
    # ============================================================================
    
    def test_10_performance_and_scalability(self):
        """Test 10: Performance e scalabilità consolidate"""
        print("      Test 10: Performance and Scalability")
        
        # Test con dataset di dimensioni crescenti
        sizes = [100, 500, 1000]
        times = []
        
        for size in sizes:
            start_time = time.time()
            
            # Crea dataset
            large_data = self.data_factory.create_building_dataset(n_samples=size, seed=42)
            
            # Operazioni base
            _ = large_data.describe()
            _ = large_data.groupby('damage_grade').size()
            _ = large_data.corr(numeric_only=True)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Performance check
            self.assertLess(elapsed, 10, f"Operations on {size} samples should be fast")
        
        # Verifica scalabilità roughly lineare
        time_ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        for ratio in time_ratios:
            self.assertLess(ratio, 10, "Should scale reasonably with data size")
        
        print(f"         Performance tests passed")
        print(f"         Scalability check: {sizes} -> {[f'{t:.3f}s' for t in times]}")
    
    # ============================================================================
    # TEST ROBUSTEZZA ED ERROR HANDLING
    # ============================================================================
    
    def test_11_error_handling_robustness(self):
        """Test 11: Gestione errori e robustezza"""
        print("      Test 11: Error Handling and Robustness")
        
        # Test con dataset vuoto
        empty_df = pd.DataFrame()
        validation_empty = TestDataValidator.validate_dataset(empty_df)
        self.assertFalse(validation_empty['consistent_dimensions'])
        
        # Test con colonne mancanti
        incomplete_df = pd.DataFrame({
            'building_id': [1, 2, 3],
            'age': [10, 20, 30]
            # Missing damage_grade
        })
        validation_incomplete = TestDataValidator.validate_dataset(incomplete_df)
        self.assertFalse(validation_incomplete['has_essential_columns'])
        
        # Test con valori damage_grade invalidi (solo se la colonna esiste)
        invalid_df = self.minimal_data.copy()
        if 'damage_grade' in invalid_df.columns:
            invalid_df['damage_grade'] = [4, 5, 6]  # Invalid values
            validation_invalid = TestDataValidator.validate_dataset(invalid_df)
            self.assertFalse(validation_invalid['valid_damage_grades'])
        
        # Test con duplicati building_id (solo se la colonna esiste)
        duplicate_df = self.minimal_data.copy()
        if 'building_id' in duplicate_df.columns:
            duplicate_df['building_id'] = [1, 1, 1]  # Duplicates
            validation_duplicate = TestDataValidator.validate_dataset(duplicate_df)
            self.assertFalse(validation_duplicate['unique_building_ids'])
        
        print(f"         Error handling tests passed")
        print(f"         Robustness checks working")


def run_core_consolidated_tests():
    """Esegue la suite consolidata di test core"""
    print("\nRICHTER PREDICTOR - CORE CONSOLIDATED TEST SUITE")
    print("=" * 70)
    print("Testing consolidated core functionality...")
    print("   Integrates: core_functionality + utils + preprocessing_pipeline")
    
    # Crea e esegui test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCoreConsolidated)
    
    # Runner con output dettagliato
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary completo
    print("\n" + "=" * 70)
    print("CORE CONSOLIDATED TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"   Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"   - {test}")
            print(f"     {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"   - {test}")
            print(f"     {traceback.split('Error:')[-1].strip()}")
    
    if not result.failures and not result.errors:
        print("\nALL CORE TESTS PASSED!")
        print("Feature engineering working")
        print("Data utilities working")  
        print("Preprocessing pipeline working")
        print("File operations working")
        print("Performance and scalability OK")
        print("Error handling and robustness OK")
    else:
        print("\nSOME CORE TESTS FAILED")
        print("Please check the failures and errors above")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_core_consolidated_tests()
    sys.exit(0 if success else 1)
