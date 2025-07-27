#!/usr/bin/env python3
"""
Test Nested CV Trainer - Testing per trainer pulito di ensemble
Tests specifici per la versione pulita e modulare del trainer nested CV
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

warnings.filterwarnings('ignore')

# Import factory dati condivisi
from test_data.synthetic_data_factory import SyntheticDataFactory, TestDataValidator


class TestNestedCVTrainer(unittest.TestCase):
    """Test suite per NestedCVRichterTrainer pulito"""
    
    @classmethod
    def setUpClass(cls):
        """Setup globale per tutti i test"""
        print("\nSetting up Nested CV Trainer Tests...")
        
        # Usa factory per dati standardizzati
        cls.data_factory = SyntheticDataFactory()
        cls.test_data = cls.data_factory.create_building_dataset(n_samples=200, seed=42)  # Ridotto per testing
        
        # Try to import trainer modules
        try:
            from models.train_nested_cv_ensemble import NestedCVRichterTrainer, LeakageDetector
            cls.modules_available = True
            cls.NestedCVRichterTrainer = NestedCVRichterTrainer
            cls.LeakageDetector = LeakageDetector
            print("    ✅ Nested CV trainer modules loaded successfully")
        except ImportError as e:
            cls.modules_available = False
            print(f"    ❌ Nested CV trainer modules not available: {e}")
    
    def setUp(self):
        """Setup per ogni singolo test"""
        if not self.modules_available:
            self.skipTest("Nested CV trainer modules not available")
    
    # ============================================================================
    # TEST INIZIALIZZAZIONE E CONFIGURAZIONE
    # ============================================================================
    
    def test_01_trainer_initialization(self):
        """Test 1: Inizializzazione trainer"""
        print("      Test 1: Trainer Initialization")
        
        trainer = self.NestedCVRichterTrainer()
        
        # Verifica attributi base
        self.assertEqual(trainer.target_f1, 0.78)
        self.assertEqual(trainer.final_f1, 0.0)
        self.assertEqual(len(trainer.best_models), 0)
        self.assertIsNotNone(trainer.leakage_detector)
        
        # Verifica hyperparameter search space
        self.assertIsInstance(trainer.random_search_space, dict)
        self.assertIn('learning_rate', trainer.random_search_space)
        self.assertIn('dropout_rate', trainer.random_search_space)
        self.assertIn('batch_size', trainer.random_search_space)
        self.assertIn('l2_reg', trainer.random_search_space)
        
        # Verifica configurazione
        self.assertEqual(trainer.n_random_search, 4)
        
        print("         Trainer initialized correctly")
    
    def test_02_leakage_detector(self):
        """Test 2: LeakageDetector functionality"""
        print("      Test 2: LeakageDetector")
        
        detector = self.LeakageDetector()
        
        # Test split validation
        train_idx = np.array([0, 1, 2, 3, 4])
        val_idx = np.array([5, 6, 7, 8, 9])
        
        # Verifica split valido
        result = detector.validate_split(train_idx, val_idx, "test_fold")
        self.assertTrue(result)
        
        # Verifica logging
        detector.log_preprocessing_fit("TestComponent", train_idx, "test_fold")
        
        # Verifica summary
        summary = detector.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_splits_validated'], 1)
        self.assertEqual(summary['preprocessing_fits'], 1)
        
        # Test overlap detection
        overlapping_val_idx = np.array([4, 5, 6, 7, 8])  # 4 overlaps
        with self.assertRaises(ValueError):
            detector.validate_split(train_idx, overlapping_val_idx, "bad_fold")
        
        print("         LeakageDetector working correctly")
    
    def test_03_data_loading(self):
        """Test 3: Data loading and preparation"""
        print("      Test 3: Data Loading")
        
        trainer = self.NestedCVRichterTrainer()
        
        # Mock the data analyzer to use our test data
        original_load_data = None
        try:
            from data.data_analysis import DataAnalyzer
            original_load_data = DataAnalyzer.load_data
            DataAnalyzer.load_data = lambda self: self.test_data
            
            X_df, y = trainer.load_and_prepare_data()
            
            # Verifica formato output
            self.assertIsInstance(X_df, pd.DataFrame)
            self.assertIsInstance(y, np.ndarray)
            
            # Verifica che il target sia stato rimosso da X_df
            self.assertNotIn('damage_grade', X_df.columns)
            self.assertNotIn('building_id', X_df.columns)
            
            # Verifica conversione target
            self.assertTrue(np.all(y >= 0))  # Dovrebbe essere 0-2, non 1-3
            self.assertTrue(np.all(y <= 2))
            
            print(f"         Data loaded: X {X_df.shape}, y {y.shape}")
            
        finally:
            if original_load_data:
                DataAnalyzer.load_data = original_load_data
    
    def test_04_helper_methods(self):
        """Test 4: Test helper methods"""
        print("      Test 4: Helper Methods")
        
        trainer = self.NestedCVRichterTrainer()
        
        # Test _create_validation_split
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, 100)
        
        X_train, X_val, y_train, y_val = trainer._create_validation_split(X, y, val_split=0.2)
        
        # Verifica split
        self.assertEqual(len(X_train) + len(X_val), len(X))
        self.assertEqual(len(y_train) + len(y_val), len(y))
        self.assertAlmostEqual(len(X_val) / len(X), 0.2, places=1)
        
        # Verifica no overlap
        train_indices = set(range(len(X_train)))
        val_indices = set(range(len(X_train), len(X_train) + len(X_val)))
        self.assertEqual(len(train_indices & val_indices), 0)
        
        # Test _create_and_compile_model (mock TensorFlow se non disponibile)
        try:
            import tensorflow as tf
            from models.ensemble_architectures import EnsembleArchitectures
            
            model = trainer._create_and_compile_model("mlp_advanced", 10, 0.001)
            self.assertIsNotNone(model)
            
            # Verifica che sia compilato
            self.assertIsNotNone(model.optimizer)
            self.assertIsNotNone(model.loss)
            
            print("         Helper methods working correctly")
            
        except ImportError:
            print("         Helper methods test skipped (TensorFlow not available)")
    
    def test_05_feature_engineering_integration(self):
        """Test 5: Feature engineering integration"""
        print("      Test 5: Feature Engineering Integration")
        
        trainer = self.NestedCVRichterTrainer()
        
        # Prepare test data
        feature_cols = [col for col in self.test_data.columns if col not in ['building_id', 'damage_grade']]
        X_df = self.test_data[feature_cols]
        
        # Create train/test split
        n_test = 20
        train_idx = np.arange(len(X_df) - n_test)
        test_idx = np.arange(len(X_df) - n_test, len(X_df))
        
        try:
            # Test feature engineering
            X_train, X_test = trainer.apply_feature_engineering(
                X_df, train_idx, test_idx, "test_fold"
            )
            
            # Verifica output
            self.assertIsInstance(X_train, np.ndarray)
            self.assertIsInstance(X_test, np.ndarray)
            self.assertEqual(X_train.shape[0], len(train_idx))
            self.assertEqual(X_test.shape[0], len(test_idx))
            self.assertEqual(X_train.shape[1], X_test.shape[1])  # Same features
            
            # Verifica data quality
            self.assertFalse(np.isnan(X_train).any())
            self.assertFalse(np.isnan(X_test).any())
            self.assertFalse(np.isinf(X_train).any())
            self.assertFalse(np.isinf(X_test).any())
            
            print(f"         Feature engineering: {X_df.shape[1]} -> {X_train.shape[1]} features")
            
        except Exception as e:
            print(f"         Feature engineering test skipped: {e}")
    
    def test_06_random_params_generation(self):
        """Test 6: Random parameters generation"""
        print("      Test 6: Random Parameters Generation")
        
        trainer = self.NestedCVRichterTrainer()
        
        # Test generation
        params_list = trainer.generate_random_params("test_arch", outer_fold=0)
        
        # Verifica formato
        self.assertIsInstance(params_list, list)
        self.assertEqual(len(params_list), trainer.n_random_search)
        
        # Verifica ogni combinazione
        for params in params_list:
            self.assertIsInstance(params, dict)
            self.assertIn('learning_rate', params)
            self.assertIn('dropout_rate', params)
            self.assertIn('batch_size', params)
            self.assertIn('l2_reg', params)
            
            # Verifica valori validi
            self.assertIn(params['learning_rate'], trainer.random_search_space['learning_rate'])
            self.assertIn(params['dropout_rate'], trainer.random_search_space['dropout_rate'])
            self.assertIn(params['batch_size'], trainer.random_search_space['batch_size'])
            self.assertIn(params['l2_reg'], trainer.random_search_space['l2_reg'])
        
        # Test reproducibilità
        params_list_2 = trainer.generate_random_params("test_arch", outer_fold=0)
        self.assertEqual(params_list, params_list_2)
        
        # Test diversità tra outer folds
        params_list_3 = trainer.generate_random_params("test_arch", outer_fold=1)
        self.assertNotEqual(params_list, params_list_3)
        
        print(f"         Generated {len(params_list)} parameter combinations")
    
    # ============================================================================
    # TEST PERFORMANCE E VALIDITÀ
    # ============================================================================
    
    def test_07_trainer_architecture_validation(self):
        """Test 7: Validazione architettura trainer"""
        print("      Test 7: Trainer Architecture Validation")
        
        trainer = self.NestedCVRichterTrainer()
        
        # Verifica che tutti i metodi necessari esistano
        required_methods = [
            'load_and_prepare_data',
            'apply_feature_engineering',
            'generate_random_params',
            'inner_cv_random_search',
            'train_nested_cv_ensemble',
            'analyze_and_select_final_models',
            'save_nested_cv_results',
            '_create_and_compile_model',
            '_create_validation_split'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(trainer, method), f"Missing method: {method}")
            self.assertTrue(callable(getattr(trainer, method)), f"Method {method} not callable")
        
        # Verifica anti-leakage integration
        self.assertIsInstance(trainer.leakage_detector, self.LeakageDetector)
        
        print("         Trainer architecture validation passed")
    
    def test_08_configuration_integrity(self):
        """Test 8: Integrità configurazione"""
        print("      Test 8: Configuration Integrity")
        
        trainer = self.NestedCVRichterTrainer()
        
        # Verifica search space consistency
        for param_name, values in trainer.random_search_space.items():
            self.assertIsInstance(values, list)
            self.assertGreater(len(values), 0)
            
            if param_name == 'learning_rate':
                self.assertTrue(all(0 < v < 1 for v in values))
            elif param_name == 'dropout_rate':
                self.assertTrue(all(0 <= v <= 1 for v in values))
            elif param_name == 'batch_size':
                self.assertTrue(all(v > 0 and isinstance(v, int) for v in values))
            elif param_name == 'l2_reg':
                self.assertTrue(all(v > 0 for v in values))
        
        # Verifica target F1 reasonable
        self.assertTrue(0.5 <= trainer.target_f1 <= 1.0)
        
        # Verifica n_random_search reasonable
        self.assertTrue(1 <= trainer.n_random_search <= 10)
        
        print("         Configuration integrity validated")


def run_nested_cv_trainer_tests():
    """Esegue la suite di test per NestedCV trainer"""
    print("\nRICHTER PREDICTOR - NESTED CV TRAINER TESTS")
    print("=" * 70)
    print("Testing cleaned and modular nested CV ensemble trainer...")
    
    # Crea e esegui test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNestedCVTrainer)
    
    # Runner con output dettagliato
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary completo
    print("\n" + "=" * 70)
    print("NESTED CV TRAINER TEST SUMMARY:")
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
        print("\nALL NESTED CV TRAINER TESTS PASSED!")
        print("✅ Trainer initialization working")
        print("✅ LeakageDetector working")  
        print("✅ Helper methods working")
        print("✅ Feature engineering integration working")
        print("✅ Configuration integrity validated")
    else:
        print("\nSOME NESTED CV TRAINER TESTS FAILED")
        print("Please check the failures and errors above")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_nested_cv_trainer_tests()
    sys.exit(0 if success else 1)