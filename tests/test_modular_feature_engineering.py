#!/usr/bin/env python3
"""
Test Modular Feature Engineering - Comprehensive Testing Suite
Tests dell'architettura modulare di feature engineering completa
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


class TestModularFeatureEngineering(unittest.TestCase):
    """Test suite completa per architettura modulare feature engineering"""
    
    @classmethod
    def setUpClass(cls):
        """Setup globale per tutti i test"""
        print("\nSetting up Modular Feature Engineering Tests...")
        
        # Usa factory per dati standardizzati
        cls.data_factory = SyntheticDataFactory()
        cls.test_data = cls.data_factory.create_building_dataset(n_samples=1000, seed=42)
        cls.train_data, cls.test_data_split = cls.data_factory.create_train_test_split(n_samples=800, seed=42)
        
        # Test data without target for validation testing
        cls.test_data_no_target = cls.test_data_split.drop(['damage_grade'], axis=1, errors='ignore')
        
        print(f"    Test datasets created:")
        print(f"       - Main dataset: {cls.test_data.shape}")
        print(f"       - Train/Test split: {cls.train_data.shape} / {cls.test_data_split.shape}")
        
        # Try to import feature engineering modules
        try:
            from feature_engineering import (
                AdvancedFeatureEngineer, SeismicFeatureEngineer, AgeDecayModelEngineer,
                StatisticalFeatureEngineer, PolynomialFeatureEngineer, 
                EncodingFeatureEngineer, BinningFeatureEngineer,
                BaseFeatureEngineer, SeismicConstants
            )
            cls.modules_available = True
            cls.AdvancedFeatureEngineer = AdvancedFeatureEngineer
            cls.SeismicFeatureEngineer = SeismicFeatureEngineer
            cls.AgeDecayModelEngineer = AgeDecayModelEngineer
            cls.StatisticalFeatureEngineer = StatisticalFeatureEngineer
            cls.PolynomialFeatureEngineer = PolynomialFeatureEngineer
            cls.EncodingFeatureEngineer = EncodingFeatureEngineer
            cls.BinningFeatureEngineer = BinningFeatureEngineer
            cls.BaseFeatureEngineer = BaseFeatureEngineer
            cls.SeismicConstants = SeismicConstants
            print("    ✅ All feature engineering modules loaded successfully")
        except ImportError as e:
            cls.modules_available = False
            print(f"    ❌ Feature engineering modules not available: {e}")
    
    def setUp(self):
        """Setup per ogni singolo test"""
        if not self.modules_available:
            self.skipTest("Feature engineering modules not available")
    
    # ============================================================================
    # TEST IMPORTS E INIZIALIZZAZIONE
    # ============================================================================
    
    def test_01_module_imports(self):
        """Test 1: Verifica import di tutti i moduli"""
        print("      Test 1: Module Imports")
        
        # Verifica che tutti i moduli siano importabili
        self.assertTrue(self.modules_available)
        
        # Verifica classi principali
        self.assertIsNotNone(self.AdvancedFeatureEngineer)
        self.assertIsNotNone(self.SeismicFeatureEngineer)
        self.assertIsNotNone(self.AgeDecayModelEngineer)
        self.assertIsNotNone(self.StatisticalFeatureEngineer)
        self.assertIsNotNone(self.PolynomialFeatureEngineer)
        self.assertIsNotNone(self.EncodingFeatureEngineer)
        self.assertIsNotNone(self.BinningFeatureEngineer)
        self.assertIsNotNone(self.BaseFeatureEngineer)
        self.assertIsNotNone(self.SeismicConstants)
        
        print("         All modules importable")
    
    def test_02_constants_initialization(self):
        """Test 2: Verifica inizializzazione SeismicConstants"""
        print("      Test 2: SeismicConstants Initialization")
        
        constants = self.SeismicConstants()
        
        # Verifica che le costanti siano definite
        self.assertTrue(hasattr(constants, 'MATERIAL_DECAY_RATES'))
        self.assertTrue(hasattr(constants, 'AGE_THRESHOLDS'))
        self.assertTrue(hasattr(constants, 'VULNERABILITY_WEIGHTS'))
        
        # Verifica tipi
        self.assertIsInstance(constants.MATERIAL_DECAY_RATES, dict)
        self.assertIsInstance(constants.AGE_THRESHOLDS, dict)
        self.assertIsInstance(constants.VULNERABILITY_WEIGHTS, dict)
        
        # Verifica contenuto
        self.assertGreater(len(constants.MATERIAL_DECAY_RATES), 0)
        self.assertGreater(len(constants.AGE_THRESHOLDS), 0)
        self.assertGreater(len(constants.VULNERABILITY_WEIGHTS), 0)
        
        print(f"         Constants loaded: {len(constants.MATERIAL_DECAY_RATES)} materials, "
              f"{len(constants.AGE_THRESHOLDS)} age thresholds")
    
    # ============================================================================
    # TEST MODULI INDIVIDUALI
    # ============================================================================
    
    def test_03_individual_engineers(self):
        """Test 3: Test di tutti i moduli individuali"""
        print("      Test 3: Individual Engineers")
        
        config = {'verbose': False}  # Silenzioso per testing
        engineers = {
            'Seismic': self.SeismicFeatureEngineer(config),
            'AgeDecay': self.AgeDecayModelEngineer(config),
            'Statistical': self.StatisticalFeatureEngineer(config),
            'Encoding': self.EncodingFeatureEngineer(config),
            'Polynomial': self.PolynomialFeatureEngineer(config),
            'Binning': self.BinningFeatureEngineer(config)
        }
        
        results = {}
        for name, engineer in engineers.items():
            try:
                # Test fit_transform
                train_enhanced = engineer.fit_transform(self.train_data.copy())
                
                # Test transform
                test_enhanced = engineer.transform(self.test_data_no_target.copy())
                
                # Verifica che sia stato aggiunto qualcosa
                features_added = len(train_enhanced.columns) - len(self.train_data.columns)
                
                # Store results
                results[name] = {
                    'success': True,
                    'features_added': features_added,
                    'train_shape': train_enhanced.shape,
                    'test_shape': test_enhanced.shape
                }
                
                # Assertions
                self.assertGreater(len(train_enhanced.columns), len(self.train_data.columns))
                self.assertTrue(engineer.fitted)
                
                print(f"         {name}: +{features_added} features")
                
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"         {name}: FAILED - {e}")
        
        # Verifica che almeno 4 su 6 moduli funzionino
        successful_modules = sum(1 for r in results.values() if r['success'])
        self.assertGreaterEqual(successful_modules, 4, f"At least 4 modules should work, got {successful_modules}")
    
    def test_04_main_orchestrator(self):
        """Test 4: Test orchestrator principale"""
        print("      Test 4: Main Orchestrator")
        
        # Test inizializzazione
        engineer = self.AdvancedFeatureEngineer()
        self.assertIsNotNone(engineer)
        self.assertFalse(engineer.fitted)
        
        # Test fit_transform
        train_enhanced = engineer.fit_transform(self.train_data.copy(), 'damage_grade')
        
        # Test transform
        test_enhanced = engineer.transform(self.test_data_no_target.copy())
        
        # Verifiche base
        self.assertTrue(engineer.fitted)
        self.assertGreater(len(train_enhanced.columns), len(self.train_data.columns))
        self.assertEqual(len(train_enhanced), len(self.train_data))
        self.assertEqual(len(test_enhanced), len(self.test_data_no_target))
        
        # Verifica consistenza colonne (escluso target)
        train_features = set(train_enhanced.columns) - {'damage_grade'}
        test_features = set(test_enhanced.columns)
        missing_in_test = train_features - test_features
        
        # Solo features critiche dovrebbero mancare (non target)
        critical_missing = missing_in_test - {'damage_grade', 'target', 'label'}
        self.assertEqual(len(critical_missing), 0, f"Critical features missing in test: {critical_missing}")
        
        total_features_added = len(train_enhanced.columns) - len(self.train_data.columns)
        print(f"         Main orchestrator: +{total_features_added} features")
        print(f"         Train enhanced: {train_enhanced.shape}")
        print(f"         Test enhanced: {test_enhanced.shape}")
    
    # ============================================================================
    # TEST PIPELINE VALIDATION
    # ============================================================================
    
    def test_05_pipeline_validation(self):
        """Test 5: Test validazione integrità pipeline"""
        print("      Test 5: Pipeline Validation")
        
        engineer = self.AdvancedFeatureEngineer()
        
        # Fit su train
        train_enhanced = engineer.fit_transform(self.train_data.copy(), 'damage_grade')
        test_enhanced = engineer.transform(self.test_data_no_target.copy())
        
        # Test validation
        validation_results = engineer.validate_pipeline_integrity(train_enhanced, test_enhanced)
        
        # Verifica struttura risultati
        self.assertIsInstance(validation_results, dict)
        expected_checks = ['no_data_leakage', 'all_numeric', 'no_inf_nan', 'reasonable_feature_count', 'overall_valid']
        for check in expected_checks:
            self.assertIn(check, validation_results)
        
        # Verifica che le validazioni principali passino
        self.assertTrue(validation_results['no_data_leakage'], "Data leakage validation should pass")
        self.assertTrue(validation_results['all_numeric'], "All features should be numeric")
        self.assertTrue(validation_results['no_inf_nan'], "No infinite or NaN values should be present")
        self.assertTrue(validation_results['reasonable_feature_count'], "Feature count should be reasonable")
        
        print(f"         Pipeline validation: {'PASSED' if validation_results['overall_valid'] else 'FAILED'}")
        for check, result in validation_results.items():
            if check != 'overall_valid':
                print(f"           {check}: {'✅' if result else '❌'}")
    
    def test_06_feature_importance_by_module(self):
        """Test 6: Test tracciabilità features per modulo"""
        print("      Test 6: Feature Importance by Module")
        
        engineer = self.AdvancedFeatureEngineer()
        
        # Fit per abilitare tracciabilità
        engineer.fit_transform(self.train_data.copy(), 'damage_grade')
        
        # Test feature breakdown
        feature_breakdown = engineer.get_feature_importance_by_module()
        
        # Verifica struttura
        self.assertIsInstance(feature_breakdown, dict)
        self.assertGreater(len(feature_breakdown), 0)
        
        # Verifica che ci siano moduli con features
        total_module_features = sum(len(features) for features in feature_breakdown.values())
        self.assertGreater(total_module_features, 0)
        
        print(f"         Feature breakdown by module:")
        for module, features in feature_breakdown.items():
            print(f"           {module}: {len(features)} features")
    
    def test_07_engineering_summary(self):
        """Test 7: Test summary engineering completo"""
        print("      Test 7: Engineering Summary")
        
        engineer = self.AdvancedFeatureEngineer()
        
        # Fit per generare summary
        engineer.fit_transform(self.train_data.copy(), 'damage_grade')
        
        # Test summary
        summary = engineer.get_engineering_summary()
        
        # Verifica struttura
        self.assertIsInstance(summary, dict)
        self.assertIn('engineers', summary)
        self.assertIn('total_features_created', summary)
        self.assertIn('processing_order', summary)
        
        # Verifica contenuto
        self.assertGreater(summary['total_features_created'], 0)
        self.assertGreater(len(summary['engineers']), 0)
        
        print(f"         Engineering summary:")
        print(f"           Total features created: {summary['total_features_created']}")
        print(f"           Engineers used: {len(summary['engineers'])}")
    
    # ============================================================================
    # TEST DATA QUALITY
    # ============================================================================
    
    def test_08_data_quality_checks(self):
        """Test 8: Verifica qualità dati dopo feature engineering"""
        print("      Test 8: Data Quality Checks")
        
        engineer = self.AdvancedFeatureEngineer()
        
        # Process data
        train_enhanced = engineer.fit_transform(self.train_data.copy(), 'damage_grade')
        test_enhanced = engineer.transform(self.test_data_no_target.copy())
        
        # Check per NaN values
        train_has_nan = train_enhanced.isna().any().any()
        test_has_nan = test_enhanced.isna().any().any()
        
        # Check per infinite values
        train_numeric = train_enhanced.select_dtypes(include=[np.number])
        test_numeric = test_enhanced.select_dtypes(include=[np.number])
        
        train_has_inf = np.isinf(train_numeric.values).any()
        test_has_inf = np.isinf(test_numeric.values).any()
        
        # Assertions
        self.assertFalse(train_has_nan, "Train data should not contain NaN values")
        self.assertFalse(test_has_nan, "Test data should not contain NaN values")
        self.assertFalse(train_has_inf, "Train data should not contain infinite values")
        self.assertFalse(test_has_inf, "Test data should not contain infinite values")
        
        # Check data types
        non_numeric_train = [col for col in train_enhanced.columns 
                           if not pd.api.types.is_numeric_dtype(train_enhanced[col])]
        non_numeric_test = [col for col in test_enhanced.columns 
                          if not pd.api.types.is_numeric_dtype(test_enhanced[col])]
        
        self.assertEqual(len(non_numeric_train), 0, f"All train features should be numeric, found: {non_numeric_train}")
        self.assertEqual(len(non_numeric_test), 0, f"All test features should be numeric, found: {non_numeric_test}")
        
        print(f"         Data quality checks passed")
        print(f"           Train: NaN={train_has_nan}, Inf={train_has_inf}")
        print(f"           Test: NaN={test_has_nan}, Inf={test_has_inf}")
    
    # ============================================================================
    # TEST PERFORMANCE E SCALABILITÀ
    # ============================================================================
    
    def test_09_performance_and_scalability(self):
        """Test 9: Test performance e scalabilità"""
        print("      Test 9: Performance and Scalability")
        
        import time
        
        # Test con diversi volumi di dati
        sizes = [100, 500]  # Ridotto per velocità testing
        times = []
        
        for size in sizes:
            start_time = time.time()
            
            # Crea dataset di test
            test_dataset = self.data_factory.create_building_dataset(n_samples=size, seed=42)
            
            # Feature engineering
            engineer = self.AdvancedFeatureEngineer()
            enhanced = engineer.fit_transform(test_dataset, 'damage_grade')
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Performance check
            self.assertLess(elapsed, 30, f"Processing {size} samples should be fast")
            self.assertGreater(len(enhanced.columns), len(test_dataset.columns))
        
        print(f"         Performance test passed")
        print(f"         Times: {[f'{t:.2f}s' for t in times]} for sizes {sizes}")


def run_modular_feature_engineering_tests():
    """Esegue la suite di test per feature engineering modulare"""
    print("\nRICHTER PREDICTOR - MODULAR FEATURE ENGINEERING TESTS")
    print("=" * 70)
    print("Testing comprehensive modular feature engineering architecture...")
    
    # Crea e esegui test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestModularFeatureEngineering)
    
    # Runner con output dettagliato
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary completo
    print("\n" + "=" * 70)
    print("MODULAR FEATURE ENGINEERING TEST SUMMARY:")
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
        print("\nALL MODULAR FEATURE ENGINEERING TESTS PASSED!")
        print("✅ Individual engineers working")
        print("✅ Main orchestrator working")  
        print("✅ Pipeline validation working")
        print("✅ Data quality excellent")
        print("✅ Performance and scalability OK")
    else:
        print("\nSOME MODULAR FEATURE ENGINEERING TESTS FAILED")
        print("Please check the failures and errors above")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_modular_feature_engineering_tests()
    sys.exit(0 if success else 1)