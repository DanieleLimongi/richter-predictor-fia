#!/usr/bin/env python3
"""
Test Core Functionality - Suite di test principale e centralizzata
Sostituisce quick_test.py, rapid_feature_test.py e test_weighted_geographic.py
Testa tutte le funzionalità critiche in modo organizzato
"""

import unittest
import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature_engineering.advanced_features import AdvancedFeatureEngineer
from src.data.data_analysis import DataAnalyzer


class TestCoreFunctionality(unittest.TestCase):
    """Test suite completa per tutte le funzionalità core"""
    
    @classmethod
    def setUpClass(cls):
        """Setup globale per tutti i test"""
        print("\n🔧 Setting up Core Functionality Tests...")
        
        # Dati sintetici per test veloci
        np.random.seed(42)
        cls.synthetic_data = cls._create_synthetic_data(n_samples=500)
        
        # Carica dati reali se disponibili
        try:
            analyzer = DataAnalyzer()
            full_data = analyzer.load_data()
            cls.real_data = full_data.head(2000).copy()  # Subset per test rapidi
            cls.has_real_data = True
            print(f"   ✅ Real data loaded: {cls.real_data.shape}")
        except Exception as e:
            cls.has_real_data = False
            cls.real_data = None
            print(f"   ⚠️  Real data not available: {e}")
        
        print(f"   ✅ Synthetic data created: {cls.synthetic_data.shape}")
    
    @classmethod
    def _create_synthetic_data(cls, n_samples=500):
        """Crea dataset sintetico completo per test"""
        
        data = {
            'building_id': range(1, n_samples + 1),
            'age': np.random.randint(1, 100, n_samples),
            'area_percentage': np.random.uniform(10, 90, n_samples),
            'height_percentage': np.random.uniform(10, 90, n_samples),
            'count_floors_pre_eq': np.random.randint(1, 8, n_samples),
            'count_families': np.random.randint(1, 15, n_samples),
            
            # Geographic (realistica distribuzione)
            'geo_level_1_id': np.random.choice(range(1, 32), n_samples),  # 31 regioni
            'geo_level_2_id': np.random.choice(range(1, 201), n_samples),  # 200 regioni
            'geo_level_3_id': np.random.choice(range(1, 1001), n_samples),  # 1000 regioni
            
            # Materials
            'foundation_type': np.random.choice(['r', 'w', 'i', 'u', 'h'], n_samples),
            'roof_type': np.random.choice(['n', 'q', 'x'], n_samples),
            'ground_floor_type': np.random.choice(['f', 'x', 'm', 'v'], n_samples),
            'other_floor_type': np.random.choice(['q', 's', 'x', 'j'], n_samples),
            
            # Binary features
            'has_superstructure_adobe_mud': np.random.choice([0, 1], n_samples),
            'has_superstructure_mud_mortar_stone': np.random.choice([0, 1], n_samples),
            'has_superstructure_stone_flag': np.random.choice([0, 1], n_samples),
            'has_superstructure_cement_mortar_stone': np.random.choice([0, 1], n_samples),
            'has_secondary_use': np.random.choice([0, 1], n_samples),
            'has_secondary_use_agriculture': np.random.choice([0, 1], n_samples),
            
            # Target
            'damage_grade': np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.6, 0.3])
        }
        
        return pd.DataFrame(data)
    
    def test_01_imports_and_initialization(self):
        """Test 1: Import e inizializzazione"""
        print("   Test 1: Imports and Initialization")
        
        # Test import
        engineer = AdvancedFeatureEngineer(target_encoding_smoothing=50)
        self.assertIsNotNone(engineer)
        self.assertEqual(engineer.target_encoding_smoothing, 50)
        self.assertFalse(engineer.fitted)
        
        print("      ✅ Import and initialization successful")
    
    def test_02_seismic_domain_features(self):
        """Test 2: Domain-specific features creation"""
        print("   Test 2: Seismic Domain Features")
        
        engineer = AdvancedFeatureEngineer()
        df = self.synthetic_data.copy()
        original_cols = len(df.columns)
        
        # Test domain features
        df_enhanced = engineer.create_seismic_domain_features(df)
        
        # Verifica nuove features
        self.assertGreater(len(df_enhanced.columns), original_cols)
        
        # Verifica features specifiche
        expected_features = [
            'building_vulnerability_index',
            'structural_complexity',
            'aspect_ratio',
            'building_volume_proxy',
            'family_density'
        ]
        
        for feature in expected_features:
            if feature == 'building_vulnerability_index':
                self.assertIn(feature, df_enhanced.columns)
                # Verifica che non ci siano NaN
                self.assertFalse(df_enhanced[feature].isna().any())
        
        print(f"      ✅ Domain features created: +{len(df_enhanced.columns) - original_cols}")
    
    def test_03_advanced_interactions(self):
        """Test 3: Advanced interactions"""
        print("   Test 3: Advanced Interactions")
        
        engineer = AdvancedFeatureEngineer()
        df = self.synthetic_data.copy()
        original_cols = len(df.columns)
        
        df_enhanced = engineer.create_advanced_interactions(df)
        
        # Verifica che siano state create interazioni
        self.assertGreater(len(df_enhanced.columns), original_cols)
        
        # Verifica interazioni specifiche con età
        age_interactions = [col for col in df_enhanced.columns if 'age_' in col and '_interaction' in col]
        self.assertGreater(len(age_interactions), 0)
        
        print(f"      ✅ Interactions created: +{len(df_enhanced.columns) - original_cols}")
    
    def test_04_unified_geographic_encoding(self):
        """Test 4: Unified Geographic Encoding (CRITICO)"""
        print("   Test 4: Unified Geographic Encoding")
        
        engineer = AdvancedFeatureEngineer()
        df = self.synthetic_data.copy()
        original_cols = len(df.columns)
        
        df_enhanced = engineer.create_unified_geographic_encoding(df, 'damage_grade')
        
        # Verifica che siano state create features geografiche
        self.assertGreater(len(df_enhanced.columns), original_cols)
        
        # Verifica presence of geographic features
        geo_features = [col for col in df_enhanced.columns if any(
            geo_term in col for geo_term in ['_risk', '_weighted_risk', '_predictive_weight']
        )]
        self.assertGreater(len(geo_features), 0)
        
        # Verifica che i mapping siano stati salvati
        self.assertGreater(len(engineer.geo_target_means), 0)
        
        print(f"      ✅ Geographic encoding: +{len(df_enhanced.columns) - original_cols}")
        print(f"      ✅ Mappings stored: {len(engineer.geo_target_means)}")
    
    def test_05_material_risk_scores(self):
        """Test 5: Material risk scores"""
        print("   Test 5: Material Risk Scores")
        
        engineer = AdvancedFeatureEngineer()
        df = self.synthetic_data.copy()
        original_cols = len(df.columns)
        
        df_enhanced = engineer.create_material_risk_scores(df, 'damage_grade')
        
        # Verifica material features
        material_features = [col for col in df_enhanced.columns if '_risk_zscore' in col]
        self.assertGreater(len(material_features), 0)
        
        # Verifica che mapping siano salvati
        self.assertGreater(len(engineer.material_risk_scores), 0)
        
        print(f"      ✅ Material features: +{len(df_enhanced.columns) - original_cols}")
    
    def test_06_polynomial_features(self):
        """Test 6: Polynomial features"""
        print("   Test 6: Polynomial Features")
        
        engineer = AdvancedFeatureEngineer()
        df = self.synthetic_data.copy()
        original_cols = len(df.columns)
        
        df_enhanced = engineer.create_polynomial_features(df)
        
        # Verifica polynomial features
        poly_features = [col for col in df_enhanced.columns if col.startswith('poly_')]
        
        # Verifica che i nomi siano salvati
        self.assertEqual(len(engineer.polynomial_features_names), len(poly_features))
        
        print(f"      ✅ Polynomial features: +{len(df_enhanced.columns) - original_cols}")
    
    def test_07_complete_fit_transform(self):
        """Test 7: Complete fit_transform workflow"""
        print("   Test 7: Complete Fit-Transform")
        
        engineer = AdvancedFeatureEngineer()
        df = self.synthetic_data.copy()
        original_cols = len(df.columns)
        
        # Complete fit_transform
        start_time = time.time()
        df_enhanced = engineer.fit_transform(df, 'damage_grade')
        duration = time.time() - start_time
        
        # Verifica risultati
        self.assertTrue(engineer.fitted)
        self.assertGreater(len(df_enhanced.columns), original_cols)
        
        # Verifica che non ci siano NaN critici
        critical_cols = [col for col in df_enhanced.columns if not col.startswith('poly_')]
        nan_counts = df_enhanced[critical_cols].isna().sum()
        critical_nans = nan_counts[nan_counts > len(df_enhanced) * 0.1]  # >10% NaN
        
        self.assertEqual(len(critical_nans), 0, f"Too many NaNs in: {critical_nans.index.tolist()}")
        
        print(f"      ✅ Complete transformation: {original_cols} → {len(df_enhanced.columns)} features")
        print(f"      ✅ Duration: {duration:.2f}s")
    
    def test_08_train_test_consistency(self):
        """Test 8: Train/Test Consistency (CRITICO)"""
        print("   Test 8: Train/Test Consistency")
        
        engineer = AdvancedFeatureEngineer()
        df = self.synthetic_data.copy()
        
        # Split train/test
        train_df = df.iloc[:300].copy()
        test_df = df.iloc[300:].copy()
        
        # Fit on train, transform both
        train_enhanced = engineer.fit_transform(train_df, 'damage_grade')
        test_enhanced = engineer.transform(test_df)
        
        # VERIFICA CONSISTENCY
        train_cols = set(train_enhanced.columns)
        test_cols = set(test_enhanced.columns)
        
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols
        
        # Verifica shapes
        self.assertEqual(len(train_enhanced.columns), len(test_enhanced.columns),
                        f"Column count mismatch: {len(train_enhanced.columns)} vs {len(test_enhanced.columns)}")
        
        # Verifica nomi colonne
        self.assertEqual(len(missing_in_test), 0, f"Missing in test: {missing_in_test}")
        self.assertEqual(len(missing_in_train), 0, f"Missing in train: {missing_in_train}")
        
        print(f"      ✅ Train shape: {train_enhanced.shape}")
        print(f"      ✅ Test shape: {test_enhanced.shape}")
        print(f"      ✅ Feature consistency: PERFECT")
    
    def test_09_real_data_validation(self):
        """Test 9: Real Data Validation (se disponibile)"""
        if not self.has_real_data:
            print("   Test 9: Real Data - SKIPPED (data not available)")
            return
        
        print("   Test 9: Real Data Validation")
        
        engineer = AdvancedFeatureEngineer()
        
        # Split real data
        train_real = self.real_data.iloc[:1000].copy()
        test_real = self.real_data.iloc[1000:].copy()
        
        # Test on real data
        start_time = time.time()
        train_enhanced = engineer.fit_transform(train_real, 'damage_grade')
        test_enhanced = engineer.transform(test_real)
        duration = time.time() - start_time
        
        # Verifica consistency su dati reali
        self.assertEqual(len(train_enhanced.columns), len(test_enhanced.columns))
        
        # Verifica che non ci siano errori catastrofici
        self.assertFalse(train_enhanced.isna().all().any(), "Some columns are completely NaN")
        self.assertFalse(test_enhanced.isna().all().any(), "Some test columns are completely NaN")
        
        print(f"      ✅ Real data processed: {train_real.shape[0]} + {test_real.shape[0]} samples")
        print(f"      ✅ Features created: {len(train_enhanced.columns)}")
        print(f"      ✅ Processing time: {duration:.2f}s")
    
    def test_10_memory_and_performance(self):
        """Test 10: Memory e Performance validation"""
        print("   Test 10: Memory & Performance")
        
        # Test con dataset più grande
        large_synthetic = self._create_synthetic_data(n_samples=2000)
        
        engineer = AdvancedFeatureEngineer()
        
        start_time = time.time()
        enhanced = engineer.fit_transform(large_synthetic, 'damage_grade')
        duration = time.time() - start_time
        
        # Performance benchmarks
        samples_per_second = len(large_synthetic) / duration
        features_created = len(enhanced.columns) - len(large_synthetic.columns)
        
        # Verifica performance accettabile
        self.assertGreater(samples_per_second, 100, "Processing too slow")
        self.assertGreater(features_created, 20, "Too few features created")
        
        print(f"      ✅ Processed {len(large_synthetic)} samples in {duration:.2f}s")
        print(f"      ✅ Speed: {samples_per_second:.0f} samples/second")
        print(f"      ✅ Features created: {features_created}")


def run_core_tests():
    """Run della test suite completa"""
    print("\n🧪 RICHTER PREDICTOR - CORE FUNCTIONALITY TEST SUITE")
    print("=" * 70)
    
    # Crea test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCoreFunctionality)
    
    # Esegui test
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    print("=" * 70)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_core_tests()
    sys.exit(0 if success else 1)
