#!/usr/bin/env python3
"""
Test Suite per EnsembleArchitectures
Verifica funzionamento completo della classe per architetture ensemble
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import warnings

# Sopprimi warnings TensorFlow
warnings.filterwarnings('ignore')

# Setup path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.ensemble_architectures import EnsembleArchitectures

class TestEnsembleArchitectures(unittest.TestCase):
    """Test completo per la classe EnsembleArchitectures"""
    
    def setUp(self):
        """Setup per ogni test"""
        self.input_dim = 100  # Dimensione input tipica dopo feature engineering
        self.n_classes = 3    # 3 livelli di damage
        self.ensemble = EnsembleArchitectures(self.input_dim, self.n_classes)
        
        # Dati di test
        np.random.seed(42)
        self.X_test = np.random.randn(50, self.input_dim).astype(np.float32)
        self.y_test = np.random.randint(0, 3, 50)
    
    def test_01_initialization(self):
        """Test 1: Inizializzazione classe"""
        print("      Test 1: Initialization")
        
        # Verifica parametri base
        self.assertEqual(self.ensemble.input_dim, self.input_dim)
        self.assertEqual(self.ensemble.n_classes, self.n_classes)
        
        # Verifica registry architetture
        archs = self.ensemble.get_available_architectures()
        expected_archs = ['deep_narrow', 'wide_shallow', 'residual_like', 
                         'regularized', 'swish_activation', 'attention_like']
        
        self.assertEqual(len(archs), 6)
        for arch in expected_archs:
            self.assertIn(arch, archs)
        
        print(f"         ✅ Input: {self.input_dim} → Output: {self.n_classes}")
        print(f"         ✅ Architectures: {len(archs)}")
    
    def test_02_individual_architectures(self):
        """Test 2: Creazione singole architetture"""
        print("      Test 2: Individual Architectures")
        
        results = {}
        
        for arch_name in self.ensemble.get_available_architectures():
            # Crea architettura
            model = self.ensemble.create_architecture(arch_name)
            
            # Test struttura
            self.assertIsNotNone(model)
            self.assertEqual(model.input_shape[1:], (self.input_dim,))
            self.assertEqual(model.output_shape[1:], (self.n_classes,))
            
            # Test forward pass
            predictions = model(self.X_test[:10])
            self.assertEqual(predictions.shape, (10, self.n_classes))
            
            # Test probabilità (somma = 1)
            prob_sums = np.sum(predictions.numpy(), axis=1)
            np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)
            
            # Raccogli statistiche
            results[arch_name] = {
                'layers': len(model.layers),
                'params': model.count_params()
            }
        
        print(f"         ✅ All {len(results)} architectures created")
        for name, stats in results.items():
            print(f"           - {name}: {stats['layers']} layers, {stats['params']:,} params")
    
    def test_03_ensemble_creation(self):
        """Test 3: Creazione ensemble completo"""
        print("      Test 3: Ensemble Creation")
        
        # Crea ensemble completo
        models = self.ensemble.create_ensemble_models(n_models=6)
        
        self.assertEqual(len(models), 6)
        
        arch_names = []
        total_params = 0
        
        for arch_name, model in models:
            # Verifica nome architettura
            self.assertIn(arch_name, self.ensemble.get_available_architectures())
            arch_names.append(arch_name)
            
            # Verifica modello funzionante
            self.assertIsNotNone(model)
            self.assertEqual(model.input_shape[1:], (self.input_dim,))
            self.assertEqual(model.output_shape[1:], (self.n_classes,))
            
            # Test prediction
            pred = model(self.X_test[:5])
            self.assertEqual(pred.shape, (5, self.n_classes))
            
            total_params += model.count_params()
        
        # Verifica diversità (tutte architetture diverse)
        self.assertEqual(len(set(arch_names)), 6)
        
        print(f"         ✅ Ensemble: {len(models)} diverse architectures")
        print(f"         ✅ Total parameters: {total_params:,}")
        print(f"         ✅ Architectures: {arch_names}")
    
    def test_04_optimizers_and_losses(self):
        """Test 4: Ottimizzatori e loss functions"""
        print("      Test 4: Optimizers & Loss Functions")
        
        # Test ottimizzatori
        optimizers = EnsembleArchitectures.get_diverse_optimizers()
        self.assertEqual(len(optimizers), 6)
        
        # Verifica diversità ottimizzatori
        optimizer_types = [type(opt).__name__ for opt in optimizers]
        unique_types = set(optimizer_types)
        self.assertGreaterEqual(len(unique_types), 3)
        
        # Test loss functions
        loss_functions = EnsembleArchitectures.get_diverse_loss_functions()
        self.assertEqual(len(loss_functions), 6)
        
        # Conta focal loss (callable) vs standard (string)
        focal_count = sum(1 for loss in loss_functions if callable(loss))
        standard_count = sum(1 for loss in loss_functions if isinstance(loss, str))
        
        self.assertGreaterEqual(focal_count, 2)
        self.assertGreaterEqual(standard_count, 2)
        
        print(f"         ✅ Optimizers: {len(optimizers)} ({len(unique_types)} types)")
        print(f"         ✅ Loss functions: {focal_count} focal + {standard_count} standard")
        print(f"         ✅ Optimizer types: {unique_types}")
    
    def test_05_model_compilation(self):
        """Test 5: Compilazione modelli"""
        print("      Test 5: Model Compilation")
        
        # Prendi primi 3 modelli per test veloce
        models = self.ensemble.create_ensemble_models(n_models=3)
        optimizers = EnsembleArchitectures.get_diverse_optimizers()[:3]
        loss_functions = EnsembleArchitectures.get_diverse_loss_functions()[:3]
        
        compiled_count = 0
        
        for i, (arch_name, model) in enumerate(models):
            try:
                # Compila modello
                model.compile(
                    optimizer=optimizers[i],
                    loss=loss_functions[i],
                    metrics=['accuracy']
                )
                
                # Verifica compilazione
                self.assertIsNotNone(model.optimizer)
                self.assertIsNotNone(model.compiled_loss)
                
                compiled_count += 1
                
            except Exception as e:
                self.fail(f"Compilation failed for {arch_name}: {e}")
        
        print(f"         ✅ Compiled models: {compiled_count}/3")
        print(f"         ✅ All models ready for training")
    
    def test_06_error_handling(self):
        """Test 6: Gestione errori"""
        print("      Test 6: Error Handling")
        
        # Test architettura inesistente
        with self.assertRaises(ValueError):
            self.ensemble.create_architecture('non_existent_architecture')
        
        # Test dimensioni input invalide
        with self.assertRaises(ValueError):
            invalid_ensemble = EnsembleArchitectures(input_dim=-1, n_classes=3)
        
        with self.assertRaises(ValueError):
            invalid_ensemble = EnsembleArchitectures(input_dim=100, n_classes=0)
        
        print(f"         ✅ Invalid architecture: properly rejected")
        print(f"         ✅ Invalid dimensions: properly handled")
    
    def test_07_backward_compatibility(self):
        """Test 7: Compatibilità funzioni legacy"""
        print("      Test 7: Backward Compatibility")
        
        from src.models.ensemble_architectures import (
            create_deep_narrow_architecture,
            create_wide_shallow_architecture,
            get_ensemble_architectures
        )
        
        # Test wrapper functions
        model1 = create_deep_narrow_architecture(self.input_dim)
        model2 = create_wide_shallow_architecture(self.input_dim)
        ensemble_models = get_ensemble_architectures(self.input_dim, n_models=3)
        
        # Verifica funzionamento
        self.assertIsNotNone(model1)
        self.assertIsNotNone(model2)
        self.assertEqual(len(ensemble_models), 3)
        
        # Test predictions
        pred1 = model1(self.X_test[:5])
        pred2 = model2(self.X_test[:5])
        
        self.assertEqual(pred1.shape, (5, 3))
        self.assertEqual(pred2.shape, (5, 3))
        
        print(f"         ✅ Legacy wrapper functions: working")
        print(f"         ✅ Backward compatibility: maintained")
    
    def test_08_ensemble_diversity(self):
        """Test 8: Verifica diversità ensemble"""
        print("      Test 8: Ensemble Diversity")
        
        models = self.ensemble.create_ensemble_models(n_models=6)
        
        # Genera predizioni per confrontare diversità
        predictions = []
        for arch_name, model in models:
            pred = model(self.X_test)
            predictions.append(pred.numpy())
        
        # Calcola correlazione tra predizioni per verificare diversità
        correlations = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                # Correlazione tra prime colonne di output
                corr = np.corrcoef(predictions[i][:, 0], predictions[j][:, 0])[0, 1]
                correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations)
        
        # Diversità buona se correlazione media < 0.8
        self.assertLess(avg_correlation, 0.8, 
                       f"Models too similar: avg correlation {avg_correlation:.3f}")
        
        print(f"         ✅ Model diversity: avg correlation {avg_correlation:.3f}")
        print(f"         ✅ Ensemble diversity: {'HIGH' if avg_correlation < 0.6 else 'MEDIUM'}")


def run_ensemble_tests():
    """Esegui tutti i test per EnsembleArchitectures"""
    print("\n🎯 RICHTER PREDICTOR - ENSEMBLE ARCHITECTURES TEST SUITE")
    print("=" * 70)
    print("🔧 Testing EnsembleArchitectures class...")
    
    # Crea test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnsembleArchitectures)
    
    # Esegui test con output dettagliato
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary dettagliato
    print("=" * 70)
    print("📊 ENSEMBLE ARCHITECTURES TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"   Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}")
            print(f"   {traceback}")
    
    if result.errors:
        print("💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}")
            print(f"   {traceback}")
    
    if not result.failures and not result.errors:
        print("🎉 ALL ENSEMBLE ARCHITECTURE TESTS PASSED!")
        print("🚀 EnsembleArchitectures class is ready for production!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = run_ensemble_tests()
    
    # Exit code per CI/CD
    exit_code = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1
    sys.exit(exit_code)
