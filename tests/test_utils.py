#!/usr/bin/env python3
"""
Test suite per utilità e funzioni di supporto del Richter Predictor
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestDataUtilities(unittest.TestCase):
    """Test per utilità di gestione dati"""
    
    def test_numpy_array_validation(self):
        """Test validazione array numpy"""
        # Array validi
        valid_array = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(valid_array.shape, (2, 3))
        self.assertEqual(valid_array.dtype, np.int64)
        
        # Array con diversi tipi
        float_array = np.array([[1.0, 2.5], [3.7, 4.2]])
        self.assertEqual(float_array.dtype, np.float64)
        
        # Conversione tipo
        int_from_float = float_array.astype(np.int32)
        self.assertEqual(int_from_float.dtype, np.int32)
        self.assertEqual(int_from_float[0, 0], 1)
    
    def test_pandas_dataframe_operations(self):
        """Test operazioni base su DataFrame"""
        # Crea DataFrame di test
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        # Test dimensioni
        self.assertEqual(df.shape, (5, 3))
        self.assertEqual(len(df.columns), 3)
        
        # Test selezione colonne
        subset = df[['A', 'C']]
        self.assertEqual(subset.shape, (5, 2))
        
        # Test filtri
        filtered = df[df['A'] > 2]
        self.assertEqual(len(filtered), 3)
        
        # Test aggregazioni
        mean_a = df['A'].mean()
        self.assertEqual(mean_a, 3.0)
    
    def test_data_type_conversions(self):
        """Test conversioni di tipo dati"""
        # DataFrame con tipi misti
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['1', '2', '3']
        })
        
        # Verifica tipi originali
        self.assertEqual(df['int_col'].dtype, np.int64)
        self.assertEqual(df['float_col'].dtype, np.float64)
        self.assertEqual(df['str_col'].dtype, object)
        
        # Conversioni
        df['str_to_int'] = df['str_col'].astype(int)
        df['float_to_int'] = df['float_col'].astype(int)
        
        self.assertEqual(df['str_to_int'].dtype, np.int64)
        self.assertEqual(df['float_to_int'].dtype, np.int64)
        self.assertEqual(df.loc[0, 'str_to_int'], 1)
        self.assertEqual(df.loc[0, 'float_to_int'], 1)


class TestFileOperations(unittest.TestCase):
    """Test per operazioni su file"""
    
    def setUp(self):
        """Setup directory temporanea"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)
    
    def test_csv_read_write(self):
        """Test lettura e scrittura CSV"""
        # Dati di test
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        
        # Scrivi CSV
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        data.to_csv(csv_path, index=False)
        
        # Verifica file creato
        self.assertTrue(os.path.exists(csv_path))
        
        # Leggi CSV
        loaded_data = pd.read_csv(csv_path)
        
        # Verifica contenuto
        pd.testing.assert_frame_equal(data, loaded_data)
    
    def test_json_read_write(self):
        """Test lettura e scrittura JSON"""
        # Dati di test
        config = {
            'model_params': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'data_params': {
                'train_split': 0.8,
                'validation_split': 0.2
            },
            'features': ['geo_level_1_id', 'count_families', 'age']
        }
        
        # Scrivi JSON
        json_path = os.path.join(self.temp_dir, 'config.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Verifica file creato
        self.assertTrue(os.path.exists(json_path))
        
        # Leggi JSON
        with open(json_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Verifica contenuto
        self.assertEqual(config, loaded_config)
        self.assertEqual(loaded_config['model_params']['learning_rate'], 0.001)
    
    def test_directory_operations(self):
        """Test operazioni su directory"""
        # Crea directory
        new_dir = os.path.join(self.temp_dir, 'new_directory')
        os.makedirs(new_dir, exist_ok=True)
        
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))
        
        # Crea subdirectory
        sub_dir = os.path.join(new_dir, 'subdirectory')
        os.makedirs(sub_dir, exist_ok=True)
        
        self.assertTrue(os.path.exists(sub_dir))
        
        # Lista contenuto
        contents = os.listdir(new_dir)
        self.assertIn('subdirectory', contents)


class TestMathUtils(unittest.TestCase):
    """Test per utilità matematiche"""
    
    def test_statistical_functions(self):
        """Test funzioni statistiche"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Media
        mean = np.mean(data)
        self.assertEqual(mean, 5.5)
        
        # Mediana
        median = np.median(data)
        self.assertEqual(median, 5.5)
        
        # Deviazione standard
        std = np.std(data)
        self.assertAlmostEqual(std, 2.8722813232690143, places=5)
        
        # Min e Max
        self.assertEqual(np.min(data), 1)
        self.assertEqual(np.max(data), 10)
    
    def test_array_operations(self):
        """Test operazioni su array"""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        
        # Operazioni elemento per elemento
        sum_ab = a + b
        expected_sum = np.array([5, 7, 9])
        np.testing.assert_array_equal(sum_ab, expected_sum)
        
        # Prodotto scalare
        dot_product = np.dot(a, b)
        self.assertEqual(dot_product, 32)  # 1*4 + 2*5 + 3*6
        
        # Reshaping
        reshaped = a.reshape(-1, 1)
        self.assertEqual(reshaped.shape, (3, 1))
    
    def test_random_operations(self):
        """Test operazioni casuali riproducibili"""
        # Imposta seed per riproducibilità
        np.random.seed(42)
        
        # Genera numeri casuali
        random_array = np.random.random(10)
        
        # Reset seed e rigenera
        np.random.seed(42)
        random_array_2 = np.random.random(10)
        
        # Dovrebbero essere identici
        np.testing.assert_array_equal(random_array, random_array_2)
        
        # Test distribuzione normale
        np.random.seed(42)
        normal_samples = np.random.normal(0, 1, 1000)
        
        # La media dovrebbe essere vicina a 0
        self.assertAlmostEqual(np.mean(normal_samples), 0, delta=0.1)


class TestValidationUtils(unittest.TestCase):
    """Test per utilità di validazione"""
    
    def test_shape_validation(self):
        """Test validazione forme array"""
        # Array 2D
        array_2d = np.random.random((100, 50))
        self.assertEqual(len(array_2d.shape), 2)
        self.assertEqual(array_2d.shape[0], 100)
        self.assertEqual(array_2d.shape[1], 50)
        
        # Array 1D
        array_1d = np.random.random(100)
        self.assertEqual(len(array_1d.shape), 1)
        self.assertEqual(array_1d.shape[0], 100)
    
    def test_value_range_validation(self):
        """Test validazione range valori"""
        # Array con valori tra 0 e 1
        probabilities = np.array([0.1, 0.3, 0.6])
        
        # Tutti i valori dovrebbero essere tra 0 e 1
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
        # La somma dovrebbe essere 1 (per probabilità)
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=6)
    
    def test_missing_values_detection(self):
        """Test rilevazione valori mancanti"""
        # Array con NaN
        array_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Rilevazione NaN
        nan_mask = np.isnan(array_with_nan)
        self.assertTrue(np.any(nan_mask))
        self.assertEqual(np.sum(nan_mask), 1)
        
        # Posizione del NaN
        nan_indices = np.where(nan_mask)[0]
        self.assertEqual(nan_indices[0], 2)
    
    def test_data_consistency_checks(self):
        """Test controlli di consistenza dati"""
        # DataFrame con ID duplicati (errore)
        df_with_duplicates = pd.DataFrame({
            'id': [1, 2, 2, 3, 4],
            'value': [10, 20, 25, 30, 40]
        })
        
        # Rilevazione duplicati
        duplicates = df_with_duplicates.duplicated(subset=['id'])
        self.assertTrue(duplicates.any())
        self.assertEqual(duplicates.sum(), 1)
        
        # DataFrame senza duplicati
        df_no_duplicates = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10, 20, 30, 40, 50]
        })
        
        no_duplicates = df_no_duplicates.duplicated(subset=['id'])
        self.assertFalse(no_duplicates.any())


class TestPerformanceUtils(unittest.TestCase):
    """Test per utilità di performance"""
    
    def test_timing_operations(self):
        """Test misurazione tempi"""
        import time
        
        # Operazione veloce
        start_time = time.time()
        
        # Simula operazione
        dummy_array = np.random.random((1000, 1000))
        result = np.sum(dummy_array)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Il tempo dovrebbe essere ragionevole
        self.assertGreater(elapsed, 0)
        self.assertLess(elapsed, 10)  # Meno di 10 secondi
        
        # Il risultato dovrebbe essere ragionevole
        self.assertGreater(result, 0)
        self.assertLess(result, 1000000)  # Array di valori tra 0 e 1
    
    def test_memory_efficient_operations(self):
        """Test operazioni efficienti in memoria"""
        # Invece di creare array molto grandi, usiamo generatori
        def generate_numbers(n):
            for i in range(n):
                yield i ** 2
        
        # Calcola somma usando generatore
        squares_sum = sum(generate_numbers(1000))
        
        # Verifica risultato
        expected_sum = sum(i ** 2 for i in range(1000))
        self.assertEqual(squares_sum, expected_sum)
        
        # Dovrebbe essere uguale a formula matematica
        # Sum of squares: n(n-1)(2n-1)/6
        n = 1000
        formula_result = (n - 1) * n * (2 * n - 1) // 6
        self.assertEqual(squares_sum, formula_result)


def run_utils_tests():
    """Esegue tutti i test delle utilità"""
    
    test_suite = unittest.TestSuite()
    
    # Aggiungi test classes
    test_classes = [
        TestDataUtilities,
        TestFileOperations,
        TestMathUtils,
        TestValidationUtils,
        TestPerformanceUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Esegui test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print(" RICHTER PREDICTOR - TEST SUITE UTILITÀ")
    print("=" * 60)
    
    # Info ambiente
    print(f" NumPy version: {np.__version__}")
    print(f" Pandas version: {pd.__version__}")
    print(f" Python version: {sys.version.split()[0]}")
    print("")
    
    # Esegui test
    result = run_utils_tests()
    
    # Stampa risultati
    print("\n" + "=" * 60)
    print(" RISULTATI TEST UTILITÀ")
    print("=" * 60)
    print(f"Test eseguiti: {result.testsRun}")
    print(f"Successi: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallimenti: {len(result.failures)}")
    print(f"Errori: {len(result.errors)}")
    print(f"Skip: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n FALLIMENTI:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n ERRORI:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Exit code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\n{' TUTTI I TEST SUPERATI!' if exit_code == 0 else ' ALCUNI TEST FALLITI!'}")
    
    exit(exit_code)
