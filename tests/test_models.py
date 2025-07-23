#!/usr/bin/env python3
"""
Test suite per i modelli di training del Richter Predictor
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import json

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow non disponibile, alcuni test saranno skippati")


class TestModelArchitecture(unittest.TestCase):
    """Test per l'architettura dei modelli"""
    
    def setUp(self):
        """Setup per i test dei modelli"""
        if TF_AVAILABLE:
            tf.random.set_seed(42)
            np.random.seed(42)
    
    @unittest.skipUnless(TF_AVAILABLE, "TensorFlow non disponibile")
    def test_model_creation(self):
        """Test creazione modello base"""
        # Parametri di test
        input_dim = 82
        num_classes = 3
        
        # Crea modello semplice per test
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compila il modello
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Verifica struttura
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, num_classes))
    
    @unittest.skipUnless(TF_AVAILABLE, "TensorFlow non disponibile")
    def test_model_prediction(self):
        """Test predizione del modello"""
        # Crea modello di test
        input_dim = 82
        num_classes = 3
        batch_size = 10
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Dati di test
        X_test = np.random.random((batch_size, input_dim)).astype(np.float32)
        
        # Predizione
        predictions = model.predict(X_test, verbose=0)
        
        # Verifica output
        self.assertEqual(predictions.shape, (batch_size, num_classes))
        self.assertTrue(np.allclose(predictions.sum(axis=1), 1.0, atol=1e-6))  # Softmax sum = 1
        self.assertTrue(np.all(predictions >= 0))  # Probabilità non negative
    
    @unittest.skipUnless(TF_AVAILABLE, "TensorFlow non disponibile")
    def test_model_training_step(self):
        """Test singolo step di training"""
        # Parametri
        input_dim = 82
        num_classes = 3
        batch_size = 32
        
        # Crea modello
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Dati sintetici
        X_train = np.random.random((batch_size, input_dim)).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(
            np.random.randint(0, num_classes, batch_size), 
            num_classes
        )
        
        # Training di un epoch
        history = model.fit(X_train, y_train, epochs=1, verbose=0)
        
        # Verifica che il training sia andato a buon fine
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)
        self.assertEqual(len(history.history['loss']), 1)


class TestDataLoading(unittest.TestCase):
    """Test per il caricamento e validazione dei dati"""
    
    def setUp(self):
        """Setup dati di test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crea file CSV di test
        self.train_values_path = os.path.join(self.temp_dir, 'train_values.csv')
        self.train_labels_path = os.path.join(self.temp_dir, 'train_labels.csv')
        
        # Dati di test
        np.random.seed(42)
        n_samples = 100
        
        # Train values
        train_values = pd.DataFrame({
            'building_id': range(1, n_samples + 1),
            'geo_level_1_id': np.random.randint(1, 32, n_samples),
            'geo_level_2_id': np.random.randint(1, 100, n_samples),
            'geo_level_3_id': np.random.randint(1, 200, n_samples),
            'count_families': np.random.randint(1, 10, n_samples),
            'count_floors_pre_eq': np.random.randint(1, 5, n_samples),
            'age': np.random.randint(0, 100, n_samples),
            'foundation_type': np.random.choice(['r', 'w', 'i', 'u', 'h'], n_samples),
            'roof_type': np.random.choice(['n', 'q', 'x'], n_samples),
            'ground_floor_type': np.random.choice(['f', 'm', 'v', 'x', 'z'], n_samples),
            'other_floor_type': np.random.choice(['j', 'q', 's', 'x'], n_samples),
            'position': np.random.choice(['j', 'o', 's', 't'], n_samples),
            'plan_configuration': np.random.choice(['a', 'c', 'd', 'f', 'm'], n_samples),
            'land_surface_condition': np.random.choice(['n', 'o', 't'], n_samples),
            'legal_ownership_status': np.random.choice(['a', 'r', 'v', 'w'], n_samples),
            'has_superstructure_adobe_mud': np.random.choice([0, 1], n_samples),
            'has_superstructure_mud_mortar_stone': np.random.choice([0, 1], n_samples),
            'has_superstructure_stone_flag': np.random.choice([0, 1], n_samples),
            'has_superstructure_cement_mortar_stone': np.random.choice([0, 1], n_samples),
            'has_superstructure_mud_mortar_brick': np.random.choice([0, 1], n_samples),
            'has_superstructure_cement_mortar_brick': np.random.choice([0, 1], n_samples),
            'has_superstructure_timber': np.random.choice([0, 1], n_samples),
            'has_superstructure_bamboo': np.random.choice([0, 1], n_samples),
            'has_superstructure_rc_non_engineered': np.random.choice([0, 1], n_samples),
            'has_superstructure_rc_engineered': np.random.choice([0, 1], n_samples),
            'has_superstructure_other': np.random.choice([0, 1], n_samples)
        })
        
        # Train labels
        train_labels = pd.DataFrame({
            'building_id': range(1, n_samples + 1),
            'damage_grade': np.random.choice([1, 2, 3], n_samples)
        })
        
        # Salva i file
        train_values.to_csv(self.train_values_path, index=False)
        train_labels.to_csv(self.train_labels_path, index=False)
    
    def tearDown(self):
        """Cleanup file temporanei"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_csv_files(self):
        """Test caricamento file CSV"""
        # Carica train values
        train_values = pd.read_csv(self.train_values_path)
        self.assertEqual(len(train_values), 100)
        self.assertIn('building_id', train_values.columns)
        self.assertIn('geo_level_1_id', train_values.columns)
        
        # Carica train labels
        train_labels = pd.read_csv(self.train_labels_path)
        self.assertEqual(len(train_labels), 100)
        self.assertIn('building_id', train_labels.columns)
        self.assertIn('damage_grade', train_labels.columns)
    
    def test_data_merge(self):
        """Test merge dei dati"""
        train_values = pd.read_csv(self.train_values_path)
        train_labels = pd.read_csv(self.train_labels_path)
        
        # Merge
        merged_data = train_values.merge(train_labels, on='building_id', how='inner')
        
        self.assertEqual(len(merged_data), 100)
        self.assertIn('damage_grade', merged_data.columns)
        self.assertEqual(len(merged_data['building_id'].unique()), 100)
    
    def test_target_values_validation(self):
        """Test validazione valori target"""
        train_labels = pd.read_csv(self.train_labels_path)
        
        # Verifica che damage_grade sia in [1, 2, 3]
        damage_grades = train_labels['damage_grade'].unique()
        for grade in damage_grades:
            self.assertIn(grade, [1, 2, 3])
    
    def test_missing_values_detection(self):
        """Test rilevazione valori mancanti"""
        train_values = pd.read_csv(self.train_values_path)
        
        # Introduce alcuni NaN per test
        train_values_with_nan = train_values.copy()
        train_values_with_nan.loc[0, 'age'] = np.nan
        train_values_with_nan.loc[1, 'count_families'] = np.nan
        
        # Controlla rilevazione NaN
        missing_values = train_values_with_nan.isnull().sum()
        self.assertEqual(missing_values['age'], 1)
        self.assertEqual(missing_values['count_families'], 1)
        self.assertEqual(missing_values['geo_level_1_id'], 0)


class TestTrainingValidation(unittest.TestCase):
    """Test per validazione del processo di training"""
    
    @unittest.skipUnless(TF_AVAILABLE, "TensorFlow non disponibile")
    def test_cross_validation_splits(self):
        """Test split per cross-validation"""
        from sklearn.model_selection import StratifiedKFold
        
        # Dati di test
        n_samples = 1000
        X = np.random.random((n_samples, 50))
        y = np.random.choice([0, 1, 2], n_samples)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        splits = list(cv.split(X, y))
        
        # Verifica numero di split
        self.assertEqual(len(splits), 5)
        
        # Verifica dimensioni split
        for train_idx, val_idx in splits:
            self.assertGreater(len(train_idx), len(val_idx))
            self.assertEqual(len(train_idx) + len(val_idx), n_samples)
            
            # Verifica che non ci siano sovrapposizioni
            self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
    
    def test_accuracy_calculation(self):
        """Test calcolo accuracy"""
        from sklearn.metrics import accuracy_score
        
        # Predizioni e target di test
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2])
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Verifica accuracy
        expected_accuracy = 7/9  # 7 predizioni corrette su 9 (0,1,2,3,4,6,8)
        self.assertAlmostEqual(accuracy, expected_accuracy, places=4)
    
    def test_f1_score_calculation(self):
        """Test calcolo F1-score"""
        from sklearn.metrics import f1_score
        
        # Predizioni e target di test
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2])
        
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Verifica che gli score siano ragionevoli
        self.assertGreaterEqual(f1_weighted, 0.0)
        self.assertLessEqual(f1_weighted, 1.0)
        self.assertGreaterEqual(f1_macro, 0.0)
        self.assertLessEqual(f1_macro, 1.0)


class TestModelSaveLoad(unittest.TestCase):
    """Test per salvataggio e caricamento modelli"""
    
    def setUp(self):
        """Setup directory temporanea"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(TF_AVAILABLE, "TensorFlow non disponibile")
    def test_model_save_load(self):
        """Test salvataggio e caricamento modello"""
        # Crea modello
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Salva modello
        model_path = os.path.join(self.temp_dir, 'test_model.keras')
        model.save(model_path)
        
        # Verifica che il file sia stato creato
        self.assertTrue(os.path.exists(model_path))
        
        # Carica modello
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Verifica struttura
        self.assertEqual(len(loaded_model.layers), len(model.layers))
        self.assertEqual(loaded_model.input_shape, model.input_shape)
        self.assertEqual(loaded_model.output_shape, model.output_shape)
    
    def test_json_config_save_load(self):
        """Test salvataggio e caricamento configurazione JSON"""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'model_architecture': {
                'hidden_layers': [128, 64],
                'dropout_rate': 0.3
            }
        }
        
        # Salva configurazione
        config_path = os.path.join(self.temp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Carica configurazione
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Verifica uguaglianza
        self.assertEqual(config, loaded_config)
        self.assertEqual(loaded_config['epochs'], 100)
        self.assertEqual(loaded_config['model_architecture']['hidden_layers'], [128, 64])


def run_model_tests():
    """Esegue tutti i test per i modelli"""
    
    test_suite = unittest.TestSuite()
    
    # Aggiungi test classes
    test_classes = [
        TestModelArchitecture,
        TestDataLoading,
        TestTrainingValidation,
        TestModelSaveLoad
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Esegui test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print(" RICHTER PREDICTOR - TEST SUITE MODELLI")
    print("=" * 60)
    
    # Controlla disponibilità TensorFlow
    if TF_AVAILABLE:
        print("TensorFlow disponibile")
        tf_version = tf.__version__
        print(f" TensorFlow version: {tf_version}")
        
        # Controlla GPU
        if tf.config.list_physical_devices('GPU'):
            print(" GPU disponibile")
        else:
            print(" Usando CPU")
    else:
        print("  TensorFlow non disponibile - alcuni test saranno skippati")
    
    print("")
    
    # Esegui test
    result = run_model_tests()
    
    # Stampa risultati
    print("\n" + "=" * 60)
    print(" RISULTATI TEST MODELLI")
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
