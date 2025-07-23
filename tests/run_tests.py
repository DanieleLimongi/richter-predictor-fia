#!/usr/bin/env python3
"""
Test runner principale per l'intera suite di test del Richter Predictor
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Aggiungi tests al path
tests_dir = Path(__file__).parent
sys.path.append(str(tests_dir))

# Import dei test modules
try:
    from test_preprocessing_pipeline import run_test_suite as run_preprocessing_tests
    from test_models import run_model_tests
    from test_utils import run_utils_tests
    TESTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Impossibile importare alcuni test: {e}")
    TESTS_AVAILABLE = False


class TestRunner:
    """Runner principale per tutti i test"""
    
    def __init__(self):
        self.total_tests = 0
        self.total_failures = 0
        self.total_errors = 0
        self.total_skipped = 0
        self.start_time = None
        self.results = {}
    
    def run_all_tests(self, verbose=True):
        """Esegue tutti i test disponibili"""
        if verbose:
            print(" RICHTER PREDICTOR - SUITE COMPLETA DI TEST")
            print("=" * 70)
            print(f" Data: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f" Directory test: {tests_dir}")
            print("=" * 70)
        
        self.start_time = time.time()
        
        # Test suite da eseguire
        test_suites = [
            ("Preprocessing Pipeline", run_preprocessing_tests),
            ("Modelli ML", run_model_tests),
            ("Utilità", run_utils_tests)
        ]
        
        for suite_name, test_function in test_suites:
            if verbose:
                print(f"\n ESECUZIONE: {suite_name}")
                print("-" * 50)
            
            try:
                result = test_function()
                self.results[suite_name] = result
                
                # Accumula statistiche
                self.total_tests += result.testsRun
                self.total_failures += len(result.failures)
                self.total_errors += len(result.errors)
                self.total_skipped += len(result.skipped) if hasattr(result, 'skipped') else 0
                
                if verbose:
                    self._print_suite_summary(suite_name, result)
                    
            except Exception as e:
                if verbose:
                    print(f" Errore nell'esecuzione di {suite_name}: {e}")
                self.results[suite_name] = None
        
        total_time = time.time() - self.start_time
        
        if verbose:
            self._print_final_summary(total_time)
        
        return self.results
    
    def _print_suite_summary(self, suite_name, result):
        """Stampa riassunto di una singola suite"""
        if result:
            successes = result.testsRun - len(result.failures) - len(result.errors)
            success_rate = (successes / result.testsRun * 100) if result.testsRun > 0 else 0
            
            print(f" {suite_name}:")
            print(f"    Successi: {successes}/{result.testsRun} ({success_rate:.1f}%)")
            if result.failures:
                print(f"    Fallimenti: {len(result.failures)}")
            if result.errors:
                print(f"    Errori: {len(result.errors)}")
            if hasattr(result, 'skipped') and result.skipped:
                print(f"     Skip: {len(result.skipped)}")
        else:
            print(f" {suite_name}: Fallito")
    
    def _print_final_summary(self, total_time):
        """Stampa riassunto finale"""
        print("\n" + "=" * 70)
        print(" RIASSUNTO FINALE TEST SUITE")
        print("=" * 70)
        
        total_successes = self.total_tests - self.total_failures - self.total_errors
        overall_success_rate = (total_successes / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"  Tempo totale: {total_time:.2f} secondi")
        print(f" Test totali: {self.total_tests}")
        print(f" Successi: {total_successes} ({overall_success_rate:.1f}%)")
        print(f" Fallimenti: {self.total_failures}")
        print(f" Errori: {self.total_errors}")
        print(f"⏭  Skip: {self.total_skipped}")
        
        # Status generale
        if self.total_failures == 0 and self.total_errors == 0:
            print("\n TUTTI I TEST SONO PASSATI CON SUCCESSO!")
            status = "PASS"
        else:
            print(f"\n ALCUNI TEST HANNO FALLITO - Rivedere i risultati")
            status = "FAIL"
        
        print(f" Status generale: {status}")
        
        # Dettagli per suite
        print("\n DETTAGLI PER SUITE:")
        for suite_name, result in self.results.items():
            if result:
                successes = result.testsRun - len(result.failures) - len(result.errors)
                suite_status = " PASS" if len(result.failures) == 0 and len(result.errors) == 0 else " FAIL"
                print(f"   {suite_name}: {successes}/{result.testsRun} - {suite_status}")
            else:
                print(f"   {suite_name}:  ERRORE")
    
    def run_specific_test(self, test_name, verbose=True):
        """Esegue un test specifico"""
        test_map = {
            'preprocessing': run_preprocessing_tests,
            'models': run_model_tests,
            'utils': run_utils_tests
        }
        
        if test_name.lower() not in test_map:
            available_tests = ', '.join(test_map.keys())
            print(f" Test '{test_name}' non trovato.")
            print(f"Test disponibili: {available_tests}")
            return None
        
        if verbose:
            print(f" Esecuzione test: {test_name}")
            print("-" * 50)
        
        test_function = test_map[test_name.lower()]
        result = test_function()
        
        if verbose:
            self._print_suite_summary(test_name, result)
        
        return result


def main():
    """Funzione principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test runner per Richter Predictor')
    parser.add_argument('--test', '-t', 
                       choices=['all', 'preprocessing', 'models', 'utils'],
                       default='all',
                       help='Specifica quale test eseguire')
    parser.add_argument('--quiet', '-q', 
                       action='store_true',
                       help='Output ridotto')
    parser.add_argument('--ci', 
                       action='store_true',
                       help='Modalità CI (Continuous Integration)')
    
    args = parser.parse_args()
    
    # Crea runner
    runner = TestRunner()
    
    # Modalità verbose
    verbose = not (args.quiet or args.ci)
    
    # Esegui test
    if args.test == 'all':
        results = runner.run_all_tests(verbose=verbose)
        success = all(
            result and len(result.failures) == 0 and len(result.errors) == 0 
            for result in results.values() if result
        )
    else:
        result = runner.run_specific_test(args.test, verbose=verbose)
        success = result and len(result.failures) == 0 and len(result.errors) == 0
    
    # CI mode: stampa solo risultato finale
    if args.ci:
        total_tests = runner.total_tests
        total_failures = runner.total_failures
        total_errors = runner.total_errors
        
        print(f"CI_RESULT: {total_tests} tests, {total_failures} failures, {total_errors} errors")
        if success:
            print("CI_STATUS: PASS")
        else:
            print("CI_STATUS: FAIL")
    
    # Exit code per CI/CD
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    if not TESTS_AVAILABLE:
        print(" Impossibile eseguire i test - moduli di test non disponibili")
        sys.exit(1)
    
    main()
