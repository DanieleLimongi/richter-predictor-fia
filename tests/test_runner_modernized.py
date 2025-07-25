#!/usr/bin/env python3
"""
Test Runner Modernized - Nuovo test runner che sostituisce tutti i vecchi test
Organizza ed esegue tutti i test in modo sistematico
"""

import sys
import time
import traceback
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import test suites
try:
    from test_core_functionality import run_core_tests
    from test_integration import run_integration_tests
    from test_feature_engineering import TestAdvancedFeatureEngineer
    CORE_TESTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import some test modules: {e}")
    CORE_TESTS_AVAILABLE = False


class ModernizedTestRunner:
    """Test runner modernizzato per la suite completa"""
    
    def __init__(self):
        self.total_tests_run = 0
        self.total_failures = 0
        self.total_errors = 0
        self.test_results = {}
        self.start_time = None
    
    def run_all_tests(self, include_slow=False):
        """Esegue tutti i test disponibili"""
        
        print("🧪 RICHTER PREDICTOR - MODERNIZED TEST SUITE")
        print("=" * 80)
        print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Project root: {project_root}")
        print("=" * 80)
        
        self.start_time = time.time()
        all_passed = True
        
        # 1. CORE FUNCTIONALITY TESTS (Sempre eseguiti)
        print("\n1️⃣  CORE FUNCTIONALITY TESTS")
        print("-" * 50)
        
        try:
            success = run_core_tests()
            self.test_results['core'] = success
            if not success:
                all_passed = False
                print("❌ Core tests FAILED")
            else:
                print("✅ Core tests PASSED")
        except Exception as e:
            print(f"💥 Core tests ERROR: {e}")
            self.test_results['core'] = False
            all_passed = False
        
        # 2. INTEGRATION TESTS (Sempre eseguiti)
        print("\n2️⃣  INTEGRATION TESTS")
        print("-" * 50)
        
        try:
            success = run_integration_tests()
            self.test_results['integration'] = success
            if not success:
                all_passed = False
                print("❌ Integration tests FAILED")
            else:
                print("✅ Integration tests PASSED")
        except Exception as e:
            print(f"💥 Integration tests ERROR: {e}")
            self.test_results['integration'] = False
            all_passed = False
        
        # 3. LEGACY TESTS (Solo se richiesti)
        if include_slow:
            print("\n3️⃣  LEGACY COMPATIBILITY TESTS")
            print("-" * 50)
            
            # Test preprocessing pipeline se esiste
            try:
                from test_preprocessing_pipeline import run_test_suite
                success = run_test_suite()
                self.test_results['preprocessing'] = success
                if not success:
                    all_passed = False
                print(f"{'✅' if success else '❌'} Preprocessing tests {'PASSED' if success else 'FAILED'}")
            except ImportError:
                print("⚠️  Preprocessing tests not available")
            except Exception as e:
                print(f"💥 Preprocessing tests ERROR: {e}")
                all_passed = False
            
            # Test ensemble architectures se esiste
            try:
                from test_ensemble_architectures import run_ensemble_tests
                success = run_ensemble_tests()
                self.test_results['ensemble'] = success
                if not success:
                    all_passed = False
                print(f"{'✅' if success else '❌'} Ensemble tests {'PASSED' if success else 'FAILED'}")
            except ImportError:
                print("⚠️  Ensemble tests not available")
            except Exception as e:
                print(f"💥 Ensemble tests ERROR: {e}")
                all_passed = False
        
        # SUMMARY
        self.print_final_summary(all_passed)
        
        return all_passed
    
    def run_quick_tests(self):
        """Esegue solo i test essenziali e veloci"""
        
        print("⚡ RICHTER PREDICTOR - QUICK TEST SUITE")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Solo core functionality (test più importanti)
        print("Running essential tests only...")
        
        try:
            success = run_core_tests()
            
            duration = time.time() - self.start_time
            
            print("\n" + "=" * 60)
            print("⚡ QUICK TEST SUMMARY:")
            print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
            print(f"   Duration: {duration:.2f}s")
            print("=" * 60)
            
            return success
            
        except Exception as e:
            print(f"💥 Quick tests failed with error: {e}")
            traceback.print_exc()
            return False
    
    def print_final_summary(self, all_passed):
        """Stampa summary finale dei test"""
        
        duration = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 80)
        
        print(f"⏱️  Total duration: {duration:.2f} seconds")
        print(f"🎯 Overall result: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
        
        print(f"\n📈 Test suite breakdown:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name.capitalize()}: {status}")
        
        if all_passed:
            print(f"\n🎉 SUCCESS: All test suites completed successfully!")
            print(f"   System is ready for production use.")
        else:
            print(f"\n⚠️  WARNING: Some tests failed!")
            print(f"   Review failures before proceeding to training.")
        
        print("=" * 80)
    
    def cleanup_old_tests(self):
        """Identifica e suggerisce pulizia dei test obsoleti"""
        
        print("\n🧹 CLEANUP RECOMMENDATIONS")
        print("-" * 50)
        
        # Test files che possono essere rimossi/sostituiti
        obsolete_files = [
            "quick_test.py",           # Sostituito da test_core_functionality
            "rapid_feature_test.py",   # Sostituito da test_core_functionality  
            "test_weighted_geographic.py"  # Feature ora integrata
        ]
        
        print("Obsolete test files (can be removed):")
        for file in obsolete_files:
            file_path = Path(__file__).parent / file
            if file_path.exists():
                print(f"   📁 {file} - ⚠️  Can be removed (replaced by modern tests)")
            else:
                print(f"   📁 {file} - ✅ Already removed")
        
        print(f"\nModern test files (keep these):")
        modern_files = [
            "test_core_functionality.py",
            "test_integration.py", 
            "test_runner_modernized.py"
        ]
        
        for file in modern_files:
            print(f"   📁 {file} - ✅ Modern test suite")


def main():
    """Main entry point"""
    
    runner = ModernizedTestRunner()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'quick':
            success = runner.run_quick_tests()
        elif command == 'all':
            success = runner.run_all_tests(include_slow=True)
        elif command == 'cleanup':
            runner.cleanup_old_tests()
            return 0
        else:
            print(f"Unknown command: {command}")
            print("Usage: python test_runner_modernized.py [quick|all|cleanup]")
            return 1
    else:
        # Default: run core tests
        success = runner.run_all_tests(include_slow=False)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
