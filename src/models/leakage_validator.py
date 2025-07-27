"""
Validatore Anti-Leakage per Nested CV
Strumenti avanzati per rilevare e prevenire data leakage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
import hashlib
import warnings
from datetime import datetime
import json

class AdvancedLeakageValidator:
    """Validatore avanzato per data leakage con controlli multipli"""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_history = []
        self.preprocessing_tracking = {}
        self.data_fingerprints = {}
        self.warnings_log = []
        
    def create_data_fingerprint(self, data: np.ndarray, name: str) -> str:
        """Crea fingerprint unico per dataset"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Hash robusto basato su shape, alcuni valori e statistiche
        shape_str = f"{data.shape}"
        sample_values = data.flatten()[:min(1000, len(data.flatten()))]
        stats = f"{np.mean(sample_values):.6f}_{np.std(sample_values):.6f}_{np.min(sample_values):.6f}_{np.max(sample_values):.6f}"
        
        fingerprint_str = f"{shape_str}_{stats}_{len(sample_values)}"
        fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
        
        self.data_fingerprints[name] = {
            'fingerprint': fingerprint,
            'shape': data.shape,
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        }
        
        return fingerprint
    
    def validate_cv_split(self, train_idx: np.ndarray, val_idx: np.ndarray, 
                         split_name: str, parent_data_name: str = None) -> Dict[str, Any]:
        """Validazione completa di uno split CV"""
        
        validation_result = {
            'split_name': split_name,
            'timestamp': datetime.now().isoformat(),
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'total_size': len(train_idx) + len(val_idx),
            'leakage_detected': False,
            'warnings': [],
            'errors': []
        }
        
        # Test 1: Overlap diretto
        overlap = set(train_idx) & set(val_idx)
        if overlap:
            error_msg = f"CRITICAL: Direct index overlap detected! {len(overlap)} indices appear in both train and validation"
            validation_result['errors'].append(error_msg)
            validation_result['leakage_detected'] = True
            
            if self.strict_mode:
                raise ValueError(f"DATA LEAKAGE: {split_name} - {error_msg}")
        
        # Test 2: Indici fuori range
        max_allowed_idx = validation_result['total_size'] - 1
        invalid_train = train_idx[train_idx > max_allowed_idx]
        invalid_val = val_idx[val_idx > max_allowed_idx]
        
        if len(invalid_train) > 0 or len(invalid_val) > 0:
            error_msg = f"Invalid indices detected! Train: {len(invalid_train)}, Val: {len(invalid_val)} out of range"
            validation_result['errors'].append(error_msg)
            
            if self.strict_mode:
                raise ValueError(f"INDEX ERROR: {split_name} - {error_msg}")
        
        # Test 3: Distribuzione size ragionevole
        train_pct = len(train_idx) / validation_result['total_size']
        val_pct = len(val_idx) / validation_result['total_size']
        
        if train_pct < 0.5 or train_pct > 0.95:
            warning_msg = f"Unusual split ratio: train={train_pct:.1%}, val={val_pct:.1%}"
            validation_result['warnings'].append(warning_msg)
            self.warnings_log.append(f"{split_name}: {warning_msg}")
        
        # Test 4: Indici duplicati all'interno dei set
        train_duplicates = len(train_idx) - len(set(train_idx))
        val_duplicates = len(val_idx) - len(set(val_idx))
        
        if train_duplicates > 0 or val_duplicates > 0:
            error_msg = f"Duplicate indices! Train: {train_duplicates}, Val: {val_duplicates}"
            validation_result['errors'].append(error_msg)
            
            if self.strict_mode:
                raise ValueError(f"DUPLICATE INDICES: {split_name} - {error_msg}")
        
        # Store validation
        self.validation_history.append(validation_result)
        
        # Print risultato
        status = "FAIL" if validation_result['leakage_detected'] or validation_result['errors'] else "PASS"
        warning_info = f" ({len(validation_result['warnings'])} warnings)" if validation_result['warnings'] else ""
        print(f"   {status} {split_name}: {len(train_idx)} train, {len(val_idx)} val{warning_info}")
        
        return validation_result
    
    def track_preprocessing_fit(self, component_name: str, data_identifier: str, 
                              train_indices: np.ndarray, fold_info: str) -> str:
        """Traccia fit di componenti preprocessing per rilevare riuso inappropriato"""
        
        # Crea chiave unica per questo fit
        indices_hash = hashlib.md5(train_indices.tobytes()).hexdigest()[:16]
        fit_key = f"{component_name}_{fold_info}_{indices_hash}"
        
        # Check se stesso componente già fittato con dati diversi
        if component_name in self.preprocessing_tracking:
            previous_fits = self.preprocessing_tracking[component_name]
            
            for prev_fit in previous_fits:
                if prev_fit['indices_hash'] != indices_hash:
                    # Stesso componente, dati diversi - potenziale leakage se riusato
                    warning_msg = f"Component {component_name} fitted multiple times with different data"
                    self.warnings_log.append(f"{fold_info}: {warning_msg}")
                    
                    if self.strict_mode:
                        print(f"   WARNING: {warning_msg}")
        
        # Store fit info
        fit_info = {
            'fit_key': fit_key,
            'data_identifier': data_identifier,
            'indices_hash': indices_hash,
            'indices_size': len(train_indices),
            'fold_info': fold_info,
            'timestamp': datetime.now().isoformat()
        }
        
        if component_name not in self.preprocessing_tracking:
            self.preprocessing_tracking[component_name] = []
        
        self.preprocessing_tracking[component_name].append(fit_info)
        
        print(f"   Tracked: {component_name} fit on {len(train_indices)} samples ({fold_info})")
        return fit_key
    
    def validate_data_consistency(self, train_data: np.ndarray, val_data: np.ndarray, 
                                split_name: str) -> Dict[str, Any]:
        """Valida consistenza tra train e validation data"""
        
        consistency_result = {
            'split_name': split_name,
            'feature_count_match': train_data.shape[1] == val_data.shape[1],
            'train_features': train_data.shape[1],
            'val_features': val_data.shape[1],
            'data_type_match': train_data.dtype == val_data.dtype,
            'warnings': [],
            'errors': []
        }
        
        # Test feature count
        if not consistency_result['feature_count_match']:
            error_msg = f"Feature count mismatch! Train: {train_data.shape[1]}, Val: {val_data.shape[1]}"
            consistency_result['errors'].append(error_msg)
            
            if self.strict_mode:
                raise ValueError(f"FEATURE MISMATCH: {split_name} - {error_msg}")
        
        # Test data types
        if not consistency_result['data_type_match']:
            warning_msg = f"Data type mismatch! Train: {train_data.dtype}, Val: {val_data.dtype}"
            consistency_result['warnings'].append(warning_msg)
            self.warnings_log.append(f"{split_name}: {warning_msg}")
        
        # Test per valori identici (potenziale leakage)
        if train_data.shape[1] == val_data.shape[1]:
            # Sample per performance
            sample_size = min(100, train_data.shape[0], val_data.shape[0])
            train_sample = train_data[:sample_size]
            val_sample = val_data[:sample_size]
            
            # Check righe identiche
            identical_rows = 0
            for i in range(sample_size):
                if np.allclose(train_sample[i], val_sample[i], atol=1e-10):
                    identical_rows += 1
            
            if identical_rows > sample_size * 0.1:  # >10% righe identiche
                warning_msg = f"Suspicious: {identical_rows}/{sample_size} identical rows between train/val"
                consistency_result['warnings'].append(warning_msg)
                self.warnings_log.append(f"{split_name}: {warning_msg}")
        
        # Store fingerprints
        train_fp = self.create_data_fingerprint(train_data, f"{split_name}_train")
        val_fp = self.create_data_fingerprint(val_data, f"{split_name}_val")
        
        consistency_result['train_fingerprint'] = train_fp
        consistency_result['val_fingerprint'] = val_fp
        
        return consistency_result
    
    def validate_target_distribution(self, y_train: np.ndarray, y_val: np.ndarray, 
                                   split_name: str, tolerance: float = 0.1) -> Dict[str, Any]:
        """Valida che la distribuzione del target sia ragionevole tra train e val"""
        
        train_dist = np.bincount(y_train) / len(y_train)
        val_dist = np.bincount(y_val) / len(y_val)
        
        # Pad per stesso numero di classi
        max_classes = max(len(train_dist), len(val_dist))
        train_dist = np.pad(train_dist, (0, max_classes - len(train_dist)))
        val_dist = np.pad(val_dist, (0, max_classes - len(val_dist)))
        
        # KL divergence approssimata
        distribution_diff = np.abs(train_dist - val_dist).max()
        
        target_result = {
            'split_name': split_name,
            'train_distribution': train_dist.tolist(),
            'val_distribution': val_dist.tolist(),
            'max_difference': distribution_diff,
            'within_tolerance': distribution_diff <= tolerance,
            'warnings': []
        }
        
        if not target_result['within_tolerance']:
            warning_msg = f"Target distribution shift: max diff {distribution_diff:.3f} > tolerance {tolerance}"
            target_result['warnings'].append(warning_msg)
            self.warnings_log.append(f"{split_name}: {warning_msg}")
            
            print(f"   WARNING: {warning_msg}")
        
        return target_result
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Report completo di tutte le validazioni"""
        
        total_validations = len(self.validation_history)
        total_errors = sum(len(v['errors']) for v in self.validation_history)
        total_warnings = len(self.warnings_log)
        leakage_detections = sum(1 for v in self.validation_history if v['leakage_detected'])
        
        report = {
            'validation_summary': {
                'total_validations': total_validations,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'leakage_detections': leakage_detections,
                'validation_passed': total_errors == 0 and leakage_detections == 0
            },
            'preprocessing_tracking': {
                'components_tracked': len(self.preprocessing_tracking),
                'total_fits': sum(len(fits) for fits in self.preprocessing_tracking.values()),
                'components': list(self.preprocessing_tracking.keys())
            },
            'data_fingerprints': {
                'total_fingerprints': len(self.data_fingerprints),
                'fingerprints': self.data_fingerprints
            },
            'validation_history': self.validation_history,
            'warnings_log': self.warnings_log,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def print_summary(self):
        """Stampa summary delle validazioni"""
        
        report = self.get_comprehensive_report()
        summary = report['validation_summary']
        
        print(f"\nANTI-LEAKAGE VALIDATION SUMMARY")
        print(f"=" * 40)
        print(f"   Validations: {summary['total_validations']}")
        print(f"   Errors: {summary['total_errors']}")
        print(f"   Warnings: {summary['total_warnings']}")
        print(f"   Leakage detected: {summary['leakage_detections']}")
        print(f"   Status: {'PASSED' if summary['validation_passed'] else 'FAILED'}")
        
        if summary['total_warnings'] > 0:
            print(f"\nWarnings:")
            for warning in self.warnings_log[-5:]:  # Ultimi 5
                print(f"   - {warning}")
        
        print(f"\nPreprocessing tracking:")
        print(f"   Components: {report['preprocessing_tracking']['components_tracked']}")
        print(f"   Total fits: {report['preprocessing_tracking']['total_fits']}")

def validate_nested_cv_implementation(trainer_instance) -> bool:
    """Funzione helper per validare implementazione nested CV"""
    
    print("VALIDATING NESTED CV IMPLEMENTATION...")
    
    # Test con dati dummy
    np.random.seed(42)
    X_dummy = np.random.randn(1000, 50)
    y_dummy = np.random.randint(0, 3, 1000)
    
    validator = AdvancedLeakageValidator(strict_mode=True)
    
    try:
        # Simula outer CV
        from sklearn.model_selection import StratifiedKFold
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_dummy, y_dummy)):
            # Valida outer split
            validator.validate_cv_split(train_idx, test_idx, f"outer_fold_{fold+1}")
            
            # Simula inner CV
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42+fold)
            X_train = X_dummy[train_idx]
            y_train = y_dummy[train_idx]
            
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train)):
                validator.validate_cv_split(
                    inner_train_idx, inner_val_idx, 
                    f"inner_fold_{fold+1}_{inner_fold+1}"
                )
                
                # Track preprocessing
                validator.track_preprocessing_fit(
                    "DummyPreprocessor", 
                    f"fold_{fold}", 
                    inner_train_idx, 
                    f"outer_{fold+1}_inner_{inner_fold+1}"
                )
                
                # Valida data consistency
                X_inner_train = X_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]
                validator.validate_data_consistency(
                    X_inner_train, X_inner_val, 
                    f"inner_{fold+1}_{inner_fold+1}"
                )
                
                # Valida target distribution
                y_inner_train = y_train[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]
                validator.validate_target_distribution(
                    y_inner_train, y_inner_val, 
                    f"inner_{fold+1}_{inner_fold+1}"
                )
        
        # Print summary
        validator.print_summary()
        
        # Return success
        report = validator.get_comprehensive_report()
        return report['validation_summary']['validation_passed']
        
    except Exception as e:
        print(f"FAILED: Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Anti-Leakage Validator...")
    success = validate_nested_cv_implementation(None)
    print(f"Validation {'PASSED' if success else 'FAILED'}")