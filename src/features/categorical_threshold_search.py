# categorical_threshold_search.py
"""
Random search per trovare le soglie ottimali per raggruppare categorie rare
delle feature geo_level_2_id e geo_level_3_id come "OTHER".

Obiettivo: ridurre la cardinalità delle feature categoriche mantenendo
le performance del modello.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import random
from typing import Dict, Tuple, List
import logging


class CategoricalThresholdOptimizer:
    """
    Classe per ottimizzare le soglie di frequenza per le feature categoriche
    geo_level_2_id e geo_level_3_id usando random search.
    """
    
    def __init__(self, data_path: str = "data/raw", random_state: int = 42, 
                 use_nested_cv: bool = True, outer_splits: int = 5, inner_splits: int = 3):
        """
        Inizializza l'ottimizzatore.
        
        Args:
            data_path: percorso della cartella contenente i dati raw
            random_state: seed per la riproducibilità
            use_nested_cv: se utilizzare nested K-fold o K-fold semplice
            outer_splits: numero di fold esterni per nested CV
            inner_splits: numero di fold interni per nested CV
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.use_nested_cv = use_nested_cv
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.df = None
        self.le_geo2 = LabelEncoder()
        self.le_geo3 = LabelEncoder()
        
        # Configura logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Carica i dati
        self._load_data()
        
    def _load_data(self):
        """Carica i dati di training."""
        self.logger.info("Caricamento dati...")
        
        train_values = pd.read_csv(self.data_path / "train_values.csv")
        train_labels = pd.read_csv(self.data_path / "train_labels.csv")
        
        self.df = train_values.merge(train_labels, on="building_id", how="left")
        self.logger.info(f"Dati caricati: {self.df.shape}")
        
    def _apply_threshold(self, series: pd.Series, threshold: float) -> pd.Series:
        """
        Applica una soglia di frequenza percentuale a una serie categorica.
        Le categorie con frequenza < threshold vengono mappate a -1 (OTHER).
        
        Args:
            series: serie pandas con i valori categorici
            threshold: soglia percentuale (es. 0.01 = 1%)
            
        Returns:
            serie con valori sotto la soglia mappati a -1
        """
        # Calcola frequenze percentuali
        value_counts = series.value_counts(normalize=True)
        
        # Identifica valori sotto la soglia
        rare_values = value_counts[value_counts < threshold].index
        
        # Mappa valori rari a -1 (OTHER)
        result = series.copy()
        result[result.isin(rare_values)] = -1
        
        return result
        
    def _prepare_features(self, geo2_threshold: float, geo3_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara le feature per la valutazione con le soglie specificate.
        
        Args:
            geo2_threshold: soglia per geo_level_2_id
            geo3_threshold: soglia per geo_level_3_id
            
        Returns:
            tuple (X, y) con feature preparate e target
        """
        df_temp = self.df.copy()
        
        # Applica soglie
        df_temp['geo_level_2_id'] = self._apply_threshold(df_temp['geo_level_2_id'], geo2_threshold)
        df_temp['geo_level_3_id'] = self._apply_threshold(df_temp['geo_level_3_id'], geo3_threshold)
        
        # Prepara feature base (usa solo alcune feature per velocizzare la valutazione)
        feature_cols = [
            'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
            'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage',
            'land_surface_condition', 'foundation_type', 'roof_type',
            'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration'
        ]
        
        X = df_temp[feature_cols].copy()
        y = df_temp['damage_grade'].values
        
        # Encoding delle feature categoriche
        categorical_cols = [
            'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
            'land_surface_condition', 'foundation_type', 'roof_type',
            'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration'
        ]
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
        return X.values, y
        
    def evaluate_thresholds(self, geo2_threshold: float, geo3_threshold: float, 
                          cv_folds: int = 3, scoring: str = 'f1_micro') -> float:
        """
        Valuta una coppia di soglie usando K-fold semplice o nested CV.
        
        Args:
            geo2_threshold: soglia per geo_level_2_id
            geo3_threshold: soglia per geo_level_3_id  
            cv_folds: numero di fold per cross-validation (solo per K-fold semplice)
            scoring: metrica di scoring
            
        Returns:
            score medio di cross-validation
        """
        X, y = self._prepare_features(geo2_threshold, geo3_threshold)
        
        if self.use_nested_cv:
            return self._nested_cv_evaluation(X, y, scoring)
        else:
            return self._simple_cv_evaluation(X, y, cv_folds, scoring)
    
    def _simple_cv_evaluation(self, X, y, cv_folds: int, scoring: str) -> float:
        """K-Fold cross-validation semplice (implementazione originale)."""
        model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return scores.mean()
    
    def _nested_cv_evaluation(self, X, y, scoring: str) -> float:
        """Nested K-Fold Cross-Validation per valutazione robusta."""
        
        # OUTER LOOP: Per la stima finale delle performance
        outer_cv = StratifiedKFold(
            n_splits=self.outer_splits, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        # INNER LOOP: Per l'ottimizzazione degli iperparametri
        inner_cv = StratifiedKFold(
            n_splits=self.inner_splits, 
            shuffle=True, 
            random_state=self.random_state + 1
        )
        
        # Griglia iperparametri ottimizzata per velocità
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        outer_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            # Split outer
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # INNER LOOP: Grid search per ottimizzazione iperparametri
            rf = RandomForestClassifier(random_state=self.random_state)
            
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            
            # Fit con i migliori parametri trovati
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Valutazione sul test set dell'outer loop
            best_model = grid_search.best_estimator_
            
            if scoring == 'accuracy':
                outer_score = best_model.score(X_test_outer, y_test_outer)
            else:
                from sklearn.metrics import f1_score
                y_pred = best_model.predict(X_test_outer)
                outer_score = f1_score(y_test_outer, y_pred, average='micro')
            
            outer_scores.append(outer_score)
        
        return np.mean(outer_scores)
        
    def random_search(self, n_iterations: int = 50, 
                     geo2_range: Tuple[float, float] = (0.001, 0.05),
                     geo3_range: Tuple[float, float] = (0.0005, 0.02)) -> Dict:
        """
        Esegue random search per trovare le soglie ottimali.
        
        Args:
            n_iterations: numero di iterazioni del random search
            geo2_range: range di soglie per geo_level_2_id (min, max)
            geo3_range: range di soglie per geo_level_3_id (min, max)
            
        Returns:
            dizionario con i risultati migliori
        """
        self.logger.info(f"Inizio random search con {n_iterations} iterazioni...")
        
        best_score = -np.inf
        best_params = {}
        history = []
        
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        for i in range(n_iterations):
            # Genera soglie random
            geo2_threshold = random.uniform(geo2_range[0], geo2_range[1])
            geo3_threshold = random.uniform(geo3_range[0], geo3_range[1])
            
            try:
                # Valuta le soglie
                score = self.evaluate_thresholds(geo2_threshold, geo3_threshold)
                
                # Calcola statistiche delle soglie per il logging
                geo2_categories = len(self.df['geo_level_2_id'].value_counts())
                geo3_categories = len(self.df['geo_level_3_id'].value_counts())
                
                geo2_remaining = len(self._apply_threshold(self.df['geo_level_2_id'], geo2_threshold).value_counts())
                geo3_remaining = len(self._apply_threshold(self.df['geo_level_3_id'], geo3_threshold).value_counts())
                
                result = {
                    'iteration': i + 1,
                    'geo2_threshold': geo2_threshold,
                    'geo3_threshold': geo3_threshold,
                    'score': score,
                    'geo2_categories_original': geo2_categories,
                    'geo2_categories_remaining': geo2_remaining,
                    'geo3_categories_original': geo3_categories,
                    'geo3_categories_remaining': geo3_remaining
                }
                
                history.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = result.copy()
                    
                self.logger.info(
                    f"Iter {i+1:2d}: geo2_th={geo2_threshold:.4f}, geo3_th={geo3_threshold:.4f}, "
                    f"score={score:.4f} (geo2: {geo2_categories}→{geo2_remaining}, "
                    f"geo3: {geo3_categories}→{geo3_remaining})"
                )
                
            except Exception as e:
                self.logger.error(f"Errore nell'iterazione {i+1}: {e}")
                continue
                
        self.logger.info(f"Miglior score: {best_score:.4f}")
        self.logger.info(f"Miglior configurazione: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history
        }
        
    def get_category_stats(self, geo2_threshold: float, geo3_threshold: float) -> Dict:
        """
        Ottieni statistiche dettagliate sulle categorie con le soglie specificate.
        
        Args:
            geo2_threshold: soglia per geo_level_2_id
            geo3_threshold: soglia per geo_level_3_id
            
        Returns:
            dizionario con statistiche dettagliate
        """
        geo2_orig = self.df['geo_level_2_id']
        geo3_orig = self.df['geo_level_3_id']
        
        geo2_filtered = self._apply_threshold(geo2_orig, geo2_threshold)
        geo3_filtered = self._apply_threshold(geo3_orig, geo3_threshold)
        
        return {
            'geo_level_2_id': {
                'original_categories': len(geo2_orig.value_counts()),
                'remaining_categories': len(geo2_filtered.value_counts()),
                'reduction_pct': (1 - len(geo2_filtered.value_counts()) / len(geo2_orig.value_counts())) * 100,
                'other_samples': (geo2_filtered == -1).sum(),
                'other_pct': (geo2_filtered == -1).mean() * 100
            },
            'geo_level_3_id': {
                'original_categories': len(geo3_orig.value_counts()),
                'remaining_categories': len(geo3_filtered.value_counts()),
                'reduction_pct': (1 - len(geo3_filtered.value_counts()) / len(geo3_orig.value_counts())) * 100,
                'other_samples': (geo3_filtered == -1).sum(),
                'other_pct': (geo3_filtered == -1).mean() * 100
            }
        }


def main():
    """Esempio di utilizzo della classe."""
    optimizer = CategoricalThresholdOptimizer()
    
    # Esegui random search
    results = optimizer.random_search(n_iterations=30)
    
    # Salva risultati
    output_dir = Path("reports/threshold_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "threshold_search_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    # Mostra statistiche delle soglie migliori
    best_params = results['best_params']
    stats = optimizer.get_category_stats(
        best_params['geo2_threshold'],
        best_params['geo3_threshold']
    )
    
    print("\n=== RISULTATI OTTIMIZZAZIONE ===")
    print(f"Miglior score: {results['best_score']:.4f}")
    print(f"Soglia geo_level_2_id: {best_params['geo2_threshold']:.4f}")
    print(f"Soglia geo_level_3_id: {best_params['geo3_threshold']:.4f}")
    
    print("\n=== STATISTICHE CATEGORIE ===")
    for feature, stat in stats.items():
        print(f"\n{feature}:")
        print(f"  Categorie originali: {stat['original_categories']}")
        print(f"  Categorie rimanenti: {stat['remaining_categories']}")
        print(f"  Riduzione: {stat['reduction_pct']:.1f}%")
        print(f"  Campioni 'OTHER': {stat['other_samples']} ({stat['other_pct']:.1f}%)")


if __name__ == "__main__":
    main()
