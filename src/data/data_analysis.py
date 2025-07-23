#!/usr/bin/env python3
"""
Data Analysis & Feature Classification per **Richter's Predictor** dataset.

Questo script si concentra sull'analisi automatica e intelligente dei dati raw,
complementando eda.py che si occupa di visualizzazioni e documentazione:

- Classificazione automatica dei tipi di features (numeric, categorical, geographic, binary)
- Analisi statistica approfondita delle distribuzioni e correlazioni
- Identificazione automatica di outliers e valori anomali
- Strategie di preprocessing personalizzate per ogni tipo di feature
- Generazione di feature mapping per la pipeline TensorFlow

Usage (dalla root del progetto):
    python src/data/data_analysis.py

Workflow raccomandato:
1. Esegui questo script per l'analisi automatica dei tipi e classificazione features
2. Esegui eda.py per visualizzazioni e documentazione dettagliata basate sui risultati

Output generati:
* Analisi colonne -> reports/eda/column_analysis.json (tipi automatici)
* Statistiche numeriche -> reports/eda/numeric_stats.csv (descrittive complete)
* Mapping features -> reports/eda/feature_mapping.json (per pipeline TF)
* Suggerimenti preprocessing -> reports/eda/preprocessing_suggestions.json

Questo script fornisce la base analitica intelligente che eda.py utilizza
per creare visualizzazioni mirate e documentazione strutturata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Analizzatore per i dati del Richter Predictor"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data/raw"
        self.train_values_path = self.data_dir / "train_values.csv"
        self.train_labels_path = self.data_dir / "train_labels.csv"
        self.reports_dir = self.project_root / "reports/eda"
        
        # Crea directory per i report
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Carica i dati raw"""
        print(" Caricamento dati raw...")
        
        # Carica features e target
        self.features = pd.read_csv(self.train_values_path)
        self.labels = pd.read_csv(self.train_labels_path)
        
        # Merge
        self.df = self.features.merge(self.labels, on='building_id')
        
        print(f"Dati caricati: {self.df.shape}")
        print(f"Features: {self.features.shape[1] - 1}")  # -1 per building_id
        print(f"Target: damage_grade")
        
        return self.df
        
    def analyze_data_types(self):
        """Analizza i tipi di dati e identifica automaticamente le categorie"""
        print("\n ANALISI TIPI DI DATI")
        print("=" * 50)
        
        # Analizza ogni colonna
        analysis = {}
        
        for col in self.df.columns:
            if col == 'building_id':
                continue
                
            col_data = self.df[col]
            dtype = col_data.dtype
            unique_vals = col_data.nunique()
            total_vals = len(col_data)
            missing_vals = col_data.isnull().sum()
            
            # Determina il tipo logico
            if col == 'damage_grade':
                col_type = 'target'
            elif col.startswith('geo_level_'):
                # Variabili geografiche sono categoriche speciali (ID gerarchici)
                col_type = 'geographic'
            elif dtype in ['int64', 'float64'] and unique_vals > 20:
                col_type = 'numeric'
            elif dtype == 'object' or unique_vals <= 20:
                col_type = 'categorical'
            else:
                col_type = 'unknown'
            
            # Se ha solo 2 valori unici, potrebbe essere binaria
            if unique_vals == 2 and not col.startswith('geo_level_'):
                col_type = 'binary'
                
            analysis[col] = {
                'dtype': str(dtype),
                'unique_values': unique_vals,
                'missing_values': missing_vals,
                'missing_percent': (missing_vals / total_vals) * 100,
                'type': col_type,
                'sample_values': list(col_data.dropna().unique()[:5])
            }
            
        # Salva analisi
        analysis_file = self.reports_dir / "column_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
            
        # Stampa riassunto
        type_counts = {}
        for col, info in analysis.items():
            col_type = info['type']
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
            
        print(f"Riassunto tipi:")
        for col_type, count in type_counts.items():
            print(f"   {col_type}: {count} colonne")
            
        # Stampa dettagli per ogni tipo
        for col_type in ['numeric', 'categorical', 'geographic', 'binary', 'target']:
            cols = [col for col, info in analysis.items() if info['type'] == col_type]
            if cols:
                print(f"\n Colonne {col_type.upper()}:")
                for col in cols[:10]:  # Mostra solo le prime 10
                    info = analysis[col]
                    print(f"   {col}: {info['unique_values']} valori unici, {info['missing_percent']:.1f}% missing")
                if len(cols) > 10:
                    print(f"   ... e altre {len(cols) - 10} colonne")
                    
        return analysis
        
    def analyze_target(self):
        """Analizza la distribuzione del target"""
        print("\nANALISI TARGET")
        print("=" * 50)
        
        target_col = 'damage_grade'
        target_data = self.df[target_col]
        
        # Distribuzione
        value_counts = target_data.value_counts().sort_index()
        print(f" Distribuzione {target_col}:")
        for val, count in value_counts.items():
            percent = (count / len(target_data)) * 100
            print(f"   Classe {val}: {count:,} campioni ({percent:.1f}%)")
            
        # Bilanciamento
        min_class = value_counts.min()
        max_class = value_counts.max()
        imbalance_ratio = max_class / min_class
        print(f" Ratio sbilanciamento: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2:
            print(" Dataset sbilanciato! Considera stratified sampling.")
        else:
            print(" Dataset relativamente bilanciato.")
            
        return value_counts
        
    def analyze_numeric_features(self, analysis):
        """Analizza le features numeriche"""
        print("\n ANALISI FEATURES NUMERICHE")
        print("=" * 50)
        
        numeric_cols = [col for col, info in analysis.items() if info['type'] == 'numeric']
        
        if not numeric_cols:
            print(" Nessuna feature numerica trovata!")
            return
            
        numeric_data = self.df[numeric_cols]
        
        # Statistiche descrittive
        stats = numeric_data.describe()
        print(f" Statistiche descrittive per {len(numeric_cols)} features numeriche:")
        print(stats)
        
        # Salva statistiche
        stats_file = self.reports_dir / "numeric_stats.csv"
        stats.to_csv(stats_file)
        
        # Identifica outliers usando IQR
        outlier_info = {}
        for col in numeric_cols:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_data[(numeric_data[col] < lower_bound) | 
                                  (numeric_data[col] > upper_bound)][col]
            
            outlier_info[col] = {
                'count': len(outliers),
                'percent': (len(outliers) / len(numeric_data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        # Mostra outliers
        print(f"\n Outliers (IQR method):")
        for col, info in outlier_info.items():
            if info['percent'] > 5:  # Mostra solo se > 5%
                print(f"   {col}: {info['count']} outliers ({info['percent']:.1f}%)")
                
        return numeric_data, outlier_info
        
    def analyze_categorical_features(self, analysis):
        """Analizza le features categoriche"""
        print("\n ANALISI FEATURES CATEGORICHE")
        print("=" * 50)
        
        categorical_cols = [col for col, info in analysis.items() if info['type'] == 'categorical']
        
        if not categorical_cols:
            print(" Nessuna feature categorica trovata!")
            return
            
        print(f" Analisi di {len(categorical_cols)} features categoriche:")
        
        cardinality_info = {}
        for col in categorical_cols:
            unique_vals = self.df[col].nunique()
            value_counts = self.df[col].value_counts()
            
            # Calcola entropia (diversità)
            probs = value_counts / len(self.df)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            
            cardinality_info[col] = {
                'cardinality': unique_vals,
                'entropy': entropy,
                'most_common': value_counts.head(3).to_dict(),
                'least_common_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
            }
            
            print(f"   {col}: {unique_vals} categorie, entropia={entropy:.2f}")
            
            # Mostra categorie rare (< 1%)
            rare_threshold = len(self.df) * 0.01
            rare_categories = value_counts[value_counts < rare_threshold]
            if len(rare_categories) > 0:
                print(f"     → {len(rare_categories)} categorie rare (< 1%)")
                
        # Salva info cardinalità
        cardinality_file = self.reports_dir / "categorical_cardinality.json"
        with open(cardinality_file, 'w') as f:
            json.dump(cardinality_info, f, indent=2, default=str)
            
        return cardinality_info
        
    def analyze_geographic_features(self, analysis):
        """Analizza le features geografiche con strategia specifica"""
        print("\n  ANALISI FEATURES GEOGRAFICHE")
        print("=" * 50)
        
        geo_cols = [col for col, info in analysis.items() if info['type'] == 'geographic']
        
        if not geo_cols:
            print("essuna feature geografica trovata!")
            return
            
        print(f" Analisi di {len(geo_cols)} features geografiche:")
        
        geo_info = {}
        for col in geo_cols:
            unique_vals = self.df[col].nunique()
            value_counts = self.df[col].value_counts()
            
            # Calcola correlazione con target (può essere utile anche se categorica)
            target_corr = self.df[[col, 'damage_grade']].corr().iloc[0, 1]
            
            geo_info[col] = {
                'cardinality': unique_vals,
                'target_correlation': target_corr,
                'most_frequent': value_counts.head(3).to_dict(),
                'coverage_top_10': (value_counts.head(10).sum() / len(self.df)) * 100
            }
            
            # Determina strategia preprocessing
            if unique_vals <= 50:
                strategy = "One-hot encoding"
            elif unique_vals <= 1000:
                strategy = "Embedding (media cardinalità)"
            else:
                strategy = "Embedding o Drop (alta cardinalità)"
                
            print(f"   {col}: {unique_vals} regioni, corr={target_corr:.3f}")
            print(f"     → Strategia: {strategy}")
            
            # Mostra copertura top categorie
            coverage = geo_info[col]['coverage_top_10']
            print(f"     → Top 10 coprono {coverage:.1f}% dei dati")
            
        # Salva info geografiche
        geo_file = self.reports_dir / "geographic_analysis.json"
        with open(geo_file, 'w') as f:
            json.dump(geo_info, f, indent=2, default=str)
            
        return geo_info
        
    def analyze_correlations(self, analysis):
        """Analizza correlazioni tra features"""
        print("\n ANALISI CORRELAZIONI")
        print("=" * 50)
        
        # Solo features numeriche per correlazione Pearson
        numeric_cols = [col for col, info in analysis.items() if info['type'] == 'numeric']
        
        if len(numeric_cols) < 2:
            print(" Insufficienti features numeriche per correlazione!")
            return
            
        # Calcola correlazioni
        numeric_data = self.df[numeric_cols + ['damage_grade']]
        corr_matrix = numeric_data.corr()
        
        # Salva matrice correlazioni
        corr_file = self.reports_dir / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_file)
        
        # Trova correlazioni forti con il target
        target_corr = corr_matrix['damage_grade'].abs().sort_values(ascending=False)
        print(f" Correlazioni più forti con damage_grade:")
        for col, corr in target_corr.head(10).items():
            if col != 'damage_grade':
                print(f"   {col}: {corr:.3f}")
                
        # Trova correlazioni forti tra features (multicollinearità)
        print(f"\n Potenziale multicollinearità (|corr| > 0.8):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8 and col1 != 'damage_grade' and col2 != 'damage_grade':
                    high_corr_pairs.append((col1, col2, corr_val))
                    
        if high_corr_pairs:
            for col1, col2, corr_val in high_corr_pairs[:5]:  # Mostra solo i primi 5
                print(f"   {col1} ↔ {col2}: {corr_val:.3f}")
        else:
            print("   Nessuna multicollinearità significativa trovata")
            
        return corr_matrix
        
    def suggest_preprocessing_strategy(self, analysis):
        """Suggerisce strategie di preprocessing basate sull'analisi"""
        print("\n SUGGERIMENTI PREPROCESSING")
        print("=" * 50)
        
        suggestions = {
            'numeric_features': [],
            'categorical_features': [],
            'geographic_features': [],
            'binary_features': [],
            'preprocessing_steps': []
        }
        
        # Analizza ogni tipo
        for col, info in analysis.items():
            col_type = info['type']
            
            if col_type == 'numeric':
                suggestions['numeric_features'].append(col)
                suggestions['preprocessing_steps'].append(
                    f" {col}: Normalizzazione con tf.keras.layers.Normalization"
                )
                
            elif col_type == 'categorical':
                suggestions['categorical_features'].append(col)
                if info['unique_values'] > 50:
                    suggestions['preprocessing_steps'].append(
                        f" {col}: Embedding (alta cardinalità: {info['unique_values']} categorie)"
                    )
                else:
                    suggestions['preprocessing_steps'].append(
                        f" {col}: One-hot encoding o IntegerLookup ({info['unique_values']} categorie)"
                    )
                    
            elif col_type == 'geographic':
                suggestions['geographic_features'].append(col)
                if info['unique_values'] <= 50:
                    suggestions['preprocessing_steps'].append(
                        f" {col}: One-hot encoding ({info['unique_values']} regioni)"
                    )
                elif info['unique_values'] <= 1000:
                    suggestions['preprocessing_steps'].append(
                        f" {col}: Embedding a bassa dimensione ({info['unique_values']} regioni)"
                    )
                else:
                    suggestions['preprocessing_steps'].append(
                        f" {col}: Embedding o Drop ({info['unique_values']} regioni - molto frammentato)"
                    )
                    
            elif col_type == 'binary':
                suggestions['binary_features'].append(col)
                suggestions['preprocessing_steps'].append(
                    f" {col}: Mantieni come binario o converti a 0/1"
                )
                
        # Suggerimenti generali
        print(" Strategia consigliata:")
        print(f"    Features numeriche ({len(suggestions['numeric_features'])}): tf.keras.layers.Normalization")
        print(f"    Features categoriche ({len(suggestions['categorical_features'])}): IntegerLookup + Embedding/OneHot")
        print(f"    Features geografiche ({len(suggestions['geographic_features'])}): Embedding gerarchico")
        print(f"    Features binarie ({len(suggestions['binary_features'])}): Mantieni come sono")
        
        print(f"\n Pipeline preprocessing dettagliata:")
        for step in suggestions['preprocessing_steps'][:10]:  # Mostra primi 10
            print(f"   {step}")
            
        if len(suggestions['preprocessing_steps']) > 10:
            print(f"   ... e altre {len(suggestions['preprocessing_steps']) - 10} features")
            
        # Salva suggerimenti
        suggestions_file = self.reports_dir / "preprocessing_suggestions.json"
        with open(suggestions_file, 'w') as f:
            json.dump(suggestions, f, indent=2, default=str)
            
        return suggestions
        
    def generate_feature_mapping(self, analysis):
        """Genera il mapping delle features per la pipeline TensorFlow"""
        print("\n  GENERAZIONE FEATURE MAPPING")
        print("=" * 50)
        
        mapping = {
            'numeric_features': [],
            'categorical_features': [],
            'geographic_features': [],
            'binary_features': [],
            'target_feature': 'damage_grade'
        }
        
        for col, info in analysis.items():
            col_type = info['type']
            
            if col_type == 'numeric':
                mapping['numeric_features'].append(col)
            elif col_type == 'categorical':
                mapping['categorical_features'].append(col)
            elif col_type == 'geographic':
                mapping['geographic_features'].append(col)
            elif col_type == 'binary':
                mapping['binary_features'].append(col)
                
        print(f" Feature mapping generato:")
        print(f"   Numeriche: {len(mapping['numeric_features'])}")
        print(f"   Categoriche: {len(mapping['categorical_features'])}")
        print(f"   Geografiche: {len(mapping['geographic_features'])}")
        print(f"   Binarie: {len(mapping['binary_features'])}")
        print(f"   Target: {mapping['target_feature']}")
        
        # Salva mapping
        mapping_file = self.reports_dir / "feature_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
            
        print(f"\n Feature mapping salvato: {mapping_file}")
        
        return mapping
        
    def run_full_analysis(self):
        """Esegue l'analisi completa"""
        print(" ANALISI ESPLORATIVA DATI - RICHTER PREDICTOR")
        print("=" * 80)
        
        # 1. Carica dati
        self.load_data()
        
        # 2. Analizza tipi
        analysis = self.analyze_data_types()
        
        # 3. Analizza target
        self.analyze_target()
        
        # 4. Analizza features numeriche
        self.analyze_numeric_features(analysis)
        
        # 5. Analizza features categoriche
        self.analyze_categorical_features(analysis)
        
        # 6. Analizza features geografiche
        geo_info = self.analyze_geographic_features(analysis)
        
        # 7. Analizza correlazioni
        self.analyze_correlations(analysis)
        
        # 8. Suggerimenti preprocessing
        suggestions = self.suggest_preprocessing_strategy(analysis)
        
        # 9. Genera feature mapping
        mapping = self.generate_feature_mapping(analysis)
        
        print(f"\n ANALISI COMPLETATA!")
        print(f" Report salvati in: {self.reports_dir}")
        
        return {
            'analysis': analysis,
            'geo_info': geo_info,
            'suggestions': suggestions,
            'mapping': mapping
        }

def main():
    """Funzione principale"""
    analyzer = DataAnalyzer()
    results = analyzer.run_full_analysis()
    
    print(f"\n PROSSIMI PASSI:")
    print(f"1. Rivedi i report generati in reports/eda/")
    print(f"2. Usa feature_mapping.json per build_preprocessing_model()")
    print(f"3. Implementa la pipeline TensorFlow basata sui suggerimenti")
    
    return results

if __name__ == "__main__":
    results = main()
