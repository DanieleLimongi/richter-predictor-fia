# make_dataset_tf.py
"""
Crea un dataset "interim" adatto alla pipeline TensorFlow/GPU.
Operazioni svolte:
  1. Carica train_values.csv e train_labels.csv
  2. Effettua il merge su building_id
  3. Applica ottimizzazione delle soglie per geo_level_2_id e geo_level_3_id
  4. Converte i tipi di colonna (numeriche -> float32, categoriche -> int32)
  5. Salva in due formati:
       • Parquet (facile da ispezionare con pandas)
       • TFRecord (opzionale, per tf.data più veloce)

NOTA: non fa One‑Hot, non fa scaling. Tutta la trasformazione
verrà affidata ai Keras preprocessing layer.
"""

from pathlib import Path
import pandas as pd
import tensorflow as tf
import numpy as np
import json
import sys
import os

# Aggiungi il path per importare il modulo di ottimizzazione
sys.path.append(str(Path(__file__).parent.parent))
from features.categorical_threshold_search import CategoricalThresholdOptimizer

# ------------------------------------------------------------
# Caricamento configurazione
# ------------------------------------------------------------
CONFIG_PATH = Path("config/dataset_config.json")
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    print(" Configurazione caricata da config/dataset_config.json")
else:
    # Configurazione di default
    config = {
        "threshold_optimization": {
            "enable": True,
            "random_search": {"n_iterations": 30},
            "manual_thresholds": {"use_manual": False}
        },
        "paths": {
            "raw_data": "data/raw",
            "interim_data": "data/interim_tf",
            "reports": "reports/threshold_optimization"
        }
    }
    print("  Configurazione di default utilizzata (config/dataset_config.json non trovato)")

# ------------------------------------------------------------
# Percorsi (da configurazione)
# ------------------------------------------------------------
RAW_DIR = Path(config["paths"]["raw_data"])
INTERIM_DIR = Path(config["paths"]["interim_data"])
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path(config["paths"]["reports"])
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_VALUES = RAW_DIR / "train_values.csv"
TRAIN_LABELS = RAW_DIR / "train_labels.csv"

PARQUET_OUT = INTERIM_DIR / "train_interim.parquet"
TFRECORD_OUT = INTERIM_DIR / "train_interim.tfrecord"
META_OUT = INTERIM_DIR / "dtype_mapping.json"
THRESHOLD_RESULTS = REPORTS_DIR / "threshold_search_results.json"

# ------------------------------------------------------------
# Funzioni di utilità per l'ottimizzazione delle soglie
# ------------------------------------------------------------
def apply_categorical_thresholds(df, config):
    """
    Applica ottimizzazione delle soglie per geo_level_2_id e geo_level_3_id
    usando la configurazione fornita.
    
    Args:
        df: DataFrame con i dati
        config: dizionario di configurazione
        
    Returns:
        DataFrame con soglie applicate, dizionario con parametri usati
    """
    df_processed = df.copy()
    threshold_config = config["threshold_optimization"]
    
    # Controlla se usare soglie manuali
    if threshold_config["manual_thresholds"]["use_manual"]:
        print(" Utilizzo soglie manuali dalla configurazione...")
        geo2_threshold = threshold_config["manual_thresholds"]["geo_level_2_id"]
        geo3_threshold = threshold_config["manual_thresholds"]["geo_level_3_id"]
        
    elif threshold_config["enable"]:
        print(" Esecuzione ottimizzazione soglie categoriche...")
        
        # Controllo se esistono già risultati salvati
        if THRESHOLD_RESULTS.exists():
            print(" Caricamento risultati precedenti...")
            with open(THRESHOLD_RESULTS, 'r') as f:
                results = json.load(f)
            best_params = results['best_params']
            print(f" Risultati caricati - Score: {results['best_score']:.4f}")
        else:
            print(" Avvio random search per ottimizzazione soglie...")
            
            # PARAMETRI NESTED CV dalla configurazione
            validation_config = threshold_config.get("validation", {})
            use_nested_cv = validation_config.get("use_nested_cv", True)
            outer_splits = validation_config.get("outer_splits", 5)
            inner_splits = validation_config.get("inner_splits", 3)
            
            optimizer = CategoricalThresholdOptimizer(
                data_path=str(RAW_DIR),
                use_nested_cv=use_nested_cv,
                outer_splits=outer_splits,
                inner_splits=inner_splits
            )
            
            # Parametri da configurazione
            n_iterations = threshold_config["random_search"]["n_iterations"]
            geo2_range = (
                threshold_config["geo_level_2_id"]["threshold_range"]["min"],
                threshold_config["geo_level_2_id"]["threshold_range"]["max"]
            )
            geo3_range = (
                threshold_config["geo_level_3_id"]["threshold_range"]["min"],
                threshold_config["geo_level_3_id"]["threshold_range"]["max"]
            )
            
            results = optimizer.random_search(
                n_iterations=n_iterations,
                geo2_range=geo2_range,
                geo3_range=geo3_range
            )
            
            # Salva risultati
            with open(THRESHOLD_RESULTS, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            best_params = results['best_params']
            print(f" Ottimizzazione completata - Score: {results['best_score']:.4f}")
            
        geo2_threshold = best_params['geo2_threshold']
        geo3_threshold = best_params['geo3_threshold']
        
    else:
        print(" Ottimizzazione soglie disabilitata - uso valori di default")
        geo2_threshold = 0.01  # 1%
        geo3_threshold = 0.005  # 0.5%
            
    print(f" Applicazione soglie:")
    print(f"   - geo_level_2_id: {geo2_threshold:.4f} ({geo2_threshold*100:.2f}%)")
    print(f"   - geo_level_3_id: {geo3_threshold:.4f} ({geo3_threshold*100:.2f}%)")
    
    # Applica soglie
    def _apply_threshold(series, threshold):
        value_counts = series.value_counts(normalize=True)
        rare_values = value_counts[value_counts < threshold].index
        result = series.copy()
        result[result.isin(rare_values)] = -999  # Valore speciale per "OTHER"
        return result
    
    # Statistiche prima dell'applicazione
    geo2_orig_cats = df_processed['geo_level_2_id'].nunique()
    geo3_orig_cats = df_processed['geo_level_3_id'].nunique()
    
    # Applica soglie
    df_processed['geo_level_2_id'] = _apply_threshold(df_processed['geo_level_2_id'], geo2_threshold)
    df_processed['geo_level_3_id'] = _apply_threshold(df_processed['geo_level_3_id'], geo3_threshold)
    
    # Statistiche dopo l'applicazione
    geo2_new_cats = df_processed['geo_level_2_id'].nunique()
    geo3_new_cats = df_processed['geo_level_3_id'].nunique()
    geo2_other_pct = (df_processed['geo_level_2_id'] == -999).mean() * 100
    geo3_other_pct = (df_processed['geo_level_3_id'] == -999).mean() * 100
    
    print(f" Risultati applicazione soglie:")
    print(f"   - geo_level_2_id: {geo2_orig_cats} → {geo2_new_cats} categorie ({geo2_other_pct:.1f}% OTHER)")
    print(f"   - geo_level_3_id: {geo3_orig_cats} → {geo3_new_cats} categorie ({geo3_other_pct:.1f}% OTHER)")
    
    params_used = {
        'geo2_threshold': geo2_threshold,
        'geo3_threshold': geo3_threshold,
        'geo2_categories_before': geo2_orig_cats,
        'geo2_categories_after': geo2_new_cats,
        'geo3_categories_before': geo3_orig_cats,  
        'geo3_categories_after': geo3_new_cats,
        'geo2_other_percentage': geo2_other_pct,
        'geo3_other_percentage': geo3_other_pct,
        'optimization_enabled': threshold_config["enable"],
        'manual_thresholds_used': threshold_config["manual_thresholds"]["use_manual"]
    }
    
    return df_processed, params_used

# ------------------------------------------------------------
# 1. Caricamento CSV e merge
# ------------------------------------------------------------
print(" Caricamento CSV…")
X = pd.read_csv(TRAIN_VALUES)
y = pd.read_csv(TRAIN_LABELS)

df = X.merge(y, on="building_id", how="left", validate="one_to_one")
print(f" Shape merged: {df.shape}")

# ------------------------------------------------------------
# 2. Ottimizzazione soglie categoriche (NUOVA FUNZIONALITÀ)
# ------------------------------------------------------------
df, threshold_params = apply_categorical_thresholds(df, config)

# ------------------------------------------------------------
# 3. Definizione mapping tipi
# ------------------------------------------------------------
cat_cols = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "legal_ownership_status",
    "geo_level_1_id", "geo_level_2_id", "geo_level_3_id",
]

num_cols = [
    "age", "area_percentage", "height_percentage", "count_floors_pre_eq", "count_families",
]

binary_cols = [c for c in df.columns if c.startswith("has_") or c.startswith("has_secondary")]

target_col = "damage_grade"

# Casting
print(" Casting tipi di dato…")
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes.astype("int32")
for col in num_cols:
    df[col] = df[col].astype("float32")
for col in binary_cols:
    df[col] = df[col].astype("int8")

df[target_col] = df[target_col].astype("int8")

# Salva mapping (include anche i parametri delle soglie)
mapping = {
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "binary_cols": binary_cols,
    "target": target_col,
    "threshold_params": threshold_params  # AGGIUNTO: parametri delle soglie
}
META_OUT.write_text(json.dumps(mapping, indent=2))
print(f" Salvato mapping colonna → lista in {META_OUT}")

# ------------------------------------------------------------
# 4. Salvataggio Parquet
# ------------------------------------------------------------
print(" Scrittura Parquet…")
df.to_parquet(PARQUET_OUT, index=False)
print(f" Parquet scritto → {PARQUET_OUT}")
