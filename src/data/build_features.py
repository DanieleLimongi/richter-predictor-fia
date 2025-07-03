#!/usr/bin/env python
"""build_features.py – Step 1 of the pipeline

Trasforma il dataset *interim* in un dataset *processed* pronto per il modello.

Operazioni implementate
-----------------------
1. Lettura di `data/interim/train_interim.parquet` e split X/y.
2. Selezione di tre gruppi di colonne:
   • *continue_cols*  → variabili numeriche continue con skew elevato.  
   • *binary_cols*    → flag 0/1 (pass-through).                     
   • *cat_cols*       → categoriche classiche + geo_level_*_id.
3. Pipeline `cont_pipeline`:
   • `PowerTransformer(method='yeo-johnson')` per ridurre skew.  
   • `StandardScaler()` per riportare media 0 / var 1.
4. Pipeline `cat_pipeline`:
   • `OneHotEncoder(handle_unknown='ignore', sparse=True)`.
5. `ColumnTransformer` che applica:
   • cont_pipeline alle continue,  
   • cat_pipeline alle categoriche,  
   • "passthrough" alle binarie.
6. Salvataggio:
   • `X_train_processed.npz` (sparse CSR)  
   • `y_train.csv`                     
   • `column_transformer.pkl` (joblib dump)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
from scipy import sparse

###############################################################################
# Percorsi
###############################################################################
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models/artifacts")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = INTERIM_DIR / "train_interim.parquet"

###############################################################################
# 1. Lettura
###############################################################################
print("\n🔹  Lettura dell'interim …")
df = pd.read_parquet(INPUT_FILE)
print(f"    righe: {len(df):,}  colonne: {df.shape[1]}")

target_col = "damage_grade"
X = df.drop(columns=[target_col])
y = df[target_col]

###############################################################################
# 2. Definizione gruppi di feature
###############################################################################
# Colonne numeriche continue con skew > 1 (dalla tua EDA)
continue_cols = [
    "age", "area_percentage", "height_percentage",
    "count_floors_pre_eq", "count_families",
]

# Flag binarie (0/1) – tutte le colonne che iniziano con 'has_' oppure 'has_secondary_'
binary_cols = [c for c in X.columns if c.startswith("has_")]

# Categoriche classiche + geo-ID
cat_cols = [
    "land_surface_condition", "foundation_type", "roof_type",
    "ground_floor_type", "other_floor_type", "position",
    "plan_configuration", "legal_ownership_status",
    "geo_level_1_id", "geo_level_2_id", "geo_level_3_id",
]

print("    continue_cols :", len(continue_cols))
print("    binary_cols   :", len(binary_cols))
print("    cat_cols      :", len(cat_cols))

###############################################################################
# 3. Pipeline per le continue (Yeo-Johnson + StandardScaler)
###############################################################################
cont_pipeline = Pipeline(steps=[
    ("yeojohnson", PowerTransformer(method="yeo-johnson")),
    ("scaler", StandardScaler()),
])

###############################################################################
# 4. Pipeline per le categoriche (One-Hot)
###############################################################################
cat_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

###############################################################################
# 5. ColumnTransformer
###############################################################################
preprocessor = ColumnTransformer(
    transformers=[
        ("cont", cont_pipeline, continue_cols),
        ("cat", cat_pipeline, cat_cols),
        ("bin", "passthrough", binary_cols),
    ],
    sparse_threshold=0.3,
)

###############################################################################
# 6. Fit & transform
###############################################################################
print("\n⚙️  Fitting ColumnTransformer …")
X_proc = preprocessor.fit_transform(X)
print("    shape finale:", X_proc.shape)

###############################################################################
# 7. Salvataggio
###############################################################################
print("\n💾  Salvataggio matrix + target + transformer …")

sparse.save_npz(PROCESSED_DIR / "X_train_processed.npz", X_proc)
y.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
joblib.dump(preprocessor, MODEL_DIR / "column_transformer.pkl")

print("✅  build_features completato. File in data/processed/ e models/artifacts/\n")
