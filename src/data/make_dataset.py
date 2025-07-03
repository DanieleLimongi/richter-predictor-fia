#!/usr/bin/env python
"""make_dataset.py – Step 0 of the pipeline

* Merge raw CSVs (values + labels)
* Cast classical and geo categoricals to pandas 'category'
* Save single Parquet file in data/interim/

Why keep an *interim* layer?
• Raw → Interim: irreversible, always‑needed fixes (merge, dtype casts).  
• Interim → Processed: experiment‑dependent steps (scaling, encoding…).  
  You can regenerate *processed* quickly without re‑reading big CSVs.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

###############################################################################
# Helpers
###############################################################################


def _human_path(path: Path) -> str:
    """Return a nice relative path for printing if possible."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


###############################################################################
# Paths
###############################################################################
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_VALUES_CSV = RAW_DIR / "train_values.csv"
TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
OUTPUT_FILE = INTERIM_DIR / "train_interim.parquet"

###############################################################################
# 1. Load CSVs
###############################################################################
print("Loading raw CSVs …")
X_train = pd.read_csv(TRAIN_VALUES_CSV)
y_train = pd.read_csv(TRAIN_LABELS_CSV)
print(f"  rows in values : {len(X_train):,}")
print(f"  rows in labels : {len(y_train):,}")

###############################################################################
# 2. Merge on building_id
###############################################################################
print("Merging on building_id …")
df = X_train.merge(y_train, on="building_id", how="left", validate="one_to_one")
print(f"  merged rows     : {len(df):,}")

###############################################################################
# 3. Cast categorical columns
###############################################################################
cat_cols = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status",
]
geo_cols = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]

print("Casting categorical columns → category dtype …")
for col in cat_cols + geo_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

###############################################################################
# 4. Save Parquet
###############################################################################
print("Saving interim dataset →", _human_path(OUTPUT_FILE))
df.to_parquet(OUTPUT_FILE, index=False)
print("✅  Interim dataset written. Run src/features/build_features.py next.")
