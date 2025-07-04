# make_dataset_tf.py
"""
Crea un dataset "interim" adatto alla pipeline TensorFlow/GPU.
Operazioni svolte:
  1. Carica train_values.csv e train_labels.csv
  2. Effettua il merge su building_id
  3. Converte i tipi di colonna (numeriche -> float32, categoriche -> int32)
  4. Salva in due formati:
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

# ------------------------------------------------------------
# Percorsi
# ------------------------------------------------------------
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim_tf")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_VALUES = RAW_DIR / "train_values.csv"
TRAIN_LABELS = RAW_DIR / "train_labels.csv"

PARQUET_OUT = INTERIM_DIR / "train_interim.parquet"
TFRECORD_OUT = INTERIM_DIR / "train_interim.tfrecord"
META_OUT = INTERIM_DIR / "dtype_mapping.json"

# ------------------------------------------------------------
# 1. Caricamento CSV e merge
# ------------------------------------------------------------
print("Caricamento CSV…")
X = pd.read_csv(TRAIN_VALUES)
y = pd.read_csv(TRAIN_LABELS)

df = X.merge(y, on="building_id", how="left", validate="one_to_one")
print("Shape merged:", df.shape)

# ------------------------------------------------------------
# 2. Definizione mapping tipi
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
print("Casting tipi di dato…")
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes.astype("int32")
for col in num_cols:
    df[col] = df[col].astype("float32")
for col in binary_cols:
    df[col] = df[col].astype("int8")

df[target_col] = df[target_col].astype("int8")

# Salva mapping (utile per i preprocessing layer)
mapping = {
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "binary_cols": binary_cols,
    "target": target_col,
}
META_OUT.write_text(json.dumps(mapping, indent=2))
print("Salvato mapping colonna → lista in", META_OUT)

# ------------------------------------------------------------
# 3. Salvataggio Parquet
# ------------------------------------------------------------
print("Scrittura Parquet…")
df.to_parquet(PARQUET_OUT, index=False)
print("Parquet scritto →", PARQUET_OUT)

# ------------------------------------------------------------
# 4. (Opz.) Salvataggio TFRecord
# ------------------------------------------------------------
print("Scrittura TFRecord…")

def _serialize_example(row):
    feature = {}
    # numeriche float32
    for col in num_cols:
        feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[col]]))
    # binarie
    for col in binary_cols:
        feature[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[col])]))
    # categoriche int32
    for col in cat_cols:
        feature[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[col])]))
    # target
    feature[target_col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[target_col])]))

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with tf.io.TFRecordWriter(str(TFRECORD_OUT)) as writer:
    for _, row in df.iterrows():
        writer.write(_serialize_example(row))
print("TFRecord scritto →", TFRECORD_OUT)

print("Dataset interim TensorFlow pronto.")
