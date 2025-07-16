# src/models/evaluate_model_tf.py

import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from pathlib import Path

# Config paths
MODEL_PATH = Path("models/keras_tf/richter_gpu_best.keras")
PREPROC_PATH = Path("models/keras_tf/preproc_tf.keras")
TFRECORD_FILE = Path("data/interim_tf/train_interim.tfrecord")
MAPPING_JSON = Path("data/interim_tf/dtype_mapping.json")

BATCH_SIZE = 1024
VALID_FRAC = 0.1

# Load mappings
mapping = json.loads(MAPPING_JSON.read_text())
cat_cols    = mapping["cat_cols"]
num_cols    = mapping["num_cols"]
binary_cols = mapping["binary_cols"]

# Reload model and preprocessing
model = tf.keras.models.load_model(MODEL_PATH)
print("Modello caricato:", MODEL_PATH)

# TFRecord decoding
def build_feature_spec():
    spec = {}
    for col in num_cols:    spec[col] = tf.io.FixedLenFeature([], tf.float32)
    for col in binary_cols: spec[col] = tf.io.FixedLenFeature([], tf.int64)
    for col in cat_cols:    spec[col] = tf.io.FixedLenFeature([], tf.int64)
    spec["damage_grade"] = tf.io.FixedLenFeature([], tf.int64)
    return spec

def parse_fn(proto):
    ex = tf.io.parse_single_example(proto, build_feature_spec())
    label = ex.pop("damage_grade")
    label = tf.cast(label, tf.int32) - 1
    for col in binary_cols:
        ex[col] = tf.cast(ex[col], tf.int8)
    return ex, label

ds = tf.data.TFRecordDataset(str(TFRECORD_FILE))
ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).shuffle(50_000, seed=42)

val_size = int(260_601 * VALID_FRAC)
val_ds = ds.take(val_size).batch(BATCH_SIZE)

# Collect predictions
y_true, y_pred = [], []

for x_batch, y_batch in val_ds:
    preds = model.predict(x_batch)
    pred_labels = np.argmax(preds, axis=1)
    y_true.extend(y_batch.numpy())
    y_pred.extend(pred_labels)

# Compute F1-micro
f1_micro = f1_score(y_true, y_pred, average="micro")
print("F1 Micro:", f1_micro)
