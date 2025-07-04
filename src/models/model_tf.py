# src/models/model_tf.py
"""
Training finale con GPU RTX-4070.
  • carica il SavedModel di preprocessing
  • costruisce una MLP in mixed-precision
  • tf.data dal TFRecord (shuffle-batch-prefetch)
  • barra di avanzamento tqdm + timer a fine epoca
  • salva best e modello finale in .keras
"""

from pathlib import Path
import json, time
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
from tqdm.keras import TqdmCallback

# ────────── configurazione ───────────────────────────────────
PREPROC_FILE   = Path("models/preproc_tf.keras")
MAPPING_JSON  = Path("data/interim_tf/dtype_mapping.json")
TFRECORD_FILE = Path("data/interim_tf/train_interim.tfrecord")

MODEL_DIR     = Path("models/keras_tf"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR       = Path("logs/fit_gpu")

BATCH_SIZE = 1024
EPOCHS     = 200
VALID_FRAC = 0.1

mixed_precision.set_global_policy("mixed_float16")   # tensor-core

# ────────── 1. carica preprocessing SavedModel ──────────────
print("► Carico preprocessing SavedModel …")
preproc_model = tf.keras.models.load_model(PREPROC_FILE, compile=False)  # Functional

# ────────── 2. crea Input da mapping json ───────────────────
mapping = json.loads(MAPPING_JSON.read_text())
cat_cols    = mapping["cat_cols"]
num_cols    = mapping["num_cols"]
binary_cols = mapping["binary_cols"]

inputs = {}
for col in num_cols:
    inputs[col] = layers.Input(shape=(1,), name=col, dtype="float32")
for col in binary_cols:
    inputs[col] = layers.Input(shape=(1,), name=col, dtype="int8")
for col in cat_cols:
    inputs[col] = layers.Input(shape=(1,), name=col, dtype="int32")

x = preproc_model(inputs)                 # vettore denso float32

# ────────── 3. MLP finale ───────────────────────────────────
y = layers.Dense(256, activation="relu")(x)
y = layers.Dropout(0.3)(y)
y = layers.Dense(128, activation="relu")(y)
y = layers.Dropout(0.2)(y)
out = layers.Dense(3, activation="softmax", dtype="float32")(y)  # cast back

model = models.Model(list(inputs.values()), out, name="richter_mlp_gpu")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
model.summary()

# ────────── 4. tf.data pipeline dal TFRecord ────────────────
def build_feature_spec():
    spec = {}
    for col in num_cols:    spec[col] = tf.io.FixedLenFeature([], tf.float32)
    for col in binary_cols: spec[col] = tf.io.FixedLenFeature([], tf.int64)
    for col in cat_cols:    spec[col] = tf.io.FixedLenFeature([], tf.int64)
    spec["damage_grade"] = tf.io.FixedLenFeature([], tf.int64)
    return spec

feature_spec = build_feature_spec()

def parse_fn(proto):
    ex = tf.io.parse_single_example(proto, feature_spec)

    # 1. estrai e rimappa il label
    label = ex.pop("damage_grade")          # 1,2,3
    label = tf.cast(label, tf.int32) - 1    # 0,1,2

    # 2. cast binarie a int8
    for col in binary_cols:
        ex[col] = tf.cast(ex[col], tf.int8)

    return ex, label

ds = tf.data.TFRecordDataset(str(TFRECORD_FILE))
ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).shuffle(50_000, seed=42)

val_size = int(260_601 * VALID_FRAC)
val_ds   = ds.take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_ds = ds.skip(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ────────── 5. callback (barra + timer) ─────────────────────
class TimeHistory(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None): self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self.t0
        print(f"Epoch {epoch+1:02d} – {dt:.1f}s | "
              f"loss {logs['loss']:.4f}  val_loss {logs['val_loss']:.4f} "
              f"acc {logs['acc']:.4f}")

cb = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_DIR / "richter_gpu_best.keras", save_best_only=True),
    callbacks.TensorBoard(LOG_DIR.as_posix()),
    TqdmCallback(verbose=0),
    TimeHistory(),
]

# ────────── 6. training ─────────────────────────────────────
print("► Avvio training su GPU …")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

# ────────── 7. salvataggio finale ───────────────────────────
model.save(MODEL_DIR / "richter_gpu_final.keras")
print("✅ Modello best e finale salvati in", MODEL_DIR)
