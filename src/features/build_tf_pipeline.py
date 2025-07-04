# build_tf_pipeline.py (FIX v1.3 – salva in formato .keras)
"""
Costruisce e salva il preprocessing model in formato **Keras V3 (.keras)**
anziché SavedModel. Così può essere caricato direttamente con
`keras.models.load_model()` in Keras 3.
"""

from pathlib import Path
import json, math
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# ─────────── percorsi ───────────────────────────────────────
DATA_PARQUET = Path("data/interim_tf/train_interim.parquet")
META_JSON    = Path("data/interim_tf/dtype_mapping.json")
MODEL_DIR    = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
PREPROC_FILE = MODEL_DIR / "preproc_tf.keras"   # <‑‑ nuovo formato

print("Carico Parquet…")
df = pd.read_parquet(DATA_PARQUET)
print("Shape:", df.shape)

# ─────────── mapping colonne ────────────────────────────────
mapping      = json.loads(META_JSON.read_text())
cat_cols     = mapping["cat_cols"]
num_cols     = mapping["num_cols"]
binary_cols  = mapping["binary_cols"]
label_col    = mapping["target"]

def emb_dim(card:int)->int:
    return int(min(50, round(1.6*math.sqrt(card))))

inputs, encoded = {}, []
lookup_layers   = {}

# numeriche
for col in num_cols:
    inp  = Input(shape=(1,), name=col, dtype="float32")
    norm = layers.Normalization(name=f"norm_{col}")(inp)
    inputs[col] = inp; encoded.append(norm)

# binarie
# binarie 0/1 -> float32  ✅ senza Lambda
for col in binary_cols:
    inp  = Input(shape=(1,), name=col, dtype="int8")
    cast = layers.Rescaling(scale=1.0, offset=0.0, name=f"cast_{col}")(inp)
    inputs[col] = inp
    encoded.append(cast)


# categoriche
onehot_cols, embed_cols = [], []
for col in cat_cols:
    (onehot_cols if df[col].nunique() <= 50 else embed_cols).append((col, int(df[col].nunique())))

# one‑hot
for col, card in onehot_cols:
    inp = Input(shape=(1,), name=col, dtype="int32")
    lookup = layers.IntegerLookup(max_tokens=card+2, output_mode="one_hot", name=f"lookup_{col}")
    onehot = lookup(inp)
    inputs[col] = inp; encoded.append(onehot)
    lookup_layers[col] = lookup

# embedding
for col, card in embed_cols:
    inp = Input(shape=(1,), name=col, dtype="int32")
    lookup = layers.IntegerLookup(max_tokens=card+2, output_mode="int", name=f"lookup_{col}")
    idx = lookup(inp)
    emb = layers.Embedding(card+2, emb_dim(card))(idx)
    flat = layers.Flatten()(emb)
    inputs[col] = inp; encoded.append(flat)
    lookup_layers[col] = lookup

# ─────────── adattamento layer ──────────────────────────────
print("Adatto Normalization…")
feat_ds = tf.data.Dataset.from_tensor_slices({c: df[c].values for c in df.columns if c != label_col}).batch(2048)
for layer in encoded:
    if isinstance(layer, layers.Normalization):
        col = layer.name.replace("norm_", "")
        layer.adapt(feat_ds.map(lambda x: x[col]))

print("Adatto IntegerLookup…")
for col, lookup in lookup_layers.items():
    lookup.adapt(df[col].values)

# ─────────── build e salva model ────────────────────────────
concat = layers.Concatenate(name="concat_features")(encoded)
preproc_model = Model(inputs, concat, name="preprocessing")
preproc_model.save(PREPROC_FILE)   # salva in formato .keras
print("✅ Preprocessing salvato in", PREPROC_FILE)

