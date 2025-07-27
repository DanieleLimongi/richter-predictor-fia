#!/usr/bin/env python3
"""
Debug script per analizzare la sparsity prima e dopo preprocessing
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os

# Setup path
sys.path.append('src')
sys.path.append('src/data')
sys.path.append('src/preprocessing')

from data.data_analysis import DataAnalyzer
from preprocessing.main_pipeline import RichterPreprocessingPipeline

def analyze_sparsity(data, name):
    """Calcola e stampa statistiche di sparsity"""
    if isinstance(data, dict):
        # Dict di tensori
        total_elements = 0
        zero_elements = 0
        
        for key, tensor in data.items():
            np_array = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
            total_elements += np_array.size
            zero_elements += np.sum(np_array == 0)
            
        sparsity = (zero_elements / total_elements) * 100
        print(f"{name}:")
        print(f"   Total elements: {total_elements:,}")
        print(f"   Zero elements: {zero_elements:,}")
        print(f"   Sparsity: {sparsity:.1f}%")
        
    elif isinstance(data, np.ndarray):
        # Numpy array
        total_elements = data.size
        zero_elements = np.sum(data == 0)
        sparsity = (zero_elements / total_elements) * 100
        print(f"{name}:")
        print(f"   Shape: {data.shape}")
        print(f"   Total elements: {total_elements:,}")
        print(f"   Zero elements: {zero_elements:,}")
        print(f"   Sparsity: {sparsity:.1f}%")
        
    elif isinstance(data, pd.DataFrame):
        # Pandas DataFrame
        total_elements = data.size
        zero_elements = (data == 0).sum().sum()
        null_elements = data.isnull().sum().sum()
        sparsity = ((zero_elements + null_elements) / total_elements) * 100
        print(f"{name}:")
        print(f"   Shape: {data.shape}")
        print(f"   Total elements: {total_elements:,}")
        print(f"   Zero elements: {zero_elements:,}")
        print(f"   Null elements: {null_elements:,}")
        print(f"   Sparsity: {sparsity:.1f}%")

def main():
    print("Debug Sparsity Analysis")
    print("="*50)
    
    # 1. Carica dati raw
    print("\n1️⃣ Loading raw data...")
    analyzer = DataAnalyzer()
    df = analyzer.load_data()
    
    # Solo features (senza target)
    feature_cols = [col for col in df.columns if col not in ['building_id', 'damage_grade']]
    X_df_raw = df[feature_cols]
    
    analyze_sparsity(X_df_raw, "RAW DATA")
    
    # 2. Converti in tensori come fa train_advanced_ensemble
    print("\n2️⃣ Converting to tensors...")
    data_dict = {}
    for col in X_df_raw.columns:
        if X_df_raw[col].dtype == 'object':
            # Categorical features
            data_dict[col] = tf.constant(X_df_raw[col].astype(str).values)
        else:
            # Numeric features
            data_dict[col] = tf.constant(X_df_raw[col].astype(np.float32).values)
    
    analyze_sparsity(data_dict, "TENSOR DATA (pre-preprocessing)")
    
    # 3. Applica preprocessing
    print("\n3️⃣ Applying preprocessing pipeline...")
    pipeline = RichterPreprocessingPipeline()
    pipeline.setup_preprocessors(
        force_embedding_categorical=False,
        add_binary_count=True,
        group_binary_correlated=True,
        outlier_detection=True
    )
    
    pipeline.fit(data_dict)
    processed = pipeline.transform(data_dict)
    
    analyze_sparsity(processed, "PROCESSED DATA (post-preprocessing)")
    
    # 4. Aggrega come fa train_advanced_ensemble
    print("\n4️⃣ Aggregating features...")
    arrays = []
    for tensor in processed.values():
        np_array = tensor.numpy()
        if len(np_array.shape) > 1:
            np_array = np_array.reshape(np_array.shape[0], -1)
        else:
            np_array = np_array.reshape(-1, 1)
        arrays.append(np_array)
    
    X_final = np.concatenate(arrays, axis=1).astype(np.float32)
    X_final = np.nan_to_num(X_final)
    
    analyze_sparsity(X_final, "FINAL AGGREGATED DATA")
    
    # 5. Analisi per tipo di feature
    print("\n5️⃣ Breakdown by feature type:")
    for i, (key, tensor) in enumerate(processed.items()):
        np_array = tensor.numpy()
        if len(np_array.shape) > 1:
            np_array = np_array.reshape(np_array.shape[0], -1)
        else:
            np_array = np_array.reshape(-1, 1)
        
        total_elements = np_array.size
        zero_elements = np.sum(np_array == 0)
        sparsity = (zero_elements / total_elements) * 100
        print(f"   {key}: {np_array.shape} -> {sparsity:.1f}% sparse")

if __name__ == "__main__":
    main()
