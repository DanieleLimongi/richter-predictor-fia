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
from feature_engineering import AdvancedFeatureEngineer

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
    print("\n1 Loading raw data...")
    analyzer = DataAnalyzer()
    df = analyzer.load_data()
    
    # Solo features (senza target)
    feature_cols = [col for col in df.columns if col not in ['building_id', 'damage_grade']]
    X_df_raw = df[feature_cols]
    
    analyze_sparsity(X_df_raw, "RAW DATA")
    
    # 2. Analisi dati raw
    print("\n2 Raw data analysis...")
    
    # Check data types
    numeric_cols = X_df_raw.select_dtypes(include=[np.number]).columns
    categorical_cols = X_df_raw.select_dtypes(include=[object]).columns
    
    print(f"   Numeric columns: {len(numeric_cols)}")
    print(f"   Categorical columns: {len(categorical_cols)}")
    
    analyze_sparsity(X_df_raw, "RAW FEATURES ONLY")
    
    # 3. Applica feature engineering
    print("\n3 Applying new modular feature engineering...")
    
    # Merge back with target for feature engineering
    y = df['damage_grade']
    df_with_target = X_df_raw.copy()
    df_with_target['damage_grade'] = y
    
    # Use new feature engineering architecture
    engineer = AdvancedFeatureEngineer()
    df_enhanced = engineer.fit_transform(df_with_target, 'damage_grade')
    
    # Remove target and building_id
    X_enhanced = df_enhanced.drop(['damage_grade', 'building_id'], axis=1, errors='ignore')
    
    analyze_sparsity(X_enhanced, "ENHANCED DATA (post-feature-engineering)")
    
    # 4. Final preprocessing
    print("\n4 Final data preparation...")
    
    # Ensure all numeric
    for col in X_enhanced.columns:
        if not pd.api.types.is_numeric_dtype(X_enhanced[col]):
            X_enhanced[col] = pd.to_numeric(X_enhanced[col], errors='coerce')
    
    # Clean data
    X_enhanced = X_enhanced.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    # Convert to numpy
    X_final = X_enhanced.values.astype(np.float32)
    
    analyze_sparsity(X_final, "FINAL NUMPY DATA")
    
    # 5. Feature breakdown
    print("\n5 Feature breakdown:")
    print(f"   Original features: {len(X_df_raw.columns)}")
    print(f"   Enhanced features: {len(X_enhanced.columns)}")
    print(f"   Features added: +{len(X_enhanced.columns) - len(X_df_raw.columns)}")
    
    # Sample some feature statistics
    if len(X_enhanced.columns) > 10:
        sample_features = X_enhanced.columns[:10]
        for feature in sample_features:
            values = X_enhanced[feature].values
            sparsity = (np.sum(values == 0) / len(values)) * 100
            print(f"   {feature}: {sparsity:.1f}% sparse")

if __name__ == "__main__":
    main()
