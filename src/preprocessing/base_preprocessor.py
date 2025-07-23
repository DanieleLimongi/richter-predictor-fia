#!/usr/bin/env python3
"""
Base Preprocessor - Richter Predictor
Classe base astratta per tutti i preprocessori specializzati
"""

import tensorflow as tf
from tensorflow.keras import layers
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path


class BasePreprocessor(ABC):
    """
    Classe base astratta per preprocessori specializzati.
    
    Ogni preprocessore gestisce un tipo specifico di features:
    - NumericPreprocessor: age, area_percentage, height_percentage
    - GeographicPreprocessor: geo_level_1_id, geo_level_2_id, geo_level_3_id  
    - CategoricalPreprocessor: foundation_type, roof_type, etc.
    - BinaryPreprocessor: has_superstructure_*
    """
    
    def __init__(self, feature_names: List[str], name: str):
        """
        Inizializza il preprocessore base.
        
        Args:
            feature_names: Lista nomi delle features gestite
            name: Nome identificativo del preprocessore
        """
        self.feature_names = feature_names
        self.name = name
        self.is_fitted = False
        self.metadata = {}
        
        print(f" {self.name} inizializzato con {len(feature_names)} features:")
        for feature in feature_names[:5]:  # Mostra prime 5
            print(f"   • {feature}")
        if len(feature_names) > 5:
            print(f"   • ... e altre {len(feature_names) - 5}")
    
    @abstractmethod
    def fit(self, data_dict: Dict[str, tf.Tensor]) -> 'BasePreprocessor':
        """
        Adatta il preprocessore sui dati di training.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            self per method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applica preprocessing ai dati.
        
        Args:
            data_dict: Dict con {feature_name: tf.Tensor}
            
        Returns:
            Dict con features preprocessate
        """
        pass
    
    @abstractmethod
    def get_output_specs(self) -> Dict[str, Tuple[int, str]]:
        """
        Restituisce specifiche degli output per costruzione modello.
        
        Returns:
            Dict con {feature_name: (output_dim, encoding_type)}
        """
        pass
    
    def get_keras_layers(self, inputs: Dict[str, layers.Input]) -> Dict[str, tf.Tensor]:
        """
        Crea layer Keras per questo preprocessore.
        
        Args:
            inputs: Dict con {feature_name: Input layer}
            
        Returns:
            Dict con {feature_name: processed tensor}
        """
        # Implementazione default - può essere sovrascritta
        return {}
    
    def save_metadata(self, filepath: Path):
        """Salva metadati del preprocessore"""
        metadata = {
            'name': self.name,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f" {self.name} metadata salvati: {filepath}")
    
    def load_metadata(self, filepath: Path):
        """Carica metadati del preprocessore"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.metadata = data.get('metadata', {})
        self.is_fitted = data.get('is_fitted', False)
        
        print(f" {self.name} metadata caricati: {filepath}")
    
    def _check_fitted(self):
        """Verifica che il preprocessore sia fitted"""
        if not self.is_fitted:
            raise ValueError(f"{self.name} deve essere fitted prima di transform()")
    
    def _filter_available_features(self, data_dict: Dict[str, tf.Tensor]) -> List[str]:
        """
        Filtra features disponibili nei dati.
        
        Args:
            data_dict: Dict con dati
            
        Returns:
            Lista features disponibili e gestite da questo preprocessore
        """
        available = []
        for feature in self.feature_names:
            if feature in data_dict:
                available.append(feature)
            else:
                print(f"     {feature}: non trovata nei dati")
        
        print(f"    {self.name}: {len(available)}/{len(self.feature_names)} features disponibili")
        return available


class PreprocessorPipeline:
    """
    Pipeline che coordina tutti i preprocessori specializzati.
    """
    
    def __init__(self, preprocessors: List[BasePreprocessor]):
        """
        Inizializza la pipeline.
        
        Args:
            preprocessors: Lista di preprocessori specializzati
        """
        self.preprocessors = preprocessors
        self.is_fitted = False
        
        print(f" Pipeline inizializzata con {len(preprocessors)} preprocessori:")
        for proc in preprocessors:
            print(f"    {proc.name}")
    
    def fit(self, data_dict: Dict[str, tf.Tensor]) -> 'PreprocessorPipeline':
        """Adatta tutti i preprocessori"""
        print(f"\n Fitting pipeline completa...")
        
        for preprocessor in self.preprocessors:
            print(f"\n Fitting {preprocessor.name}...")
            preprocessor.fit(data_dict)
        
        self.is_fitted = True
        print(f"\n Pipeline fitted con successo!")
        return self
    
    def transform(self, data_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Applica tutti i preprocessori"""
        if not self.is_fitted:
            raise ValueError("Pipeline deve essere fitted prima di transform()")
        
        processed = {}
        
        for preprocessor in self.preprocessors:
            preprocessor_output = preprocessor.transform(data_dict)
            processed.update(preprocessor_output)
        
        return processed
    
    def get_all_output_specs(self) -> Dict[str, Tuple[int, str]]:
        """Ottieni specifiche output di tutti i preprocessori"""
        all_specs = {}
        
        for preprocessor in self.preprocessors:
            specs = preprocessor.get_output_specs()
            all_specs.update(specs)
        
        return all_specs
    
    def build_keras_model(self) -> tf.keras.Model:
        """
        Costruisce modello Keras completo con tutti i preprocessori.
        
        Returns:
            Modello Keras di preprocessing
        """
        if not self.is_fitted:
            raise ValueError("Pipeline deve essere fitted prima di build_keras_model()")
        
        print(f"\n  Costruzione modello Keras completo...")
        
        # Crea inputs per tutte le features
        inputs = {}
        all_outputs = []
        
        for preprocessor in self.preprocessors:
            # Inputs per questo preprocessor
            proc_inputs = {}
            
            for feature in preprocessor.feature_names:
                # Determina tipo input basato sul preprocessore
                if isinstance(preprocessor, type(self.preprocessors[0])):  # Esempio
                    dtype = tf.float32
                else:
                    dtype = tf.int32
                
                proc_inputs[feature] = tf.keras.Input(
                    shape=(1,), 
                    name=feature, 
                    dtype=dtype
                )
                inputs[feature] = proc_inputs[feature]
            
            # Ottieni layers processati
            proc_outputs = preprocessor.get_keras_layers(proc_inputs)
            all_outputs.extend(proc_outputs.values())
        
        # Concatena tutti gli output
        if len(all_outputs) > 1:
            concatenated = layers.Concatenate(name='preprocessed_features')(all_outputs)
        else:
            concatenated = all_outputs[0]
        
        # Crea modello finale
        model = tf.keras.Model(
            inputs=inputs, 
            outputs=concatenated, 
            name='richter_preprocessor_pipeline'
        )
        
        print(f" Modello Keras creato!")
        print(f"    Input features: {len(inputs)}")
        print(f"    Output shape: {concatenated.shape}")
        
        return model
    
    def save_all_metadata(self, output_dir: Path):
        """Salva metadati di tutti i preprocessori"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for preprocessor in self.preprocessors:
            filename = f"{preprocessor.name.lower()}_metadata.json"
            preprocessor.save_metadata(output_dir / filename)
        
        # Salva anche metadata della pipeline
        pipeline_metadata = {
            'preprocessors': [proc.name for proc in self.preprocessors],
            'is_fitted': self.is_fitted,
            'total_features': sum(len(proc.feature_names) for proc in self.preprocessors)
        }
        
        with open(output_dir / "pipeline_metadata.json", 'w') as f:
            json.dump(pipeline_metadata, f, indent=2)
        
        print(f" Tutti i metadata salvati in: {output_dir}")
