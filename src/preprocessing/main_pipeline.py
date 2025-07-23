#!/usr/bin/env python3
"""
Main Pipeline - Richter Predictor
Integra tutti i preprocessori specializzati in una pipeline unificata
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime

from .base_preprocessor import PreprocessorPipeline
from .geographic_preprocessor import GeographicPreprocessor
from .numeric_preprocessor import NumericPreprocessor  
from .categorical_preprocessor import CategoricalPreprocessor
from .binary_preprocessor import BinaryPreprocessor


class RichterPreprocessingPipeline:
    """
    Pipeline di preprocessing completa per il dataset Richter.
    
    Integra tutti i preprocessori specializzati e crea il modello 
    TensorFlow finale per il preprocessing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza la pipeline completa.
        
        Args:
            config_path: Path al file di configurazione del dataset
        """
        self.config_path = config_path
        self.dataset_config = self._load_dataset_config()
        
        # Feature lists basate sulla configurazione
        self.feature_lists = self._initialize_feature_lists()
        
        # Preprocessori specializzati
        self.preprocessors = {}
        self.pipeline = None
        
        # Modello Keras finale
        self.preprocessing_model = None
        
        # Metadati
        self.is_fitted = False
        self.fit_timestamp = None
        self.preprocessing_stats = {}
    
    def _load_dataset_config(self) -> Dict:
        """Carica configurazione del dataset"""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Configurazione di fallback
        return {
            "feature_classification": {
                "numeric": ["count_families", "count_floors_pre_eq", "age"],
                "geographic": ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"],
                "categorical": [
                    "foundation_type", "roof_type", "ground_floor_type",
                    "other_floor_type", "position", "plan_configuration",
                    "land_surface_condition", "legal_ownership_status"
                ],
                "binary": [
                    "has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
                    "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone",
                    "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
                    "has_superstructure_timber", "has_superstructure_bamboo",
                    "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
                    "has_superstructure_other"
                ]
            }
        }
    
    def _initialize_feature_lists(self) -> Dict[str, List[str]]:
        """Inizializza le liste di features per tipo"""
        classification = self.dataset_config.get("feature_classification", {})
        
        return {
            "numeric": classification.get("numeric", []),
            "geographic": classification.get("geographic", []),  
            "categorical": classification.get("categorical", []),
            "binary": classification.get("binary", [])
        }
    
    def setup_preprocessors(self, 
                           force_embedding_categorical: bool = False,
                           add_binary_count: bool = True,
                           group_binary_correlated: bool = True,
                           outlier_detection: bool = True) -> 'RichterPreprocessingPipeline':
        """
        Configura tutti i preprocessori specializzati.
        
        Args:
            force_embedding_categorical: Forza embedding per tutte le categoriche
            add_binary_count: Aggiungi feature di conteggio binarie
            group_binary_correlated: Raggruppa binarie correlate
            outlier_detection: Abilita rilevamento outliers numerici
            
        Returns:
            self per method chaining
        """
        print("  Configurando preprocessori specializzati...")
        
        # Geographic Preprocessor
        if self.feature_lists["geographic"]:
            self.preprocessors["geographic"] = GeographicPreprocessor(
                self.feature_lists["geographic"]
            )
            print(f"    Geographic: {len(self.feature_lists['geographic'])} features")
        
        # Numeric Preprocessor  
        if self.feature_lists["numeric"]:
            self.preprocessors["numeric"] = NumericPreprocessor(
                self.feature_lists["numeric"],
                handle_outliers=outlier_detection
            )
            print(f"    Numeric: {len(self.feature_lists['numeric'])} features")
        
        # Categorical Preprocessor
        if self.feature_lists["categorical"]:
            self.preprocessors["categorical"] = CategoricalPreprocessor(
                self.feature_lists["categorical"],
                force_embedding=force_embedding_categorical
            )
            print(f"    Categorical: {len(self.feature_lists['categorical'])} features")
        
        # Binary Preprocessor
        if self.feature_lists["binary"]:
            self.preprocessors["binary"] = BinaryPreprocessor(
                self.feature_lists["binary"],
                add_count_feature=add_binary_count,
                group_correlated=group_binary_correlated
            )
            print(f"    Binary: {len(self.feature_lists['binary'])} features")
        
        # Crea pipeline coordinatrice
        self.pipeline = PreprocessorPipeline(list(self.preprocessors.values()))
        
        print(f"    {len(self.preprocessors)} preprocessori configurati!")
        return self
    
    def fit(self, train_data: Dict[str, tf.Tensor]) -> 'RichterPreprocessingPipeline':
        """
        Adatta tutti i preprocessori sui dati di training.
        
        Args:
            train_data: Dict con {feature_name: tf.Tensor}
            
        Returns:
            self per method chaining
        """
        if self.pipeline is None:
            raise ValueError("Pipeline non configurata! Chiama setup_preprocessors() prima.")
        
        print(" Fitting pipeline completa...")
        start_time = datetime.now()
        
        # Fit della pipeline
        self.pipeline.fit(train_data)
        
        # Salva timestamp e statistiche
        self.fit_timestamp = datetime.now()
        fit_duration = (self.fit_timestamp - start_time).total_seconds()
        
        self.preprocessing_stats = {
            "fit_duration_seconds": fit_duration,
            "fit_timestamp": self.fit_timestamp.isoformat(),
            "total_features_processed": sum(len(fl) for fl in self.feature_lists.values()),
            "preprocessors_used": list(self.preprocessors.keys())
        }
        
        print(f"    Pipeline fitted in {fit_duration:.2f}s!")
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Applica preprocessing completo ai dati.
        
        Args:
            data: Dict con {feature_name: tf.Tensor}
            
        Returns:
            Dict con features preprocessate
        """
        if not self.is_fitted:
            raise ValueError("Pipeline non fittata! Chiama fit() prima.")
        
        return self.pipeline.transform(data)
    
    def build_keras_model(self, input_shapes: Optional[Dict[str, Tuple]] = None) -> Model:
        """
        Costruisce modello Keras completo per preprocessing.
        
        Args:
            input_shapes: Dict con {feature_name: shape} se specifiche
            
        Returns:
            Modello Keras per preprocessing
        """
        if not self.is_fitted:
            raise ValueError("Pipeline non fittata! Chiama fit() prima.")
        
        print("  Costruendo modello Keras...")
        
        # Crea input layers
        inputs = {}
        
        for feature_type, features in self.feature_lists.items():
            for feature in features:
                if input_shapes and feature in input_shapes:
                    shape = input_shapes[feature]
                else:
                    # Shape di default basata sul tipo
                    if feature_type in ["numeric", "categorical", "geographic"]:
                        shape = (1,)  # Scalar input
                    else:  # binary
                        shape = (1,)
                
                inputs[feature] = layers.Input(
                    shape=shape,
                    name=f'input_{feature}',
                    dtype=tf.float32 if feature_type == "numeric" else tf.int32
                )
        
        # Applica preprocessing per tipo
        all_outputs = []
        
        for preprocessor_name, preprocessor in self.preprocessors.items():
            # Ottieni output layers dal preprocessore
            preprocessor_outputs = preprocessor.get_keras_layers(inputs)
            
            for feature_name, output_tensor in preprocessor_outputs.items():
                all_outputs.append(output_tensor)
                print(f"   ➕ {feature_name}: {output_tensor.shape}")
        
        # Concatena tutti gli output
        if len(all_outputs) == 1:
            final_output = all_outputs[0]
        else:
            final_output = layers.Concatenate(name='preprocessing_output')(all_outputs)
        
        # Crea modello
        self.preprocessing_model = Model(
            inputs=inputs,
            outputs=final_output,
            name='richter_preprocessing'
        )
        
        print(f"    Modello creato! Output shape: {final_output.shape}")
        
        return self.preprocessing_model
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Ottieni riassunto completo del preprocessing.
        
        Returns:
            Dict con analisi dettagliata
        """
        if not self.is_fitted:
            return {"error": "Pipeline non fittata"}
        
        summary = {
            "pipeline_info": {
                "fit_timestamp": self.fit_timestamp.isoformat() if self.fit_timestamp else None,
                "total_preprocessors": len(self.preprocessors),
                "preprocessing_stats": self.preprocessing_stats
            },
            "feature_breakdown": {},
            "output_specifications": {},
            "recommendations": []
        }
        
        # Analisi per preprocessore
        for name, preprocessor in self.preprocessors.items():
            summary["feature_breakdown"][name] = {
                "feature_count": len(preprocessor.feature_names),
                "features": preprocessor.feature_names,
                "metadata": preprocessor.metadata
            }
            
            # Specifiche output
            output_specs = preprocessor.get_output_specs()
            summary["output_specifications"][name] = output_specs
            
            # Analisi specializzate
            if hasattr(preprocessor, 'analyze_geographic_hierarchy'):
                geo_analysis = preprocessor.analyze_geographic_hierarchy()
                summary["feature_breakdown"][name]["geographic_analysis"] = geo_analysis
            
            if hasattr(preprocessor, 'analyze_categorical_distributions'):
                cat_analysis = preprocessor.analyze_categorical_distributions()
                summary["feature_breakdown"][name]["categorical_analysis"] = cat_analysis
            
            if hasattr(preprocessor, 'analyze_binary_patterns'):
                bin_analysis = preprocessor.analyze_binary_patterns()
                summary["feature_breakdown"][name]["binary_analysis"] = bin_analysis
        
        return summary
    
    def save_pipeline(self, save_dir: str) -> str:
        """
        Salva pipeline completa su disco.
        
        Args:
            save_dir: Directory dove salvare
            
        Returns:
            Path del file salvato
        """
        if not self.is_fitted:
            raise ValueError("Pipeline non fittata! Chiama fit() prima.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Salva summary
        summary = self.get_preprocessing_summary()
        summary_path = os.path.join(save_dir, "preprocessing_summary.json")
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Salva modello Keras se esiste
        if self.preprocessing_model:
            model_path = os.path.join(save_dir, "preprocessing_model.keras")
            self.preprocessing_model.save(model_path)
            print(f"    Modello salvato: {model_path}")
        
        print(f"   Pipeline salvata: {summary_path}")
        return summary_path
    
    def get_feature_engineering_suggestions(self) -> List[str]:
        """
        Ottieni suggerimenti per feature engineering.
        
        Returns:
            Lista di suggerimenti dall'analisi
        """
        suggestions = []
        
        for name, preprocessor in self.preprocessors.items():
            if hasattr(preprocessor, 'suggest_feature_engineering'):
                processor_suggestions = preprocessor.suggest_feature_engineering()
                suggestions.extend([
                    f"[{name.upper()}] {suggestion}" 
                    for suggestion in processor_suggestions
                ])
        
        return suggestions
