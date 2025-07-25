"""
Advanced Feature Engineering per Richter Predictor
Genera features intelligenti basate su domain knowledge sismico per massimizzare F1-score
INTEGRATO con preprocessing classes esistenti per evitare ridondanza
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer
from sklearn.feature_selection import mutual_info_classif
import warnings
import sys
import os
# Aggiungi path per utilizzare le classi di preprocessing esistenti
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))

warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Feature Engineering avanzato che massimizza le performance"""
    
    def __init__(self, target_encoding_smoothing=100):
        self.target_encoding_smoothing = target_encoding_smoothing
        self.geo_target_means = {}
        self.material_risk_scores = {}
        self.polynomial_features_names = []  # Per tracking polynomial features nel test
        self.binning_info = {}  # Per tracking binning cuts
        self.created_features = set()  # Track tutte le features create in fit_transform
        self.fitted = False
        
    def create_seismic_domain_features(self, df):
        """Features basate su DOMAIN KNOWLEDGE sismico"""
        
        print("   Creating seismic domain features...")
        
        # 1. VULNERABILITY INDEX (età + altezza + materiali)
        if 'age' in df.columns and 'count_floors_pre_eq' in df.columns:
            df['building_vulnerability_index'] = (
                df['age'] * 0.4 + 
                df['count_floors_pre_eq'] * 0.6
            )
        
        # 2. STRUCTURAL COMPLEXITY SCORE
        structure_cols = [col for col in df.columns if 'superstructure' in col]
        if structure_cols:
            df['structural_complexity'] = df[structure_cols].sum(axis=1)
            df['has_mixed_structure'] = (df['structural_complexity'] > 2).astype(int)
            
            # Combinazioni specifiche pericolose
            if 'has_superstructure_adobe_mud' in df.columns and 'has_superstructure_mud_mortar_stone' in df.columns:
                df['has_weak_materials'] = (
                    df['has_superstructure_adobe_mud'] | 
                    df['has_superstructure_mud_mortar_stone']
                ).astype(int)
        
        # 3. SIZE-BASED RISK FACTORS
        if 'area_percentage' in df.columns and 'height_percentage' in df.columns:
            # Building aspect ratio (critico per stabilità sismica)
            df['aspect_ratio'] = df['height_percentage'] / (df['area_percentage'] + 1e-8)
            
            # Volume proxy
            df['building_volume_proxy'] = df['area_percentage'] * df['height_percentage']
            
            # Size category binning
            df['size_category'] = pd.cut(
                df['area_percentage'], 
                bins=[0, 33, 66, 100], 
                labels=[0, 1, 2]
            ).astype(float)
        
        # 4. POPULATION DENSITY RISK
        if 'count_families' in df.columns and 'area_percentage' in df.columns:
            df['family_density'] = df['count_families'] / (df['area_percentage'] + 1e-8)
            df['overcrowding_risk'] = (df['family_density'] > df['family_density'].quantile(0.8)).astype(int)
        
        return df
    
    def create_advanced_interactions(self, df):
        """Interazioni avanzate tra features correlate"""
        
        print("   Creating advanced interactions...")
        
        # Age-based interactions (età è critica per vulnerabilità)
        if 'age' in df.columns:
            for col in ['area_percentage', 'height_percentage', 'count_floors_pre_eq', 'count_families']:
                if col in df.columns:
                    df[f'age_{col}_interaction'] = df['age'] * df[col]
                    
                    # Non-linear interactions
                    df[f'age_sqrt_{col}'] = np.sqrt(df['age'] + 1) * df[col]
        
        # Geometric interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        important_pairs = [
            ('area_percentage', 'height_percentage'),
            ('count_floors_pre_eq', 'height_percentage'), 
            ('age', 'count_floors_pre_eq'),
            ('count_families', 'count_floors_pre_eq')
        ]
        
        for col1, col2 in important_pairs:
            if col1 in numeric_cols and col2 in numeric_cols:
                # Ratio
                df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                
                # Product
                df[f'{col1}_{col2}_product'] = df[col1] * df[col2]
                
                # Difference
                df[f'{col1}_{col2}_diff'] = np.abs(df[col1] - df[col2])
        
        return df
    
    def create_unified_geographic_encoding(self, df, target_col='damage_grade'):
        """
        UNIFIED GEOGRAPHIC ENCODING: Combina approccio standard + weighted intelligentemente
        - Per geo_level_1,2: weighted encoding (più importanti per capacità predittiva)
        - Per geo_level_3: standard encoding (granularità troppo alta per weighting)
        - Riduce ridondanza mantenendo informazione massimale
        """
        
        print("   Creating unified geographic encoding...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        geo_cols = [col for col in df.columns if col.startswith('geo_level_')]
        
        if not geo_cols or target_col not in df.columns:
            return df
        
        # 1. DETERMINA STRATEGIA PER OGNI LIVELLO GEOGRAFICO
        strategy_map = {}
        
        for geo_col in geo_cols:
            unique_values = df[geo_col].nunique()
            
            # Strategia basata su cardinalità e livello
            if 'geo_level_1' in geo_col or 'geo_level_2' in geo_col:
                # Livelli alti: sempre weighted (più informativi)
                strategy_map[geo_col] = 'weighted'
            elif unique_values > 100:
                # Troppi valori unici: standard encoding
                strategy_map[geo_col] = 'standard'
            else:
                # Medio-piccolo: weighted se abbastanza campioni
                min_samples_per_level = len(df) / unique_values
                if min_samples_per_level >= 20:
                    strategy_map[geo_col] = 'weighted'
                else:
                    strategy_map[geo_col] = 'standard'
            
            print(f"     {geo_col} ({unique_values} levels) -> {strategy_map[geo_col]} encoding")
        
        # 2. APPLICA ENCODING BASATO SULLA STRATEGIA
        for geo_col in geo_cols:
            if strategy_map[geo_col] == 'weighted':
                # WEIGHTED ENCODING con RF importance
                self._apply_weighted_encoding(df, geo_col, target_col)
            else:
                # STANDARD TARGET ENCODING
                self._apply_standard_encoding(df, geo_col, target_col)
        
        return df
    
    def _apply_weighted_encoding(self, df, geo_col, target_col):
        """Applica weighted encoding per un singolo geo_col"""
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        try:
            # RF importance
            le = LabelEncoder()
            geo_encoded = le.fit_transform(df[geo_col])
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8, n_jobs=-1)
            rf.fit(geo_encoded.reshape(-1, 1), df[target_col])
            rf_importance = rf.feature_importances_[0]
            
            # Statistiche per zona
            geo_stats = df.groupby(geo_col)[target_col].agg(['mean', 'count', 'std']).fillna(0)
            global_mean = df[target_col].mean()
            global_std = df[target_col].std()
            
            # Peso zona
            geo_stats['deviation'] = np.abs(geo_stats['mean'] - global_mean) / global_std
            geo_stats['freq_weight'] = np.log1p(geo_stats['count'])
            geo_stats['zone_weight'] = (
                geo_stats['deviation'] * geo_stats['freq_weight'] * rf_importance
            )
            
            # Normalizza pesi
            if geo_stats['zone_weight'].max() > 0:
                geo_stats['zone_weight'] = geo_stats['zone_weight'] / geo_stats['zone_weight'].max()
            else:
                geo_stats['zone_weight'] = 0.5
            
            # Target encoding pesato con smoothing adattivo
            base_smoothing = self.target_encoding_smoothing
            
            for geo_value in geo_stats.index:
                count = geo_stats.loc[geo_value, 'count']
                zone_weight = geo_stats.loc[geo_value, 'zone_weight']
                
                # Smoothing adattivo: maggiore peso → meno smoothing
                adaptive_smoothing = base_smoothing * (1 - zone_weight)
                
                smoothed_mean = (
                    geo_stats.loc[geo_value, 'mean'] * count + global_mean * adaptive_smoothing
                ) / (count + adaptive_smoothing)
                
                # Salva mapping per test
                mapping_key = f"{geo_col}_weighted"
                if mapping_key not in self.geo_target_means:
                    self.geo_target_means[mapping_key] = {}
                self.geo_target_means[mapping_key][geo_value] = smoothed_mean
                
                # Salva peso per test
                weight_key = f"{geo_col}_weights"
                if weight_key not in self.geo_target_means:
                    self.geo_target_means[weight_key] = {}
                self.geo_target_means[weight_key][geo_value] = zone_weight
            
            # Crea features
            df[f"{geo_col}_weighted_risk"] = df[geo_col].map(self.geo_target_means[mapping_key]).fillna(global_mean)
            df[f"{geo_col}_predictive_weight"] = df[geo_col].map(self.geo_target_means[weight_key]).fillna(0.5)
            df[f"{geo_col}_weighted_deviation"] = np.abs(df[f"{geo_col}_weighted_risk"] - global_mean)
            
        except Exception as e:
            print(f"     Warning: Weighted encoding failed for {geo_col}: {e}")
            # Fallback to standard
            self._apply_standard_encoding(df, geo_col, target_col)
    
    def _apply_standard_encoding(self, df, geo_col, target_col):
        """Applica standard target encoding per un singolo geo_col"""
        
        geo_stats = df.groupby(geo_col)[target_col].agg(['mean', 'count']).fillna(0)
        global_mean = df[target_col].mean()
        
        # Standard smoothing
        smoothing = self.target_encoding_smoothing
        
        for geo_value in geo_stats.index:
            count = geo_stats.loc[geo_value, 'count']
            geo_mean = geo_stats.loc[geo_value, 'mean']
            
            smoothed_mean = (geo_mean * count + global_mean * smoothing) / (count + smoothing)
            
            # Salva mapping per test
            if geo_col not in self.geo_target_means:
                self.geo_target_means[geo_col] = {}
            self.geo_target_means[geo_col][geo_value] = smoothed_mean
        
        # Crea feature
        df[f"{geo_col}_risk"] = df[geo_col].map(self.geo_target_means[geo_col]).fillna(global_mean)

    def create_material_risk_scores(self, df, target_col='damage_grade'):
        """Risk scoring per materiali di costruzione"""
        
        print("   Creating material risk scores...")
        
        material_cols = ['foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type']
        
        for col in material_cols:
            if col in df.columns and target_col in df.columns:
                # Target encoding per materiali
                material_risk = df.groupby(col)[target_col].mean()
                
                # Z-score rispetto alla media globale
                global_mean = df[target_col].mean()
                global_std = df[target_col].std()
                
                material_risk_zscore = (material_risk - global_mean) / global_std
                
                df[f'{col}_risk_zscore'] = df[col].map(material_risk_zscore).fillna(0)
                
                # Salva per test set
                self.material_risk_scores[col] = material_risk_zscore.to_dict()
        
        return df
    
    def create_polynomial_features(self, df, degree=2, max_features=30):
        """Polynomial features selettive per evitare curse of dimensionality"""
        
        print("   Creating polynomial features...")
        
        # Seleziona solo le top features più correlate
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_col = 'damage_grade'
        
        if target_col in df.columns:
            # Calcola correlazioni solo per colonne senza NaN
            valid_numeric_cols = []
            for col in numeric_cols:
                if not df[col].isna().all() and df[col].notna().sum() > 0:
                    valid_numeric_cols.append(col)
            
            if len(valid_numeric_cols) < 2:
                return df  # Non abbastanza features valide
            
            # Calcola correlazioni
            correlations = df[valid_numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            
            # Rimuovi NaN dalla correlazione
            correlations = correlations.dropna()
            
            # Prendi top 8 features per polynomial
            top_features = correlations.head(8).index.tolist()
            
            if len(top_features) >= 2:
                # Riempi NaN con mediana per polynomial features
                poly_data = df[top_features].copy()
                for col in top_features:
                    if poly_data[col].isna().any():
                        median_val = poly_data[col].median()
                        if pd.isna(median_val):
                            median_val = 0  # Fallback se tutto è NaN
                        poly_data[col] = poly_data[col].fillna(median_val)
                
                poly = PolynomialFeatures(
                    degree=degree, 
                    include_bias=False, 
                    interaction_only=True  # Solo interazioni, non potenze
                )
                
                poly_features = poly.fit_transform(poly_data)
                poly_names = poly.get_feature_names_out(top_features)
                
                # Prendi solo le nuove features (non quelle originali)
                original_count = len(top_features)
                new_poly_features = poly_features[:, original_count:]
                new_poly_names = [f"poly_{name}" for name in poly_names[original_count:]]
                
                # Limita numero per evitare overfitting
                if len(new_poly_names) > max_features:
                    # Seleziona le più correlate
                    poly_df_temp = pd.DataFrame(new_poly_features, columns=new_poly_names)
                    poly_corr = poly_df_temp.corrwith(df[target_col]).abs().sort_values(ascending=False)
                    
                    top_poly_features = poly_corr.head(max_features).index
                    selected_indices = [new_poly_names.index(feat) for feat in top_poly_features]
                    
                    new_poly_features = new_poly_features[:, selected_indices]
                    new_poly_names = [new_poly_names[i] for i in selected_indices]
                
                # Aggiungi al dataframe
                poly_df = pd.DataFrame(new_poly_features, columns=new_poly_names, index=df.index)
                df = pd.concat([df, poly_df], axis=1)
                
                # Salva nomi per test consistency
                self.polynomial_features_names = new_poly_names
        
        return df
    
    def create_binning_features(self, df):
        """Binning intelligente per features continue"""
        
        print("   Creating binning features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'damage_grade' and df[col].nunique() > 20:
                
                # Binning quantile-based
                try:
                    binned_series = pd.qcut(
                        df[col], 
                        q=5, 
                        labels=[0, 1, 2, 3, 4],
                        duplicates='drop',
                        retbins=True
                    )
                    
                    df[f'{col}_binned'] = binned_series[0].astype(float)
                    
                    # Salva bin edges per test consistency
                    if hasattr(self, 'fitted') and not self.fitted:  # Solo in fase di training
                        self.binning_info[col] = {
                            'type': 'quantile',
                            'bins': binned_series[1],  # Bin edges
                            'labels': [0, 1, 2, 3, 4]
                        }
                        
                except:
                    # Fallback con cut normale
                    try:
                        binned_series = pd.cut(
                            df[col], 
                            bins=5, 
                            labels=[0, 1, 2, 3, 4],
                            retbins=True
                        )
                        
                        df[f'{col}_binned'] = binned_series[0].astype(float)
                        
                        # Salva bin edges per test consistency
                        if hasattr(self, 'fitted') and not self.fitted:
                            self.binning_info[col] = {
                                'type': 'uniform',
                                'bins': binned_series[1],
                                'labels': [0, 1, 2, 3, 4]
                            }
                            
                    except:
                        # Se fallisce anche questo, salta
                        continue
        
        return df
    
    def create_aggregation_features(self, df):
        """Features aggregate intelligenti"""
        
        print("   Creating aggregation features...")
        
        # Binary features aggregation
        binary_cols = [col for col in df.columns if col.startswith('has_')]
        if binary_cols:
            df['total_binary_features'] = df[binary_cols].sum(axis=1)
            df['binary_diversity_score'] = (df[binary_cols] > 0).sum(axis=1)
            df['binary_density'] = df['total_binary_features'] / len(binary_cols)
        
        # Superstructure aggregations
        superstructure_cols = [col for col in df.columns if 'superstructure' in col]
        if superstructure_cols:
            df['superstructure_count'] = df[superstructure_cols].sum(axis=1)
            df['has_multiple_superstructures'] = (df['superstructure_count'] > 1).astype(int)
        
        # Secondary use aggregations
        secondary_cols = [col for col in df.columns if 'secondary_use' in col]
        if secondary_cols:
            df['secondary_use_count'] = df[secondary_cols].sum(axis=1)
            df['has_secondary_use_flag'] = (df['secondary_use_count'] > 0).astype(int)
        
        return df
    
    def fit_transform(self, train_df, target_col='damage_grade'):
        """Fit e transform completo per training"""
        
        print("Advanced Feature Engineering - Training...")
        df = train_df.copy()
        original_features = len(df.columns)
        
        # 1. Domain-specific features
        df = self.create_seismic_domain_features(df)
        print(f"      Seismic domain features: +{len(df.columns) - original_features}")
        
        # 2. Advanced interactions
        prev_count = len(df.columns)
        df = self.create_advanced_interactions(df)
        print(f"      Advanced interactions: +{len(df.columns) - prev_count}")
        
        # 3. Aggregation features
        prev_count = len(df.columns)
        df = self.create_aggregation_features(df)
        print(f"      Aggregation features: +{len(df.columns) - prev_count}")
        
        # 4. Unified intelligent geographic encoding (combina standard + weighted)
        prev_count = len(df.columns)
        df = self.create_unified_geographic_encoding(df, target_col)
        print(f"      Unified geographic encoding: +{len(df.columns) - prev_count}")
        
        # 5. Material risk scores
        prev_count = len(df.columns)
        df = self.create_material_risk_scores(df, target_col)
        print(f"      Material risk scores: +{len(df.columns) - prev_count}")
        
        # 6. Polynomial features
        prev_count = len(df.columns)
        df = self.create_polynomial_features(df)
        print(f"      Polynomial features: +{len(df.columns) - prev_count}")
        
        # 7. Binning features
        prev_count = len(df.columns)
        df = self.create_binning_features(df)
        print(f"      Binning features: +{len(df.columns) - prev_count}")
        
        total_new_features = len(df.columns) - original_features
        print(f"      TOTAL NEW FEATURES: +{total_new_features}")
        
        # Track tutte le features create (escluse quelle originali)
        original_columns = set(train_df.columns)
        self.created_features = set(df.columns) - original_columns
        
        self.fitted = True
        return df
    
    def transform(self, test_df):
        """Transform per test set usando mapping salvati"""
        
        if not self.fitted:
            raise ValueError("Feature engineer must be fitted before transform!")
        
        print("Advanced Feature Engineering - Test...")
        df = test_df.copy()
        
        # Applica stesso feature engineering ma senza target encoding
        df = self.create_seismic_domain_features(df)
        df = self.create_advanced_interactions(df)
        df = self.create_aggregation_features(df)
        
        # Applica binning usando le info salvate dal training
        df = self._apply_test_binning(df)
        
        # IMPORTANTE: Crea polynomial features consistenti per train/test
        # Per il test set, creiamo le stesse polynomial features con valori dummy
        if hasattr(self, 'polynomial_features_names') and self.polynomial_features_names:
            for poly_name in self.polynomial_features_names:
                if poly_name not in df.columns:
                    df[poly_name] = 0.0  # Dummy values - polynomial requires target correlation
        
        # Applica mapping geografici e materiali salvati (UNIFIED APPROACH)
        global_mean = 2.0  # Default medio
        
        for geo_col, mapping in self.geo_target_means.items():
            if geo_col.endswith('_weighted') and geo_col.startswith('geo_level_'):
                # Weighted geographic features
                base_col = geo_col.replace('_weighted', '')
                if base_col in df.columns:
                    df[f'{base_col}_weighted_risk'] = df[base_col].map(mapping).fillna(global_mean)
                    
            elif geo_col.endswith('_weights') and geo_col.startswith('geo_level_'):
                # Predictive weights
                base_col = geo_col.replace('_weights', '')
                if base_col in df.columns:
                    df[f'{base_col}_predictive_weight'] = df[base_col].map(mapping).fillna(0.5)
                    # Calculate weighted deviation
                    weighted_risk_col = f'{base_col}_weighted_risk'
                    if weighted_risk_col in df.columns:
                        df[f'{base_col}_weighted_deviation'] = np.abs(df[weighted_risk_col] - global_mean)
                        
            elif geo_col.startswith('geo_level_') and not any(suffix in geo_col for suffix in ['_weighted', '_weights', '_std']):
                # Standard geographic encoding
                if geo_col in df.columns:
                    df[f'{geo_col}_risk'] = df[geo_col].map(mapping).fillna(global_mean)
        
        # Applica material risk mapping (CORREZIONE: usa self.material_risk_scores)
        for material_col, mapping in self.material_risk_scores.items():
            if material_col in df.columns:
                df[f'{material_col}_risk_zscore'] = df[material_col].map(mapping).fillna(0)
        
        # CRITICAL: Assicurati che tutte le features create nel training siano presenti nel test
        missing_features = self.created_features - set(df.columns)
        for missing_feature in missing_features:
            print(f"      Adding missing feature: {missing_feature}")
            df[missing_feature] = 0.0  # Default value
        
        print(f"   Test features created: {len(df.columns)} total")
        return df
    
    def _apply_test_binning(self, df):
        """Applica binning al test set usando le info dal training"""
        
        print("   Applying binning features...")
        
        for col, binning_config in self.binning_info.items():
            if col in df.columns:
                try:
                    # Usa i bin edges salvati dal training
                    bins = binning_config['bins']
                    labels = binning_config['labels']
                    
                    # Applica cut con i bin dal training
                    df[f'{col}_binned'] = pd.cut(
                        df[col], 
                        bins=bins, 
                        labels=labels,
                        include_lowest=True
                    ).astype(float)
                    
                except Exception as e:
                    # Fallback: usa valore medio se binning fallisce
                    df[f'{col}_binned'] = 2.0  # Valore centrale
        
        return df
