#!/usr/bin/env python3
"""
Sparsity Analysis for Richter Predictor FIA

This module analyzes data sparsity (percentage of zeros) throughout the ML pipeline
to monitor memory usage, model performance implications, and feature engineering impact.

Key capabilities:
- Raw data sparsity analysis
- Post-feature engineering sparsity tracking
- Before/after comparison reports
- Per-feature sparsity breakdown
- Performance optimization insights

Usage:
    python src/data/sparsity_analysis.py
    
    # From Docker:
    ./docker-helper.sh debug-sparse

Output:
    - Console: Real-time sparsity analysis
    - reports/sparsity/: Detailed analysis reports
    - reports/sparsity/sparsity_summary.json: Machine-readable results

Why sparsity matters:
- High sparsity (>80%) may benefit from sparse matrix operations
- Low sparsity (<20%) indicates dense feature representations
- Sudden sparsity changes reveal feature engineering issues
- Memory usage scales with density, not just feature count
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union

# Setup path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_analysis import DataAnalyzer
from feature_engineering import AdvancedFeatureEngineer


class SparsityAnalyzer:
    """Comprehensive sparsity analysis for ML pipeline"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "reports" / "sparsity"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'analysis': {},
            'summary': {},
            'recommendations': []
        }
    
    def analyze_data_sparsity(self, data: Union[pd.DataFrame, np.ndarray, Dict], 
                             name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive sparsity statistics for different data types
        
        Args:
            data: Input data (DataFrame, numpy array, or dict)
            name: Descriptive name for this analysis
            
        Returns:
            Dictionary with sparsity statistics
        """
        analysis = {'name': name, 'type': type(data).__name__}
        
        if isinstance(data, dict):
            # Dictionary of tensors/arrays
            total_elements = 0
            zero_elements = 0
            
            for key, tensor in data.items():
                np_array = tensor.numpy() if hasattr(tensor, 'numpy') else tensor
                total_elements += np_array.size
                zero_elements += np.sum(np_array == 0)
            
            analysis.update({
                'shape': f"Dict with {len(data)} tensors",
                'total_elements': int(total_elements),
                'zero_elements': int(zero_elements),
                'sparsity_percent': round((zero_elements / total_elements) * 100, 2) if total_elements > 0 else 0
            })
            
        elif isinstance(data, np.ndarray):
            # Numpy array
            total_elements = data.size
            zero_elements = np.sum(data == 0)
            
            analysis.update({
                'shape': list(data.shape),
                'dtype': str(data.dtype),
                'total_elements': int(total_elements),
                'zero_elements': int(zero_elements),
                'sparsity_percent': round((zero_elements / total_elements) * 100, 2) if total_elements > 0 else 0,
                'memory_mb': round(data.nbytes / (1024 * 1024), 2)
            })
            
        elif isinstance(data, pd.DataFrame):
            # Pandas DataFrame
            total_elements = data.size
            zero_elements = (data == 0).sum().sum()
            null_elements = data.isnull().sum().sum()
            
            # Per-column sparsity for top sparse columns
            column_sparsity = {}
            for col in data.columns:
                col_zeros = (data[col] == 0).sum()
                col_nulls = data[col].isnull().sum()
                col_total = len(data[col])
                col_sparsity = ((col_zeros + col_nulls) / col_total) * 100
                column_sparsity[col] = round(col_sparsity, 2)
            
            # Top 10 most sparse columns
            top_sparse = dict(sorted(column_sparsity.items(), 
                                   key=lambda x: x[1], reverse=True)[:10])
            
            analysis.update({
                'shape': list(data.shape),
                'total_elements': int(total_elements),
                'zero_elements': int(zero_elements),
                'null_elements': int(null_elements),
                'sparsity_percent': round(((zero_elements + null_elements) / total_elements) * 100, 2) if total_elements > 0 else 0,
                'memory_mb': round(data.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                'top_sparse_columns': top_sparse,
                'avg_column_sparsity': round(np.mean(list(column_sparsity.values())), 2)
            })
        
        return analysis
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """Print formatted sparsity analysis"""
        print(f"\n{analysis['name']}:")
        print(f"   Type: {analysis['type']}")
        print(f"   Shape: {analysis['shape']}")
        print(f"   Total elements: {analysis['total_elements']:,}")
        print(f"   Zero elements: {analysis['zero_elements']:,}")
        
        if 'null_elements' in analysis:
            print(f"   Null elements: {analysis['null_elements']:,}")
        
        print(f"   Sparsity: {analysis['sparsity_percent']:.2f}%")
        
        if 'memory_mb' in analysis:
            print(f"   Memory usage: {analysis['memory_mb']:.2f} MB")
        
        if 'top_sparse_columns' in analysis and analysis['top_sparse_columns']:
            print(f"   Top sparse columns:")
            for col, sparsity in list(analysis['top_sparse_columns'].items())[:5]:
                print(f"      {col}: {sparsity:.1f}%")
    
    def generate_recommendations(self) -> list:
        """Generate optimization recommendations based on sparsity analysis"""
        recommendations = []
        
        if 'enhanced_data' in self.results['analysis']:
            enhanced = self.results['analysis']['enhanced_data']
            sparsity = enhanced['sparsity_percent']
            
            if sparsity > 80:
                recommendations.append({
                    'type': 'high_sparsity',
                    'message': f"Very high sparsity ({sparsity:.1f}%) detected. Consider sparse matrix operations or feature selection.",
                    'action': 'Use scipy.sparse matrices or implement feature pruning'
                })
            elif sparsity > 50:
                recommendations.append({
                    'type': 'medium_sparsity',
                    'message': f"Moderate sparsity ({sparsity:.1f}%) detected. Monitor memory usage during training.",
                    'action': 'Consider batch size optimization'
                })
            elif sparsity < 10:
                recommendations.append({
                    'type': 'low_sparsity',
                    'message': f"Low sparsity ({sparsity:.1f}%) indicates dense features. Good for standard ML operations.",
                    'action': 'Standard dense matrix operations are optimal'
                })
        
        # Feature count recommendations
        if 'enhanced_data' in self.results['analysis'] and 'raw_data' in self.results['analysis']:
            raw_features = self.results['analysis']['raw_data']['shape'][1] if isinstance(self.results['analysis']['raw_data']['shape'], list) else 40
            enhanced_features = self.results['analysis']['enhanced_data']['shape'][1] if isinstance(self.results['analysis']['enhanced_data']['shape'], list) else 280
            
            feature_ratio = enhanced_features / raw_features
            if feature_ratio > 10:
                recommendations.append({
                    'type': 'feature_explosion',
                    'message': f"Feature count increased {feature_ratio:.1f}x ({raw_features} → {enhanced_features}). Consider feature selection.",
                    'action': 'Implement feature importance analysis and pruning'
                })
        
        return recommendations
    
    def save_results(self):
        """Save analysis results to JSON file"""
        # Generate recommendations
        self.results['recommendations'] = self.generate_recommendations()
        
        # Calculate summary statistics
        if self.results['analysis']:
            sparsities = [analysis['sparsity_percent'] for analysis in self.results['analysis'].values()]
            self.results['summary'] = {
                'min_sparsity': min(sparsities),
                'max_sparsity': max(sparsities),
                'avg_sparsity': round(np.mean(sparsities), 2),
                'analysis_count': len(sparsities)
            }
        
        # Save to file
        output_file = self.reports_dir / "sparsity_summary.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    def run_complete_analysis(self):
        """Run complete sparsity analysis pipeline"""
        print("Richter Predictor FIA - Sparsity Analysis")
        print("=" * 60)
        
        try:
            # 1. Load raw data
            print("\n1. Loading raw data...")
            analyzer = DataAnalyzer()
            df = analyzer.load_data()
            
            # Extract features only
            feature_cols = [col for col in df.columns if col not in ['building_id', 'damage_grade']]
            X_df_raw = df[feature_cols]
            
            # Analyze raw data
            raw_analysis = self.analyze_data_sparsity(X_df_raw, "Raw Data")
            self.results['analysis']['raw_data'] = raw_analysis
            self.print_analysis(raw_analysis)
            
            # 2. Data type breakdown
            print("\n2. Data type analysis...")
            numeric_cols = X_df_raw.select_dtypes(include=[np.number]).columns
            categorical_cols = X_df_raw.select_dtypes(include=[object]).columns
            
            print(f"   Numeric columns: {len(numeric_cols)}")
            print(f"   Categorical columns: {len(categorical_cols)}")
            
            # 3. Apply feature engineering
            print("\n3. Applying modular feature engineering...")
            
            # Prepare data with target
            y = df['damage_grade']
            df_with_target = X_df_raw.copy()
            df_with_target['damage_grade'] = y
            
            # Apply feature engineering
            engineer = AdvancedFeatureEngineer()
            df_enhanced = engineer.fit_transform(df_with_target, 'damage_grade')
            
            # Remove target and building_id
            X_enhanced = df_enhanced.drop(['damage_grade', 'building_id'], axis=1, errors='ignore')
            
            # Analyze enhanced data
            enhanced_analysis = self.analyze_data_sparsity(X_enhanced, "Enhanced Data (Post Feature Engineering)")
            self.results['analysis']['enhanced_data'] = enhanced_analysis
            self.print_analysis(enhanced_analysis)
            
            # 4. Final preprocessing
            print("\n4. Final data preparation...")
            
            # Ensure numeric types
            for col in X_enhanced.columns:
                if not pd.api.types.is_numeric_dtype(X_enhanced[col]):
                    X_enhanced[col] = pd.to_numeric(X_enhanced[col], errors='coerce')
            
            # Clean data
            X_enhanced_clean = X_enhanced.fillna(0.0).replace([np.inf, -np.inf], 0.0)
            
            # Convert to numpy
            X_final = X_enhanced_clean.values.astype(np.float32)
            
            # Analyze final data
            final_analysis = self.analyze_data_sparsity(X_final, "Final NumPy Data (ML-Ready)")
            self.results['analysis']['final_data'] = final_analysis
            self.print_analysis(final_analysis)
            
            # 5. Feature transformation summary
            print("\n5. Feature transformation summary:")
            original_features = len(X_df_raw.columns)
            final_features = len(X_enhanced.columns)
            features_added = final_features - original_features
            
            print(f"   Original features: {original_features}")
            print(f"   Enhanced features: {final_features}")
            print(f"   Features added: +{features_added}")
            print(f"   Growth ratio: {final_features/original_features:.1f}x")
            
            # Memory comparison
            raw_memory = raw_analysis.get('memory_mb', 0)
            enhanced_memory = enhanced_analysis.get('memory_mb', 0)
            if raw_memory > 0 and enhanced_memory > 0:
                memory_ratio = enhanced_memory / raw_memory
                print(f"   Memory usage: {raw_memory:.1f} MB → {enhanced_memory:.1f} MB ({memory_ratio:.1f}x)")
            
            # 6. Show recommendations
            recommendations = self.generate_recommendations()
            if recommendations:
                print("\n6. Optimization recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec['message']}")
                    print(f"      Action: {rec['action']}")
            
            # Save results
            self.save_results()
            
            print(f"\nSparsity analysis completed successfully!")
            print(f"Check {self.reports_dir} for detailed reports.")
            
        except Exception as e:
            print(f"Error during sparsity analysis: {e}")
            raise


def main():
    """Main entry point"""
    analyzer = SparsityAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()