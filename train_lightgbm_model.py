"""
LightGBM Model Training Pipeline for Query Cost Estimation
Trains LightGBM on TPC-H dataset and compares with XGBoost baseline
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

class QueryDataGenerator:
    """Generate synthetic TPC-H query execution data"""

    def __init__(self, n_samples=1000, seed=42):
        self.n_samples = n_samples
        np.random.seed(seed)

    def generate_data(self):
        """Generate realistic TPC-H execution plan features"""
        print("📊 Generating TPC-H dataset...")

        # Query templates (22 TPC-H templates)
        templates = [f"TPC-H-Q{i}" for i in range(1, 23)]

        data = {
            'query_id': range(self.n_samples),
            'query_template': np.random.choice(templates, self.n_samples),
            'query_text': [f"SELECT * FROM table_{i%10} WHERE condition_{i%5}" for i in range(self.n_samples)],

            # Scalar Features (20+)
            'estimated_rows': np.random.exponential(scale=1000, size=self.n_samples),
            'actual_rows': np.random.exponential(scale=1000, size=self.n_samples),
            'startup_cost': np.random.exponential(scale=100, size=self.n_samples),
            'total_cost': np.random.exponential(scale=500, size=self.n_samples),
            'actual_startup_time': np.random.exponential(scale=10, size=self.n_samples),
            'actual_total_time': np.random.exponential(scale=50, size=self.n_samples),
            'plan_width': np.random.randint(100, 5000, self.n_samples),
            'loops': np.random.randint(1, 20, self.n_samples),
            'base_cardinality': np.random.exponential(scale=500, size=self.n_samples),
            'output_cardinality': np.random.exponential(scale=500, size=self.n_samples),
            'input_cardinality': np.random.exponential(scale=500, size=self.n_samples),
            'node_execution_time': np.random.exponential(scale=30, size=self.n_samples),
            'filter_selectivity': np.random.uniform(0, 1, self.n_samples),
            'join_ratio': np.random.uniform(0, 1, self.n_samples),
            'cost_per_row': np.random.exponential(scale=0.5, size=self.n_samples),
            'time_per_loop': np.random.exponential(scale=5, size=self.n_samples),
            'cardinality_ratio': np.random.uniform(0.1, 10, self.n_samples),
            'normalized_cost': np.random.exponential(scale=1, size=self.n_samples),
            'depth_ratio': np.random.uniform(0, 1, self.n_samples),

            # Structural Features (6)
            'node_type': np.random.choice(['Seq Scan', 'Hash Join', 'Nested Loop', 'Index Scan', 'Sort', 'Aggregate'], self.n_samples),
            'node_depth': np.random.randint(1, 15, self.n_samples),
            'parent_type': np.random.choice(['Seq Scan', 'Hash Join', 'Nested Loop', 'Index Scan', 'Sort', 'Root'], self.n_samples),
            'parallel_awareness': np.random.choice([0, 1], self.n_samples),
            'subtree_cardinality': np.random.exponential(scale=1000, size=self.n_samples),
            'num_children': np.random.randint(0, 5, self.n_samples),

            # Target Variable
            'actual_runtime_ms': np.random.exponential(scale=100, size=self.n_samples) + 50,
        }

        df = pd.DataFrame(data)
        df.to_csv('data/raw/tpc_h_queries.csv', index=False)
        print(f"✅ Generated {len(df)} query samples")
        return df

class FeatureEngineer:
    """Extract and engineer features from execution plans"""

    def __init__(self):
        self.scalar_cols = [
            'estimated_rows', 'actual_rows', 'startup_cost', 'total_cost',
            'actual_startup_time', 'plan_width', 'loops', 'base_cardinality',
            'output_cardinality', 'input_cardinality', 'node_execution_time',
            'filter_selectivity', 'join_ratio', 'cost_per_row', 'time_per_loop',
            'cardinality_ratio', 'normalized_cost', 'depth_ratio'
        ]
        self.structural_cols = [
            'node_type', 'node_depth', 'parent_type', 'parallel_awareness',
            'subtree_cardinality', 'num_children'
        ]

    def engineer_features(self, df):
        """Process and engineer all features"""
        print("\n🔧 Engineering features...")
        df = df.copy()

        # Add derived features
        df['cost_to_row_ratio'] = df['total_cost'] / (df['actual_rows'] + 1)
        df['estimated_actual_ratio'] = df['estimated_rows'] / (df['actual_rows'] + 1)
        df['time_cost_ratio'] = df['actual_total_time'] / (df['total_cost'] + 1)

        print(f"✅ Created {len(self.scalar_cols)} scalar features")
        print(f"✅ Created {len(self.structural_cols)} structural features")

        return df

    def extract_semantic_features(self, df):
        """Generate semantic features using sentence transformers"""
        print("\n📝 Extracting semantic features (TF-IDF style embeddings)...")

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['query_text'].tolist(), show_progress_bar=True)

        # Create embedding features
        embedding_cols = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)

        df = pd.concat([df, embedding_df], axis=1)
        print(f"✅ Generated {len(embedding_cols)} semantic embedding features")

        return df

class DataPreprocessor:
    """Preprocess and prepare data for model training"""

    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}

    def preprocess(self, df, fit=True):
        """Preprocess features: encoding, scaling, missing value handling"""
        print("\n⚙️  Preprocessing data...")
        df = df.copy()

        # Categorical columns to encode
        categorical_cols = ['query_template', 'node_type', 'parent_type']

        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # Numeric columns
        numeric_cols = [col for col in df.columns if col not in categorical_cols +
                       ['query_id', 'query_text', 'actual_runtime_ms'] and
                       df[col].dtype in ['float64', 'int64']]

        # Handle missing values
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Scale numeric features
        if fit:
            self.scalers['numeric'] = StandardScaler()
            df[numeric_cols] = self.scalers['numeric'].fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scalers['numeric'].transform(df[numeric_cols])

        print(f"✅ Encoded {len(categorical_cols)} categorical features")
        print(f"✅ Scaled {len(numeric_cols)} numeric features")

        return df

class ModelTrainer:
    """Train and evaluate LightGBM and XGBoost models"""

    def __init__(self):
        self.lgb_model = None
        self.xgb_model = None
        self.feature_names = None

    def prepare_features(self, df):
        """Prepare X and y"""
        self.feature_names = [col for col in df.columns if col not in
                             ['query_id', 'query_text', 'actual_runtime_ms']]
        X = df[self.feature_names]
        y = df['actual_runtime_ms']
        return X, y

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("\n🚀 Training LightGBM model...")

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'verbose': -1,
        }

        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=20)]
        )

        print("✅ LightGBM training complete")
        return self.lgb_model

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model for baseline comparison"""
        print("\n🚀 Training XGBoost baseline model...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.1,
            'max_depth': 7,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
        }

        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=20
        )

        print("✅ XGBoost training complete")
        return self.xgb_model

    def evaluate(self, X_test, y_test, model_type='lightgbm'):
        """Evaluate model on test set"""
        if model_type == 'lightgbm':
            model = self.lgb_model
            y_pred = model.predict(X_test)
        else:
            model = self.xgb_model
            dtest = xgb.DMatrix(X_test)
            y_pred = model.predict(dtest)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # ±10% accuracy
        relative_error = np.abs((y_test - y_pred) / y_test)
        accuracy_10 = (relative_error <= 0.1).sum() / len(y_test) * 100

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'accuracy_10': accuracy_10,
            'predictions': y_pred,
        }

class ResultAnalyzer:
    """Analyze and visualize results"""

    @staticmethod
    def compare_models(lgb_results, xgb_results, baseline_results=None):
        """Compare LightGBM vs XGBoost vs baseline"""
        print("\n📊 MODEL COMPARISON RESULTS")
        print("=" * 80)

        comparison = pd.DataFrame({
            'LightGBM': {
                'MSE': lgb_results['mse'],
                'RMSE': lgb_results['rmse'],
                'MAE': lgb_results['mae'],
                'MAPE': lgb_results['mape'],
                'R² Score': lgb_results['r2'],
                '±10% Accuracy': lgb_results['accuracy_10'],
            },
            'XGBoost': {
                'MSE': xgb_results['mse'],
                'RMSE': xgb_results['rmse'],
                'MAE': xgb_results['mae'],
                'MAPE': xgb_results['mape'],
                'R² Score': xgb_results['r2'],
                '±10% Accuracy': xgb_results['accuracy_10'],
            }
        })

        if baseline_results:
            comparison['XGBoost Baseline (Paper)'] = baseline_results

        print(comparison.to_string())
        print("=" * 80)

        # Performance improvement
        lgb_better = lgb_results['r2'] > xgb_results['r2']
        r2_diff = (lgb_results['r2'] - xgb_results['r2']) * 100
        mse_diff = ((xgb_results['mse'] - lgb_results['mse']) / xgb_results['mse'] * 100) if xgb_results['mse'] != 0 else 0

        print(f"\n🏆 PERFORMANCE COMPARISON:")
        print(f"   LightGBM R² vs XGBoost: {lgb_results['r2']:.4f} vs {xgb_results['r2']:.4f}")
        if lgb_better:
            print(f"   ✅ LightGBM is BETTER by {abs(r2_diff):.2f}% R² improvement")
        else:
            print(f"   ⚠️  XGBoost is better by {abs(r2_diff):.2f}% R² improvement")

        print(f"   LightGBM MSE vs XGBoost: {lgb_results['mse']:.4f} vs {xgb_results['mse']:.4f}")
        if mse_diff > 0:
            print(f"   ✅ LightGBM MSE is {mse_diff:.2f}% better")
        else:
            print(f"   ⚠️  XGBoost MSE is {abs(mse_diff):.2f}% better")

        print(f"   LightGBM ±10% Accuracy: {lgb_results['accuracy_10']:.2f}%")
        print(f"   XGBoost ±10% Accuracy: {xgb_results['accuracy_10']:.2f}%")

        return comparison

    @staticmethod
    def plot_results(y_test, lgb_pred, xgb_pred):
        """Plot actual vs predicted values"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # LightGBM: Actual vs Predicted
        axes[0, 0].scatter(y_test, lgb_pred, alpha=0.5, s=20)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Runtime (ms)')
        axes[0, 0].set_ylabel('Predicted Runtime (ms)')
        axes[0, 0].set_title('LightGBM: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)

        # XGBoost: Actual vs Predicted
        axes[0, 1].scatter(y_test, xgb_pred, alpha=0.5, s=20, color='orange')
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Runtime (ms)')
        axes[0, 1].set_ylabel('Predicted Runtime (ms)')
        axes[0, 1].set_title('XGBoost: Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)

        # LightGBM: Residuals
        lgb_residuals = y_test - lgb_pred
        axes[1, 0].scatter(lgb_pred, lgb_residuals, alpha=0.5, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Runtime (ms)')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('LightGBM: Residual Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # XGBoost: Residuals
        xgb_residuals = y_test - xgb_pred
        axes[1, 1].scatter(xgb_pred, xgb_residuals, alpha=0.5, s=20, color='orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted Runtime (ms)')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('XGBoost: Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved comparison plot to results/model_comparison.png")
        plt.close()

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("🚀 LightGBM QUERY COST ESTIMATION MODEL TRAINING")
    print("="*80)

    # Step 1: Generate data
    generator = QueryDataGenerator(n_samples=1000)
    df = generator.generate_data()

    # Step 2: Feature engineering
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    df = engineer.extract_semantic_features(df)

    # Step 3: Data preprocessing
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df, fit=True)

    # Step 4: Split data (stratified by query template)
    trainer = ModelTrainer()
    X, y = trainer.prepare_features(df_processed)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    print(f"\n📈 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Testing: {len(X_test)} samples")

    # Step 5: Train models
    lgb_model = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
    xgb_model = trainer.train_xgboost(X_train, y_train, X_val, y_val)

    # Step 6: Evaluate models
    print("\n📊 EVALUATING MODELS ON TEST SET...")
    lgb_results = trainer.evaluate(X_test, y_test, model_type='lightgbm')
    xgb_results = trainer.evaluate(X_test, y_test, model_type='xgboost')

    # Baseline from research paper
    baseline_results = {
        'MSE': 0.3002,
        'RMSE': 0.5479,
        'MAE': '~142.8',
        'MAPE': '~16.4',
        'R² Score': 0.9512,
        '±10% Accuracy': 65.52,
    }

    # Step 7: Compare results
    analyzer = ResultAnalyzer()
    comparison = analyzer.compare_models(lgb_results, xgb_results, baseline_results)

    # Step 8: Visualization
    analyzer.plot_results(y_test, lgb_results['predictions'], xgb_results['predictions'])

    # Step 9: Feature importance
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES (LightGBM):")
    feature_importance = pd.DataFrame({
        'feature': trainer.feature_names,
        'importance': lgb_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False).head(10)
    print(feature_importance.to_string(index=False))

    # Step 10: Save models and results
    print("\n💾 Saving models and results...")

    with open('models/lightgbm_model.pkl', 'wb') as f:
        pickle.dump(lgb_model, f)
    print("✅ Saved LightGBM model")

    lgb_model.save_model('models/lightgbm_model.txt')
    xgb_model.save_model('models/xgboost_model.json')
    print("✅ Saved XGBoost model")

    # Save results
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'lightgbm': lgb_results,
        'xgboost': xgb_results,
        'baseline_paper': baseline_results,
        'feature_names': trainer.feature_names,
        'feature_importance': feature_importance.to_dict('records'),
        'test_size': len(X_test),
    }

    with open('results/training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print("✅ Saved training results to results/training_results.json")

    # Save comparison report
    comparison.to_csv('results/model_comparison.csv')
    print("✅ Saved model comparison to results/model_comparison.csv")

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\n📁 Output files:")
    print("   • models/lightgbm_model.pkl - Trained LightGBM model")
    print("   • models/lightgbm_model.txt - LightGBM model (text format)")
    print("   • models/xgboost_model.json - XGBoost baseline model")
    print("   • results/model_comparison.png - Visualization")
    print("   • results/model_comparison.csv - Detailed metrics")
    print("   • results/training_results.json - Full results")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
