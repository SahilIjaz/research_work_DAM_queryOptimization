"""
Simple LightGBM Model Training - Fast Version
Trains LightGBM on TPC-H dataset without heavy transformers
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("\n" + "="*80)
print("🚀 LightGBM QUERY COST ESTIMATION - SIMPLE TRAINING")
print("="*80)

# Step 1: Generate TPC-H synthetic data
print("\n📊 Generating TPC-H dataset...")
np.random.seed(42)
n_samples = 1000

templates = [f"TPC-H-Q{i}" for i in range(1, 23)]

data = {
    'query_id': range(n_samples),
    'query_template': np.random.choice(templates, n_samples),

    # Scalar Features
    'estimated_rows': np.random.exponential(scale=1000, size=n_samples),
    'actual_rows': np.random.exponential(scale=1000, size=n_samples),
    'startup_cost': np.random.exponential(scale=100, size=n_samples),
    'total_cost': np.random.exponential(scale=500, size=n_samples),
    'actual_startup_time': np.random.exponential(scale=10, size=n_samples),
    'actual_total_time': np.random.exponential(scale=50, size=n_samples),
    'plan_width': np.random.randint(100, 5000, n_samples),
    'loops': np.random.randint(1, 20, n_samples),
    'base_cardinality': np.random.exponential(scale=500, size=n_samples),
    'output_cardinality': np.random.exponential(scale=500, size=n_samples),
    'input_cardinality': np.random.exponential(scale=500, size=n_samples),
    'filter_selectivity': np.random.uniform(0, 1, n_samples),
    'join_ratio': np.random.uniform(0, 1, n_samples),
    'cost_per_row': np.random.exponential(scale=0.5, size=n_samples),
    'time_per_loop': np.random.exponential(scale=5, size=n_samples),

    # Structural Features
    'node_type': np.random.choice(['Seq Scan', 'Hash Join', 'Nested Loop', 'Index Scan'], n_samples),
    'node_depth': np.random.randint(1, 15, n_samples),
    'parallel_awareness': np.random.choice([0, 1], n_samples),
    'subtree_cardinality': np.random.exponential(scale=1000, size=n_samples),
    'num_children': np.random.randint(0, 5, n_samples),

    # Target
    'actual_runtime_ms': np.random.exponential(scale=100, size=n_samples) + 50,
}

df = pd.DataFrame(data)
df.to_csv('data/raw/tpc_h_queries.csv', index=False)
print(f"✅ Generated {len(df)} query samples")

# Step 2: Feature Engineering
print("\n🔧 Engineering features...")
df['cost_to_row_ratio'] = df['total_cost'] / (df['actual_rows'] + 1)
df['estimated_actual_ratio'] = df['estimated_rows'] / (df['actual_rows'] + 1)
df['time_cost_ratio'] = df['actual_total_time'] / (df['total_cost'] + 1)
print(f"✅ Created derived features")

# Step 3: Data Preprocessing
print("\n⚙️  Preprocessing data...")
df_processed = df.copy()

# Encode categorical
label_encoders = {}
for col in ['query_template', 'node_type']:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

# Scale numeric
numeric_cols = [col for col in df_processed.columns if col not in
               ['query_id', 'actual_runtime_ms', 'query_template', 'node_type'] and
               df_processed[col].dtype in ['float64', 'int64']]

scaler = StandardScaler()
df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
print(f"✅ Encoded categoricals and scaled {len(numeric_cols)} numeric features")

# Step 4: Train-Test Split
print("\n📈 Splitting data (80-20 stratified)...")
feature_cols = [col for col in df_processed.columns if col not in ['query_id', 'actual_runtime_ms']]
X = df_processed[feature_cols]
y = df_processed['actual_runtime_ms']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Testing: {len(X_test)} samples")

# Step 5: Train LightGBM
print("\n🚀 Training LightGBM model...")
train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
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

lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=50)]
)
print("✅ LightGBM training complete")

# Step 6: Train XGBoost baseline
print("\n🚀 Training XGBoost baseline...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.1,
    'max_depth': 7,
    'subsample': 0.8,
}

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    verbose_eval=False
)
print("✅ XGBoost training complete")

# Step 7: Evaluate Models
print("\n📊 EVALUATING MODELS ON TEST SET...")

dtest = xgb.DMatrix(X_test)

lgb_pred = lgb_model.predict(X_test)
xgb_pred = xgb_model.predict(dtest)

def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    rel_error = np.abs((y_true - y_pred) / (y_true + 1e-10))
    accuracy_10 = (rel_error <= 0.1).sum() / len(y_true) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'accuracy_10': accuracy_10,
        'predictions': y_pred,
    }

lgb_results = calc_metrics(y_test, lgb_pred)
xgb_results = calc_metrics(y_test, xgb_pred)

# Step 8: Compare Results
print("\n" + "="*80)
print("📊 MODEL COMPARISON RESULTS")
print("="*80)

comparison = pd.DataFrame({
    'LightGBM': {
        'MSE': f"{lgb_results['mse']:.4f}",
        'RMSE': f"{lgb_results['rmse']:.4f}",
        'MAE': f"{lgb_results['mae']:.2f}",
        'MAPE': f"{lgb_results['mape']:.2f}%",
        'R² Score': f"{lgb_results['r2']:.4f}",
        '±10% Accuracy': f"{lgb_results['accuracy_10']:.2f}%",
    },
    'XGBoost': {
        'MSE': f"{xgb_results['mse']:.4f}",
        'RMSE': f"{xgb_results['rmse']:.4f}",
        'MAE': f"{xgb_results['mae']:.2f}",
        'MAPE': f"{xgb_results['mape']:.2f}%",
        'R² Score': f"{xgb_results['r2']:.4f}",
        '±10% Accuracy': f"{xgb_results['accuracy_10']:.2f}%",
    },
    'Paper Baseline (XGBoost)': {
        'MSE': '0.3002',
        'RMSE': '0.5479',
        'MAE': '142.8',
        'MAPE': '16.4%',
        'R² Score': '0.9512',
        '±10% Accuracy': '65.52%',
    }
})

print("\n" + comparison.to_string())
print("="*80)

# Performance Analysis
print("\n🏆 PERFORMANCE ANALYSIS:")
print("-" * 80)

r2_diff = (lgb_results['r2'] - xgb_results['r2'])
mse_improvement = ((xgb_results['mse'] - lgb_results['mse']) / xgb_results['mse'] * 100)

print(f"\nLightGBM vs XGBoost:")
print(f"  R² Score: {lgb_results['r2']:.4f} vs {xgb_results['r2']:.4f}")

if r2_diff > 0:
    print(f"  ✅ LightGBM WINS - {abs(r2_diff)*100:.2f}% better R²")
elif r2_diff < 0:
    print(f"  ⚠️  XGBoost WINS - {abs(r2_diff)*100:.2f}% better R²")
else:
    print(f"  🔄 TIED performance")

print(f"\n  MSE: {lgb_results['mse']:.4f} vs {xgb_results['mse']:.4f}")
if mse_improvement > 0:
    print(f"  ✅ LightGBM {mse_improvement:.2f}% better MSE")
else:
    print(f"  ⚠️  XGBoost {abs(mse_improvement):.2f}% better MSE")

print(f"\n  ±10% Accuracy: {lgb_results['accuracy_10']:.2f}% vs {xgb_results['accuracy_10']:.2f}%")
print(f"  MAE: {lgb_results['mae']:.2f} ms vs {xgb_results['mae']:.2f} ms")

print(f"\nvs Research Paper Baseline:")
print(f"  Paper R² = 0.9512")
if lgb_results['r2'] > 0.9512:
    print(f"  ✅ LightGBM EXCEEDS baseline by {(lgb_results['r2']-0.9512)*100:.2f}%")
else:
    print(f"  ⚠️  LightGBM is {(0.9512-lgb_results['r2'])*100:.2f}% below baseline")

print(f"  Paper ±10% Accuracy = 65.52%")
if lgb_results['accuracy_10'] > 65.52:
    print(f"  ✅ LightGBM EXCEEDS baseline by {lgb_results['accuracy_10']-65.52:.2f}%")
else:
    print(f"  ⚠️  LightGBM is {65.52-lgb_results['accuracy_10']:.2f}% below baseline")

print("\n" + "="*80)

# Step 9: Feature Importance
print("\n📊 TOP 10 MOST IMPORTANT FEATURES (LightGBM):")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False).head(10)
print(feature_importance.to_string(index=False))

# Step 10: Save models
print("\n💾 Saving models...")
with open('models/lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
lgb_model.save_model('models/lightgbm_model.txt')
xgb_model.save_model('models/xgboost_model.json')
print("✅ Saved models to models/")

# Step 11: Save results
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'lightgbm': {k: v for k, v in lgb_results.items() if k != 'predictions'},
    'xgboost': {k: v for k, v in xgb_results.items() if k != 'predictions'},
    'comparison': comparison.to_dict(),
    'feature_importance': feature_importance.to_dict('records'),
}

with open('results/training_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

comparison.to_csv('results/model_comparison.csv')
print("✅ Saved results to results/")

# Step 12: Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LightGBM
axes[0, 0].scatter(y_test, lgb_pred, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Runtime (ms)')
axes[0, 0].set_ylabel('Predicted Runtime (ms)')
axes[0, 0].set_title(f'LightGBM (R²={lgb_results["r2"]:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# XGBoost
axes[0, 1].scatter(y_test, xgb_pred, alpha=0.5, s=20, color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Runtime (ms)')
axes[0, 1].set_ylabel('Predicted Runtime (ms)')
axes[0, 1].set_title(f'XGBoost (R²={xgb_results["r2"]:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# Residuals LightGBM
lgb_residuals = y_test.values - lgb_pred
axes[1, 0].scatter(lgb_pred, lgb_residuals, alpha=0.5, s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Runtime (ms)')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('LightGBM: Residuals')
axes[1, 0].grid(True, alpha=0.3)

# Residuals XGBoost
xgb_residuals = y_test.values - xgb_pred
axes[1, 1].scatter(xgb_pred, xgb_residuals, alpha=0.5, s=20, color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Runtime (ms)')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('XGBoost: Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization to results/model_comparison.png")
plt.close()

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\n📁 Output files created:")
print("   • data/raw/tpc_h_queries.csv")
print("   • models/lightgbm_model.pkl")
print("   • models/lightgbm_model.txt")
print("   • models/xgboost_model.json")
print("   • results/model_comparison.csv")
print("   • results/model_comparison.png")
print("   • results/training_results.json")
print("\n" + "="*80 + "\n")
