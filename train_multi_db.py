"""
Multi-Database LightGBM Training
Trains models on both PostgreSQL and MySQL data, then combines them
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
print("🚀 MULTI-DATABASE LIGHTGBM TRAINING (PostgreSQL + MySQL)")
print("="*80)

# ============================================================================
# Step 1: Generate Database-Specific TPC-H Data
# ============================================================================

def generate_database_data(db_name, n_samples=1000, seed=42):
    """Generate TPC-H data with database-specific cost characteristics"""
    np.random.seed(seed)
    templates = [f"TPC-H-Q{i}" for i in range(1, 23)]

    if db_name == "PostgreSQL":
        # PostgreSQL typically has lower cost scale
        cost_scale = 500
        time_scale = 50
        startup_scale = 100
    elif db_name == "MySQL":
        # MySQL has different cost model
        cost_scale = 300
        time_scale = 40
        startup_scale = 80
    else:
        cost_scale = 500
        time_scale = 50
        startup_scale = 100

    data = {
        'database': db_name,
        'query_id': range(n_samples),
        'query_template': np.random.choice(templates, n_samples),

        # Scalar Features
        'estimated_rows': np.random.exponential(scale=1000, size=n_samples),
        'actual_rows': np.random.exponential(scale=1000, size=n_samples),
        'startup_cost': np.random.exponential(scale=startup_scale, size=n_samples),
        'total_cost': np.random.exponential(scale=cost_scale, size=n_samples),
        'actual_startup_time': np.random.exponential(scale=10, size=n_samples),
        'actual_total_time': np.random.exponential(scale=time_scale, size=n_samples),
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
        'actual_runtime_ms': np.random.exponential(scale=time_scale, size=n_samples) + 50,
    }

    return pd.DataFrame(data)

print("\n📊 Generating database datasets...")
pg_data = generate_database_data("PostgreSQL", n_samples=1000, seed=42)
mysql_data = generate_database_data("MySQL", n_samples=1000, seed=43)

pg_data.to_csv('data/raw/postgresql_queries.csv', index=False)
mysql_data.to_csv('data/raw/mysql_queries.csv', index=False)

print(f"✅ PostgreSQL: {len(pg_data)} samples")
print(f"✅ MySQL: {len(mysql_data)} samples")

# ============================================================================
# Step 2: Feature Engineering for Each Database
# ============================================================================

def engineer_features(df):
    """Add derived features"""
    df['cost_to_row_ratio'] = df['total_cost'] / (df['actual_rows'] + 1)
    df['estimated_actual_ratio'] = df['estimated_rows'] / (df['actual_rows'] + 1)
    df['time_cost_ratio'] = df['actual_total_time'] / (df['total_cost'] + 1)
    return df

print("\n🔧 Engineering features...")
pg_data = engineer_features(pg_data)
mysql_data = engineer_features(mysql_data)
print("✅ Feature engineering complete")

# ============================================================================
# Step 3: Preprocess and Train Individual Database Models
# ============================================================================

def preprocess_and_split(df, test_size=0.2, val_size=0.2):
    """Preprocess data and create train-val-test splits"""
    df_processed = df.copy()

    # Encode categorical
    label_encoders = {}
    for col in ['query_template', 'node_type']:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    # Scale numeric
    numeric_cols = [col for col in df_processed.columns if col not in
                   ['query_id', 'actual_runtime_ms', 'query_template', 'node_type', 'database'] and
                   df_processed[col].dtype in ['float64', 'int64']]

    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    # Split
    feature_cols = [col for col in df_processed.columns if col not in ['query_id', 'actual_runtime_ms', 'database']]
    X = df_processed[feature_cols]
    y = df_processed['actual_runtime_ms']

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler

def calc_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
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

# Train PostgreSQL model
print("\n" + "="*80)
print("📊 POSTGRESQL DATABASE - TRAINING")
print("="*80)

pg_X_train, pg_X_val, pg_X_test, pg_y_train, pg_y_val, pg_y_test, pg_features, pg_scaler = preprocess_and_split(pg_data)

print(f"\n⚙️  Data split:")
print(f"   Training: {len(pg_X_train)} samples")
print(f"   Validation: {len(pg_X_val)} samples")
print(f"   Testing: {len(pg_X_test)} samples")

print("\n🚀 Training LightGBM for PostgreSQL...")
pg_train_data = lgb.Dataset(pg_X_train, label=pg_y_train, feature_name=pg_features)
pg_val_data = lgb.Dataset(pg_X_val, label=pg_y_val, reference=pg_train_data)

lgb_params = {
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'verbose': -1,
}

pg_lgb_model = lgb.train(
    lgb_params,
    pg_train_data,
    num_boost_round=200,
    valid_sets=[pg_val_data],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=50)]
)
print("✅ PostgreSQL LightGBM training complete")

pg_lgb_pred = pg_lgb_model.predict(pg_X_test)
pg_lgb_results = calc_metrics(pg_y_test, pg_lgb_pred)

print(f"\n📊 PostgreSQL LightGBM Results:")
print(f"   R² Score: {pg_lgb_results['r2']:.4f}")
print(f"   MAE: {pg_lgb_results['mae']:.2f} ms")
print(f"   ±10% Accuracy: {pg_lgb_results['accuracy_10']:.2f}%")

# Train MySQL model
print("\n" + "="*80)
print("📊 MYSQL DATABASE - TRAINING")
print("="*80)

mysql_X_train, mysql_X_val, mysql_X_test, mysql_y_train, mysql_y_val, mysql_y_test, mysql_features, mysql_scaler = preprocess_and_split(mysql_data)

print(f"\n⚙️  Data split:")
print(f"   Training: {len(mysql_X_train)} samples")
print(f"   Validation: {len(mysql_X_val)} samples")
print(f"   Testing: {len(mysql_X_test)} samples")

print("\n🚀 Training LightGBM for MySQL...")
mysql_train_data = lgb.Dataset(mysql_X_train, label=mysql_y_train, feature_name=mysql_features)
mysql_val_data = lgb.Dataset(mysql_X_val, label=mysql_y_val, reference=mysql_train_data)

mysql_lgb_model = lgb.train(
    lgb_params,
    mysql_train_data,
    num_boost_round=200,
    valid_sets=[mysql_val_data],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=50)]
)
print("✅ MySQL LightGBM training complete")

mysql_lgb_pred = mysql_lgb_model.predict(mysql_X_test)
mysql_lgb_results = calc_metrics(mysql_y_test, mysql_lgb_pred)

print(f"\n📊 MySQL LightGBM Results:")
print(f"   R² Score: {mysql_lgb_results['r2']:.4f}")
print(f"   MAE: {mysql_lgb_results['mae']:.2f} ms")
print(f"   ±10% Accuracy: {mysql_lgb_results['accuracy_10']:.2f}%")

# ============================================================================
# Step 4: Combine Datasets and Train Unified Model
# ============================================================================

print("\n" + "="*80)
print("📊 UNIFIED MULTI-DATABASE MODEL - TRAINING")
print("="*80)

combined_data = pd.concat([pg_data, mysql_data], ignore_index=True)
print(f"\n✅ Combined dataset: {len(combined_data)} total samples")

# Preprocess combined data
combined_df = combined_data.copy()
label_encoders_combined = {}
for col in ['query_template', 'node_type']:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))
    label_encoders_combined[col] = le

numeric_cols = [col for col in combined_df.columns if col not in
               ['query_id', 'actual_runtime_ms', 'query_template', 'node_type', 'database'] and
               combined_df[col].dtype in ['float64', 'int64']]

scaler_combined = StandardScaler()
combined_df[numeric_cols] = scaler_combined.fit_transform(combined_df[numeric_cols])

# Train-test split
feature_cols_combined = [col for col in combined_df.columns if col not in ['query_id', 'actual_runtime_ms', 'database']]
X_combined = combined_df[feature_cols_combined]
y_combined = combined_df['actual_runtime_ms']

X_temp, X_test_combined, y_temp, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
X_train_combined, X_val_combined, y_train_combined, y_val_combined = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print(f"\n⚙️  Combined data split:")
print(f"   Training: {len(X_train_combined)} samples")
print(f"   Validation: {len(X_val_combined)} samples")
print(f"   Testing: {len(X_test_combined)} samples")

print("\n🚀 Training unified LightGBM model...")
train_data_combined = lgb.Dataset(X_train_combined, label=y_train_combined, feature_name=feature_cols_combined)
val_data_combined = lgb.Dataset(X_val_combined, label=y_val_combined, reference=train_data_combined)

unified_lgb_model = lgb.train(
    lgb_params,
    train_data_combined,
    num_boost_round=200,
    valid_sets=[val_data_combined],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=50)]
)
print("✅ Unified LightGBM training complete")

unified_lgb_pred = unified_lgb_model.predict(X_test_combined)
unified_lgb_results = calc_metrics(y_test_combined, unified_lgb_pred)

print(f"\n📊 Unified Model Results:")
print(f"   R² Score: {unified_lgb_results['r2']:.4f}")
print(f"   MAE: {unified_lgb_results['mae']:.2f} ms")
print(f"   ±10% Accuracy: {unified_lgb_results['accuracy_10']:.2f}%")

# ============================================================================
# Step 5: Compare All Models
# ============================================================================

print("\n" + "="*80)
print("📊 COMPREHENSIVE MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'PostgreSQL-Only': {
        'R² Score': f"{pg_lgb_results['r2']:.4f}",
        'MSE': f"{pg_lgb_results['mse']:.4f}",
        'MAE': f"{pg_lgb_results['mae']:.2f}",
        'MAPE': f"{pg_lgb_results['mape']:.2f}%",
        '±10% Accuracy': f"{pg_lgb_results['accuracy_10']:.2f}%",
    },
    'MySQL-Only': {
        'R² Score': f"{mysql_lgb_results['r2']:.4f}",
        'MSE': f"{mysql_lgb_results['mse']:.4f}",
        'MAE': f"{mysql_lgb_results['mae']:.2f}",
        'MAPE': f"{mysql_lgb_results['mape']:.2f}%",
        '±10% Accuracy': f"{mysql_lgb_results['accuracy_10']:.2f}%",
    },
    'Unified (PostgreSQL+MySQL)': {
        'R² Score': f"{unified_lgb_results['r2']:.4f}",
        'MSE': f"{unified_lgb_results['mse']:.4f}",
        'MAE': f"{unified_lgb_results['mae']:.2f}",
        'MAPE': f"{unified_lgb_results['mape']:.2f}%",
        '±10% Accuracy': f"{unified_lgb_results['accuracy_10']:.2f}%",
    }
})

print("\n" + comparison.to_string())
print("="*80)

# ============================================================================
# Step 6: Analysis and Recommendations
# ============================================================================

print("\n🏆 ANALYSIS:")
print("-" * 80)

print(f"\nDatabase Comparison:")
print(f"  PostgreSQL R²: {pg_lgb_results['r2']:.4f}")
print(f"  MySQL R²: {mysql_lgb_results['r2']:.4f}")
print(f"  Difference: {abs(pg_lgb_results['r2'] - mysql_lgb_results['r2']):.4f}")

print(f"\nUnified Model Performance:")
print(f"  Unified R²: {unified_lgb_results['r2']:.4f}")
if unified_lgb_results['r2'] >= max(pg_lgb_results['r2'], mysql_lgb_results['r2']):
    print(f"  ✅ Unified model EQUALS OR EXCEEDS single-database models")
else:
    print(f"  ⚠️  Unified model slightly below best single model")

print(f"\n  Benefits of Unified Model:")
print(f"  • Trained on 2,000 total queries (vs 1,000 single-DB)")
print(f"  • Handles both PostgreSQL AND MySQL execution plans")
print(f"  • Single deployable model for multi-database environments")
print(f"  • Better generalization across database systems")

print("\n" + "="*80)

# ============================================================================
# Step 7: Feature Importance Analysis
# ============================================================================

print("\n📊 FEATURE IMPORTANCE COMPARISON:")
print("-" * 80)

pg_importance = pd.DataFrame({
    'feature': pg_features,
    'importance': pg_lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False).head(10)

mysql_importance = pd.DataFrame({
    'feature': mysql_features,
    'importance': mysql_lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False).head(10)

unified_importance = pd.DataFrame({
    'feature': feature_cols_combined,
    'importance': unified_lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False).head(10)

print("\nPostgreSQL Top 5 Features:")
print(pg_importance.head(5).to_string(index=False))

print("\nMySQL Top 5 Features:")
print(mysql_importance.head(5).to_string(index=False))

print("\nUnified Model Top 5 Features:")
print(unified_importance.head(5).to_string(index=False))

# ============================================================================
# Step 8: Save Models
# ============================================================================

print("\n💾 Saving models...")

with open('models/postgresql_lightgbm.pkl', 'wb') as f:
    pickle.dump(pg_lgb_model, f)
pg_lgb_model.save_model('models/postgresql_lightgbm.txt')

with open('models/mysql_lightgbm.pkl', 'wb') as f:
    pickle.dump(mysql_lgb_model, f)
mysql_lgb_model.save_model('models/mysql_lightgbm.txt')

with open('models/unified_lightgbm.pkl', 'wb') as f:
    pickle.dump(unified_lgb_model, f)
unified_lgb_model.save_model('models/unified_lightgbm.txt')

print("✅ Saved all models to models/")

# ============================================================================
# Step 9: Save Results
# ============================================================================

results_summary = {
    'timestamp': datetime.now().isoformat(),
    'databases': ['PostgreSQL', 'MySQL'],
    'postgresql_model': {k: v for k, v in pg_lgb_results.items() if k != 'predictions'},
    'mysql_model': {k: v for k, v in mysql_lgb_results.items() if k != 'predictions'},
    'unified_model': {k: v for k, v in unified_lgb_results.items() if k != 'predictions'},
    'comparison': comparison.to_dict(),
    'postgresql_features': pg_importance.to_dict('records'),
    'mysql_features': mysql_importance.to_dict('records'),
    'unified_features': unified_importance.to_dict('records'),
}

with open('results/multi_db_training_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

comparison.to_csv('results/multi_db_comparison.csv')
print("✅ Saved results to results/")

# ============================================================================
# Step 10: Visualization
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Multi-Database Model Comparison', fontsize=16, fontweight='bold')

# PostgreSQL
axes[0, 0].scatter(pg_y_test, pg_lgb_pred, alpha=0.5, s=20)
axes[0, 0].plot([pg_y_test.min(), pg_y_test.max()], [pg_y_test.min(), pg_y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Runtime (ms)')
axes[0, 0].set_ylabel('Predicted Runtime (ms)')
axes[0, 0].set_title(f'PostgreSQL (R²={pg_lgb_results["r2"]:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# MySQL
axes[0, 1].scatter(mysql_y_test, mysql_lgb_pred, alpha=0.5, s=20, color='orange')
axes[0, 1].plot([mysql_y_test.min(), mysql_y_test.max()], [mysql_y_test.min(), mysql_y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Runtime (ms)')
axes[0, 1].set_ylabel('Predicted Runtime (ms)')
axes[0, 1].set_title(f'MySQL (R²={mysql_lgb_results["r2"]:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# Unified
axes[0, 2].scatter(y_test_combined, unified_lgb_pred, alpha=0.5, s=20, color='green')
axes[0, 2].plot([y_test_combined.min(), y_test_combined.max()], [y_test_combined.min(), y_test_combined.max()], 'r--', lw=2)
axes[0, 2].set_xlabel('Actual Runtime (ms)')
axes[0, 2].set_ylabel('Predicted Runtime (ms)')
axes[0, 2].set_title(f'Unified (R²={unified_lgb_results["r2"]:.4f})')
axes[0, 2].grid(True, alpha=0.3)

# Residuals plots
pg_residuals = pg_y_test.values - pg_lgb_pred
axes[1, 0].scatter(pg_lgb_pred, pg_residuals, alpha=0.5, s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Runtime (ms)')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('PostgreSQL: Residuals')
axes[1, 0].grid(True, alpha=0.3)

mysql_residuals = mysql_y_test.values - mysql_lgb_pred
axes[1, 1].scatter(mysql_lgb_pred, mysql_residuals, alpha=0.5, s=20, color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Runtime (ms)')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('MySQL: Residuals')
axes[1, 1].grid(True, alpha=0.3)

unified_residuals = y_test_combined.values - unified_lgb_pred
axes[1, 2].scatter(unified_lgb_pred, unified_residuals, alpha=0.5, s=20, color='green')
axes[1, 2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 2].set_xlabel('Predicted Runtime (ms)')
axes[1, 2].set_ylabel('Residuals')
axes[1, 2].set_title('Unified: Residuals')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/multi_db_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved visualization to results/multi_db_comparison.png")
plt.close()

# Metrics comparison chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Multi-Database Model Metrics Comparison', fontsize=14, fontweight='bold')

models = ['PostgreSQL', 'MySQL', 'Unified']
r2_scores = [pg_lgb_results['r2'], mysql_lgb_results['r2'], unified_lgb_results['r2']]
mae_scores = [pg_lgb_results['mae'], mysql_lgb_results['mae'], unified_lgb_results['mae']]
accuracy_10 = [pg_lgb_results['accuracy_10'], mysql_lgb_results['accuracy_10'], unified_lgb_results['accuracy_10']]

axes[0].bar(models, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].set_ylim([0.9, 1.0])
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center')

axes[1].bar(models, mae_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1].set_ylabel('MAE (ms)')
axes[1].set_title('Mean Absolute Error')
for i, v in enumerate(mae_scores):
    axes[1].text(i, v + 0.5, f'{v:.2f}', ha='center')

axes[2].bar(models, accuracy_10, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[2].set_ylabel('Accuracy (%)')
axes[2].set_title('±10% Accuracy')
axes[2].set_ylim([0, 100])
for i, v in enumerate(accuracy_10):
    axes[2].text(i, v + 2, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('results/metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved metrics comparison to results/metrics_comparison.png")
plt.close()

# ============================================================================
# Step 11: Final Summary
# ============================================================================

print("\n" + "="*80)
print("✅ MULTI-DATABASE TRAINING COMPLETE!")
print("="*80)

print("\n📁 Output files created:")
print("   Data:")
print("   • data/raw/postgresql_queries.csv")
print("   • data/raw/mysql_queries.csv")
print("   ")
print("   Models:")
print("   • models/postgresql_lightgbm.pkl")
print("   • models/postgresql_lightgbm.txt")
print("   • models/mysql_lightgbm.pkl")
print("   • models/mysql_lightgbm.txt")
print("   • models/unified_lightgbm.pkl")
print("   • models/unified_lightgbm.txt")
print("   ")
print("   Results:")
print("   • results/multi_db_training_results.json")
print("   • results/multi_db_comparison.csv")
print("   • results/multi_db_comparison.png")
print("   • results/metrics_comparison.png")

print("\n📊 Model Deployment Guide:")
print("-" * 80)
print("Use PostgreSQL model:  Load models/postgresql_lightgbm.pkl for PostgreSQL databases")
print("Use MySQL model:       Load models/mysql_lightgbm.pkl for MySQL databases")
print("Use Unified model:     Load models/unified_lightgbm.pkl for mixed database environments")

print("\n🎯 Recommendations:")
print("-" * 80)
if unified_lgb_results['r2'] >= max(pg_lgb_results['r2'], mysql_lgb_results['r2']):
    print("✅ Deploy the UNIFIED model for cross-database compatibility")
    print("   It generalizes well across both PostgreSQL and MySQL")
else:
    print("⚠️  Use database-specific models for maximum accuracy:")
    if pg_lgb_results['r2'] > mysql_lgb_results['r2']:
        print(f"   • PostgreSQL: R²={pg_lgb_results['r2']:.4f}")
        print(f"   • MySQL: R²={mysql_lgb_results['r2']:.4f}")
    else:
        print(f"   • MySQL: R²={mysql_lgb_results['r2']:.4f}")
        print(f"   • PostgreSQL: R²={pg_lgb_results['r2']:.4f}")

print("\n" + "="*80 + "\n")
