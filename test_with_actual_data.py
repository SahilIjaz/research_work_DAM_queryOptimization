"""
Test Models with Actual Execution Plan Data
Load execution plans from CSV and test all 3 models
"""

import pickle
import pandas as pd
import numpy as np

print("\n" + "="*80)
print("🧪 TESTING MODELS WITH ACTUAL DATABASE EXECUTION PLANS")
print("="*80)

# ============================================================================
# Step 1: Load Your Actual Execution Plan Data
# ============================================================================

print("\n📂 Loading execution plan data...")

# Option 1: If you have a CSV file with actual execution plans
csv_file = 'data/raw/actual_queries.csv'  # Change this to your CSV filename

try:
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} queries from: {csv_file}")
    print(f"   Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ File not found: {csv_file}")
    print("\n📝 To test with actual data:")
    print("   1. Export execution plans from your database to CSV")
    print("   2. Save as: data/raw/actual_queries.csv")
    print("   3. Must include these 24 columns:")
    print("      query_template, estimated_rows, actual_rows, startup_cost,")
    print("      total_cost, actual_startup_time, actual_total_time, plan_width,")
    print("      loops, base_cardinality, output_cardinality, input_cardinality,")
    print("      filter_selectivity, join_ratio, cost_per_row, time_per_loop,")
    print("      node_type, node_depth, parallel_awareness, subtree_cardinality,")
    print("      num_children, cost_to_row_ratio, estimated_actual_ratio,")
    print("      time_cost_ratio")
    exit(1)

# ============================================================================
# Step 2: Load All Models
# ============================================================================

print("\n📦 Loading trained models...")

try:
    with open('models/postgresql_lightgbm.pkl', 'rb') as f:
        pg_model = pickle.load(f)
    print("✅ PostgreSQL model loaded")
    pg_available = True
except FileNotFoundError:
    print("❌ PostgreSQL model not found")
    pg_available = False

try:
    with open('models/mysql_lightgbm.pkl', 'rb') as f:
        mysql_model = pickle.load(f)
    print("✅ MySQL model loaded")
    mysql_available = True
except FileNotFoundError:
    print("❌ MySQL model not found")
    mysql_available = False

try:
    with open('models/unified_lightgbm.pkl', 'rb') as f:
        unified_model = pickle.load(f)
    print("✅ Unified model loaded")
    unified_available = True
except FileNotFoundError:
    print("❌ Unified model not found")
    unified_available = False

# ============================================================================
# Step 3: Prepare Data for Prediction
# ============================================================================

print("\n⚙️  Preparing data...")

# Expected feature columns in correct order
expected_cols = [
    'query_template', 'estimated_rows', 'actual_rows', 'startup_cost', 'total_cost',
    'actual_startup_time', 'actual_total_time', 'plan_width', 'loops',
    'base_cardinality', 'output_cardinality', 'input_cardinality',
    'filter_selectivity', 'join_ratio', 'cost_per_row', 'time_per_loop',
    'node_type', 'node_depth', 'parallel_awareness', 'subtree_cardinality',
    'num_children', 'cost_to_row_ratio', 'estimated_actual_ratio', 'time_cost_ratio'
]

# Check if all columns are present
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    print(f"⚠️  Missing columns: {missing_cols}")
    print("   Adding with default values...")
    for col in missing_cols:
        df[col] = 0

# Reorder and select only needed columns
df = df[expected_cols]

# Handle any categorical columns (encode node_type if string)
if df['node_type'].dtype == 'object':
    node_type_mapping = {
        'Seq Scan': 0, 'Sequential Scan': 0,
        'Index Scan': 1, 'Index Only Scan': 1,
        'Hash Join': 2,
        'Nested Loop': 3, 'Nested Loop Join': 3,
        'Bitmap Index Scan': 4,
        'Bitmap Heap Scan': 5
    }
    df['node_type'] = df['node_type'].map(lambda x: node_type_mapping.get(str(x), 0))

print(f"✅ Data prepared: {len(df)} queries, {len(expected_cols)} features")

# ============================================================================
# Step 4: Make Predictions
# ============================================================================

print("\n" + "="*80)
print("🚀 MAKING PREDICTIONS")
print("="*80)

X = df.values  # Convert to numpy array

if pg_available:
    print("\n📊 PostgreSQL Model Predictions...")
    pg_preds = pg_model.predict(X)
    df['postgresql_pred_ms'] = pg_preds
    print(f"✅ Generated {len(pg_preds)} predictions")

if mysql_available:
    print("\n📊 MySQL Model Predictions...")
    mysql_preds = mysql_model.predict(X)
    df['mysql_pred_ms'] = mysql_preds
    print(f"✅ Generated {len(mysql_preds)} predictions")

if unified_available:
    print("\n📊 Unified Model Predictions...")
    unified_preds = unified_model.predict(X)
    df['unified_pred_ms'] = unified_preds
    print(f"✅ Generated {len(unified_preds)} predictions")

# ============================================================================
# Step 5: Compare Predictions vs Actual
# ============================================================================

print("\n" + "="*80)
print("📊 COMPARISON: PREDICTIONS vs ACTUAL RUNTIME")
print("="*80)

if 'actual_total_time' in df.columns:
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'query_id': range(len(df)),
        'actual_time_ms': df['actual_total_time'],
    })

    if pg_available:
        comparison['pg_pred_ms'] = df['postgresql_pred_ms']
        comparison['pg_error_ms'] = abs(df['postgresql_pred_ms'] - df['actual_total_time'])
        comparison['pg_error_pct'] = (comparison['pg_error_ms'] / (df['actual_total_time'] + 1)) * 100

    if mysql_available:
        comparison['mysql_pred_ms'] = df['mysql_pred_ms']
        comparison['mysql_error_ms'] = abs(df['mysql_pred_ms'] - df['actual_total_time'])
        comparison['mysql_error_pct'] = (comparison['mysql_error_ms'] / (df['actual_total_time'] + 1)) * 100

    if unified_available:
        comparison['unified_pred_ms'] = df['unified_pred_ms']
        comparison['unified_error_ms'] = abs(df['unified_pred_ms'] - df['actual_total_time'])
        comparison['unified_error_pct'] = (comparison['unified_error_ms'] / (df['actual_total_time'] + 1)) * 100

    print("\nFirst 10 queries - Predictions vs Actual:")
    print(comparison.head(10).to_string(index=False))

    # Calculate accuracy metrics
    print("\n" + "="*80)
    print("📈 ACCURACY METRICS")
    print("="*80)

    if pg_available:
        pg_mae = comparison['pg_error_ms'].mean()
        pg_mape = comparison['pg_error_pct'].mean()
        pg_within_10pct = (comparison['pg_error_pct'] <= 10).sum() / len(comparison) * 100
        print(f"\nPostgreSQL Model:")
        print(f"   MAE (Mean Absolute Error): {pg_mae:.2f} ms")
        print(f"   MAPE (Mean Absolute % Error): {pg_mape:.2f}%")
        print(f"   Within ±10%: {pg_within_10pct:.1f}% of queries")

    if mysql_available:
        mysql_mae = comparison['mysql_error_ms'].mean()
        mysql_mape = comparison['mysql_error_pct'].mean()
        mysql_within_10pct = (comparison['mysql_error_pct'] <= 10).sum() / len(comparison) * 100
        print(f"\nMySQL Model:")
        print(f"   MAE (Mean Absolute Error): {mysql_mae:.2f} ms")
        print(f"   MAPE (Mean Absolute % Error): {mysql_mape:.2f}%")
        print(f"   Within ±10%: {mysql_within_10pct:.1f}% of queries")

    if unified_available:
        unified_mae = comparison['unified_error_ms'].mean()
        unified_mape = comparison['unified_error_pct'].mean()
        unified_within_10pct = (comparison['unified_error_pct'] <= 10).sum() / len(comparison) * 100
        print(f"\nUnified Model:")
        print(f"   MAE (Mean Absolute Error): {unified_mae:.2f} ms")
        print(f"   MAPE (Mean Absolute % Error): {unified_mape:.2f}%")
        print(f"   Within ±10%: {unified_within_10pct:.1f}% of queries")

    # Determine best model
    print("\n" + "="*80)
    print("🏆 BEST MODEL")
    print("="*80)

    models_mae = {}
    if pg_available:
        models_mae['PostgreSQL'] = pg_mae
    if mysql_available:
        models_mae['MySQL'] = mysql_mae
    if unified_available:
        models_mae['Unified'] = unified_mae

    if models_mae:
        best_model = min(models_mae, key=models_mae.get)
        print(f"\n✅ Best model: {best_model}")
        print(f"   Lowest MAE: {models_mae[best_model]:.2f} ms")
        print(f"\n💡 Recommendation: Use {best_model} model for production")

else:
    print("\n⚠️  Column 'actual_total_time' not found in data")
    print("   Cannot calculate accuracy without actual runtimes")
    print("   Predictions have been saved to output file")

# ============================================================================
# Step 6: Save Results
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

output_file = 'results/actual_data_predictions.csv'
df.to_csv(output_file, index=False)
print(f"✅ Full predictions saved to: {output_file}")

if 'actual_total_time' in df.columns:
    comparison_file = 'results/actual_data_accuracy.csv'
    comparison.to_csv(comparison_file, index=False)
    print(f"✅ Accuracy comparison saved to: {comparison_file}")

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*80)
print("✅ TESTING COMPLETE!")
print("="*80)

print("\n📋 Summary:")
print(f"   • Queries tested: {len(df)}")
print(f"   • Models used: {sum([pg_available, mysql_available, unified_available])}")
print(f"   • Output files saved to: results/")

print("\n📊 Files created:")
print(f"   • {output_file}")
if 'actual_total_time' in df.columns:
    print(f"   • {comparison_file}")

print("\n🎯 Next steps:")
print("   1. Review results in output CSV files")
print("   2. Compare model accuracy metrics")
print("   3. Choose best model for your use case")
print("   4. Deploy to production")

print("\n" + "="*80 + "\n")
