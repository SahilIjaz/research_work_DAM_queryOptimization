"""
Test Script - Load and Test Trained Models
Tests all 3 models (PostgreSQL, MySQL, Unified) with sample data
"""

import pickle
import numpy as np
import pandas as pd

print("\n" + "="*80)
print("🧪 MODEL TESTING - SAMPLE DATA")
print("="*80)

# ============================================================================
# Step 1: Load All Models
# ============================================================================

print("\n📦 Loading models...")

try:
    with open('models/postgresql_lightgbm.pkl', 'rb') as f:
        pg_model = pickle.load(f)
    print("✅ PostgreSQL model loaded")
except FileNotFoundError:
    print("❌ PostgreSQL model not found")
    pg_model = None

try:
    with open('models/mysql_lightgbm.pkl', 'rb') as f:
        mysql_model = pickle.load(f)
    print("✅ MySQL model loaded")
except FileNotFoundError:
    print("❌ MySQL model not found")
    mysql_model = None

try:
    with open('models/unified_lightgbm.pkl', 'rb') as f:
        unified_model = pickle.load(f)
    print("✅ Unified model loaded")
except FileNotFoundError:
    print("❌ Unified model not found")
    unified_model = None

# ============================================================================
# Step 2: Create Sample Test Data (23 features)
# ============================================================================

print("\n📊 Creating sample test data...")

# Feature names and order (24 features - query_template is first)
feature_names = [
    'query_template', 'estimated_rows', 'actual_rows', 'startup_cost', 'total_cost',
    'actual_startup_time', 'actual_total_time', 'plan_width', 'loops',
    'base_cardinality', 'output_cardinality', 'input_cardinality',
    'filter_selectivity', 'join_ratio', 'cost_per_row', 'time_per_loop',
    'node_type', 'node_depth', 'parallel_awareness', 'subtree_cardinality',
    'num_children', 'cost_to_row_ratio', 'estimated_actual_ratio', 'time_cost_ratio'
]

# Create 5 sample test cases (24 features each)
test_samples = np.array([
    # Sample 1: Simple sequential scan (fast query)
    [1, 100, 95, 10.0, 50.0, 2.0, 5.0, 512, 1, 100, 95, 100, 0.95, 1.0, 0.5, 5.0, 0, 1, 1, 100, 0, 0.526, 1.053, 0.1],

    # Sample 2: Index scan with filter (medium query)
    [2, 1000, 950, 50.0, 450.0, 10.5, 45.2, 1024, 3, 500, 480, 490, 0.85, 0.95, 0.45, 15.0, 1, 5, 1, 510, 2, 0.474, 1.053, 0.1],

    # Sample 3: Hash join (slow query)
    [3, 5000, 4800, 200.0, 1500.0, 50.0, 200.0, 2048, 5, 2000, 1900, 1950, 0.75, 0.80, 0.3, 40.0, 1, 8, 1, 2000, 3, 0.313, 1.042, 0.13],

    # Sample 4: Nested loop join (complex query)
    [4, 10000, 9500, 300.0, 2500.0, 80.0, 350.0, 4096, 10, 5000, 4500, 4800, 0.60, 0.70, 0.25, 35.0, 2, 10, 0, 4800, 4, 0.263, 1.053, 0.14],

    # Sample 5: Multi-table join with aggregation
    [5, 50000, 45000, 500.0, 5000.0, 150.0, 800.0, 8192, 20, 10000, 8000, 9000, 0.50, 0.60, 0.1, 40.0, 3, 12, 1, 8500, 5, 0.111, 1.111, 0.16],
])

test_df = pd.DataFrame(test_samples, columns=feature_names)

print(f"✅ Created {len(test_samples)} sample test cases")
print("\nSample Data:")
print(test_df.to_string())

# ============================================================================
# Step 3: Make Predictions with All Models
# ============================================================================

print("\n" + "="*80)
print("🚀 MAKING PREDICTIONS")
print("="*80)

results = []

for idx, sample in enumerate(test_samples, 1):
    sample_reshaped = sample.reshape(1, -1)

    print(f"\n📍 Test Case {idx}:")
    print(f"   Query characteristics:")
    print(f"   - Estimated rows: {sample[0]:.0f}")
    print(f"   - Actual rows: {sample[1]:.0f}")
    print(f"   - Total cost: {sample[3]:.1f}")
    print(f"   - Plan depth: {sample[16]:.0f}")

    result = {'test_case': idx, 'estimated_rows': sample[0], 'actual_rows': sample[1]}

    # PostgreSQL prediction
    if pg_model:
        pg_pred = pg_model.predict(sample_reshaped)[0]
        result['postgresql_pred_ms'] = pg_pred
        print(f"   ✅ PostgreSQL: {pg_pred:.2f} ms")

    # MySQL prediction
    if mysql_model:
        mysql_pred = mysql_model.predict(sample_reshaped)[0]
        result['mysql_pred_ms'] = mysql_pred
        print(f"   ✅ MySQL: {mysql_pred:.2f} ms")

    # Unified prediction
    if unified_model:
        unified_pred = unified_model.predict(sample_reshaped)[0]
        result['unified_pred_ms'] = unified_pred
        print(f"   ✅ Unified: {unified_pred:.2f} ms")

    results.append(result)

# ============================================================================
# Step 4: Compare Predictions
# ============================================================================

print("\n" + "="*80)
print("📊 PREDICTION COMPARISON TABLE")
print("="*80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# ============================================================================
# Step 5: Statistics
# ============================================================================

print("\n" + "="*80)
print("📈 PREDICTION STATISTICS")
print("="*80)

if pg_model:
    pg_preds = results_df['postgresql_pred_ms'].values
    print(f"\nPostgreSQL Model:")
    print(f"   Min prediction: {pg_preds.min():.2f} ms")
    print(f"   Max prediction: {pg_preds.max():.2f} ms")
    print(f"   Mean prediction: {pg_preds.mean():.2f} ms")
    print(f"   Std deviation: {pg_preds.std():.2f} ms")

if mysql_model:
    mysql_preds = results_df['mysql_pred_ms'].values
    print(f"\nMySQL Model:")
    print(f"   Min prediction: {mysql_preds.min():.2f} ms")
    print(f"   Max prediction: {mysql_preds.max():.2f} ms")
    print(f"   Mean prediction: {mysql_preds.mean():.2f} ms")
    print(f"   Std deviation: {mysql_preds.std():.2f} ms")

if unified_model:
    unified_preds = results_df['unified_pred_ms'].values
    print(f"\nUnified Model:")
    print(f"   Min prediction: {unified_preds.min():.2f} ms")
    print(f"   Max prediction: {unified_preds.max():.2f} ms")
    print(f"   Mean prediction: {unified_preds.mean():.2f} ms")
    print(f"   Std deviation: {unified_preds.std():.2f} ms")

# ============================================================================
# Step 6: Model Comparison
# ============================================================================

if pg_model and mysql_model and unified_model:
    print("\n" + "="*80)
    print("🏆 MODEL COMPARISON FOR EACH TEST CASE")
    print("="*80)

    for idx in range(len(results)):
        pg_p = results_df.loc[idx, 'postgresql_pred_ms']
        mysql_p = results_df.loc[idx, 'mysql_pred_ms']
        unified_p = results_df.loc[idx, 'unified_pred_ms']

        preds = [
            ('PostgreSQL', pg_p),
            ('MySQL', mysql_p),
            ('Unified', unified_p)
        ]

        fastest = min(preds, key=lambda x: x[1])
        slowest = max(preds, key=lambda x: x[1])

        print(f"\nTest Case {idx + 1}:")
        print(f"   Fastest: {fastest[0]} ({fastest[1]:.2f} ms)")
        print(f"   Slowest: {slowest[0]} ({slowest[1]:.2f} ms)")
        print(f"   Difference: {slowest[1] - fastest[1]:.2f} ms")

# ============================================================================
# Step 7: Save Results to CSV
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

results_df.to_csv('results/test_results.csv', index=False)
print("✅ Saved results to: results/test_results.csv")

# ============================================================================
# Step 8: Final Summary
# ============================================================================

print("\n" + "="*80)
print("✅ TESTING COMPLETE!")
print("="*80)

print("\n📋 Summary:")
print(f"   • Test cases: {len(test_samples)}")
print(f"   • Models tested: {sum([bool(pg_model), bool(mysql_model), bool(unified_model)])}")
print(f"   • Total predictions: {len(results) * sum([bool(pg_model), bool(mysql_model), bool(unified_model)])}")
print(f"   • Results saved to: results/test_results.csv")

print("\n🎯 What you can do next:")
print("   1. View results: cat results/test_results.csv")
print("   2. Compare with actual runtimes from your database")
print("   3. Adjust features based on your actual query characteristics")
print("   4. Use best-performing model for production")

print("\n" + "="*80 + "\n")
