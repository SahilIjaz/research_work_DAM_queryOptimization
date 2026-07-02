"""
Connect to Render PostgreSQL, Create Data, Extract Execution Plans, Test Models
"""

import psycopg2
from psycopg2 import sql
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("\n" + "="*80)
print("🔗 CONNECTING TO RENDER POSTGRESQL DATABASE")
print("="*80)

# ============================================================================
# DATABASE CONNECTION DETAILS
# ============================================================================

DB_CONFIG = {
    'host': 'dpg-d935co0k1i2s73dksleg-a.oregon-postgres.render.com',
    'database': 'query_analysis',
    'user': 'query_analysis_user',
    'password': '7M3FK839A8fWZ0tKPsVWmmZFMPW5aoeY',
    'port': 5432
}

# ============================================================================
# STEP 1: CONNECT TO DATABASE
# ============================================================================

print("\n📍 Connecting to Render PostgreSQL...")

try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ Connected to database successfully!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# ============================================================================
# STEP 2: CREATE SAMPLE TABLES
# ============================================================================

print("\n📊 Creating sample tables...")

try:
    # Drop tables if they exist
    cursor.execute("DROP TABLE IF EXISTS orders CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS users CASCADE;")
    cursor.execute("DROP TABLE IF EXISTS products CASCADE;")

    # Create users table
    cursor.execute("""
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100),
            status VARCHAR(20),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Create orders table
    cursor.execute("""
        CREATE TABLE orders (
            id SERIAL PRIMARY KEY,
            user_id INT REFERENCES users(id),
            total DECIMAL(10,2),
            status VARCHAR(20),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Create products table
    cursor.execute("""
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            price DECIMAL(10,2),
            category VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    print("✅ Tables created successfully!")

except Exception as e:
    print(f"❌ Error creating tables: {e}")
    conn.rollback()
    exit(1)

# ============================================================================
# STEP 3: INSERT SAMPLE DATA
# ============================================================================

print("\n📈 Inserting sample data...")

try:
    # Insert users
    cursor.execute("""
        INSERT INTO users (name, email, status) VALUES
        ('Alice Johnson', 'alice@example.com', 'active'),
        ('Bob Smith', 'bob@example.com', 'active'),
        ('Charlie Brown', 'charlie@example.com', 'inactive'),
        ('David Wilson', 'david@example.com', 'active'),
        ('Eve Davis', 'eve@example.com', 'active'),
        ('Frank Miller', 'frank@example.com', 'inactive'),
        ('Grace Lee', 'grace@example.com', 'active'),
        ('Henry Taylor', 'henry@example.com', 'active');
    """)

    # Insert orders
    cursor.execute("""
        INSERT INTO orders (user_id, total, status) VALUES
        (1, 100.00, 'completed'),
        (2, 250.50, 'completed'),
        (1, 75.25, 'pending'),
        (3, 500.00, 'completed'),
        (2, 150.75, 'completed'),
        (4, 320.00, 'pending'),
        (5, 450.25, 'completed'),
        (1, 200.00, 'completed'),
        (3, 175.50, 'pending'),
        (2, 600.00, 'completed');
    """)

    # Insert products
    cursor.execute("""
        INSERT INTO products (name, price, category) VALUES
        ('Laptop', 999.99, 'Electronics'),
        ('Mouse', 25.99, 'Electronics'),
        ('Desk', 299.99, 'Furniture'),
        ('Chair', 199.99, 'Furniture'),
        ('Monitor', 399.99, 'Electronics'),
        ('Keyboard', 79.99, 'Electronics'),
        ('Lamp', 49.99, 'Furniture'),
        ('Speaker', 149.99, 'Electronics');
    """)

    conn.commit()
    print("✅ Sample data inserted successfully!")

except Exception as e:
    print(f"❌ Error inserting data: {e}")
    conn.rollback()
    exit(1)

# ============================================================================
# STEP 4: RUN EXPLAIN QUERIES AND EXTRACT EXECUTION PLANS
# ============================================================================

print("\n" + "="*80)
print("🔍 EXTRACTING EXECUTION PLANS")
print("="*80)

queries = [
    ("Query 1: Simple SELECT", "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT * FROM users WHERE status = 'active';"),
    ("Query 2: COUNT aggregation", "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT COUNT(*) FROM orders WHERE status = 'completed';"),
    ("Query 3: JOIN operation", "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id;"),
    ("Query 4: Filter on numeric", "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT * FROM orders WHERE total > 200;"),
    ("Query 5: Multiple conditions", "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' AND o.total > 100;"),
]

execution_plans = []

for query_name, query in queries:
    print(f"\n📍 {query_name}")
    try:
        cursor.execute(query)
        result = cursor.fetchall()

        if result:
            # Handle both string and list return types
            json_data = result[0][0]
            if isinstance(json_data, list):
                plan_json = json_data
            else:
                plan_json = json.loads(json_data)

            plan = plan_json[0]['Plan']

            # Extract features
            features = {
                'query_template': len(execution_plans) + 1,
                'estimated_rows': plan.get('Rows', 0),
                'actual_rows': plan.get('Actual Rows', plan.get('Rows', 0)),
                'startup_cost': plan.get('Startup Cost', 0),
                'total_cost': plan.get('Total Cost', 0),
                'actual_startup_time': plan_json[0].get('Actual Startup Time', 0),
                'actual_total_time': plan_json[0].get('Actual Total Time', 0),
                'plan_width': plan.get('Width', 0),
                'loops': plan.get('Loops', 1),
                'base_cardinality': plan.get('Rows', 0),
                'output_cardinality': plan.get('Actual Rows', plan.get('Rows', 0)),
                'input_cardinality': plan.get('Rows', 0),
                'filter_selectivity': plan.get('Actual Rows', 0) / (plan.get('Rows', 1)),
                'join_ratio': 0.95,
                'cost_per_row': plan.get('Total Cost', 0) / (plan.get('Actual Rows', 1) + 1),
                'time_per_loop': plan_json[0].get('Actual Total Time', 0) / (plan.get('Loops', 1) + 1),
                'node_type': 0,  # Seq Scan default
                'node_depth': 1,
                'parallel_awareness': 0,
                'subtree_cardinality': plan.get('Rows', 0),
                'num_children': len(plan.get('Plans', [])),
                'cost_to_row_ratio': plan.get('Total Cost', 0) / (plan.get('Actual Rows', 1) + 1),
                'estimated_actual_ratio': plan.get('Rows', 0) / (plan.get('Actual Rows', 1) + 1),
                'time_cost_ratio': plan_json[0].get('Actual Total Time', 0) / (plan.get('Total Cost', 1) + 1),
            }

            execution_plans.append(features)
            print(f"   ✅ Extracted execution plan")
            print(f"      Actual rows: {features['actual_rows']:.0f}")
            print(f"      Total cost: {features['total_cost']:.2f}")
            print(f"      Actual time: {features['actual_total_time']:.2f} ms")

    except Exception as e:
        print(f"   ⚠️  Error: {e}")

conn.close()
print(f"\n✅ Extracted {len(execution_plans)} execution plans")

# ============================================================================
# STEP 5: CREATE DATAFRAME AND SAVE
# ============================================================================

print("\n" + "="*80)
print("💾 SAVING EXECUTION PLANS TO CSV")
print("="*80)

if execution_plans:
    df = pd.DataFrame(execution_plans)

    # Reorder columns
    expected_cols = [
        'query_template', 'estimated_rows', 'actual_rows', 'startup_cost', 'total_cost',
        'actual_startup_time', 'actual_total_time', 'plan_width', 'loops',
        'base_cardinality', 'output_cardinality', 'input_cardinality',
        'filter_selectivity', 'join_ratio', 'cost_per_row', 'time_per_loop',
        'node_type', 'node_depth', 'parallel_awareness', 'subtree_cardinality',
        'num_children', 'cost_to_row_ratio', 'estimated_actual_ratio', 'time_cost_ratio'
    ]

    df = df[expected_cols]

    # Save to CSV
    csv_file = 'data/raw/render_database_queries.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved to: {csv_file}")

    print("\n📊 Execution Plans Preview:")
    print(df.to_string())

# ============================================================================
# STEP 6: LOAD MODELS AND TEST
# ============================================================================

print("\n" + "="*80)
print("🚀 TESTING MODELS WITH REAL DATA")
print("="*80)

# Load models
print("\n📦 Loading trained models...")

try:
    with open('models/postgresql_lightgbm.pkl', 'rb') as f:
        pg_model = pickle.load(f)
    print("✅ PostgreSQL model loaded")
except:
    print("❌ PostgreSQL model not found")
    pg_model = None

try:
    with open('models/mysql_lightgbm.pkl', 'rb') as f:
        mysql_model = pickle.load(f)
    print("✅ MySQL model loaded")
except:
    print("❌ MySQL model not found")
    mysql_model = None

try:
    with open('models/unified_lightgbm.pkl', 'rb') as f:
        unified_model = pickle.load(f)
    print("✅ Unified model loaded")
except:
    print("❌ Unified model not found")
    unified_model = None

# Make predictions
if pg_model and mysql_model and unified_model:
    print("\n🔮 Making predictions...")

    X = df.values

    pg_preds = pg_model.predict(X)
    mysql_preds = mysql_model.predict(X)
    unified_preds = unified_model.predict(X)

    df['postgresql_pred_ms'] = pg_preds
    df['mysql_pred_ms'] = mysql_preds
    df['unified_pred_ms'] = unified_preds

    # Calculate accuracy
    df['pg_error_ms'] = abs(df['postgresql_pred_ms'] - df['actual_total_time'])
    df['mysql_error_ms'] = abs(df['mysql_pred_ms'] - df['actual_total_time'])
    df['unified_error_ms'] = abs(df['unified_pred_ms'] - df['actual_total_time'])

    df['pg_error_pct'] = (df['pg_error_ms'] / (df['actual_total_time'] + 1)) * 100
    df['mysql_error_pct'] = (df['mysql_error_ms'] / (df['actual_total_time'] + 1)) * 100
    df['unified_error_pct'] = (df['unified_error_ms'] / (df['actual_total_time'] + 1)) * 100

    # Display results
    print("\n" + "="*80)
    print("📊 PREDICTION RESULTS")
    print("="*80)

    results_df = df[['query_template', 'actual_total_time', 'postgresql_pred_ms', 'mysql_pred_ms', 'unified_pred_ms']]
    print("\n" + results_df.to_string(index=False))

    # Calculate metrics
    print("\n" + "="*80)
    print("📈 ACCURACY METRICS")
    print("="*80)

    pg_mae = df['pg_error_ms'].mean()
    mysql_mae = df['mysql_error_ms'].mean()
    unified_mae = df['unified_error_ms'].mean()

    pg_mape = df['pg_error_pct'].mean()
    mysql_mape = df['mysql_error_pct'].mean()
    unified_mape = df['unified_error_pct'].mean()

    pg_within_10 = (df['pg_error_pct'] <= 10).sum() / len(df) * 100
    mysql_within_10 = (df['mysql_error_pct'] <= 10).sum() / len(df) * 100
    unified_within_10 = (df['unified_error_pct'] <= 10).sum() / len(df) * 100

    print(f"\nPostgreSQL Model:")
    print(f"   MAE: {pg_mae:.2f} ms")
    print(f"   MAPE: {pg_mape:.2f}%")
    print(f"   Within ±10%: {pg_within_10:.1f}%")

    print(f"\nMySQL Model:")
    print(f"   MAE: {mysql_mae:.2f} ms")
    print(f"   MAPE: {mysql_mape:.2f}%")
    print(f"   Within ±10%: {mysql_within_10:.1f}%")

    print(f"\nUnified Model:")
    print(f"   MAE: {unified_mae:.2f} ms")
    print(f"   MAPE: {unified_mape:.2f}%")
    print(f"   Within ±10%: {unified_within_10:.1f}%")

    # Best model
    print(f"\n" + "="*80)
    print(f"🏆 BEST MODEL")
    print(f"="*80)

    models = {
        'PostgreSQL': pg_mae,
        'MySQL': mysql_mae,
        'Unified': unified_mae
    }

    best_model = min(models, key=models.get)
    print(f"\n✅ Best Model: {best_model}")
    print(f"   Lowest MAE: {models[best_model]:.2f} ms")

    # Save results
    df.to_csv('results/render_predictions.csv', index=False)
    print(f"\n✅ Results saved to: results/render_predictions.csv")

print("\n" + "="*80)
print("✅ TESTING COMPLETE!")
print("="*80 + "\n")
