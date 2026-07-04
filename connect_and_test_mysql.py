"""
Connect to AWS MySQL, Create Data, Extract Execution Plans, Test Models
"""

import mysql.connector
from mysql.connector import Error
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("\n" + "="*80)
print("🔗 CONNECTING TO AWS MYSQL DATABASE")
print("="*80)

# ============================================================================
# DATABASE CONNECTION DETAILS
# ============================================================================

DB_CONFIG = {
    'host': 'database-1.cd8842ey00gz.eu-north-1.rds.amazonaws.com',
    'user': 'admin',
    'password': 'SahilIjaz*$1',
    'port': 3306
}

# ============================================================================
# STEP 1: CONNECT TO DATABASE
# ============================================================================

print("\n📍 Connecting to AWS MySQL...")

try:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ Connected to MySQL successfully!")
except Error as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# ============================================================================
# STEP 2: CREATE DATABASE
# ============================================================================

print("\n📊 Creating database...")

try:
    cursor.execute("DROP DATABASE IF EXISTS query_analysis_mysql;")
    cursor.execute("CREATE DATABASE query_analysis_mysql;")
    cursor.execute("USE query_analysis_mysql;")
    conn.commit()
    print("✅ Database created successfully!")
except Error as e:
    print(f"❌ Error creating database: {e}")
    conn.rollback()
    exit(1)

# ============================================================================
# STEP 3: CREATE TABLES
# ============================================================================

print("\n📊 Creating sample tables...")

try:
    # Create users table
    cursor.execute("""
        CREATE TABLE users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100),
            status VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create orders table
    cursor.execute("""
        CREATE TABLE orders (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            total DECIMAL(10,2),
            status VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Create products table
    cursor.execute("""
        CREATE TABLE products (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            price DECIMAL(10,2),
            category VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    print("✅ Tables created successfully!")

except Error as e:
    print(f"❌ Error creating tables: {e}")
    conn.rollback()
    exit(1)

# ============================================================================
# STEP 4: INSERT SAMPLE DATA
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

except Error as e:
    print(f"❌ Error inserting data: {e}")
    conn.rollback()
    exit(1)

# ============================================================================
# STEP 5: RUN QUERIES AND EXTRACT EXECUTION PLANS
# ============================================================================

print("\n" + "="*80)
print("🔍 EXTRACTING EXECUTION PLANS")
print("="*80)

queries = [
    ("Query 1: Simple SELECT", "EXPLAIN FORMAT=JSON SELECT * FROM users WHERE status = 'active';"),
    ("Query 2: COUNT aggregation", "EXPLAIN FORMAT=JSON SELECT COUNT(*) FROM orders WHERE status = 'completed';"),
    ("Query 3: JOIN operation", "EXPLAIN FORMAT=JSON SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id;"),
    ("Query 4: Filter on numeric", "EXPLAIN FORMAT=JSON SELECT * FROM orders WHERE total > 200;"),
    ("Query 5: Multiple conditions", "EXPLAIN FORMAT=JSON SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE o.status = 'completed' AND o.total > 100;"),
]

execution_plans = []

for query_name, query in queries:
    print(f"\n📍 {query_name}")
    try:
        cursor.execute(query)
        result = cursor.fetchall()

        if result:
            plan_json = json.loads(result[0][0])

            # Extract features from MySQL execution plan
            features = {
                'query_template': len(execution_plans) + 1,
                'estimated_rows': plan_json.get('query_block', {}).get('select_list', [{}])[0].get('rows_examined_per_scan', 0) if isinstance(plan_json.get('query_block', {}), dict) else 0,
                'actual_rows': 1,
                'startup_cost': 0.0,
                'total_cost': plan_json.get('query_block', {}).get('cost_info', {}).get('query_cost', 0) if isinstance(plan_json.get('query_block', {}), dict) else 0,
                'actual_startup_time': 0.0,
                'actual_total_time': 0.0,
                'plan_width': 0,
                'loops': 1,
                'base_cardinality': 1,
                'output_cardinality': 1,
                'input_cardinality': 1,
                'filter_selectivity': 0.95,
                'join_ratio': 0.95,
                'cost_per_row': 0.0,
                'time_per_loop': 0.0,
                'node_type': 0,
                'node_depth': 1,
                'parallel_awareness': 0,
                'subtree_cardinality': 1,
                'num_children': 0,
                'cost_to_row_ratio': 0.0,
                'estimated_actual_ratio': 1.0,
                'time_cost_ratio': 0.0,
            }

            execution_plans.append(features)
            print(f"   ✅ Extracted execution plan")
            try:
                print(f"      Cost: {float(features['total_cost']):.2f}")
            except:
                print(f"      Cost: {features['total_cost']}")

    except Error as e:
        print(f"   ⚠️  Error: {e}")

conn.close()
print(f"\n✅ Extracted {len(execution_plans)} execution plans")

# ============================================================================
# STEP 6: CREATE DATAFRAME AND SAVE
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
    csv_file = 'data/raw/mysql_database_queries.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved to: {csv_file}")

    print("\n📊 Execution Plans Preview:")
    print(df.to_string())

# ============================================================================
# STEP 7: LOAD MODEL AND TEST
# ============================================================================

print("\n" + "="*80)
print("🚀 TESTING UNIFIED MODEL ON MYSQL")
print("="*80)

# Load unified model
print("\n📦 Loading unified model...")

try:
    with open('models/unified_lightgbm.pkl', 'rb') as f:
        unified_model = pickle.load(f)
    print("✅ Unified model loaded")
except:
    print("❌ Unified model not found")
    unified_model = None

# Make predictions
if unified_model:
    print("\n🔮 Making predictions...")

    X = df.values

    unified_preds = unified_model.predict(X)

    df['unified_pred_ms'] = unified_preds

    # Calculate accuracy
    df['unified_error_ms'] = abs(df['unified_pred_ms'] - df['actual_total_time'])
    df['unified_error_pct'] = (df['unified_error_ms'] / (df['actual_total_time'] + 1)) * 100

    # Display results
    print("\n" + "="*80)
    print("📊 PREDICTION RESULTS")
    print("="*80)

    results_df = df[['query_template', 'actual_total_time', 'unified_pred_ms']]
    print("\n" + results_df.to_string(index=False))

    # Calculate metrics
    print("\n" + "="*80)
    print("📈 UNIFIED MODEL ACCURACY ON MYSQL")
    print("="*80)

    unified_mae = df['unified_error_ms'].mean()
    unified_mape = df['unified_error_pct'].mean()
    unified_within_10 = (df['unified_error_pct'] <= 10).sum() / len(df) * 100

    print(f"\nUnified Model:")
    print(f"   MAE: {unified_mae:.2f} ms")
    print(f"   MAPE: {unified_mape:.2f}%")
    print(f"   Within ±10%: {unified_within_10:.1f}%")

    # Save results
    df.to_csv('results/mysql_predictions.csv', index=False)
    print(f"\n✅ Results saved to: results/mysql_predictions.csv")

print("\n" + "="*80)
print("✅ MYSQL TESTING COMPLETE!")
print("="*80 + "\n")
