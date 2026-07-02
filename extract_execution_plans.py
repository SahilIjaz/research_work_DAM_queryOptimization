"""
Extract Execution Plans from PostgreSQL Database
Converts EXPLAIN output to features for model prediction
"""

import json
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime

# ============================================================================
# CONFIGURATION - UPDATE THESE WITH YOUR DATABASE DETAILS
# ============================================================================

DB_CONFIG = {
    'host': 'localhost',
    'database': 'your_database',
    'user': 'your_username',
    'password': 'your_password',
    'port': 5432
}

# List of queries to analyze
QUERIES = [
    "SELECT * FROM users WHERE id = 1;",
    "SELECT * FROM orders WHERE user_id = 1;",
    "SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id;",
    # Add your actual queries here
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_plan_features(plan_json):
    """Extract features from PostgreSQL EXPLAIN JSON output"""

    if not plan_json or len(plan_json) == 0:
        return None

    plan = plan_json[0]['Plan']

    # Extract basic metrics
    features = {
        'estimated_rows': plan.get('Rows', 0),
        'startup_cost': plan.get('Startup Cost', 0),
        'total_cost': plan.get('Total Cost', 0),
        'plan_width': plan.get('Width', 0),
        'node_type': plan.get('Node Type', 'Unknown'),
    }

    # Extract actual execution info if available
    if 'Actual Rows' in plan:
        features['actual_rows'] = plan['Actual Rows']
    else:
        features['actual_rows'] = plan.get('Rows', 0)

    if 'Actual Total Time' in plan:
        features['actual_total_time'] = plan['Actual Total Time']
    else:
        features['actual_total_time'] = 0

    if 'Actual Startup Time' in plan:
        features['actual_startup_time'] = plan['Actual Startup Time']
    else:
        features['actual_startup_time'] = 0

    # Extract loops information
    features['loops'] = plan.get('Loops', 1)

    # Calculate derived features
    features['cost_per_row'] = features['total_cost'] / (features['actual_rows'] + 1)
    features['estimated_actual_ratio'] = features['estimated_rows'] / (features['actual_rows'] + 1)
    features['time_cost_ratio'] = features['actual_total_time'] / (features['total_cost'] + 1)

    # Set defaults for missing features
    features['base_cardinality'] = features['estimated_rows']
    features['output_cardinality'] = features['actual_rows']
    features['input_cardinality'] = features['estimated_rows']
    features['filter_selectivity'] = features['actual_rows'] / (features['estimated_rows'] + 1)
    features['join_ratio'] = 0.95  # Default
    features['time_per_loop'] = features['actual_total_time'] / (features['loops'] + 1)
    features['node_depth'] = 1  # Default depth
    features['parallel_awareness'] = 0
    features['subtree_cardinality'] = features['estimated_rows']
    features['num_children'] = len(plan.get('Plans', []))
    features['cost_to_row_ratio'] = features['total_cost'] / (features['actual_rows'] + 1)
    features['query_template'] = 1  # Default template ID

    return features

def connect_to_db(config):
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**config)
        print(f"✅ Connected to {config['database']} on {config['host']}")
        return conn
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def execute_explain(conn, query):
    """Execute EXPLAIN on a query and get JSON output"""
    try:
        cursor = conn.cursor()
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
        cursor.execute(explain_query)
        result = cursor.fetchall()
        cursor.close()

        if result:
            return json.loads(result[0][0])
        return None
    except Exception as e:
        print(f"⚠️  Error executing query '{query[:50]}...': {e}")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("📊 EXTRACTING EXECUTION PLANS FROM DATABASE")
    print("="*80)

    # Step 1: Connect to database
    print("\n🔌 Connecting to database...")
    conn = connect_to_db(DB_CONFIG)

    if not conn:
        print("❌ Cannot proceed without database connection")
        print("\n📝 To use this script:")
        print("   1. Update DB_CONFIG with your database credentials")
        print("   2. Add your SQL queries to QUERIES list")
        print("   3. Run this script again")
        return

    # Step 2: Extract execution plans
    print("\n📖 Extracting execution plans from queries...")
    all_features = []
    successful = 0
    failed = 0

    for idx, query in enumerate(QUERIES, 1):
        print(f"\n📍 Query {idx}: {query[:60]}...")

        plan_json = execute_explain(conn, query)

        if plan_json:
            features = extract_plan_features(plan_json)
            if features:
                all_features.append(features)
                print(f"   ✅ Extracted features")
                successful += 1
            else:
                print(f"   ⚠️  Could not extract features")
                failed += 1
        else:
            print(f"   ❌ EXPLAIN failed")
            failed += 1

    conn.close()

    # Step 3: Create DataFrame and save
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)

    if all_features:
        df = pd.DataFrame(all_features)

        # Reorder columns to match model expectations
        expected_cols = [
            'query_template', 'estimated_rows', 'actual_rows', 'startup_cost', 'total_cost',
            'actual_startup_time', 'actual_total_time', 'plan_width', 'loops',
            'base_cardinality', 'output_cardinality', 'input_cardinality',
            'filter_selectivity', 'join_ratio', 'cost_per_row', 'time_per_loop',
            'node_type', 'node_depth', 'parallel_awareness', 'subtree_cardinality',
            'num_children', 'cost_to_row_ratio', 'estimated_actual_ratio', 'time_cost_ratio'
        ]

        # Add missing columns with defaults
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_cols]

        # Encode node_type
        node_type_mapping = {'Seq Scan': 0, 'Index Scan': 1, 'Hash Join': 2, 'Nested Loop': 3}
        df['node_type'] = df['node_type'].map(lambda x: node_type_mapping.get(x, 0))

        # Save to CSV
        output_file = f'data/raw/actual_queries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(output_file, index=False)

        print(f"\n✅ Saved {len(df)} queries to: {output_file}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")

        print("\n📊 Sample of extracted data:")
        print(df.head(10).to_string())

    else:
        print("\n❌ No execution plans extracted")

if __name__ == '__main__':
    main()
