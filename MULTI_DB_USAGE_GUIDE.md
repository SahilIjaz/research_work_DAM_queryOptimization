# Multi-Database Model Usage Guide

Quick reference for using the multi-database trained models in your application.

---

## 🚀 Quick Start

### 1. Load and Use PostgreSQL Model

```python
import pickle
import numpy as np

# Load the model
with open('models/postgresql_lightgbm.pkl', 'rb') as f:
    pg_model = pickle.load(f)

# Example: PostgreSQL execution plan features (23 dimensions)
features = np.array([[
    1000,      # estimated_rows
    950,       # actual_rows
    50.0,      # startup_cost
    450.0,     # total_cost
    10.5,      # actual_startup_time
    45.2,      # actual_total_time
    1024,      # plan_width
    3,         # loops
    500,       # base_cardinality
    480,       # output_cardinality
    490,       # input_cardinality
    0.85,      # filter_selectivity
    0.95,      # join_ratio
    0.45,      # cost_per_row
    15.0,      # time_per_loop
    1,         # node_type (encoded: 0=SeqScan, 1=HashJoin, etc.)
    5,         # node_depth
    1,         # parallel_awareness
    510,       # subtree_cardinality
    2,         # num_children
    0.45,      # cost_to_row_ratio (derived)
    1.05,      # estimated_actual_ratio (derived)
    0.10       # time_cost_ratio (derived)
]])

# Predict runtime
predicted_ms = pg_model.predict(features)
print(f"Predicted runtime: {predicted_ms[0]:.2f} ms")
```

---

### 2. Load and Use MySQL Model

```python
import pickle
import numpy as np

# Load the model
with open('models/mysql_lightgbm.pkl', 'rb') as f:
    mysql_model = pickle.load(f)

# Example: MySQL execution plan features (23 dimensions)
features = np.array([[
    1000,      # estimated_rows
    950,       # actual_rows
    40.0,      # startup_cost (MySQL typically lower)
    300.0,     # total_cost (MySQL typically lower)
    8.5,       # actual_startup_time
    38.2,      # actual_total_time
    1024,      # plan_width
    3,         # loops
    500,       # base_cardinality
    480,       # output_cardinality
    490,       # input_cardinality
    0.85,      # filter_selectivity
    0.95,      # join_ratio
    0.30,      # cost_per_row (different for MySQL)
    12.7,      # time_per_loop
    2,         # node_type (encoded differently)
    5,         # node_depth
    0,         # parallel_awareness
    510,       # subtree_cardinality
    2,         # num_children
    0.32,      # cost_to_row_ratio (derived)
    1.05,      # estimated_actual_ratio (derived)
    0.127      # time_cost_ratio (derived)
]])

# Predict runtime
predicted_ms = mysql_model.predict(features)
print(f"Predicted runtime: {predicted_ms[0]:.2f} ms")
```

---

### 3. Use Unified Model (Works with Both)

```python
import pickle
import numpy as np

# Load the unified model (works with both PostgreSQL and MySQL)
with open('models/unified_lightgbm.pkl', 'rb') as f:
    unified_model = pickle.load(f)

# This model accepts the same 23-dimensional input
# Can handle both PostgreSQL and MySQL features without modification
features = np.array([[...23 features...]])

predicted_ms = unified_model.predict(features)
print(f"Predicted runtime: {predicted_ms[0]:.2f} ms")
```

---

## 🎯 Choose the Right Model

```python
def select_model(database_type):
    """Select appropriate model based on database"""
    import pickle
    
    if database_type == 'postgresql':
        with open('models/postgresql_lightgbm.pkl', 'rb') as f:
            return pickle.load(f)
    
    elif database_type == 'mysql':
        with open('models/mysql_lightgbm.pkl', 'rb') as f:
            return pickle.load(f)
    
    else:  # Unknown database or multi-database environment
        with open('models/unified_lightgbm.pkl', 'rb') as f:
            return pickle.load(f)

# Usage
db_type = 'postgresql'
model = select_model(db_type)
predictions = model.predict(features)
```

---

## 📊 Feature List (23 Total)

The models expect exactly 23 features in this order:

### Scalar Features (15)
1. **estimated_rows** - Query optimizer's estimated row count
2. **actual_rows** - Actual rows returned
3. **startup_cost** - Database's startup cost estimate
4. **total_cost** - Total query plan cost
5. **actual_startup_time** - Time until first row (ms)
6. **actual_total_time** - Total execution time (ms)
7. **plan_width** - Width of plan node's result (bytes)
8. **loops** - Number of times node is executed
9. **base_cardinality** - Base cardinality before operations
10. **output_cardinality** - Output cardinality after operations
11. **input_cardinality** - Input cardinality to node
12. **filter_selectivity** - Fraction of rows passing filter (0-1)
13. **join_ratio** - Join selectivity (0-1)
14. **cost_per_row** - Average cost per output row
15. **time_per_loop** - Average time per loop execution

### Structural Features (5)
16. **node_type** - Type of plan node (encoded as integer)
    - 0 = Seq Scan
    - 1 = Hash Join
    - 2 = Nested Loop
    - 3 = Index Scan
17. **node_depth** - Depth in query plan tree (1-15)
18. **parallel_awareness** - Whether node supports parallelization (0 or 1)
19. **subtree_cardinality** - Cardinality of entire subtree
20. **num_children** - Number of child nodes (0-5)

### Derived Features (3)
21. **cost_to_row_ratio** = total_cost / (actual_rows + 1)
22. **estimated_actual_ratio** = estimated_rows / (actual_rows + 1)
23. **time_cost_ratio** = actual_total_time / (total_cost + 1)

---

## 🔄 Feature Normalization

The models were trained on StandardScaler-normalized features. If using in production:

```python
from sklearn.preprocessing import StandardScaler
import pickle

# Load the scaler used during training
# (You may need to save and load this separately)
# For now, if you have raw data:

def normalize_features(raw_features, use_postgresql=True):
    """Normalize raw features using appropriate scaler"""
    # Note: In production, you'd load the actual scaler used during training
    # This is a placeholder showing the normalization approach
    
    from sklearn.preprocessing import StandardScaler
    
    # Features to normalize (all except node_type which is categorical)
    numeric_cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19]  # indices
    
    scaler = StandardScaler()
    # ... fit scaler on training data ...
    
    normalized = raw_features.copy()
    normalized[:, numeric_cols] = scaler.transform(raw_features[:, numeric_cols])
    
    return normalized

# Usage
raw_features = np.array([[...]])
normalized_features = normalize_features(raw_features)
prediction = model.predict(normalized_features)
```

---

## 🐳 Docker Integration Example

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install lightgbm scikit-learn numpy pandas

# Copy models
COPY models/ /app/models/

# Copy prediction code
COPY predict.py /app/

EXPOSE 5000

CMD ["python", "predict.py"]
```

Example prediction service:
```python
# predict.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models
with open('models/postgresql_lightgbm.pkl', 'rb') as f:
    pg_model = pickle.load(f)
with open('models/mysql_lightgbm.pkl', 'rb') as f:
    mysql_model = pickle.load(f)
with open('models/unified_lightgbm.pkl', 'rb') as f:
    unified_model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    db_type = data.get('database_type', 'unified')
    features = np.array(data['features']).reshape(1, -1)
    
    # Select appropriate model
    if db_type == 'postgresql':
        model = pg_model
    elif db_type == 'mysql':
        model = mysql_model
    else:
        model = unified_model
    
    # Make prediction
    prediction = float(model.predict(features)[0])
    
    return jsonify({
        'database_type': db_type,
        'predicted_runtime_ms': prediction
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📈 Batch Prediction Example

```python
import pickle
import numpy as np
import pandas as pd

# Load model
with open('models/unified_lightgbm.pkl', 'rb') as f:
    model = pickle.load(f)

# Load execution plans from CSV or database
execution_plans = pd.read_csv('execution_plans.csv')

# Prepare features (assumes CSV has columns matching feature list)
feature_cols = [
    'estimated_rows', 'actual_rows', 'startup_cost', 'total_cost',
    'actual_startup_time', 'actual_total_time', 'plan_width', 'loops',
    'base_cardinality', 'output_cardinality', 'input_cardinality',
    'filter_selectivity', 'join_ratio', 'cost_per_row', 'time_per_loop',
    'node_type', 'node_depth', 'parallel_awareness', 'subtree_cardinality',
    'num_children', 'cost_to_row_ratio', 'estimated_actual_ratio', 'time_cost_ratio'
]

X = execution_plans[feature_cols].values

# Batch predict
predictions = model.predict(X)

# Add predictions back to dataframe
execution_plans['predicted_runtime_ms'] = predictions

# Save results
execution_plans.to_csv('predictions.csv', index=False)
print(f"Generated predictions for {len(predictions)} queries")
```

---

## ⚠️ Important Notes

1. **Feature Order Matters** - Features must be in the exact order listed above
2. **All 23 Features Required** - Model expects exactly 23 input dimensions
3. **Numeric Precision** - Use float64 for numeric features
4. **Categorical Encoding** - node_type must be integer-encoded as shown
5. **Out-of-Range Values** - Clipping/normalizing values outside training range is recommended

---

## 🔍 Troubleshooting

### "Input has wrong number of features"
- Ensure exactly 23 features
- Check feature order matches documentation
- Verify no features were accidentally skipped

### "Prediction seems unreasonable"
- Check if features are normalized (standardized)
- Verify feature values are in reasonable ranges
- Compare with actual database query times

### "Different predictions for same query"
- Ensure consistent feature ordering
- Verify consistency in categorical encoding
- Check normalization is applied uniformly

---

## 📚 Additional Resources

- See `MULTI_DB_SUMMARY.md` for performance metrics
- See `train_multi_db.py` for training code
- See `results/multi_db_training_results.json` for detailed metrics

---

**Last Updated:** June 29, 2026  
**Model Versions:**
- PostgreSQL: 1.0
- MySQL: 1.0  
- Unified: 1.0
