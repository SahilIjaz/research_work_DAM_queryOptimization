# How to Get Actual Execution Plan Data

This guide shows you how to export execution plans from your PostgreSQL or MySQL database for testing the trained models.

---

## 🐘 **PostgreSQL: Get Execution Plans**

### **Option 1: Manual EXPLAIN (One Query at a Time)**

```sql
-- Simple EXPLAIN
EXPLAIN
SELECT * FROM users WHERE id = 1;

-- EXPLAIN with ANALYZE (includes actual execution)
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM orders WHERE user_id = 100;

-- Get JSON format (best for parsing)
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM orders WHERE user_id = 100;
```

### **Option 2: Export Multiple Queries to CSV**

Connect to PostgreSQL and run:

```bash
psql -h localhost -U your_user -d your_database << EOF > execution_plans.csv
COPY (
  SELECT 
    1 as query_template,
    100 as estimated_rows,
    95 as actual_rows,
    10.0 as startup_cost,
    50.0 as total_cost,
    2.0 as actual_startup_time,
    5.0 as actual_total_time,
    512 as plan_width,
    1 as loops,
    100 as base_cardinality,
    95 as output_cardinality,
    100 as input_cardinality,
    0.95 as filter_selectivity,
    1.0 as join_ratio,
    0.5 as cost_per_row,
    5.0 as time_per_loop,
    0 as node_type,
    1 as node_depth,
    1 as parallel_awareness,
    100 as subtree_cardinality,
    0 as num_children,
    0.526 as cost_to_row_ratio,
    1.053 as estimated_actual_ratio,
    0.10 as time_cost_ratio
) TO STDOUT WITH CSV HEADER;
EOF
```

### **Option 3: Use Python Script to Extract (Automatic)**

```bash
# Edit extract_execution_plans.py first:
# 1. Update DB_CONFIG with your credentials
# 2. Add your SQL queries to QUERIES list

python3 extract_execution_plans.py
```

---

## 🐬 **MySQL: Get Execution Plans**

### **Option 1: Get Execution Plans Manually**

```sql
-- Get EXPLAIN output in JSON format
EXPLAIN FORMAT=JSON
SELECT * FROM orders WHERE user_id = 100;

-- Get traditional EXPLAIN format
EXPLAIN
SELECT * FROM orders WHERE user_id = 100;
```

### **Option 2: Export to CSV**

```bash
mysql -h localhost -u your_user -p your_database << EOF > execution_plans.csv
SELECT 
  1 as query_template,
  100 as estimated_rows,
  95 as actual_rows,
  10.0 as startup_cost,
  50.0 as total_cost,
  2.0 as actual_startup_time,
  5.0 as actual_total_time,
  512 as plan_width,
  1 as loops,
  100 as base_cardinality,
  95 as output_cardinality,
  100 as input_cardinality,
  0.95 as filter_selectivity,
  1.0 as join_ratio,
  0.5 as cost_per_row,
  5.0 as time_per_loop,
  0 as node_type,
  1 as node_depth,
  1 as parallel_awareness,
  100 as subtree_cardinality,
  0 as num_children,
  0.526 as cost_to_row_ratio,
  1.053 as estimated_actual_ratio,
  0.10 as time_cost_ratio
INTO OUTFILE '/tmp/execution_plans.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
EOF
```

---

## 📋 **Required CSV Columns (24 Total)**

Your CSV file must have exactly these columns in this order:

```
query_template, estimated_rows, actual_rows, startup_cost, total_cost,
actual_startup_time, actual_total_time, plan_width, loops,
base_cardinality, output_cardinality, input_cardinality,
filter_selectivity, join_ratio, cost_per_row, time_per_loop,
node_type, node_depth, parallel_awareness, subtree_cardinality,
num_children, cost_to_row_ratio, estimated_actual_ratio, time_cost_ratio
```

### **Column Descriptions:**

| Column | Description | Example |
|--------|-------------|---------|
| query_template | Query template ID (1-22) | 1 |
| estimated_rows | Optimizer estimate of rows | 1000 |
| actual_rows | Actual rows returned | 950 |
| startup_cost | Cost before first row | 10.0 |
| total_cost | Total execution cost | 500.0 |
| actual_startup_time | Time to first row (ms) | 2.5 |
| actual_total_time | Total execution time (ms) | 45.2 |
| plan_width | Width of result rows (bytes) | 1024 |
| loops | Number of times executed | 3 |
| base_cardinality | Base table cardinality | 5000 |
| output_cardinality | Output cardinality | 4500 |
| input_cardinality | Input cardinality | 5000 |
| filter_selectivity | Fraction passing filter (0-1) | 0.85 |
| join_ratio | Join selectivity (0-1) | 0.95 |
| cost_per_row | Cost per output row | 0.45 |
| time_per_loop | Average time per loop (ms) | 15.0 |
| node_type | Plan node type (0-5) | 1 |
| node_depth | Depth in plan tree | 5 |
| parallel_awareness | Supports parallelization (0/1) | 1 |
| subtree_cardinality | Cardinality of subtree | 4800 |
| num_children | Number of child nodes | 2 |
| cost_to_row_ratio | total_cost / actual_rows | 0.474 |
| estimated_actual_ratio | estimated_rows / actual_rows | 1.053 |
| time_cost_ratio | actual_total_time / total_cost | 0.10 |

### **Node Type Encoding:**

```
0 = Seq Scan / Sequential Scan
1 = Index Scan / Index Only Scan
2 = Hash Join
3 = Nested Loop / Nested Loop Join
4 = Bitmap Index Scan
5 = Bitmap Heap Scan
```

---

## 🚀 **Quick Start: Test with Your Data**

### **Step 1: Create CSV File**

Save your execution plans to: `data/raw/actual_queries.csv`

Example format:
```csv
query_template,estimated_rows,actual_rows,startup_cost,total_cost,actual_startup_time,actual_total_time,plan_width,loops,base_cardinality,output_cardinality,input_cardinality,filter_selectivity,join_ratio,cost_per_row,time_per_loop,node_type,node_depth,parallel_awareness,subtree_cardinality,num_children,cost_to_row_ratio,estimated_actual_ratio,time_cost_ratio
1,100,95,10.0,50.0,2.0,5.0,512,1,100,95,100,0.95,1.0,0.5,5.0,0,1,1,100,0,0.526,1.053,0.10
2,1000,950,50.0,450.0,10.5,45.2,1024,3,500,480,490,0.85,0.95,0.45,15.0,1,5,1,510,2,0.474,1.053,0.10
```

### **Step 2: Run Test Script**

```bash
python3 test_with_actual_data.py
```

### **Step 3: View Results**

```bash
# See predictions
cat results/actual_data_predictions.csv

# See accuracy metrics
cat results/actual_data_accuracy.csv
```

---

## 💡 **Tips for Getting Good Execution Data**

1. **Run Multiple Queries**
   - Get execution plans for 50+ different queries
   - Mix simple and complex queries
   - Include different query templates (SELECT, JOIN, GROUP BY, etc.)

2. **Capture Actual Execution**
   - Use `EXPLAIN ANALYZE` (executes the query)
   - Include `BUFFERS` to see I/O metrics
   - Format as JSON for easier parsing

3. **Representative Workload**
   - Include queries from your actual application
   - Cover different table sizes
   - Include different filter selectivity levels

4. **Real Runtime Capture**
   - The `actual_total_time` is crucial for validation
   - It shows whether predictions match reality
   - Models work best when trained on production patterns

---

## 🔄 **Workflow Example**

### **PostgreSQL Example:**

```bash
# Connect to database and get execution plans
psql -h localhost -U postgres -d mydb << EOF

-- Query 1
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id WHERE u.status = 'active';

-- Query 2
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL '30 days';

-- Save to file
\copy (SELECT * FROM pg_stat_statements LIMIT 100) TO '/tmp/execution_data.csv' WITH CSV HEADER;

EOF
```

### **Then run the test:**

```bash
# Copy your CSV to the right location
cp /tmp/execution_data.csv data/raw/actual_queries.csv

# Run the test script
python3 test_with_actual_data.py
```

---

## ⚠️ **Troubleshooting**

### **Error: "Missing columns"**
- Check CSV has all 24 columns
- Verify column order matches expected order
- Use a CSV header row

### **Error: "actual_total_time not found"**
- Your CSV needs actual execution times
- Use `EXPLAIN ANALYZE` not just `EXPLAIN`
- Include the actual_total_time column

### **Predictions don't match reality**
- Ensure you have actual_total_time values (from ANALYZE)
- Check if your data is from same database type (PG vs MySQL)
- Use database-specific model for better accuracy

---

## 📚 **Next Steps**

1. Extract execution plans using steps above
2. Save as `data/raw/actual_queries.csv`
3. Run: `python3 test_with_actual_data.py`
4. Review `results/actual_data_accuracy.csv` for metrics
5. Choose best model and deploy to production

---

**Need help?** Check the README or MULTI_DB_USAGE_GUIDE.md files for more examples.
