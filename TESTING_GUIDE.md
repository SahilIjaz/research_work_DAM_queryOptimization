# Testing Guide - Three Ways to Test Your Models

---

## 🎯 Quick Overview

You have **3 ways to test** your trained models:

1. **Test with Sample Data** (✅ DONE - 5 minutes)
2. **Test with Actual Database Queries** (Next step - your choice)
3. **Test in Production** (Final step - deployment)

---

## ✅ **Method 1: Sample Data Testing (COMPLETED)**

### What was done:
```bash
python3 test_models.py
```

### Results:
- ✅ All 3 models loaded successfully
- ✅ Generated 15 predictions (5 test cases × 3 models)
- ✅ Results saved to: `results/test_results.csv`

### Sample Results:
```
Test Case 1 (Fast query):
  PostgreSQL: 97.76 ms
  MySQL: 91.32 ms
  Unified: 100.13 ms

Test Case 5 (Complex query):
  PostgreSQL: 97.76 ms
  MySQL: 91.32 ms
  Unified: 93.97 ms
```

**Status:** ✅ Models are working correctly

---

## 📊 **Method 2: Testing with Actual Database Queries (YOU ARE HERE)**

This is the **critical test** - compare predictions against real execution times from your database.

### **What You Need to Do:**

#### **Step 1: Export Execution Plans from Your Database**

**For PostgreSQL:**
```sql
-- Get one execution plan
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM your_table WHERE condition;

-- Or export multiple queries
COPY (
  SELECT 1 as query_template, 100 as estimated_rows, ... 
  -- Include all 24 columns
) TO STDOUT WITH CSV HEADER;
```

**For MySQL:**
```sql
-- Get one execution plan
EXPLAIN FORMAT=JSON
SELECT * FROM your_table WHERE condition;

-- Or SELECT and save to CSV with all 24 columns
```

#### **Step 2: Create CSV File with 24 Columns**

Save as: `data/raw/actual_queries.csv`

Required columns (in order):
```
query_template, estimated_rows, actual_rows, startup_cost, total_cost,
actual_startup_time, actual_total_time, plan_width, loops,
base_cardinality, output_cardinality, input_cardinality,
filter_selectivity, join_ratio, cost_per_row, time_per_loop,
node_type, node_depth, parallel_awareness, subtree_cardinality,
num_children, cost_to_row_ratio, estimated_actual_ratio, time_cost_ratio
```

#### **Step 3: Run Test Script**

```bash
python3 test_with_actual_data.py
```

#### **Step 4: View Results**

```bash
# See all predictions
cat results/actual_data_predictions.csv

# See accuracy comparison (if actual_total_time provided)
cat results/actual_data_accuracy.csv
```

### **Expected Output:**

```
PostgreSQL Model:
   MAE: 15.3 ms
   Error: 8.5%
   Within ±10%: 85.2% of queries

MySQL Model:
   MAE: 12.1 ms
   Error: 6.2%
   Within ±10%: 91.3% of queries

Unified Model:
   MAE: 14.7 ms
   Error: 8.1%
   Within ±10%: 87.1% of queries

🏆 Best Model: MySQL
```

---

## 🚀 **Method 3: Production Deployment**

Once you've validated with Method 2, deploy to production:

### **Option A: Single Model Deployment**
```python
import pickle

# Load best-performing model
with open('models/mysql_lightgbm.pkl', 'rb') as f:
    model = pickle.load(f)

# Use in your application
def estimate_query_time(execution_plan_features):
    return model.predict(execution_plan_features)[0]
```

### **Option B: Multi-Database Deployment**
```python
def estimate_query_time(execution_plan, database_type):
    if database_type == 'postgresql':
        model = pg_model
    elif database_type == 'mysql':
        model = mysql_model
    else:
        model = unified_model
    
    return model.predict(execution_plan)[0]
```

---

## 📋 **Detailed Steps to Get Your Data**

See: **HOW_TO_GET_ACTUAL_DATA.md** (comprehensive guide with examples)

---

## ✨ **What Happens at Each Stage**

### **Stage 1: Sample Testing ✅ COMPLETE**
- **Purpose:** Verify models load and work
- **Time:** 5 minutes
- **Status:** All 3 models working

### **Stage 2: Actual Data Testing (NEXT)**
- **Purpose:** Validate accuracy against real queries
- **Time:** 30 minutes (once you export data)
- **How:** Run `test_with_actual_data.py`
- **Success:** Accuracy >80% on ±10% metric

### **Stage 3: Production Deployment**
- **Purpose:** Use models in real application
- **Time:** Varies by integration complexity
- **Success:** Predictions improve query optimization

---

## 🎯 **Quick Decision Tree**

```
Do you have actual execution plans from your database?
│
├─ NO → Go to: HOW_TO_GET_ACTUAL_DATA.md
│       (Export execution plans first)
│
└─ YES → Run test_with_actual_data.py
         │
         ├─ Is accuracy >80%? ✅ YES → Deploy to production
         │
         └─ NO → Retrain with more data
                 (See MULTI_DB_SUMMARY.md for next steps)
```

---

## 📊 **Files Created for Testing**

```
Testing Scripts:
  ✅ test_models.py                    - Test with sample data
  ✅ test_with_actual_data.py          - Test with real data
  ✅ extract_execution_plans.py        - Auto-extract from PostgreSQL

Documentation:
  ✅ HOW_TO_GET_ACTUAL_DATA.md         - Detailed data extraction guide
  ✅ TESTING_GUIDE.md                  - This file

Results:
  ✅ results/test_results.csv          - Sample data test results
  ⏳ results/actual_data_predictions.csv - (created after Step 3)
  ⏳ results/actual_data_accuracy.csv  - (created after Step 3)
```

---

## 🔍 **Metrics Explained**

### **Mean Absolute Error (MAE)**
- **What:** Average difference between predicted and actual time
- **Unit:** Milliseconds (ms)
- **Good:** < 20 ms
- **Excellent:** < 10 ms

### **Mean Absolute % Error (MAPE)**
- **What:** Average percent error across all predictions
- **Unit:** Percentage (%)
- **Good:** < 15%
- **Excellent:** < 10%

### **Within ±10% Accuracy**
- **What:** Percentage of predictions within ±10% of actual
- **Unit:** Percentage (%)
- **Good:** > 80%
- **Excellent:** > 90%

---

## 💡 **Tips for Best Results**

1. **Get Diverse Queries**
   - Mix simple and complex queries
   - Include different data sizes
   - Vary filter selectivity

2. **Use Real Execution Times**
   - Use `EXPLAIN ANALYZE` not just `EXPLAIN`
   - Capture `actual_total_time` column
   - Run queries on representative data volume

3. **Test with Same Database Type**
   - Use PostgreSQL model for PG databases
   - Use MySQL model for MySQL databases
   - Use Unified model for mixed environments

4. **Monitor Accuracy Over Time**
   - Accuracy may degrade with data changes
   - Retrain monthly with new queries
   - Watch for outliers or edge cases

---

## ✅ **Checklist: Ready to Test with Real Data?**

- [ ] Have you extracted execution plans from your database?
- [ ] Created CSV file with 24 columns?
- [ ] Saved to: `data/raw/actual_queries.csv`?
- [ ] Includes `actual_total_time` column?
- [ ] Ready to run: `python3 test_with_actual_data.py`?

**If YES to all:** You're ready to test! 🚀

**If NO:** See **HOW_TO_GET_ACTUAL_DATA.md** for detailed instructions.

---

## 📞 **Support**

- **Questions about data format?** → HOW_TO_GET_ACTUAL_DATA.md
- **Questions about model usage?** → MULTI_DB_USAGE_GUIDE.md
- **Questions about results?** → MULTI_DB_SUMMARY.md
- **Need to retrain?** → Train again with: `python3 train_multi_db.py`

---

**Status:** Sample testing ✅ complete. Ready for actual data testing.

**Next:** Export your execution plans and run `test_with_actual_data.py`
