"""Generate TPC-H SF1 with DuckDB's tpch extension and load it into PostgreSQL."""
import os
import subprocess
import sys

import duckdb

SF = 1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tpch_sf1')
TABLES = ['region', 'nation', 'supplier', 'customer', 'part',
          'partsupp', 'orders', 'lineitem']

os.makedirs(OUT, exist_ok=True)

print(f"Generating TPC-H SF{SF} with DuckDB dbgen ...", flush=True)
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch;")
con.execute(f"CALL dbgen(sf={SF})")

for t in TABLES:
    path = os.path.join(OUT, f"{t}.csv")
    con.execute(f"COPY {t} TO '{path}' (FORMAT csv, HEADER false, DELIMITER '|')")
    print(f"  exported {t}", flush=True)
con.close()

base = os.path.dirname(os.path.abspath(__file__))
print("Creating schema ...", flush=True)
subprocess.run(['psql', '-d', 'tpch', '-q', '-f',
                os.path.join(base, 'sql', 'tpch_schema.sql')], check=True)

for t in TABLES:
    path = os.path.join(OUT, f"{t}.csv")
    print(f"Loading {t} ...", flush=True)
    subprocess.run(['psql', '-d', 'tpch', '-q', '-c',
                    f"\\copy {t} FROM '{path}' (FORMAT csv, DELIMITER '|')"],
                   check=True)

print("Creating indexes + ANALYZE ...", flush=True)
subprocess.run(['psql', '-d', 'tpch', '-q', '-f',
                os.path.join(base, 'sql', 'tpch_indexes.sql')], check=True)

out = subprocess.run(['psql', '-d', 'tpch', '-t', '-c',
                      "SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC"],
                     capture_output=True, text=True)
print(out.stdout)
print("TPC-H SF1 loaded.")
