"""
Collect REAL PostgreSQL execution traces on TPC-H scale factor 1.

For each of the 22 TPC-H templates, generates parameter variants per the
TPC-H 3.0 specification, executes them with EXPLAIN (ANALYZE, FORMAT JSON),
and extracts a LEAK-FREE feature vector: only quantities available from the
query PLANNER before execution (costs, row estimates, plan structure).
The target is the measured execution time in milliseconds.


Usage: python collect_real_traces.py [--variants 46] [--timeout 120]
"""

import argparse
import json
import random

import psycopg2

REGIONS = ['AFRICA', 'AMERICA', 'ASIA', 'EUROPE', 'MIDDLE EAST']
NATION_REGION = {
    'ALGERIA': 'AFRICA', 'ARGENTINA': 'AMERICA', 'BRAZIL': 'AMERICA',
    'CANADA': 'AMERICA', 'EGYPT': 'MIDDLE EAST', 'ETHIOPIA': 'AFRICA',
    'FRANCE': 'EUROPE', 'GERMANY': 'EUROPE', 'INDIA': 'ASIA',
    'INDONESIA': 'ASIA', 'IRAN': 'MIDDLE EAST', 'IRAQ': 'MIDDLE EAST',
    'JAPAN': 'ASIA', 'JORDAN': 'MIDDLE EAST', 'KENYA': 'AFRICA',
    'MOROCCO': 'AFRICA', 'MOZAMBIQUE': 'AFRICA', 'PERU': 'AMERICA',
    'CHINA': 'ASIA', 'ROMANIA': 'EUROPE', 'SAUDI ARABIA': 'MIDDLE EAST',
    'VIETNAM': 'ASIA', 'RUSSIA': 'EUROPE', 'UNITED KINGDOM': 'EUROPE',
    'UNITED STATES': 'AMERICA',
}
NATIONS = list(NATION_REGION)
TYPE_S1 = ['STANDARD', 'SMALL', 'MEDIUM', 'LARGE', 'ECONOMY', 'PROMO']
TYPE_S2 = ['ANODIZED', 'BURNISHED', 'PLATED', 'POLISHED', 'BRUSHED']
TYPE_S3 = ['TIN', 'NICKEL', 'BRASS', 'STEEL', 'COPPER']
SEGMENTS = ['AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD']
SHIPMODES = ['REG AIR', 'AIR', 'RAIL', 'SHIP', 'TRUCK', 'MAIL', 'FOB']
CONTAINER_S1 = ['SM', 'LG', 'MED', 'JUMBO', 'WRAP']
CONTAINER_S2 = ['CASE', 'BOX', 'BAG', 'JAR', 'PKG', 'PACK', 'CAN', 'DRUM']
COLORS = [
    'almond', 'antique', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
    'blanched', 'blue', 'blush', 'brown', 'burlywood', 'burnished',
    'chartreuse', 'chiffon', 'chocolate', 'coral', 'cornflower', 'cornsilk',
    'cream', 'cyan', 'dark', 'deep', 'dim', 'dodger', 'drab', 'firebrick',
    'floral', 'forest', 'frosted', 'gainsboro', 'ghost', 'goldenrod',
    'green', 'grey', 'honeydew', 'hot', 'indian', 'ivory', 'khaki', 'lace',
    'lavender', 'lawn', 'lemon', 'light', 'lime', 'linen', 'magenta',
    'maroon', 'medium', 'metallic', 'midnight', 'mint', 'misty', 'moccasin',
    'navajo', 'navy', 'olive', 'orange', 'orchid', 'pale', 'papaya',
    'peach', 'peru', 'pink', 'plum', 'powder', 'puff', 'purple', 'red',
    'rose', 'rosy', 'royal', 'saddle', 'salmon', 'sandy', 'seashell',
    'sienna', 'sky', 'slate', 'smoke', 'snow', 'spring', 'steel', 'tan',
    'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'yellow',
]


def month_start(rng, y0, m0, y1, m1):
    months = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        months.append(f"{y}-{m:02d}-01")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return rng.choice(months)


def gen_params(template, rng):
    """Parameter substitution ranges per TPC-H 3.0 Clause 2.4.x."""
    if template == 1:
        return {'delta': rng.randint(60, 120)}
    if template == 2:
        return {'size': rng.randint(1, 50), 'type': rng.choice(TYPE_S3),
                'region': rng.choice(REGIONS)}
    if template == 3:
        return {'segment': rng.choice(SEGMENTS),
                'date': f"1995-03-{rng.randint(1, 31):02d}"}
    if template == 4:
        return {'date': month_start(rng, 1993, 1, 1997, 10)}
    if template == 5:
        return {'region': rng.choice(REGIONS),
                'date': f"{rng.randint(1993, 1997)}-01-01"}
    if template == 6:
        return {'date': f"{rng.randint(1993, 1997)}-01-01",
                'discount': round(rng.randint(2, 9) / 100, 2),
                'quantity': rng.choice([24, 25])}
    if template == 7:
        n1, n2 = rng.sample(NATIONS, 2)
        return {'nation1': n1, 'nation2': n2}
    if template == 8:
        nation = rng.choice(NATIONS)
        return {'nation': nation, 'region': NATION_REGION[nation],
                'type': f"{rng.choice(TYPE_S1)} {rng.choice(TYPE_S2)} {rng.choice(TYPE_S3)}"}
    if template == 9:
        return {'color': rng.choice(COLORS)}
    if template == 10:
        return {'date': month_start(rng, 1993, 2, 1995, 1)}
    if template == 11:
        return {'nation': rng.choice(NATIONS)}
    if template == 12:
        m1, m2 = rng.sample(SHIPMODES, 2)
        return {'m1': m1, 'm2': m2, 'date': f"{rng.randint(1993, 1997)}-01-01"}
    if template == 13:
        return {'w1': rng.choice(['special', 'pending', 'unusual', 'express']),
                'w2': rng.choice(['packages', 'requests', 'accounts', 'deposits'])}
    if template == 14:
        return {'date': month_start(rng, 1993, 1, 1997, 12)}
    if template == 15:
        return {'date': month_start(rng, 1993, 1, 1997, 10)}
    if template == 16:
        sizes = rng.sample(range(1, 51), 8)
        return {'brand': f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
                'typeprefix': f"{rng.choice(TYPE_S1)} {rng.choice(TYPE_S2)}",
                'sizes': ', '.join(str(s) for s in sorted(sizes))}
    if template == 17:
        return {'brand': f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
                'container': f"{rng.choice(CONTAINER_S1)} {rng.choice(CONTAINER_S2)}"}
    if template == 18:
        return {'q': rng.randint(312, 315)}
    if template == 19:
        return {'b1': f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
                'b2': f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
                'b3': f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
                'q1': rng.randint(1, 10), 'q2': rng.randint(10, 20),
                'q3': rng.randint(20, 30)}
    if template == 20:
        return {'color': rng.choice(COLORS),
                'date': f"{rng.randint(1993, 1997)}-01-01",
                'nation': rng.choice(NATIONS)}
    if template == 21:
        return {'nation': rng.choice(NATIONS)}
    if template == 22:
        codes = rng.sample(range(10, 35), 7)
        return {'codes': ', '.join(f"'{c}'" for c in sorted(codes))}
    raise ValueError(template)


TEMPLATES = {
1: """select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty,
       sum(l_extendedprice) as sum_base_price,
       sum(l_extendedprice*(1-l_discount)) as sum_disc_price,
       sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
       avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price,
       avg(l_discount) as avg_disc, count(*) as count_order
from lineitem
where l_shipdate <= date '1998-12-01' - interval '{delta} days'
group by l_returnflag, l_linestatus
order by l_returnflag, l_linestatus""",

2: """select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
from part, supplier, partsupp, nation, region
where p_partkey = ps_partkey and s_suppkey = ps_suppkey
  and p_size = {size} and p_type like '%{type}'
  and s_nationkey = n_nationkey and n_regionkey = r_regionkey
  and r_name = '{region}'
  and ps_supplycost = (
      select min(ps_supplycost)
      from partsupp, supplier, nation, region
      where p_partkey = ps_partkey and s_suppkey = ps_suppkey
        and s_nationkey = n_nationkey and n_regionkey = r_regionkey
        and r_name = '{region}')
order by s_acctbal desc, n_name, s_name, p_partkey
limit 100""",

3: """select l_orderkey, sum(l_extendedprice*(1-l_discount)) as revenue,
       o_orderdate, o_shippriority
from customer, orders, lineitem
where c_mktsegment = '{segment}' and c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and o_orderdate < date '{date}' and l_shipdate > date '{date}'
group by l_orderkey, o_orderdate, o_shippriority
order by revenue desc, o_orderdate
limit 10""",

4: """select o_orderpriority, count(*) as order_count
from orders
where o_orderdate >= date '{date}'
  and o_orderdate < date '{date}' + interval '3 months'
  and exists (select * from lineitem
              where l_orderkey = o_orderkey and l_commitdate < l_receiptdate)
group by o_orderpriority
order by o_orderpriority""",

5: """select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue
from customer, orders, lineitem, supplier, nation, region
where c_custkey = o_custkey and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey and n_regionkey = r_regionkey
  and r_name = '{region}'
  and o_orderdate >= date '{date}'
  and o_orderdate < date '{date}' + interval '1 year'
group by n_name
order by revenue desc""",

6: """select sum(l_extendedprice * l_discount) as revenue
from lineitem
where l_shipdate >= date '{date}'
  and l_shipdate < date '{date}' + interval '1 year'
  and l_discount between {discount} - 0.01 and {discount} + 0.01
  and l_quantity < {quantity}""",

7: """select supp_nation, cust_nation, l_year, sum(volume) as revenue
from (select n1.n_name as supp_nation, n2.n_name as cust_nation,
             extract(year from l_shipdate) as l_year,
             l_extendedprice * (1 - l_discount) as volume
      from supplier, lineitem, orders, customer, nation n1, nation n2
      where s_suppkey = l_suppkey and o_orderkey = l_orderkey
        and c_custkey = o_custkey and s_nationkey = n1.n_nationkey
        and c_nationkey = n2.n_nationkey
        and ((n1.n_name = '{nation1}' and n2.n_name = '{nation2}')
          or (n1.n_name = '{nation2}' and n2.n_name = '{nation1}'))
        and l_shipdate between date '1995-01-01' and date '1996-12-31') as shipping
group by supp_nation, cust_nation, l_year
order by supp_nation, cust_nation, l_year""",

8: """select o_year,
       sum(case when nation = '{nation}' then volume else 0 end) / sum(volume) as mkt_share
from (select extract(year from o_orderdate) as o_year,
             l_extendedprice * (1 - l_discount) as volume, n2.n_name as nation
      from part, supplier, lineitem, orders, customer, nation n1, nation n2, region
      where p_partkey = l_partkey and s_suppkey = l_suppkey
        and l_orderkey = o_orderkey and o_custkey = c_custkey
        and c_nationkey = n1.n_nationkey and n1.n_regionkey = r_regionkey
        and r_name = '{region}' and s_nationkey = n2.n_nationkey
        and o_orderdate between date '1995-01-01' and date '1996-12-31'
        and p_type = '{type}') as all_nations
group by o_year
order by o_year""",

9: """select nation, o_year, sum(amount) as sum_profit
from (select n_name as nation, extract(year from o_orderdate) as o_year,
             l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity as amount
      from part, supplier, lineitem, partsupp, orders, nation
      where s_suppkey = l_suppkey and ps_suppkey = l_suppkey
        and ps_partkey = l_partkey and p_partkey = l_partkey
        and o_orderkey = l_orderkey and s_nationkey = n_nationkey
        and p_name like '%{color}%') as profit
group by nation, o_year
order by nation, o_year desc""",

10: """select c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) as revenue,
        c_acctbal, n_name, c_address, c_phone, c_comment
from customer, orders, lineitem, nation
where c_custkey = o_custkey and l_orderkey = o_orderkey
  and o_orderdate >= date '{date}'
  and o_orderdate < date '{date}' + interval '3 months'
  and l_returnflag = 'R' and c_nationkey = n_nationkey
group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
order by revenue desc
limit 20""",

11: """select ps_partkey, sum(ps_supplycost * ps_availqty) as value
from partsupp, supplier, nation
where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = '{nation}'
group by ps_partkey
having sum(ps_supplycost * ps_availqty) > (
    select sum(ps_supplycost * ps_availqty) * 0.0001
    from partsupp, supplier, nation
    where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = '{nation}')
order by value desc""",

12: """select l_shipmode,
        sum(case when o_orderpriority = '1-URGENT' or o_orderpriority = '2-HIGH'
            then 1 else 0 end) as high_line_count,
        sum(case when o_orderpriority <> '1-URGENT' and o_orderpriority <> '2-HIGH'
            then 1 else 0 end) as low_line_count
from orders, lineitem
where o_orderkey = l_orderkey and l_shipmode in ('{m1}', '{m2}')
  and l_commitdate < l_receiptdate and l_shipdate < l_commitdate
  and l_receiptdate >= date '{date}'
  and l_receiptdate < date '{date}' + interval '1 year'
group by l_shipmode
order by l_shipmode""",

13: """select c_count, count(*) as custdist
from (select c_custkey, count(o_orderkey) as c_count
      from customer left outer join orders
        on c_custkey = o_custkey and o_comment not like '%{w1}%{w2}%'
      group by c_custkey) as c_orders
group by c_count
order by custdist desc, c_count desc""",

14: """select 100.00 * sum(case when p_type like 'PROMO%'
        then l_extendedprice * (1 - l_discount) else 0 end)
        / sum(l_extendedprice * (1 - l_discount)) as promo_revenue
from lineitem, part
where l_partkey = p_partkey
  and l_shipdate >= date '{date}'
  and l_shipdate < date '{date}' + interval '1 month'""",

15: """with revenue0 as (
    select l_suppkey as supplier_no,
           sum(l_extendedprice * (1 - l_discount)) as total_revenue
    from lineitem
    where l_shipdate >= date '{date}'
      and l_shipdate < date '{date}' + interval '3 months'
    group by l_suppkey)
select s_suppkey, s_name, s_address, s_phone, total_revenue
from supplier, revenue0
where s_suppkey = supplier_no
  and total_revenue = (select max(total_revenue) from revenue0)
order by s_suppkey""",

16: """select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt
from partsupp, part
where p_partkey = ps_partkey and p_brand <> '{brand}'
  and p_type not like '{typeprefix}%' and p_size in ({sizes})
  and ps_suppkey not in (
      select s_suppkey from supplier
      where s_comment like '%Customer%Complaints%')
group by p_brand, p_type, p_size
order by supplier_cnt desc, p_brand, p_type, p_size""",

17: """select sum(l_extendedprice) / 7.0 as avg_yearly
from lineitem, part
where p_partkey = l_partkey and p_brand = '{brand}' and p_container = '{container}'
  and l_quantity < (select 0.2 * avg(l_quantity)
                    from lineitem where l_partkey = p_partkey)""",

18: """select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
from customer, orders, lineitem
where o_orderkey in (select l_orderkey from lineitem
                     group by l_orderkey having sum(l_quantity) > {q})
  and c_custkey = o_custkey and o_orderkey = l_orderkey
group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
order by o_totalprice desc, o_orderdate
limit 100""",

19: """select sum(l_extendedprice* (1 - l_discount)) as revenue
from lineitem, part
where (p_partkey = l_partkey and p_brand = '{b1}'
   and p_container in ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
   and l_quantity >= {q1} and l_quantity <= {q1} + 10
   and p_size between 1 and 5
   and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON')
or (p_partkey = l_partkey and p_brand = '{b2}'
   and p_container in ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
   and l_quantity >= {q2} and l_quantity <= {q2} + 10
   and p_size between 1 and 10
   and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON')
or (p_partkey = l_partkey and p_brand = '{b3}'
   and p_container in ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
   and l_quantity >= {q3} and l_quantity <= {q3} + 10
   and p_size between 1 and 15
   and l_shipmode in ('AIR', 'AIR REG') and l_shipinstruct = 'DELIVER IN PERSON')""",

20: """select s_name, s_address
from supplier, nation
where s_suppkey in (
    select ps_suppkey from partsupp
    where ps_partkey in (select p_partkey from part where p_name like '{color}%')
      and ps_availqty > (
          select 0.5 * sum(l_quantity) from lineitem
          where l_partkey = ps_partkey and l_suppkey = ps_suppkey
            and l_shipdate >= date '{date}'
            and l_shipdate < date '{date}' + interval '1 year'))
  and s_nationkey = n_nationkey and n_name = '{nation}'
order by s_name""",

21: """select s_name, count(*) as numwait
from supplier, lineitem l1, orders, nation
where s_suppkey = l1.l_suppkey and o_orderkey = l1.l_orderkey
  and o_orderstatus = 'F' and l1.l_receiptdate > l1.l_commitdate
  and exists (select * from lineitem l2
              where l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey)
  and not exists (select * from lineitem l3
                  where l3.l_orderkey = l1.l_orderkey
                    and l3.l_suppkey <> l1.l_suppkey
                    and l3.l_receiptdate > l3.l_commitdate)
  and s_nationkey = n_nationkey and n_name = '{nation}'
group by s_name
order by numwait desc, s_name
limit 100""",

22: """select cntrycode, count(*) as numcust, sum(c_acctbal) as totacctbal
from (select substring(c_phone from 1 for 2) as cntrycode, c_acctbal
      from customer
      where substring(c_phone from 1 for 2) in ({codes})
        and c_acctbal > (select avg(c_acctbal) from customer
                         where c_acctbal > 0.00
                           and substring(c_phone from 1 for 2) in ({codes}))
        and not exists (select * from orders where o_custkey = c_custkey)) as custsale
group by cntrycode
order by cntrycode""",
}

SCAN_NODES = {'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan',
              'Tid Scan', 'CTE Scan', 'Subquery Scan', 'Function Scan'}


def extract_features(explain_json, template, params):
    """Leak-free features: planner estimates + plan structure only."""
    root = explain_json[0]['Plan']
    nodes, relations = [], set()

    def walk(node, depth):
        nodes.append((node, depth))
        if 'Relation Name' in node:
            relations.add(node['Relation Name'])
        for child in node.get('Plans', []):
            walk(child, depth + 1)

    walk(root, 0)

    def count(*types):
        return sum(1 for n, _ in nodes if n['Node Type'] in types)

    total_cost = root['Total Cost']
    plan_rows = root['Plan Rows']
    feats = {
        'query_template': f'Q{template}',
        'params': json.dumps(params),
        # planner cost estimates (root)
        'startup_cost': root['Startup Cost'],
        'total_cost': total_cost,
        'plan_rows': plan_rows,
        'plan_width': root['Plan Width'],
        # plan structure
        'num_nodes': len(nodes),
        'plan_depth': max(d for _, d in nodes),
        'num_relations': len(relations),
        'n_seq_scan': count('Seq Scan'),
        'n_index_scan': count('Index Scan', 'Index Only Scan'),
        'n_bitmap_scan': count('Bitmap Heap Scan', 'Bitmap Index Scan'),
        'n_hash_join': count('Hash Join'),
        'n_merge_join': count('Merge Join'),
        'n_nested_loop': count('Nested Loop'),
        'n_sort': count('Sort', 'Incremental Sort'),
        'n_aggregate': count('Aggregate', 'GroupAggregate', 'HashAggregate', 'WindowAgg'),
        'n_gather': count('Gather', 'Gather Merge'),
        'n_materialize': count('Materialize', 'Memoize'),
        'workers_planned': sum(n.get('Workers Planned', 0) for n, _ in nodes),
        # planner cardinality aggregates over the plan tree
        'sum_est_rows': sum(n['Plan Rows'] for n, _ in nodes),
        'max_est_rows': max(n['Plan Rows'] for n, _ in nodes),
        'est_scan_rows': sum(n['Plan Rows'] for n, _ in nodes
                             if n['Node Type'] in SCAN_NODES),
        'sum_est_width': sum(n['Plan Width'] for n, _ in nodes),
        # derived (planner-only)
        'cost_per_est_row': total_cost / (plan_rows + 1),
        'startup_cost_frac': root['Startup Cost'] / (total_cost + 1),
        'n_joins': count('Hash Join', 'Merge Join', 'Nested Loop'),
        # target + non-feature diagnostics
        'planning_time_ms': explain_json[0].get('Planning Time', None),
        'execution_time_ms': explain_json[0]['Execution Time'],
        'actual_root_rows': root.get('Actual Rows', None),
    }
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', type=int, default=46)
    ap.add_argument('--timeout', type=int, default=120, help='seconds')
    ap.add_argument('--out', default='data/raw/tpch_sf1_real_traces.csv')
    ap.add_argument('--plans', default='data/raw/tpch_sf1_plans.jsonl')
    args = ap.parse_args()

    rng = random.Random(42)
    jobs = []
    for t in range(1, 23):
        seen = set()
        for _ in range(args.variants):
            for _attempt in range(50):
                p = gen_params(t, rng)
                key = json.dumps(p, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    break
            jobs.append((t, p))
    rng.shuffle(jobs)  # randomize execution order to decorrelate cache state

    conn = psycopg2.connect(dbname='tpch')
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SET statement_timeout = {args.timeout * 1000}")

    # warm the OS/buffer cache once so measurements are steady-state
    for tbl in ['region', 'nation', 'supplier', 'customer', 'part',
                'partsupp', 'orders', 'lineitem']:
        cur.execute(f"SELECT count(*) FROM {tbl}")
        print(f"warmup {tbl}: {cur.fetchone()[0]} rows", flush=True)

    import csv
    rows, failures = [], 0
    plans_out = open(args.plans, 'w')
    for i, (t, p) in enumerate(jobs):
        sql = TEMPLATES[t].format(**p)
        try:
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
            plan = cur.fetchone()[0]
            feats = extract_features(plan, t, p)
            feats['query_id'] = len(rows)
            rows.append(feats)
            plans_out.write(json.dumps({'template': t, 'params': p,
                                        'plan': plan}) + '\n')
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(jobs)}] Q{t} {feats['execution_time_ms']:.1f} ms",
                      flush=True)
        except Exception as e:
            failures += 1
            conn.rollback() if not conn.autocommit else None
            print(f"[{i+1}/{len(jobs)}] Q{t} FAILED: {str(e)[:120]}", flush=True)
            cur.execute(f"SET statement_timeout = {args.timeout * 1000}")
    plans_out.close()

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone: {len(rows)} traces collected, {failures} failures")
    print(f"Features -> {args.out}\nPlans -> {args.plans}")


if __name__ == '__main__':
    main()
