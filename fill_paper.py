"""Fill RESEARCH_PAPER.md placeholders from results/real_traces_results.json."""
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(BASE, 'results', 'real_traces_results.json')))
paper = open(os.path.join(BASE, 'RESEARCH_PAPER.md')).read()

A = R['split_A_random']
B = R['split_B_unseen_templates']['results']
sigA = R['significance_A']
acc = R['accuracy_breakdown_lgb_A']
rt = R['runtime_summary']

M = {'LGB': 'LightGBM', 'XGB': 'XGBoost', 'CAT': 'CatBoost',
     'RF': 'RandomForest', 'MLP': 'MLP'}

tok = {}
for k, name in M.items():
    for split, res in (('A', A), ('B', B)):
        m = res[name]
        tok[f'{split}_{k}_R2'] = f"{m['r2']:.4f}"
        tok[f'{split}_{k}_RMSE'] = f"{m['rmse']:.1f}"
        tok[f'{split}_{k}_MAE'] = f"{m['mae']:.1f}"
        tok[f'{split}_{k}_MEDAE'] = f"{m['median_ae']:.1f}"
        tok[f'{split}_{k}_MAPE'] = f"{m['mape']:.1f}"
        tok[f'{split}_{k}_ACC10'] = f"{m['acc10']:.1f}"
        tok[f'{split}_{k}_ACC20'] = f"{m['acc20']:.1f}"

for k, name in [('XGB', 'XGBoost'), ('CAT', 'CatBoost'),
                ('RF', 'RandomForest'), ('MLP', 'MLP')]:
    s = sigA[name]
    tok[f'SIG_{k}_W'] = f"{s['wilcoxon_p']:.4f}" if s['wilcoxon_p'] >= 1e-4 else f"{s['wilcoxon_p']:.1e}"
    tok[f'SIG_{k}_T'] = f"{s['ttest_p']:.4f}" if s['ttest_p'] >= 1e-4 else f"{s['ttest_p']:.1e}"
    tok[f'SIG_{k}_YN'] = 'yes' if s['wilcoxon_p'] < 0.05 else 'no'

n = rt['n']
tok['N_TRACES'] = f"{n:,}"
tok['VARIANTS'] = '46'
tok['RT_MIN'] = f"{rt['min_ms']:.0f}"
tok['RT_MED'] = f"{rt['median_ms']:.0f}"
tok['RT_P25'] = f"{rt['p25_ms']:.0f}"
tok['RT_P75'] = f"{rt['p75_ms']:.0f}"
tok['RT_MAX'] = f"{rt['max_ms']/1000:.1f}"
tok['RT_MAX_MS'] = f"{rt['max_ms']:,.0f}"
tok['N_TEST'] = '203'
tok['N_VAL'] = '162'
tok['N_TRAIN'] = '647'
tok['HOLDOUT_TEMPLATES'] = ', '.join(R['split_B_unseen_templates']['holdout'])
tok['N_TEST_B'] = str(R['split_B_unseen_templates']['n_test'])
tok['B_BEST_R2'] = f"{max(B[m]['r2'] for m in B):.3f}"
tok['HW'] = 'Intel Core i7-8569U, 16 GB RAM'

for k in (5, 10, 15, 20, 30):
    tok[f'ACC{k}'] = f"{acc[f'±{k}%']:.1f}"

e = R['error_stats_lgb_A']
tok['ERR_MAE'] = f"{e['mae']:.1f}"

fi_rows = []
import csv
with open(os.path.join(BASE, 'results', 'feature_importance_real.csv')) as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= 10:
            break
        fi_rows.append(f"| {i+1} | `{row['feature']}` | {float(row['share_pct']):.1f}% |")
tok['FI_TABLE'] = '\n'.join(fi_rows)

tok['TRAIN_TIME'] = '2'
# narrative placeholders left for manual review get neutral defaults
tok.setdefault('TABLE1_NOTE', 'Best value per row in bold.')
tok.setdefault('SIG_NARRATIVE', '')
tok.setdefault('SPLITB_NARRATIVE', '')
tok.setdefault('PER_TMPL_NARRATIVE', '')

missing = set(re.findall(r'\{\{(\w+)\}\}', paper)) - set(tok)
for k, v in tok.items():
    paper = paper.replace('{{' + k + '}}', str(v))

open(os.path.join(BASE, 'RESEARCH_PAPER.md'), 'w').write(paper)
print("Filled. Unresolved tokens:", missing or "none")
