"""
Train and compare 5 models on REAL PostgreSQL TPC-H SF1 execution traces.

Models:  LightGBM, XGBoost, CatBoost, Random Forest, MLP
Splits:  (A) random 64/16/20 stratified by template (interpolation)
         (B) template-disjoint: 5 templates held out entirely (extrapolation)
Stats:   paired Wilcoxon signed-rank + paired t-test on per-query abs errors
Output:  results/real_traces_results.json, results/feature_importance_real.csv,
         figures under results/figures/
"""

import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RNG = 42
BASE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(os.path.join(BASE, 'data', 'raw', 'tpch_sf1_real_traces.csv'))
print(f"Loaded {len(df)} real traces, "
      f"runtime {df.execution_time_ms.min():.1f}-{df.execution_time_ms.max():.1f} ms, "
      f"median {df.execution_time_ms.median():.1f} ms")

NON_FEATURES = ['query_id', 'query_template', 'params', 'planning_time_ms',
                'execution_time_ms', 'actual_root_rows']
FEATURES = [c for c in df.columns if c not in NON_FEATURES]
print(f"{len(FEATURES)} leak-free features: {FEATURES}")

# log-compress heavy-tailed planner quantities (monotone; harmless for trees,
# essential for the MLP)
SKEWED = ['startup_cost', 'total_cost', 'plan_rows', 'sum_est_rows',
          'max_est_rows', 'est_scan_rows', 'sum_est_width', 'cost_per_est_row']
X_all = df[FEATURES].copy()
for c in SKEWED:
    X_all[c] = np.log1p(X_all[c])
y_all = df['execution_time_ms'].values
y_log = np.log1p(y_all)
templates = df['query_template'].values


def make_models():
    return {
        'LightGBM': ('lgb', dict(objective='regression', metric='mse',
                                 num_leaves=31, max_depth=7, min_data_in_leaf=20,
                                 learning_rate=0.1, feature_fraction=0.8,
                                 bagging_fraction=0.8, bagging_freq=1,
                                 verbose=-1, seed=RNG)),
        'XGBoost': ('xgb', dict(objective='reg:squarederror', max_depth=7,
                                learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=1,
                                seed=RNG)),
        'CatBoost': ('cat', dict(depth=7, learning_rate=0.1, iterations=200,
                                 loss_function='RMSE', random_seed=RNG,
                                 verbose=0)),
        'RandomForest': ('rf', None),
        'MLP': ('mlp', None),
    }


def fit_predict(kind, params, Xtr, ytr, Xval, yval, Xte):
    """All models fit log1p(runtime); predictions are back-transformed."""
    if kind == 'lgb':
        m = lgb.train(params, lgb.Dataset(Xtr, label=ytr),
                      num_boost_round=200,
                      valid_sets=[lgb.Dataset(Xval, label=yval)],
                      callbacks=[lgb.early_stopping(10, verbose=False)])
        return m, m.predict(Xte)
    if kind == 'xgb':
        m = xgb.train(params, xgb.DMatrix(Xtr, label=ytr),
                      num_boost_round=200,
                      evals=[(xgb.DMatrix(Xval, label=yval), 'val')],
                      early_stopping_rounds=10, verbose_eval=False)
        return m, m.predict(xgb.DMatrix(Xte))
    if kind == 'cat':
        m = CatBoostRegressor(**params)
        m.fit(Xtr, ytr, eval_set=(Xval, yval), early_stopping_rounds=10)
        return m, m.predict(Xte)
    if kind == 'rf':
        m = RandomForestRegressor(n_estimators=300, max_depth=None,
                                  min_samples_leaf=2, random_state=RNG,
                                  n_jobs=-1)
        m.fit(np.vstack([Xtr, Xval]), np.concatenate([ytr, yval]))
        return m, m.predict(Xte)
    if kind == 'mlp':
        sc = StandardScaler().fit(Xtr)
        m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000,
                         early_stopping=True, n_iter_no_change=15,
                         random_state=RNG)
        m.fit(sc.transform(np.vstack([Xtr, Xval])),
              np.concatenate([ytr, yval]))
        return m, m.predict(sc.transform(Xte))
    raise ValueError(kind)


def metrics(y_true, y_pred):
    rel = np.abs(y_true - y_pred) / y_true
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'median_ae': float(np.median(np.abs(y_true - y_pred))),
        'mape': float(np.mean(rel) * 100),
        'acc10': float(np.mean(rel <= 0.10) * 100),
        'acc20': float(np.mean(rel <= 0.20) * 100),
    }


def run_split(name, tr_idx, val_idx, te_idx):
    Xtr, Xval, Xte = (X_all.values[i] for i in (tr_idx, val_idx, te_idx))
    ytr, yval = y_log[tr_idx], y_log[val_idx]
    yte_ms = y_all[te_idx]
    out, preds, models = {}, {}, {}
    for label, (kind, params) in make_models().items():
        m, p_log = fit_predict(kind, params, Xtr, ytr, Xval, yval, Xte)
        p_ms = np.maximum(np.expm1(p_log), 0.0)
        out[label] = metrics(yte_ms, p_ms)
        preds[label] = p_ms
        models[label] = m
        print(f"  [{name}] {label:13s} R2={out[label]['r2']:.4f} "
              f"MAE={out[label]['mae']:.1f}ms MAPE={out[label]['mape']:.1f}% "
              f"acc10={out[label]['acc10']:.1f}%")
    # paired significance tests: LightGBM vs each baseline on per-query |err|
    err_lgb = np.abs(yte_ms - preds['LightGBM'])
    sig = {}
    for label in preds:
        if label == 'LightGBM':
            continue
        err_b = np.abs(yte_ms - preds[label])
        w_stat, w_p = stats.wilcoxon(err_lgb, err_b)
        # t-test on log-scale absolute errors (approx. normal after log)
        t_stat, t_p = stats.ttest_rel(np.log1p(err_lgb), np.log1p(err_b))
        sig[label] = {'wilcoxon_p': float(w_p), 'ttest_p': float(t_p),
                      'lgb_mae': float(err_lgb.mean()),
                      'other_mae': float(err_b.mean())}
    return out, preds, models, sig, yte_ms


# ------------------------------------------- split A: random, stratified
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.20, random_state=RNG,
                                  stratify=templates)
tr_idx, val_idx = train_test_split(tr_idx, test_size=0.20, random_state=RNG,
                                   stratify=templates[tr_idx])
print(f"\nSplit A (random stratified): train={len(tr_idx)} "
      f"val={len(val_idx)} test={len(te_idx)}")
res_A, preds_A, models_A, sig_A, yte_A = run_split('A', tr_idx, val_idx, te_idx)

# --------------------------------------- split B: unseen query templates
rng = np.random.RandomState(RNG)
holdout_templates = sorted(rng.choice(sorted(set(templates)), 5, replace=False))
print(f"\nSplit B (template-disjoint): held-out templates = {holdout_templates}")
mask_te = np.isin(templates, holdout_templates)
idx_te_B = idx[mask_te]
idx_tr_B = idx[~mask_te]
idx_tr_B, idx_val_B = train_test_split(idx_tr_B, test_size=0.20,
                                       random_state=RNG,
                                       stratify=templates[idx_tr_B])
res_B, preds_B, _, sig_B, yte_B = run_split('B', idx_tr_B, idx_val_B, idx_te_B)

# ------------------------------------------------- feature importance
lgb_model = models_A['LightGBM']
imp = pd.DataFrame({'feature': FEATURES,
                    'gain': lgb_model.feature_importance('gain')})
imp = imp.sort_values('gain', ascending=False).reset_index(drop=True)
imp['share_pct'] = 100 * imp.gain / imp.gain.sum()
imp.to_csv(os.path.join(BASE, 'results', 'feature_importance_real.csv'),
           index=False)
print("\nTop features:\n", imp.head(12).to_string(index=False))

# ------------------------------------------------- accuracy breakdown
rel_A = np.abs(yte_A - preds_A['LightGBM']) / yte_A
acc_table = {f"±{k}%": float(np.mean(rel_A <= k / 100) * 100)
             for k in (5, 10, 15, 20, 30)}

# error stats
err = np.abs(yte_A - preds_A['LightGBM'])
err_stats = {'mae': float(err.mean()), 'std': float(err.std()),
             'median': float(np.median(err)), 'min': float(err.min()),
             'max': float(err.max())}

# per-template MAPE (LightGBM, split A)
te_templates = templates[te_idx]
per_tmpl = (pd.DataFrame({'template': te_templates,
                          'rel': rel_A * 100})
            .groupby('template')['rel'].median()
            .sort_values())

# runtime distribution summary
runtime_summary = {
    'n': int(len(df)),
    'min_ms': float(df.execution_time_ms.min()),
    'p25_ms': float(df.execution_time_ms.quantile(.25)),
    'median_ms': float(df.execution_time_ms.median()),
    'p75_ms': float(df.execution_time_ms.quantile(.75)),
    'max_ms': float(df.execution_time_ms.max()),
}

json.dump({
    'features': FEATURES,
    'runtime_summary': runtime_summary,
    'split_A_random': res_A, 'significance_A': sig_A,
    'split_B_unseen_templates': {'holdout': list(map(str, holdout_templates)),
                                 'n_test': int(mask_te.sum()), **{'results': res_B}},
    'significance_B': sig_B,
    'accuracy_breakdown_lgb_A': acc_table,
    'error_stats_lgb_A': err_stats,
    'per_template_median_ape': {str(k): float(v) for k, v in per_tmpl.items()},
}, open(os.path.join(BASE, 'results', 'real_traces_results.json'), 'w'),
    indent=2)
print("\nSaved results/real_traces_results.json")

# ================================================================ figures
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3e0'
SERIES = {'LightGBM': '#2a78d6', 'XGBoost': '#1baf7a', 'CatBoost': '#eda100',
          'RandomForest': '#008300', 'MLP': '#4a3aa7'}
plt.rcParams.update({
    'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'sans-serif',
})

# --- Fig 1: predicted vs actual (log-log), LightGBM, split A
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(yte_A, preds_A['LightGBM'], s=14, alpha=0.55,
           color=SERIES['LightGBM'], edgecolors='none')
lims = [yte_A.min() * 0.7, yte_A.max() * 1.4]
ax.plot(lims, lims, ls='--', lw=1, color=INK2)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual execution time (ms)')
ax.set_ylabel('Predicted execution time (ms)')
ax.set_title(f"LightGBM on real TPC-H SF1 traces "
             f"(R² = {res_A['LightGBM']['r2']:.3f})", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig1_pred_vs_actual.png'), dpi=250)
plt.close(fig)

# --- Fig 2: relative-error CDF, all 5 models, split A
fig, ax = plt.subplots(figsize=(5.6, 3.8))
xs = np.linspace(0, 60, 400)
for label, p in preds_A.items():
    rel = np.abs(yte_A - p) / yte_A * 100
    cdf = [np.mean(rel <= x) * 100 for x in xs]
    ax.plot(xs, cdf, lw=2, color=SERIES[label], label=label)
ax.axvline(10, color=INK2, lw=0.8, ls=':')
ax.text(10.7, 8, '±10%', fontsize=8, color=INK2)
ax.set_xlim(0, 60); ax.set_ylim(0, 100)
ax.set_xlabel('Relative error threshold (%)')
ax.set_ylabel('Queries within threshold (%)')
ax.set_title('Prediction accuracy profile (test set, random split)',
             fontsize=9.5)
ax.legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig2_error_cdf.png'), dpi=250)
plt.close(fig)

# --- Fig 3: feature importance top 12 (sequential single hue)
top = imp.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.barh(top.feature, top.share_pct, color='#2a78d6', height=0.62)
for y_, v in enumerate(top.share_pct):
    ax.text(v + 0.3, y_, f"{v:.1f}%", va='center', fontsize=8, color=INK2)
ax.set_xlabel('Share of total split gain (%)')
ax.set_title('LightGBM feature importance (gain)', fontsize=9.5)
ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig3_feature_importance.png'), dpi=250)
plt.close(fig)

# --- Fig 4: per-template median relative error (LightGBM, split A)
fig, ax = plt.subplots(figsize=(6.2, 3.2))
order = per_tmpl.index.tolist()
ax.bar(order, per_tmpl.values, color='#2a78d6', width=0.62)
ax.axhline(10, color=INK2, lw=0.8, ls=':')
ax.text(len(order) - 0.4, 10.6, '10%', fontsize=8, color=INK2, ha='right')
ax.set_ylabel('Median relative error (%)')
ax.set_title('LightGBM median relative error by TPC-H template', fontsize=9.5)
ax.tick_params(axis='x', labelsize=7.5, rotation=45)
ax.grid(axis='x', visible=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'fig4_per_template.png'), dpi=250)
plt.close(fig)

print("Saved 4 figures to results/figures/")
