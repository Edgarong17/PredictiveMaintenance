"""Light-theme figures for the week-3 class slides. Uses validation data and week-3 CSV outputs only
(the test split is not touched). Run from the repo root: python docs/make_dark_figures.py"""
import os, sys, warnings
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pm_dl import build_hourly_channels, SequenceStore, SplitData, predict, COMPS
from pm_week3 import (load_raw, load_split, load_week2_best, apply_temperature, reliability_table, integrated_gradients,
                      PolicySimulator, threshold_grid)
import json

OUT = os.path.join('docs', 'figs_light'); os.makedirs(OUT, exist_ok=True)
BG, FG, MUT, GRID = '#FFFFFF', '#1A1A1A', '#6B7079', '#E4E2DD'
AMBER, TEAL, RED, BLUE = '#D98E04', '#1A9E8F', '#D4432F', '#2B7BD6'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG, 'axes.edgecolor': GRID,
                     'axes.labelcolor': MUT, 'xtick.color': MUT, 'ytick.color': MUT, 'text.color': FG, 'font.family': 'DejaVu Sans',
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 11})

tel, err, mnt, fail, mach = load_raw()
grid = build_hourly_channels(tel, err, mnt)
model, ck = load_week2_best()
store = SequenceStore(grid, mach, ck['norm_stats'])
VA = SplitData(load_split('validation', mach), store, ck['age_stats'])
D = json.load(open(os.path.join('data', 'processed', 'week3_decisions.json')))
pb = apply_temperature(predict(model, VA, 24)[0], D['temperature'])
short = ['volt', 'rotate', 'pressure', 'vibration', 'err1', 'err2', 'err3', 'err4', 'err5', 'h_c1', 'h_c2', 'h_c3', 'h_c4']

# 1) one real pre-failure window (comp1) as the "tokens" the model sees
idx = np.flatnonzero(VA.ym == 1)
i0 = idx[len(idx) // 2]
x, m, a, _, _ = VA.batch(np.array([i0]), 24)
w = x[0].numpy().T  # (13, 24)
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.imshow(np.clip(w, -3, 3), aspect='auto', cmap='magma', vmin=-3, vmax=3)
ax.set_yticks(range(13)); ax.set_yticklabels(short, fontsize=10, family='DejaVu Sans Mono')
ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(['t-23', 't-17', 't-11', 't-5', 't'], family='DejaVu Sans Mono')
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'window.png'), dpi=220); plt.close()

# 2) reliability (validation, calibrated final model)
c = reliability_table(VA.yb, pb, bins=10)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot([0, 1], [0, 1], ls=':', color=MUT, lw=1)
ax.plot(c['mean_p'], c['observed'], marker='o', color=AMBER, lw=2.2, ms=6)
ax.set_xlabel('probabilidad predicha'); ax.set_ylabel('tasa de falla observada'); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax2 = ax.inset_axes([0.58, 0.08, 0.38, 0.3])
ax2.bar(range(len(c)), c['n'], color=TEAL); ax2.set_yscale('log'); ax2.set_xticks([]); ax2.tick_params(labelsize=7)
ax2.set_title('horas por bin (log)', fontsize=8, color=MUT); ax2.grid(False); ax2.set_facecolor(BG)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'reliability.png'), dpi=220); plt.close()

# 3) integrated gradients heatmaps (validation, 200 pre-failure windows per component)
rng = np.random.default_rng(42)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
for k, (ax, comp) in enumerate(zip(axes, COMPS), start=1):
    ids = np.sort(rng.choice(np.flatnonzero(VA.ym == k), 200, replace=False))
    xx, mm, aa, _, _ = VA.batch(ids, 24)
    att = integrated_gradients(model, xx, mm, aa, target=k, steps=50).numpy()
    hm = np.abs(att).mean(0).T
    ax.imshow(hm / hm.max(), aspect='auto', cmap='magma')
    ax.set_title(comp, color=AMBER, fontsize=15, family='DejaVu Sans Mono', loc='left')
    ax.set_yticks(range(13)); ax.set_yticklabels(short, fontsize=10, family='DejaVu Sans Mono')
    ax.set_xticks([0, 12, 23]); ax.set_xticklabels(['t-23', 't-11', 't'], family='DejaVu Sans Mono'); ax.grid(False)
    [s.set_visible(False) for s in ax.spines.values()]
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'ig.png'), dpi=200); plt.close()

# 4) cost vs threshold (validation)
sim = PolicySimulator(VA.df[['machineID', 'datetime']], fail)
sw = sim.sweep(pb, threshold_grid(pb)).sort_values('threshold')
best = sw[sw.cost == sw.cost.min()]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.axvspan(best.threshold.min(), best.threshold.max(), color=TEAL, alpha=0.18, lw=0)
ax.plot(sw.threshold, sw.cost / 1e3, color=FG, lw=1.8)
ax.axhline(len(sim.fail) * 15, color=RED, ls='--', lw=1.2); ax.text(2e-5, len(sim.fail) * 15 - 130, 'operar hasta la falla: 1 770', color=RED, fontsize=10)
ax.axvline(D['threshold'], color=AMBER, lw=1.6); ax.text(D['threshold'], 1300, '  0.48', color=AMBER, fontsize=12, family='DejaVu Sans Mono')
ax.set_xscale('logit'); ax.set_xlabel('umbral (probabilidad calibrada, escala logit)'); ax.set_ylabel('costo en validación (miles)'); ax.set_ylim(300, 1900)
ax.set_xticks([1e-4, 1e-2, 0.5, 1 - 1e-2, 1 - 1e-4]); ax.set_xticklabels(['0.0001', '0.01', '0.5', '0.99', '0.9999'], family='DejaVu Sans Mono'); ax.minorticks_off()
ax.text(0.62, 470, 'meseta 0.44–0.52', color=TEAL, fontsize=10, ha='left')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'cost.png'), dpi=220); plt.close()
print('figures in', OUT)
