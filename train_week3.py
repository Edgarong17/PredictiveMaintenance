"""Trains (or loads from models/week3/) every week-3 network. The week-3 notebook calls the same
fit_or_load with the same arguments, so running this script first only pre-fills the cache.
Usage: python train_week3.py"""
import sys, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from pm_dl import build_hourly_channels, SequenceStore, SplitData
from pm_week3 import load_raw, load_split, load_week2_best, week3_runs, fit_or_load, MODEL_DIR

torch.set_num_threads(max(1, torch.get_num_threads()))
tel, err, mnt, fail, mach = load_raw()
grid = build_hourly_channels(tel, err, mnt)
_, ck = load_week2_best()
store = SequenceStore(grid, mach, ck['norm_stats'])
TR = SplitData(load_split('train', mach), store, ck['age_stats'])
VA = SplitData(load_split('validation', mach), store, ck['age_stats'])
val_es_idx = np.arange(0, len(VA), 3)
for name, arch, kw, seed in week3_runs(ck):
    print('==', name, flush=True)
    _, hist, minutes = fit_or_load(name, arch, 24, TR, VA, 'oversample', seed, kw, MODEL_DIR, val_idx=val_es_idx)
    print(f'   best val PR-AUC (ES subset) {hist.val_pr_auc.max():.4f}, epochs {len(hist)}, {minutes:.1f} min', flush=True)
