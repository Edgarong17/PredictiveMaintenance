"""Week 3: advanced model, ensembles and calibration, integrated gradients, cost-based maintenance policy.
Builds on pm_dl.py (same 13-channel raw hourly tensor, same heads, same training loop)."""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from pm_dl import (CHANNELS, COMPS, SENSORS, ERROR_TYPES, build_hourly_channels, SequenceStore, SplitData,
                   make_model, train_model, predict, Strategy, evaluate, expected_calibration_error)

DATA_DIR, OUT_DIR = 'archive', os.path.join('data', 'processed')
T_TRAIN_END = pd.Timestamp('2015-08-31')   # last training hour + 24 h margin
T_VAL, T_TEST = pd.Timestamp('2015-09-01'), pd.Timestamp('2015-11-01')
COSTS = {'failure': 15000, 'hit': 3000, 'false_alarm': 600}


# ----------------------------------------------------------------------------- data
def load_raw():
    rd = lambda f: pd.read_csv(os.path.join(DATA_DIR, f), parse_dates=['datetime'])
    machines = pd.read_csv(os.path.join(DATA_DIR, 'PdM_machines.csv'))
    return rd('PdM_telemetry.csv'), rd('PdM_errors.csv'), rd('PdM_maint.csv'), rd('PdM_failures.csv'), machines


def load_split(split, machines):
    return pd.read_parquet(os.path.join(OUT_DIR, f'{split}.parquet')).merge(
        machines[['machineID', 'model', 'age']], on='machineID', how='left', validate='many_to_one')


# ----------------------------------------------------------------------------- training with weights cached
def fit_or_load(name, arch, L, TR, VA, strategy_name, seed, kw, model_dir, epochs=16, patience=3,
                epoch_size=60000, rep=5, val_idx=None):
    """Trains (or loads) one network and caches its weights + learning curve in model_dir/name.pt."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, name + '.pt')
    kw = dict(kw)
    lam, lr, wd = kw.pop('lam', 0.5), kw.pop('lr', 2e-3), kw.pop('weight_decay', 1e-4)
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        model = make_model(arch, L, **kw)
        model.load_state_dict(ck['state_dict'])
        return model, pd.DataFrame(ck['hist']), ck['minutes']
    t0 = time.time()
    strat = Strategy(strategy_name, TR.yb, TR.ym, epoch_size, neg_ratio=4, rep=rep, seed=seed)
    model, hist = train_model(arch, L, TR, VA, strat, lam=lam, lr=lr, weight_decay=wd, epochs=epochs,
                              patience=patience, val_idx=val_idx, seed=seed, **kw)
    minutes = (time.time() - t0) / 60
    torch.save({'state_dict': model.state_dict(), 'arch': arch, 'L': L, 'kw': kw, 'lam': lam, 'lr': lr,
                'weight_decay': wd, 'seed': seed, 'strategy': strategy_name, 'hist': hist.to_dict(orient='list'),
                'minutes': minutes}, path)
    return model, hist, minutes


def load_week2_best(path=os.path.join('models', 'week2_best.pt')):
    ck = torch.load(path, weights_only=False)
    kw = dict(ck['kw'])
    for k in ('lr', 'weight_decay', 'lam'):
        kw.pop(k, None)
    model = make_model(ck['arch'], ck['L'], **kw)
    model.load_state_dict(ck['state_dict'])
    return model, ck


MODEL_DIR = os.path.join('models', 'week3')
ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]   # seed 42 = week-2 model (models/week2_best.pt), reused as member 1


def week3_runs(ck):
    """Every network trained in week 3, declared in one place (the notebook and train_week3.py share it).
    Transformer: declared budget of 2 sizes, same training recipe as the week-2 tuned TCN (oversampling,
    lambda, weight decay); lr 1e-3. Ensemble: week-2 tuned TCN configuration, 4 new seeds."""
    base = {k: ck['kw'][k] for k in ('lam', 'weight_decay')}
    runs = []
    for d in (32, 64):
        runs.append((f'Transformer_d{d}', 'Transformer',
                     {**base, 'lr': 1e-3, 'd': d, 'n_heads': 4, 'layers': 2, 'ff': 2 * d, 'dropout': 0.1}, 42))
    for s in ENSEMBLE_SEEDS[1:]:
        runs.append((f'TCN_seed{s}', 'TCN', dict(ck['kw']), s))
    return runs


# ----------------------------------------------------------------------------- uncertainty
def enable_dropout_only(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, data, L, T=30, seed=0, batch=4096):
    """T stochastic forward passes with dropout active. Returns (T, n) binary probs and mean multiclass probs."""
    torch.manual_seed(seed)
    pbs, pm_sum = [], None
    for _ in range(T):
        enable_dropout_only(model)
        pb, pm = [], []
        for i in range(0, len(data), batch):
            x, m, a, _, _ = data.batch(np.arange(i, min(i + batch, len(data))), L)
            lb, lm = model(x, m, a)
            pb.append(torch.sigmoid(lb).numpy()); pm.append(torch.softmax(lm, 1).numpy())
        pbs.append(np.concatenate(pb)); pm = np.concatenate(pm)
        pm_sum = pm if pm_sum is None else pm_sum + pm
    model.eval()
    return np.stack(pbs), pm_sum / T


def ensemble_predict(models, data, L):
    out = [predict(m, data, L) for m in models]
    pbs = np.stack([o[0] for o in out])
    pms = np.stack([o[1] for o in out])
    return pbs, pms.mean(0)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(p, y):
    """Scalar temperature T on the binary logit, minimising validation NLL. Returns T."""
    z = logit(p)
    yt = y.astype(np.float64)
    def nll(logT):
        q = 1 / (1 + np.exp(-z / np.exp(logT)))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.mean(yt * np.log(q) + (1 - yt) * np.log(1 - q))
    r = minimize_scalar(nll, bounds=(-3, 3), method='bounded')
    return float(np.exp(r.x))


def apply_temperature(p, T):
    return 1 / (1 + np.exp(-logit(p) / T))


def reliability_table(y, p, bins=10):
    """Equal-width reliability bins (probability scale), as in a standard reliability diagram."""
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for k in range(bins):
        s = b == k
        if s.any():
            rows.append({'bin': k, 'lo': edges[k], 'hi': edges[k + 1], 'n': int(s.sum()),
                         'mean_p': float(p[s].mean()), 'observed': float(y[s].mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- integrated gradients
def integrated_gradients(model, x, m, a, target, steps=50, head='multi'):
    """IG of one output w.r.t. the dynamic window x (n, L, C). Baseline = 0, i.e. the training mean
    of every channel after normalisation. target: class index (multi head) or ignored (binary head)."""
    model.eval()
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xi = (base + (k / steps) * (x - base)).requires_grad_(True)
        lb, lm = model(xi, m, a)
        out = lm[:, target] if head == 'multi' else lb
        g, = torch.autograd.grad(out.sum(), xi)
        total += g
    return ((x - base) * total / steps).detach()


# ----------------------------------------------------------------------------- maintenance policy
def split_failures(meta, failures, label_horizon=24):
    """Component failures that generate positive labels in this split (same rule as pm_dl.anticipation)."""
    lo, hi = meta['datetime'].min(), meta['datetime'].max()
    return failures[(failures['datetime'] > lo) & (failures['datetime'] <= hi + pd.Timedelta(hours=label_horizon))]


class PolicySimulator:
    """Event-level simulation of the alert policy on one split.
    - An alert is raised at hour t when p >= thr for `sustain` consecutive hours ending at t (the decision
      is taken once the evidence exists), and not within `cooldown` hours of the previous alert.
    - Alert followed by >= 1 component failure in (t, t + 24 h] -> preventive intervention (hit cost per
      component failure covered); otherwise -> false alarm.
    - Component failure with no alert in [f - 24 h, f) -> unanticipated failure."""

    def __init__(self, meta, failures, horizon=24, sustain=3, cooldown=24):
        d = meta[['machineID', 'datetime']].reset_index(drop=True)
        self.order = np.lexsort((d['datetime'].values, d['machineID'].values))
        d = d.iloc[self.order].reset_index(drop=True)
        self.horizon, self.sustain, self.cooldown = horizon, sustain, cooldown
        self.fail = split_failures(meta, failures).reset_index(drop=True)
        self.n_hours = len(d) / d['machineID'].nunique()
        self.groups = {}
        hours = ((d['datetime'] - d['datetime'].min()) / pd.Timedelta(hours=1)).to_numpy().astype(np.int64)
        t_origin = d['datetime'].min()
        for mid, pos in d.groupby('machineID').indices.items():
            f = self.fail[self.fail.machineID == mid]
            fh = ((f['datetime'] - t_origin) / pd.Timedelta(hours=1)).to_numpy()
            self.groups[mid] = (pos, hours[pos], np.sort(fh), f.index.to_numpy()[np.argsort(fh)])

    def run(self, p, thr, costs=COSTS):
        p = np.asarray(p)[self.order]
        n_alert = n_hit_alerts = n_fa = 0
        covered = np.zeros(len(self.fail), bool)
        lead = np.full(len(self.fail), np.nan)
        alerts = []   # (row index in the caller's order, hit?, indices of the failures it covers)
        for mid, (pos, hrs, fh, fidx) in self.groups.items():
            above = p[pos] >= thr
            if not above.any():
                continue
            # consecutive-run length ending at each hour
            run = np.zeros(len(above), np.int32)
            c = 0
            for i, v in enumerate(above):
                c = c + 1 if v else 0
                run[i] = c
            cand = np.flatnonzero(run >= self.sustain)
            last = -np.inf
            for i in cand:
                t = hrs[i]
                if t < last + self.cooldown:
                    continue
                last = t
                n_alert += 1
                hit = (fh > t) & (fh <= t + self.horizon)
                if hit.any():
                    n_hit_alerts += 1
                    new = hit & ~covered[fidx]
                    covered[fidx[new]] = True
                    lead[fidx[new]] = fh[new] - t
                else:
                    n_fa += 1
                alerts.append((int(self.order[pos[i]]), bool(hit.any()), fidx[hit]))
        n_fail = len(self.fail)
        n_cov = int(covered.sum())
        cost = n_cov * costs['hit'] + n_fa * costs['false_alarm'] + (n_fail - n_cov) * costs['failure']
        return {'threshold': thr, 'alerts': n_alert, 'false_alarms': n_fa, 'failures': n_fail,
                'failures_avoided': n_cov, 'failures_missed': n_fail - n_cov, 'cost': cost,
                'lead_mean_h': np.nanmean(lead) if n_cov else np.nan, 'days': self.n_hours / 24,
                '_lead': lead, '_covered': covered, '_alerts': alerts}

    def sweep(self, p, thresholds, costs=COSTS):
        rows = []
        for t in thresholds:
            r = self.run(p, t, costs)
            rows.append({k: v for k, v in r.items() if not k.startswith('_')})
        return pd.DataFrame(rows)


def run_to_failure_cost(n_failures, costs=COSTS):
    return n_failures * costs['failure']


def threshold_grid(p, n=400):
    """Candidate thresholds: quantiles of the upper tail of p (where any useful threshold lives), plus a coarse grid."""
    q = np.quantile(p, np.linspace(0.90, 0.99999, n))
    g = np.unique(np.r_[q, np.linspace(0.01, 0.99, 99)])
    return g[(g > 1e-6) & (g < 1 - 1e-6)]


# ----------------------------------------------------------------------------- calendar (age-based) replacement
def component_cycles(maint, failures, end_time):
    """Life cycles between consecutive replacements of each (machine, component), ending before end_time.
    event = 1 if the cycle ended in a failure (replacement matches a PdM_failures row), 0 if it ended in a
    scheduled replacement (right-censored life). Also the open cycle at end_time, censored."""
    fk = set(zip(failures.machineID, failures.failure, failures.datetime))
    rows = []
    for (mid, comp), g in maint.groupby(['machineID', 'comp']):
        t = np.sort(g['datetime'].values)
        t = t[t < np.datetime64(end_time)]
        for a, b in zip(t[:-1], t[1:]):
            rows.append({'machineID': mid, 'comp': comp, 'life_h': (b - a) / np.timedelta64(1, 'h'),
                         'event': int((mid, comp, pd.Timestamp(b)) in fk)})
        if len(t):
            rows.append({'machineID': mid, 'comp': comp, 'life_h': (np.datetime64(end_time) - t[-1]) / np.timedelta64(1, 'h'),
                         'event': 0})
    return pd.DataFrame(rows)


def kaplan_meier(life, event):
    """Kaplan-Meier survival S(t) at the distinct event times. Returns (times, S)."""
    order = np.argsort(life)
    life, event = np.asarray(life)[order], np.asarray(event)[order]
    times = np.unique(life[event == 1])
    S, s = [], 1.0
    for t in times:
        at_risk = (life >= t).sum()
        d = ((life == t) & (event == 1)).sum()
        s *= 1 - d / at_risk
        S.append(s)
    return times, np.array(S)


def simulate_calendar(maint, failures, comp, T, start, end, costs=COSTS):
    """Age-based calendar replacement of one component on the real timeline of every machine.
    The component is replaced (planned, cost 'hit') whenever its age reaches T hours. Real scheduled
    replacements of the plant (maint rows that are not failures) still happen and reset the age, as in
    every other policy. A real failure at f is counted as avoided if a planned replacement happened after
    the component's last real replacement and before f (the worn part had already been changed);
    otherwise it costs 'failure' and resets the age. Costs are counted for events in (start, end]."""
    fk = set(zip(failures.machineID, failures.failure, failures.datetime))
    mc = maint[maint.comp == comp]
    fc = failures[failures.failure == comp]
    start, end = np.datetime64(start), np.datetime64(end)
    Th = np.timedelta64(int(T), 'h')
    planned = failed = avoided = 0
    per_machine = {}
    for mid in sorted(set(mc.machineID) | set(fc.machineID)):
        p0, f0 = planned, failed
        sched = [t for t in mc[mc.machineID == mid].datetime.values if (mid, comp, pd.Timestamp(t)) not in fk]
        fails = list(fc[fc.machineID == mid].datetime.values)
        events = sorted([(t, 'sched') for t in sched] + [(t, 'fail') for t in fails])
        hist = [t for t, _ in events if t <= start]
        last = hist[-1] if hist else None   # last real replacement up to the window start
        replaced_since_real = False
        future = [(t, k) for t, k in events if start < t <= end]
        if last is None:
            last = start  # no history: assume new at the window start (rare)
        # planned replacements that would have happened before the window, from the last real one
        while last + Th <= start:
            last = last + Th
            replaced_since_real = True
        for t, kind in future + [(end, 'end')]:
            while last + Th < t:
                last = last + Th
                planned += 1
                replaced_since_real = True
            if kind == 'sched':
                last, replaced_since_real = t, False
            elif kind == 'fail':
                if replaced_since_real:
                    avoided += 1          # the planned replacement already removed the worn part
                else:
                    failed += 1
                    last = t              # repaired after the failure: age restarts
                replaced_since_real = False   # the next real failure needs its own planned replacement
        per_machine[mid] = (planned - p0) * costs['hit'] + (failed - f0) * costs['failure']
    cost = planned * costs['hit'] + failed * costs['failure']
    return {'comp': comp, 'T_h': T, 'planned_replacements': planned, 'failures': failed + avoided,
            'failures_avoided': avoided, 'failures_missed': failed, 'cost': cost, 'per_machine': per_machine}


def best_calendar(maint, failures, comp, start, end, grid=None, costs=COSTS):
    grid = np.arange(240, 24 * 365, 24) if grid is None else grid
    rows = pd.DataFrame([simulate_calendar(maint, failures, comp, T, start, end, costs) for T in grid]).drop(columns='per_machine')
    return rows.loc[rows.cost.idxmin()], rows


def model_cost_per_machine(sim, run, machine_of_row, costs=COSTS):
    """Split the cost of one PolicySimulator.run by machine (for bootstrap intervals)."""
    out = {}
    fail = sim.fail
    for mid, g in fail.groupby('machineID'):
        cov = run['_covered'][g.index.to_numpy()]
        out[mid] = cov.sum() * costs['hit'] + (~cov).sum() * costs['failure']
    for row, hit, _ in run['_alerts']:
        if not hit:
            mid = machine_of_row[row]
            out[mid] = out.get(mid, 0) + costs['false_alarm']
    return out


def bootstrap_annual(per_machine_costs, machines_all, days, n=2000, seed=0):
    """95 % interval of the fleet's annual cost, resampling the 100 machines with replacement.
    per_machine_costs: dict policy -> dict machineID -> cost in the period. Same resample for every policy."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(machines_all)
    mats = {k: np.array([v.get(m, 0.0) for m in ids]) for k, v in per_machine_costs.items()}
    draws = {k: [] for k in mats}
    for _ in range(n):
        s = rng.integers(0, len(ids), len(ids))
        for k, m in mats.items():
            draws[k].append(m[s].sum() * 365 / days / 1e6)
    return {k: np.array(v) for k, v in draws.items()}


def age_at_failure(failures_split, maint):
    """Hours since the previous replacement of the same component (strictly before the failure)."""
    ages = []
    for r in failures_split.itertuples(index=False):
        prev = maint[(maint.machineID == r.machineID) & (maint.comp == r.failure) & (maint.datetime < r.datetime)]
        ages.append((r.datetime - prev.datetime.max()) / pd.Timedelta(hours=1) if len(prev) else np.nan)
    return np.array(ages)
