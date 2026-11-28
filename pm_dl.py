"""Week 2: sequence datasets, deep learning architectures, training loop and metrics."""
import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, brier_score_loss

SENSORS = ['volt', 'rotate', 'pressure', 'vibration']
COMPS = ['comp1', 'comp2', 'comp3', 'comp4']
ERROR_TYPES = ['error1', 'error2', 'error3', 'error4', 'error5']
MODELS = ['model1', 'model2', 'model3', 'model4']
CHANNELS = SENSORS + [f'{e}_now' for e in ERROR_TYPES] + [f'log_hours_since_{c}' for c in COMPS]


# ----------------------------------------------------------------------------- data
def hours_since_last(times, events):
    """Hours from the last event strictly before each time (NaN if none). times/events sorted datetime64."""
    prev = np.searchsorted(events, times, side='left') - 1
    out = np.full(len(times), np.nan)
    ok = prev >= 0
    out[ok] = (times[ok] - events[prev[ok]]) / np.timedelta64(1, 'h')
    return out


def build_hourly_channels(telemetry, errors, maint):
    """One row per (machine, hour) on the raw grid, with the dynamic channels (un-normalised).
    Every channel at hour t only uses information up to t (maintenance strictly before t)."""
    tel = telemetry.sort_values(['machineID', 'datetime']).reset_index(drop=True)
    grid = tel[['machineID', 'datetime'] + SENSORS].copy()
    err = errors.assign(n=1).pivot_table(index=['machineID', 'datetime'], columns='errorID', values='n', aggfunc='sum', fill_value=0)
    err = err.reindex(columns=ERROR_TYPES, fill_value=0)
    err.columns = [f'{c}_now' for c in err.columns]
    grid = grid.merge(err.reset_index(), on=['machineID', 'datetime'], how='left')
    grid[[f'{e}_now' for e in ERROR_TYPES]] = grid[[f'{e}_now' for e in ERROR_TYPES]].fillna(0)
    times_all = grid['datetime'].values.astype('datetime64[ns]')
    for c in COMPS:
        col = np.full(len(grid), np.nan)
        for mid, pos in grid.groupby('machineID').indices.items():
            ev = np.sort(maint.loc[(maint.machineID == mid) & (maint.comp == c), 'datetime'].values.astype('datetime64[ns]'))
            col[pos] = hours_since_last(times_all[pos], ev)
        grid[f'log_hours_since_{c}'] = np.log1p(col)
    return grid


class SequenceStore:
    """Holds the normalised hourly channels of every machine in one flat array and cuts windows on demand,
    so windows of 168 h never have to be materialised for 580k samples."""

    def __init__(self, grid, machines, norm_stats):
        grid = grid.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        self.mu, self.sd = norm_stats
        X = grid[CHANNELS].to_numpy(dtype=np.float64)
        X = (X - self.mu) / self.sd
        X = np.nan_to_num(X, nan=0.0)  # no previous replacement -> train mean
        self.X = X.astype(np.float32)
        counts = grid.groupby('machineID').size()
        self.start = dict(zip(counts.index, np.r_[0, np.cumsum(counts.values)[:-1]]))
        self.t0 = grid.groupby('machineID')['datetime'].min().to_dict()
        m = machines.set_index('machineID')
        self.model_id = {k: MODELS.index(v) for k, v in m['model'].items()}
        self.age = m['age'].to_dict()

    def flat_index(self, machine_ids, datetimes):
        start = np.array([self.start[k] for k in machine_ids])
        first = np.array([self.t0[k] for k in machine_ids], dtype='datetime64[ns]')
        pos = ((datetimes.astype('datetime64[ns]') - first) / np.timedelta64(1, 'h')).astype(int)
        return start, pos

    def windows(self, start, pos, L):
        """(n, L, C) windows ending at pos (inclusive). Left edge is padded by repeating the first hour."""
        offs = np.arange(-L + 1, 1)
        idx = pos[:, None] + offs[None, :]
        idx = np.clip(idx, 0, None) + start[:, None]
        return self.X[idx]


def norm_stats_from_train(grid, train_end):
    """Global mean/std per channel computed ONLY on training-period hours."""
    tr = grid[grid['datetime'] < train_end]
    X = tr[CHANNELS].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


class SplitData:
    """Sample index of one split (rows of the week-1 parquet) plus labels and static inputs."""

    def __init__(self, df, store, age_stats):
        df = df.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        self.df = df[['machineID', 'datetime', 'y_binary', 'y_multiclass'] + [f'y_{c}' for c in COMPS]].copy()
        self.start, self.pos = store.flat_index(df['machineID'].values, df['datetime'].values)
        self.yb = df['y_binary'].to_numpy(np.float32)
        self.ym = df['y_multiclass'].to_numpy(np.int64)
        self.model = np.array([store.model_id[k] for k in df['machineID']], dtype=np.int64)
        a_mu, a_sd = age_stats
        self.age = ((np.array([store.age[k] for k in df['machineID']], dtype=np.float32) - a_mu) / a_sd).astype(np.float32)
        self.store = store

    def __len__(self):
        return len(self.yb)

    def batch(self, idx, L):
        x = torch.from_numpy(self.store.windows(self.start[idx], self.pos[idx], L))
        return (x, torch.from_numpy(self.model[idx]), torch.from_numpy(self.age[idx])[:, None],
                torch.from_numpy(self.yb[idx]), torch.from_numpy(self.ym[idx]))


# ----------------------------------------------------------------------------- models
class StaticEncoder(nn.Module):
    def __init__(self, n_models=4, emb=3):
        super().__init__()
        self.emb = nn.Embedding(n_models, emb)
        self.out_dim = emb + 1

    def forward(self, model, age):
        return torch.cat([self.emb(model), age], dim=1)


class Heads(nn.Module):
    """Shared dense layer and two outputs: binary logit and 5-class logits."""

    def __init__(self, in_dim, hidden=64, dropout=0.2, n_classes=5):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.binary = nn.Linear(hidden, 1)
        self.multi = nn.Linear(hidden, n_classes)

    def forward(self, z):
        h = self.shared(z)
        return self.binary(h).squeeze(1), self.multi(h)


class RNNNet(nn.Module):
    """LSTM or GRU (1-2 layers); static inputs concatenated to the final hidden state."""

    def __init__(self, n_ch, cell='gru', hidden=64, layers=1, dropout=0.2):
        super().__init__()
        rnn = nn.GRU if cell == 'gru' else nn.LSTM
        self.rnn = rnn(n_ch, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.static = StaticEncoder()
        self.heads = Heads(hidden + self.static.out_dim, dropout=dropout)

    def forward(self, x, model, age):
        out, _ = self.rnn(x)
        return self.heads(torch.cat([out[:, -1], self.static(model, age)], dim=1))


class CausalConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        h = self.drop(F.relu(self.conv2(F.pad(h, (self.pad, 0)))))
        return F.relu(h + self.res(x))


class TCNNet(nn.Module):
    """Temporal convolutional network: causal dilated convolutions, residual blocks.
    Dilations double until the receptive field covers the window."""

    def __init__(self, n_ch, L, channels=32, k=3, dropout=0.2):
        super().__init__()
        blocks, d, rf, c_in = [], 1, 1, n_ch
        while rf < L:
            blocks.append(CausalConvBlock(c_in, channels, k, d, dropout))
            rf += 2 * (k - 1) * d
            d *= 2
            c_in = channels
        self.receptive_field = rf
        self.tcn = nn.Sequential(*blocks)
        self.static = StaticEncoder()
        self.heads = Heads(channels + self.static.out_dim, dropout=dropout)

    def forward(self, x, model, age):
        h = self.tcn(x.transpose(1, 2))
        return self.heads(torch.cat([h[:, :, -1], self.static(model, age)], dim=1))


class CNNLSTMNet(nn.Module):
    """Hybrid: causal 1D convolutions extract local patterns, max-pooling shortens the sequence,
    an LSTM summarises it; static inputs concatenated to the final state."""

    def __init__(self, n_ch, channels=32, hidden=64, k=5, dropout=0.2):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(n_ch, channels, k)
        self.conv2 = nn.Conv1d(channels, channels, k)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.static = StaticEncoder()
        self.heads = Heads(hidden + self.static.out_dim, dropout=dropout)

    def forward(self, x, model, age):
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(F.pad(h, (self.k - 1, 0))))
        h = self.pool(F.relu(self.conv2(F.pad(h, (self.k - 1, 0)))))
        out, _ = self.lstm(self.drop(h.transpose(1, 2)))
        return self.heads(torch.cat([out[:, -1], self.static(model, age)], dim=1))


class TransformerNet(nn.Module):
    """Week 3 (3.1a): Transformer encoder over the hourly window. A 'machine token' built from the static
    inputs (model embedding + age) is prepended to the sequence (like a CLS token); sinusoidal positional
    encoding; the token's final state, concatenated with the static vector, feeds the same two heads."""

    def __init__(self, n_ch, L, d=32, n_heads=4, layers=2, ff=64, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_ch, d)
        self.static = StaticEncoder()
        self.machine_token = nn.Linear(self.static.out_dim, d)
        pos = torch.arange(L + 1, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(L + 1, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe[None])
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.heads = Heads(d + self.static.out_dim, dropout=dropout)

    def forward(self, x, model, age):
        s = self.static(model, age)
        h = torch.cat([self.machine_token(s)[:, None], self.inp(x)], dim=1) + self.pe[:, :x.shape[1] + 1]
        h = self.norm(self.encoder(h))
        return self.heads(torch.cat([h[:, 0], s], dim=1))


def make_model(arch, L, n_ch=len(CHANNELS), **kw):
    if arch == 'Transformer':
        return TransformerNet(n_ch, L, d=kw.get('d', 32), n_heads=kw.get('n_heads', 4), layers=kw.get('layers', 2),
                              ff=kw.get('ff', 64), dropout=kw.get('dropout', 0.1))
    if arch == 'GRU':
        return RNNNet(n_ch, 'gru', hidden=kw.get('hidden', 64), layers=kw.get('layers', 1), dropout=kw.get('dropout', 0.2))
    if arch == 'LSTM':
        return RNNNet(n_ch, 'lstm', hidden=kw.get('hidden', 64), layers=kw.get('layers', 1), dropout=kw.get('dropout', 0.2))
    if arch == 'TCN':
        return TCNNet(n_ch, L, channels=kw.get('channels', 32), dropout=kw.get('dropout', 0.2))
    if arch == 'CNN-LSTM':
        return CNNLSTMNet(n_ch, channels=kw.get('channels', 32), hidden=kw.get('hidden', 64), dropout=kw.get('dropout', 0.2))
    raise ValueError(arch)


# ----------------------------------------------------------------------------- losses / sampling
def focal_bce(logits, y, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    p_t = p * y + (1 - p) * (1 - y)
    a_t = alpha * y + (1 - alpha) * (1 - y)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


class Strategy:
    """Imbalance strategy: how each epoch is sampled and how the binary loss is weighted.
    - weighted: uniform sample of hours, BCE with pos_weight = n_neg / n_pos, class-weighted CE.
    - focal: uniform sample of hours, focal loss (gamma=2, alpha=0.75), class-weighted CE.
    - undersample: every positive hour + neg_ratio x random healthy hours, plain BCE / CE.
    - oversample: uniform sample + every positive (pre-failure) window repeated `rep` times, plain BCE / CE."""

    def __init__(self, name, yb, ym, epoch_size, neg_ratio=4, rep=10, seed=0):
        self.name, self.yb, self.ym, self.epoch_size = name, yb, ym, epoch_size
        self.neg_ratio, self.rep = neg_ratio, rep
        self.rng = np.random.default_rng(seed)
        self.pos_idx = np.flatnonzero(yb == 1)
        self.neg_idx = np.flatnonzero(yb == 0)
        self.prior = yb.mean()
        counts = np.bincount(ym, minlength=5).astype(float)
        self.class_w = torch.tensor(len(ym) / (5 * counts), dtype=torch.float32)
        self.pos_weight = torch.tensor(len(self.neg_idx) / len(self.pos_idx), dtype=torch.float32)

    def epoch_indices(self):
        if self.name in ('weighted', 'focal'):
            idx = self.rng.choice(len(self.yb), self.epoch_size, replace=False)
        elif self.name == 'undersample':
            n_neg = min(len(self.neg_idx), self.neg_ratio * len(self.pos_idx))
            idx = np.r_[self.pos_idx, self.rng.choice(self.neg_idx, n_neg, replace=False)]
        elif self.name == 'oversample':
            base = self.rng.choice(len(self.yb), self.epoch_size, replace=False)
            idx = np.r_[base, np.repeat(self.pos_idx, self.rep)]
        self.rng.shuffle(idx)
        return idx

    def train_positive_rate(self):
        if self.name == 'undersample':
            return 1 / (1 + self.neg_ratio)
        if self.name == 'oversample':
            n_pos = self.epoch_size * self.prior + self.rep * len(self.pos_idx)
            return n_pos / (self.epoch_size + self.rep * len(self.pos_idx))
        return self.prior

    def loss(self, lb, lm, yb, ym, lam):
        if self.name == 'weighted':
            lb_ = F.binary_cross_entropy_with_logits(lb, yb, pos_weight=self.pos_weight)
            lm_ = F.cross_entropy(lm, ym, weight=self.class_w)
        elif self.name == 'focal':
            lb_ = focal_bce(lb, yb)
            lm_ = F.cross_entropy(lm, ym, weight=self.class_w)
        else:
            lb_ = F.binary_cross_entropy_with_logits(lb, yb)
            lm_ = F.cross_entropy(lm, ym)
        return lb_ + lam * lm_, lb_.item(), lm_.item()


# ----------------------------------------------------------------------------- train / predict
@torch.no_grad()
def predict(model, data, L, idx=None, batch=4096):
    model.eval()
    idx = np.arange(len(data)) if idx is None else idx
    pb, pm = [], []
    for i in range(0, len(idx), batch):
        x, m, a, _, _ = data.batch(idx[i:i + batch], L)
        lb, lm = model(x, m, a)
        pb.append(torch.sigmoid(lb).numpy())
        pm.append(torch.softmax(lm, 1).numpy())
    return np.concatenate(pb), np.concatenate(pm)


def train_model(arch, L, train, val, strategy, lam=0.5, epochs=12, patience=3, lr=2e-3, weight_decay=1e-4,
                batch=512, seed=42, val_idx=None, verbose=True, max_minutes=None, trial=None, **arch_kw):
    """Adam + weight decay, dropout inside the model, early stopping on validation PR-AUC (binary head).
    Returns the best model (restored) and the learning-curve history."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(arch, L, **arch_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist, best, best_state, bad = [], -1, None, 0
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        idx = strategy.epoch_indices()
        tl, tb, tm, n = 0.0, 0.0, 0.0, 0
        t0 = time.time()
        for i in range(0, len(idx), batch):
            x, m, a, yb, ym = train.batch(idx[i:i + batch], L)
            lb, lm = model(x, m, a)
            loss, lb_, lm_ = strategy.loss(lb, lm, yb, ym, lam)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            k = len(yb); tl += loss.item() * k; tb += lb_ * k; tm += lm_ * k; n += k
        vi = np.arange(len(val)) if val_idx is None else val_idx
        pb, pm = predict(model, val, L, vi)
        yv = val.yb[vi]
        v_bce = F.binary_cross_entropy(torch.tensor(np.clip(pb, 1e-6, 1 - 1e-6)), torch.tensor(yv)).item()
        v_ap = average_precision_score(yv, pb)
        hist.append({'epoch': ep, 'train_loss': tl / n, 'train_bce': tb / n, 'train_ce': tm / n,
                     'val_bce': v_bce, 'val_pr_auc': v_ap, 'seconds': time.time() - t0})
        if verbose:
            print(f'  ep {ep:2d} loss {tl / n:.4f}  val PR-AUC {v_ap:.4f}  ({time.time() - t0:.0f}s)', flush=True)
        if trial is not None:
            trial.report(v_ap, ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        if v_ap > best + 1e-4:
            best, bad = v_ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if max_minutes and (time.time() - t_start) / 60 > max_minutes:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


# ----------------------------------------------------------------------------- metrics
def threshold_at_precision(y, p, target=0.80):
    pr, rc, th = precision_recall_curve(y, p)
    ok = pr[:-1] >= target
    if not ok.any():
        return np.nan
    return th[np.flatnonzero(ok)[np.argmax(rc[:-1][ok])]]


def expected_calibration_error(y, p, bins=15):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1, 2
    b = np.digitize(p, edges[1:-1])
    ece = 0.0
    for k in range(bins):
        s = b == k
        if s.any():
            ece += s.mean() * abs(p[s].mean() - y[s].mean())
    return ece


def prior_correct(p, train_rate, true_rate):
    """Undo the prior shift caused by re-sampling (Elkan / Saerens correction)."""
    odds = p / (1 - p + 1e-12) * (true_rate / (1 - true_rate)) / (train_rate / (1 - train_rate))
    return odds / (1 + odds)


def anticipation(meta, p, thr, failures, horizon=72, sustain=3, label_horizon=24):
    """Per failure event in the split: hours between the start of the first sustained alert
    (p >= thr for >= `sustain` consecutive hours) and the failure. The alert is searched in the
    `horizon` hours before the failure (72 h by default: three times the 24 h label horizon, so an
    alert raised before the label window still counts as anticipation). NaN if no sustained alert."""
    d = meta[['machineID', 'datetime']].copy()
    d['p'] = p
    d = d.sort_values(['machineID', 'datetime'])
    lo, hi = d['datetime'].min(), d['datetime'].max()
    # events = failures that create positive labels in this split (failure within 24 h after one of its hours)
    ev = failures[(failures['datetime'] > lo) & (failures['datetime'] <= hi + pd.Timedelta(hours=label_horizon))]
    ev = ev[['machineID', 'datetime']].drop_duplicates()
    by_m = {m: g.set_index('datetime')['p'] for m, g in d.groupby('machineID')}
    rows = []
    for e in ev.itertuples(index=False):
        s = by_m[e.machineID]
        w = s[(s.index >= e.datetime - pd.Timedelta(hours=horizon)) & (s.index < e.datetime)]
        if len(w) == 0:
            continue
        above = (w.values >= thr).astype(int)
        run, first = 0, None
        for i, a in enumerate(above):
            run = run + 1 if a else 0
            if run == sustain:
                first = w.index[i - sustain + 1]
                break
        rows.append({'machineID': e.machineID, 'failure_time': e.datetime,
                     'lead_hours': np.nan if first is None else (e.datetime - first) / pd.Timedelta(hours=1)})
    return pd.DataFrame(rows)


def evaluate(name, yb, ym, pb, pm, meta, failures, thr=None):
    thr = threshold_at_precision(yb, pb) if thr is None else thr
    pred = (pb >= thr).astype(int)
    tp = int(((pred == 1) & (yb == 1)).sum()); fp = int(((pred == 1) & (yb == 0)).sum())
    # Component decision uses both heads: the binary head decides whether to alert and the
    # multiclass head chooses which component (argmax over comp1..comp4) when it does.
    comp_pred = np.where(pred == 1, pm[:, 1:].argmax(1) + 1, 0)
    comp_direct = pm.argmax(1)
    lead = anticipation(meta, pb, thr, failures)
    out = {
        'model': name,
        'PR-AUC': average_precision_score(yb, pb),
        'ROC-AUC': roc_auc_score(yb, pb),
        'threshold': thr,
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(int(yb.sum()), 1),
        'FP': fp,
        'F1_macro_comp': f1_score(ym, comp_pred, labels=[1, 2, 3, 4], average='macro', zero_division=0),
        'F1_macro_comp_direct': f1_score(ym, comp_direct, labels=[1, 2, 3, 4], average='macro', zero_division=0),
        **{f'F1_{c}': v for c, v in zip(COMPS, f1_score(ym, comp_pred, labels=[1, 2, 3, 4], average=None, zero_division=0))},
        'events': len(lead),
        'events_alerted_%': 100 * lead['lead_hours'].notna().mean(),
        'lead_median_h': lead['lead_hours'].median(),
        'lead_mean_h': lead['lead_hours'].mean(),
        'Brier': brier_score_loss(yb, pb),
        'ECE': expected_calibration_error(yb, pb),
    }
    return out, lead
