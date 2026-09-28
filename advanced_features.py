from pathlib import Path

import numpy as np
import pandas as pd

from solution import EVENT_NAMES, describe, entropy


def advanced_features(data_dir, meta):
    ev = pd.read_csv(Path(data_dir) / 'data/events.csv.gz', parse_dates=['event_ts'])
    ev = ev.merge(meta[['cookie_id', 'window_start_ts', 'window_end_ts']], on='cookie_id', validate='many_to_one')
    ev = ev.loc[ev.event_ts.ge(ev.window_start_ts) & ev.event_ts.lt(ev.window_end_ts)].drop_duplicates().copy()
    ev['sec'] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    ev['query'] = ev.search_query.astype('string').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    ev = ev.sort_values(['cookie_id', 'event_ts', 'eid'], kind='mergesort')
    codes = {name: i for i, name in enumerate(EVENT_NAMES)}
    rows = []
    for cookie, g in ev.groupby('cookie_id', sort=False):
        r = {'cookie_id': cookie}
        n = len(g)
        ts = g.sec.to_numpy()
        gaps = np.diff(ts)
        within = gaps[(gaps > 0) & (gaps < 1800)]
        describe(r, 'extra_log_gap', np.log1p(within))
        if len(within):
            median = np.median(within)
            r['extra_gap_mad_relative'] = np.median(abs(within - median)) / max(median, 1)
            r['extra_gap_iqr_relative'] = (np.quantile(within, .75) - np.quantile(within, .25)) / max(median, 1)
            r['extra_gap_mode_fraction'] = pd.Series(within).value_counts().iloc[0] / len(within)
            r['extra_gap_skew'] = float(pd.Series(np.log1p(within)).skew())
            r['extra_gap_kurtosis'] = float(pd.Series(np.log1p(within)).kurtosis())
        for width in [2, 5, 15, 60]:
            r[f'extra_gap_rounded_{width}_entropy'] = entropy(np.floor(within / width))
        if len(gaps) > 1:
            adjacent = (gaps[1:] < 1800) & (gaps[:-1] < 1800)
            pairs = np.abs(gaps[1:][adjacent] - gaps[:-1][adjacent])
            describe(r, 'extra_gap_local_difference', pairs)
            r['extra_gap_similar_fraction'] = float((pairs <= 2).mean()) if len(pairs) else np.nan
        for cutoff in [60, 300, 600, 1800]:
            session = np.r_[0, np.cumsum(gaps > cutoff)]
            starts = np.r_[0, np.flatnonzero(gaps > cutoff) + 1]
            ends = np.r_[starts[1:] - 1, n - 1]
            sizes = np.bincount(session)
            duration = ts[ends] - ts[starts]
            r[f'extra_session_{cutoff}_occupied_seconds'] = duration.sum()
            r[f'extra_session_{cutoff}_single_fraction'] = float((sizes == 1).mean())
            r[f'extra_session_{cutoff}_max_size_fraction'] = sizes.max() / n
            describe(r, f'extra_session_{cutoff}_duration', duration)
            describe(r, f'extra_session_{cutoff}_rate', sizes / (1 + duration / 60))
        event = g.event_name.map(codes).to_numpy(dtype=int)
        trans = np.bincount(event[:-1] * len(codes) + event[1:], minlength=len(codes) ** 2)
        for a in range(len(codes)):
            total = trans[a * len(codes):(a + 1) * len(codes)].sum()
            for b in range(len(codes)):
                r[f'extra_transition_{a}_{b}_fraction'] = trans[a * len(codes) + b] / max(n - 1, 1)
                if a in [0, 1, 4]:
                    r[f'extra_transition_{a}_{b}_conditional'] = trans[a * len(codes) + b] / max(total, 1)
        for name in ['item_view', 'search_results_view', 'contact_phone_show', 'photo_swipe']:
            code = codes[name]
            mask = event[1:] == code
            describe(r, 'extra_gap_before_' + name, gaps[mask & (gaps < 1800)])
        for col, key in [('item_category', 'category'), ('item_location', 'location'), ('query', 'query')]:
            s = g[col].dropna()
            r['extra_' + key + '_mode'] = str(s.mode().iloc[0]) if len(s) else 'missing'
            vals = s.to_numpy()
            r['extra_' + key + '_adjacent_repeat'] = float((vals[1:] == vals[:-1]).mean()) if len(vals) > 1 else np.nan
            r['extra_' + key + '_unique_per_item'] = s.nunique() / max(g.item_id.nunique(), 1)
        viewed_at = {}
        contacted_at = {}
        delays = []
        contact_repeats = 0
        contacts = 0
        missing_view = 0
        photo_items = set()
        for t, name, item in zip(ts, g.event_name, g.item_id):
            if pd.isna(item):
                continue
            if name == 'item_view':
                viewed_at[item] = t
            if name == 'photo_swipe':
                photo_items.add(item)
            if name in ['contact_phone_show', 'contact_chat_open', 'contact_message_sent']:
                contacts += 1
                if item in contacted_at:
                    contact_repeats += 1
                if item in viewed_at:
                    delays.append(t - viewed_at[item])
                else:
                    missing_view += 1
                contacted_at[item] = t
        describe(r, 'extra_contact_delay', delays)
        r['extra_contact_without_prior_view'] = missing_view / max(contacts, 1)
        r['extra_contact_repeat_fraction'] = contact_repeats / max(contacts, 1)
        r['extra_photo_items_per_viewed_item'] = len(photo_items & set(viewed_at)) / max(len(viewed_at), 1)
        pointer = g[['pointer_x', 'pointer_y']].dropna()
        for axis in ['pointer_x', 'pointer_y']:
            s = pointer[axis].to_numpy()
            r['extra_' + axis + '_nunique'] = len(np.unique(s))
            r['extra_' + axis + '_entropy'] = entropy(s)
            r['extra_' + axis + '_range'] = np.ptp(s) if len(s) else np.nan
            r['extra_' + axis + '_std'] = np.std(s) if len(s) else np.nan
        if len(pointer) > 1:
            delta = np.diff(pointer.to_numpy(), axis=0)
            describe(r, 'extra_pointer_step', np.linalg.norm(delta, axis=1))
            r['extra_pointer_horizontal_fraction'] = float((delta[:, 1] == 0).mean())
            r['extra_pointer_vertical_fraction'] = float((delta[:, 0] == 0).mean())
        rows.append(r)
    features = meta[['cookie_id']].merge(pd.DataFrame(rows), on='cookie_id', how='left', validate='one_to_one')
    categorical = ['extra_category_mode', 'extra_location_mode', 'extra_query_mode']
    for col in categorical:
        features[col] = features[col].fillna('missing').astype(str)
    numeric = features.columns.difference(['cookie_id'] + categorical)
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan).fillna(-1)
    return features, categorical


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    columns = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
    train = pd.read_csv(root / 'files/data/train.csv', parse_dates=columns)
    test = pd.read_csv(root / 'files/data/test.csv', parse_dates=columns)
    meta = pd.concat([train.drop(columns='target'), test], ignore_index=True)
    features, categorical = advanced_features(root / 'files', meta)
    features.to_csv(root / 'advanced_features.csv', index=False)
    print(features.shape, categorical, flush=True)
