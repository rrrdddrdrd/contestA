from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SEED = 42
EVENT_NAMES = [
    'search_results_view', 'item_view', 'photo_swipe', 'seller_page_view',
    'contact_phone_show', 'contact_chat_open', 'contact_message_sent',
    'favorite_add', 'login', 'captcha_shown',
]


def precision_at_recall(y, score, recall=0.7):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind='mergesort')
    ys, ss = y[order], score[order]
    ends = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(ys)[ends]
    selected = np.arange(1, len(y) + 1)[ends]
    if y.sum() == 0:
        return float('nan')
    valid = tp / y.sum() >= recall
    return float((tp / selected)[valid].max()) if valid.any() else 0.0


def entropy(values):
    counts = pd.Series(values).value_counts().to_numpy(dtype=float)
    if len(counts) == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def describe(row, prefix, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    row[prefix + '_count'] = len(values)
    if len(values) == 0:
        for name in ['min', 'q10', 'q25', 'median', 'q75', 'q90', 'max', 'mean', 'std', 'cv']:
            row[prefix + '_' + name] = np.nan
        return
    quantiles = np.quantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    for name, value in zip(['min', 'q10', 'q25', 'median', 'q75', 'q90', 'max'], quantiles):
        row[prefix + '_' + name] = value
    row[prefix + '_mean'] = values.mean()
    row[prefix + '_std'] = values.std()
    row[prefix + '_cv'] = values.std() / (abs(values.mean()) + 1e-6)


def browser_family(ua):
    ua = str(ua).lower()
    for token, name in [('headless', 'headless'), ('python-requests', 'requests'),
                        ('scrapy', 'scrapy'), ('curl/', 'curl'), ('node-fetch', 'node'),
                        ('avito/', 'avito_app'), ('yabrowser', 'yandex'),
                        ('firefox', 'firefox'), ('chrome', 'chrome'), ('safari', 'safari')]:
        if token in ua:
            return name
    return 'other'


def os_family(ua):
    ua = str(ua).lower()
    for token, name in [('android', 'android'), ('iphone', 'ios'), ('ipad', 'ios'),
                        ('windows', 'windows'), ('macintosh', 'mac'), ('linux', 'linux')]:
        if token in ua:
            return name
    return 'other'


def diversity(row, prefix, values):
    values = pd.Series(values).dropna()
    counts = values.value_counts()
    row[prefix + '_nunique'] = len(counts)
    row[prefix + '_unique_ratio'] = len(counts) / max(len(values), 1)
    row[prefix + '_entropy'] = entropy(values)
    row[prefix + '_dominant_ratio'] = counts.iloc[0] / len(values) if len(counts) else 0


def build_features(data_dir):
    dates = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
    train = pd.read_csv(data_dir / 'data/train.csv', parse_dates=dates)
    test = pd.read_csv(data_dir / 'data/test.csv', parse_dates=dates)
    meta = pd.concat([train.drop(columns='target'), test], ignore_index=True)
    if not meta.cookie_id.is_unique:
        raise ValueError('cookie_id must be unique across train and test')
    raw = pd.read_csv(data_dir / 'data/events.csv.gz', parse_dates=['event_ts'])
    raw_count = len(raw)
    raw = raw.merge(meta[['cookie_id', 'window_start_ts', 'window_end_ts']], on='cookie_id', validate='many_to_one')
    raw = raw.loc[raw.event_ts.ge(raw.window_start_ts) & raw.event_ts.lt(raw.window_end_ts)].copy()
    in_window_count = len(raw)
    raw_counts = raw.groupby('cookie_id').size()
    ev = raw.drop_duplicates().copy()
    dedup_count = len(ev)
    ev['platform_norm'] = ev.platform.str.lower().replace({'desktop': 'web', 'iphone': 'ios'})
    ev['browser'] = ev.user_agent.map(browser_family)
    ev['os'] = ev.user_agent.map(os_family)
    ev['query_norm'] = ev.search_query.astype('string').str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    ev['sec'] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    ev = ev.sort_values(['cookie_id', 'event_ts', 'eid'], kind='mergesort')
    rows = []
    for cookie, g in ev.groupby('cookie_id', sort=False):
        n = len(g)
        row = {'cookie_id': cookie, 'n_events': n, 'duplicate_ratio': (raw_counts[cookie] - n) / raw_counts[cookie]}
        ts = g.sec.to_numpy()
        gaps = np.diff(ts)
        row['active_span'] = ts[-1] - ts[0]
        row['first_event_seconds'] = ts[0]
        row['last_event_seconds'] = ts[-1]
        row['events_per_active_minute'] = n / (row['active_span'] / 60 + 1)
        describe(row, 'gap', gaps)
        describe(row, 'session_gap', gaps[gaps <= 1800])
        for cutoff in [0, 1, 2, 5, 10, 30, 60, 300, 1800]:
            row[f'gap_le_{cutoff}_ratio'] = float((gaps <= cutoff).mean()) if len(gaps) else 0
        row['gap_entropy'] = entropy(gaps)
        row['gap_unique_ratio'] = len(np.unique(gaps)) / max(len(gaps), 1)
        row['session_count'] = 1 + int((gaps > 1800).sum())
        session_ids = np.r_[0, np.cumsum(gaps > 1800)]
        session_sizes = np.bincount(session_ids)
        describe(row, 'session_size', session_sizes)
        for width in [60, 300, 3600]:
            bins = (ts // width).astype(int)
            counts = np.bincount(bins, minlength=int(86400 / width))
            row[f'active_bins_{width}'] = int((counts > 0).sum())
            row[f'bin_{width}_max'] = counts.max()
            row[f'bin_{width}_dominant_ratio'] = counts.max() / n
            row[f'bin_{width}_entropy'] = entropy(bins)
        row['night_ratio'] = float(((ts < 6 * 3600) | (ts >= 23 * 3600)).mean())
        row['timestamp_unique_ratio'] = len(np.unique(ts)) / n
        names = g.event_name.to_numpy()
        counts = g.event_name.value_counts()
        for name in EVENT_NAMES:
            row['event_' + name + '_count'] = int(counts.get(name, 0))
            row['event_' + name + '_ratio'] = counts.get(name, 0) / n
        row['event_entropy'] = entropy(names)
        row['event_types'] = len(counts)
        row['same_event_adjacent_ratio'] = float((names[1:] == names[:-1]).mean()) if n > 1 else 0
        runs = np.r_[0, np.cumsum(names[1:] != names[:-1])]
        describe(row, 'event_run', np.bincount(runs))
        transitions = [a + '>' + b for a, b in zip(names[:-1], names[1:])]
        row['transition_entropy'] = entropy(transitions)
        row['transition_types'] = len(set(transitions))
        for name in ['search_results_view', 'item_view', 'contact_phone_show']:
            describe(row, name + '_gap', np.diff(ts[names == name]))
        item_views = counts.get('item_view', 0)
        searches = counts.get('search_results_view', 0)
        contacts = sum(counts.get(x, 0) for x in ['contact_phone_show', 'contact_chat_open', 'contact_message_sent'])
        row['contacts_per_item_view'] = contacts / max(item_views, 1)
        row['photos_per_item_view'] = counts.get('photo_swipe', 0) / max(item_views, 1)
        row['favorites_per_item_view'] = counts.get('favorite_add', 0) / max(item_views, 1)
        row['item_views_per_search'] = item_views / max(searches, 1)
        row['phone_per_chat'] = counts.get('contact_phone_show', 0) / max(counts.get('contact_chat_open', 0), 1)
        for col in ['item_id', 'item_category', 'item_location', 'seller_type', 'query_norm', 'platform_norm', 'browser', 'os']:
            diversity(row, col, g[col])
            row[col + '_missing_ratio'] = float(g[col].isna().mean())
        for col in ['platform_norm', 'browser', 'os']:
            row[col + '_mode'] = g[col].mode().iloc[0]
        for value in ['web', 'android', 'ios']:
            row['platform_' + value + '_ratio'] = float(g.platform_norm.eq(value).mean())
        for value in ['headless', 'requests', 'scrapy', 'curl', 'node', 'avito_app']:
            row['browser_' + value + '_ratio'] = float(g.browser.eq(value).mean())
        row['pro_seller_ratio'] = float(g.seller_type.dropna().eq('pro').mean()) if g.seller_type.notna().any() else np.nan
        for name in ['item_view', 'contact_phone_show', 'favorite_add']:
            diversity(row, name + '_item', g.loc[g.event_name.eq(name), 'item_id'])
        viewed = set(g.loc[g.event_name.eq('item_view'), 'item_id'].dropna())
        contacted = set(g.loc[g.event_name.isin(['contact_phone_show', 'contact_chat_open', 'contact_message_sent']), 'item_id'].dropna())
        row['contacted_item_fraction'] = len(viewed & contacted) / max(len(viewed), 1)
        items = g.item_id.dropna().to_numpy()
        row['same_item_adjacent_ratio'] = float((items[1:] == items[:-1]).mean()) if len(items) > 1 else 0
        pages = g.search_page.dropna().to_numpy()
        describe(row, 'search_page', pages)
        row['search_page_nunique'] = len(np.unique(pages))
        for cutoff in [1, 3, 5, 10]:
            row[f'search_page_gt_{cutoff}_ratio'] = float((pages > cutoff).mean()) if len(pages) else 0
        page_diffs = np.diff(pages)
        row['page_increment_ratio'] = float((page_diffs == 1).mean()) if len(page_diffs) else 0
        row['page_same_ratio'] = float((page_diffs == 0).mean()) if len(page_diffs) else 0
        row['page_decrease_ratio'] = float((page_diffs < 0).mean()) if len(page_diffs) else 0
        queries = g.query_norm.dropna()
        describe(row, 'query_length', queries.str.len().to_numpy(dtype=float))
        row['query_change_ratio'] = float((queries.to_numpy()[1:] != queries.to_numpy()[:-1]).mean()) if len(queries) > 1 else 0
        pointer = g[['pointer_x', 'pointer_y']].dropna()
        row['pointer_count'] = len(pointer)
        row['pointer_ratio'] = len(pointer) / n
        row['pointer_unique_ratio'] = len(pointer.drop_duplicates()) / max(len(pointer), 1)
        for name in EVENT_NAMES:
            mask = g.event_name.eq(name)
            row[name + '_pointer_ratio'] = float(g.loc[mask, 'pointer_x'].notna().mean()) if mask.any() else np.nan
        if len(pointer):
            p = pointer.to_numpy()
            step = np.linalg.norm(np.diff(p, axis=0), axis=1)
            diagonal = np.linalg.norm(np.ptp(p, axis=0))
            describe(row, 'pointer_step_relative', step / max(diagonal, 1))
            row['pointer_stationary_ratio'] = float((step == 0).mean()) if len(step) else 0
            row['pointer_linearity'] = abs(np.corrcoef(p.T)[0, 1]) if len(p) > 2 and (p.std(axis=0) > 0).all() else np.nan
        else:
            describe(row, 'pointer_step_relative', [])
            row['pointer_stationary_ratio'] = 0
            row['pointer_linearity'] = np.nan
        rows.append(row)
    features = meta[['cookie_id']].merge(pd.DataFrame(rows), on='cookie_id', how='left', validate='one_to_one')
    features['cookie_age_hours'] = ((meta.window_end_ts - meta.cookie_created_at).dt.total_seconds() / 3600).clip(lower=0)
    features['cookie_new_in_window'] = meta.cookie_created_at.ge(meta.window_start_ts).astype(int)
    features['n_events'] = features['n_events'].fillna(0)
    features['duplicate_ratio'] = features['duplicate_ratio'].fillna(0)
    categories = ['platform_norm_mode', 'browser_mode', 'os_mode']
    for col in categories:
        features[col] = features[col].fillna('missing').astype(str)
    numeric = features.columns.difference(['cookie_id'] + categories)
    features[numeric] = features[numeric].replace([np.inf, -np.inf], np.nan).fillna(-1)
    info = {'train_cookies': len(train), 'test_cookies': len(test), 'raw_events': raw_count,
            'events_in_window': in_window_count, 'unique_events_in_window': dedup_count,
            'duplicates_removed': in_window_count - dedup_count,
            'out_of_window_removed': raw_count - in_window_count,
            'feature_count': len(features.columns) - 1, 'categorical_features': categories}
    return train, test, features, info


def metrics(y, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return {'precision_at_recall_0.7': precision_at_recall(y, score),
            'pr_auc_average_precision': float(average_precision_score(y, score)),
            'roc_auc': float(roc_auc_score(y, score)), 'n_cookies': len(y),
            'n_bots': int(np.sum(y)), 'bot_prevalence': float(np.mean(y))}



if __name__ == '__main__':
    from train_improved import main
    main()
