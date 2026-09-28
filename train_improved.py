import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

from advanced_features import advanced_features
from solution import build_features


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'files')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--reuse-features', action='store_true')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'selected_config.json').read_text(encoding='utf-8'))
    date_columns = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
    if args.reuse_features:
        train = pd.read_csv(args.data_dir / 'data/train.csv', parse_dates=date_columns)
        test = pd.read_csv(args.data_dir / 'data/test.csv', parse_dates=date_columns)
        base = pd.read_csv(out / 'features.csv', float_precision='round_trip')
        info = json.loads((out / 'feature_info.json').read_text(encoding='utf-8'))
    else:
        train, test, base, info = build_features(args.data_dir)
        base.to_csv(out / 'features.csv', index=False)
        (out / 'feature_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    meta = pd.concat([train.drop(columns='target'), test], ignore_index=True)
    assert base.cookie_id.tolist() == meta.cookie_id.tolist()
    features = {'base': base.drop(columns='cookie_id')}
    categorical = {'base': info['categorical_features']}
    if any(spec['features'] == 'advanced' for spec in config['models']):
        if args.reuse_features:
            extra = pd.read_csv(out / 'advanced_features.csv', float_precision='round_trip')
            cats = ['extra_category_mode', 'extra_location_mode', 'extra_query_mode']
        else:
            extra, cats = advanced_features(args.data_dir, meta)
            extra.to_csv(out / 'advanced_features.csv', index=False)
        assert base.cookie_id.equals(extra.cookie_id)
        features['advanced'] = pd.concat([features['base'], extra.drop(columns='cookie_id')], axis=1)
        categorical['advanced'] = categorical['base'] + cats
    scores = []
    importances = []
    (out / 'models').mkdir(exist_ok=True)
    for i, spec in enumerate(config['models']):
        x = features[spec['features']]
        model = CatBoostClassifier(
            loss_function='Logloss', random_seed=config['random_state'], thread_count=4,
            allow_writing_files=False, verbose=False, cat_features=categorical[spec['features']],
            **spec['params'],
        )
        with threadpool_limits(limits=4):
            model.fit(x.iloc[:len(train)], train.target.to_numpy())
            scores.append(model.predict_proba(x.iloc[len(train):])[:, 1])
        model.save_model(str(out / 'models' / f'model_{i}.cbm'))
        importances.append(pd.Series(model.feature_importances_, index=x.columns) * config['weights'][i])
        print('TRAINED', spec['name'], flush=True)
    score = np.average(scores, axis=0, weights=config['weights'])
    submission = pd.DataFrame({'cookie_id': test.cookie_id, 'score': score})
    assert len(submission) == len(test) and submission.cookie_id.is_unique
    assert submission.cookie_id.equals(test.cookie_id)
    assert np.isfinite(score).all() and submission.score.between(0, 1).all()
    submission.to_csv(out / 'submission.csv', index=False)
    if importances:
        importance = pd.concat(importances, axis=1).fillna(0).sum(axis=1)
        importance.sort_values(ascending=False).rename_axis('feature').reset_index(name='importance').to_csv(out / 'feature_importance_v2.csv', index=False)
    report = {'version': 2, 'python': platform.python_version(), 'random_state': config['random_state'],
              'data': info, 'feature_set_counts': {name: frame.shape[1] for name, frame in features.items()},
              'selected': config, 'validation_status': 'temporal cross-validation used for selection; no new independent holdout',
              'submission': {'rows': len(test), 'score_min': float(score.min()), 'score_max': float(score.max()), 'valid': True}}
    (out / 'metrics_v2.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SUBMISSION', json.dumps(report['submission']), flush=True)


if __name__ == '__main__':
    main()
