#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:48:44 2026

@author: rdx
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised 'live speech' vs 'replay spoof' classifier on the acoustic liveness features.

Input: a CSV manifest with (at least) a file column and a label column, e.g.

    file,label,group
    audio/spk01_utt01.wav,live speech,spk01
    audio/spk01_utt01_replay.wav,replay spoof,spk01

  * file   (aliases: filename, path, wav, audio) - absolute, or relative to the CSV
            (or to --audio-root)
  * label  (aliases: class, target) - 0/1, or text such as 'live speech', 'live',
            'bonafide' -> class 0;  'replay spoof', 'replay', 'spoof' -> class 1
  * group  (optional; aliases: speaker, speaker_id, session) - clips sharing a group are never
            split between train and test in cross-validation. If absent, the group is the
            file-name text before the first '_' (--group-sep), e.g. spk01_utt03.wav -> spk01.

    python train_liveness_ensemble.py train   --csv labels.csv --model liveness_ensemble.joblib
    python train_liveness_ensemble.py predict --model liveness_ensemble.joblib a.wav b.wav

Extracted features are cached in <csv name>_features.csv next to the CSV, per audio file, so
editing labels/groups or adding rows only computes what is new (--refresh recomputes all).

Ensemble = soft vote (mean probability) of four deliberately different learners:
    hgb    shallow gradient-boosted trees   (non-linear thresholds / interactions)
    et     extra-trees                      (variance reduction, robust to noise)
    logreg logistic regression              (smooth, monotone, extrapolates sanely)
    svm    RBF SVM                          (smooth non-linear boundary)
Keep this file next to detectliveness.py.
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

from detectliveness import AcousticLivenessScorer, load_wav

SR = 22050
MIN_SECONDS = 1.0
CLASSES = {0: 'live speech', 1: 'replay spoof'}
FEATURES = [
    'crest_factor', 'spectral_rolloff', 'tilt', 'bispectrum', 'harmonic_high_ratio',
    'temp_dr_mean', 'temp_dr_std', 'clip_ratio', 'max_flat_run',
    'echo_amplitude', 'echo_delay_ms',
]

warnings.filterwarnings('ignore', message='n_quantiles')


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def extract_features(y: np.ndarray, sr: int = SR) -> list:
    """Raw (not hand-thresholded) liveness metrics for one clip.

    The clip is RMS-normalised first: the bispectrum grows with amplitude cubed,
    so without this the model would partly learn recording gain instead of liveness.
    """
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-8) * 0.1
    raw = AcousticLivenessScorer(sample_rate=sr).extract_raw_features(y)
    return [raw[k] for k in FEATURES]


def _extract_file(path: Path):
    try:
        y, sr = load_wav(str(path), target_sr=SR)
        if len(y) < MIN_SECONDS * sr:
            print(f'  skipped (< {MIN_SECONDS}s): {path.name}')
            return None
        return extract_features(y, sr)
    except Exception as e:  # keep going on a bad file
        print(f'  skipped ({e}): {path.name}')
        return None


FILE_COLS = ('file', 'filename', 'path', 'wav', 'audio')
LABEL_COLS = ('label', 'class', 'target')
GROUP_COLS = ('group', 'speaker', 'speaker_id', 'session')
LABEL_MAP = {
    **dict.fromkeys(('0', 'live', 'live speech', 'life speech', 'bonafide', 'bona fide',
                     'genuine', 'real'), 0),
    **dict.fromkeys(('1', 'replay', 'replay spoof', 'spoof', 'attack', 'fake'), 1),
}


def _pick_col(columns, aliases):
    lower = {str(c).strip().lower(): c for c in columns}
    return next((lower[a] for a in aliases if a in lower), None)


def read_manifest(csv_path: Path, audio_root, sep: str) -> pd.DataFrame:
    """CSV -> DataFrame[file, label, group] with resolved paths and parsed labels."""
    m = pd.read_csv(csv_path, sep=None, engine='python')  # sniffs , ; tab
    fcol, lcol = _pick_col(m.columns, FILE_COLS), _pick_col(m.columns, LABEL_COLS)
    if fcol is None or lcol is None:
        raise SystemExit(f'{csv_path} needs a file column {FILE_COLS} and a label column '
                         f'{LABEL_COLS}; found {list(m.columns)}')
    gcol = _pick_col(m.columns, GROUP_COLS)

    root = Path(audio_root) if audio_root else csv_path.parent
    paths = [Path(str(f)).expanduser() for f in m[fcol]]
    paths = [p if p.is_absolute() else root / p for p in paths]

    bad = sorted({str(v) for v in m[lcol] if str(v).strip().lower() not in LABEL_MAP})
    if bad:
        raise SystemExit(f'Unrecognised label(s) {bad}. Use 0/1 or one of: {sorted(LABEL_MAP)}')
    labels = [LABEL_MAP[str(v).strip().lower()] for v in m[lcol]]
    groups = m[gcol].astype(str).tolist() if gcol else [p.stem.split(sep)[0] for p in paths]

    df = pd.DataFrame({'file': [str(p) for p in paths], 'label': labels, 'group': groups})
    missing = ~df['file'].map(lambda f: Path(f).is_file())
    if missing.any():
        print(f'WARNING: {int(missing.sum())} listed file(s) not found, skipped, e.g. '
              f'{df.loc[missing, "file"].iloc[0]}')
        df = df[~missing]
    n_dup = int(df['file'].duplicated().sum())
    if n_dup:
        print(f'WARNING: {n_dup} duplicate file row(s) dropped')
        df = df.drop_duplicates('file')
    return df.reset_index(drop=True)


def build_table(manifest: pd.DataFrame, cache: Path, jobs: int, refresh: bool) -> pd.DataFrame:
    """Attach features to the manifest, computing only files missing from the cache."""
    cached = pd.DataFrame(columns=['file'] + FEATURES)
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)
    todo = manifest.loc[~manifest['file'].isin(cached['file']), 'file'].tolist()
    print(f'{len(manifest)} files listed: {len(manifest) - len(todo)} cached, {len(todo)} to extract')
    if todo:
        feats = Parallel(n_jobs=jobs)(delayed(_extract_file)(Path(f)) for f in todo)
        new = pd.DataFrame([dict(file=f, **dict(zip(FEATURES, x)))
                            for f, x in zip(todo, feats) if x is not None])
        cached = pd.concat([cached, new], ignore_index=True)
        cached.to_csv(cache, index=False)
    return manifest.merge(cached, on='file', how='inner')


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_members(seed: int = 0) -> dict:
    def qt():  # rank-Gaussianise: tames heavy tails (bispectrum, clip ratio) for the smooth models
        return QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=seed)

    return {
        'hgb': HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=10,
            l2_regularization=1.0, class_weight='balanced', random_state=seed),
        'et': ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight='balanced',
            n_jobs=-1, random_state=seed),
        'logreg': make_pipeline(
            qt(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)),
        'svm': make_pipeline(
            qt(), SVC(C=1.0, kernel='rbf', probability=True, class_weight='balanced',
                      random_state=seed)),
    }


def eer_and_threshold(y, score):
    """Equal error rate (miss rate == false-alarm rate) and the score threshold that gives it."""
    fpr, tpr, thr = roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def oof_scores(X, y, groups, n_splits, seed=0) -> dict:
    """Out-of-fold P(replay) for each member and for the mean-probability ensemble."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {k: np.zeros(len(y)) for k in make_members()}
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        for k, m in make_members(seed).items():
            oof[k][te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    oof['ENSEMBLE'] = np.mean([oof[k] for k in list(oof)], axis=0)
    return oof


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_train(a):
    csv_path = Path(a.csv)
    manifest = read_manifest(csv_path, a.audio_root, a.group_sep)
    cache = csv_path.with_name(csv_path.stem + '_features.csv')
    df = build_table(manifest, cache, a.jobs, a.refresh)
    if df.empty:
        raise SystemExit('No usable clips.')
    X, y, groups = df[FEATURES].to_numpy(float), df['label'].to_numpy(int), df['group'].to_numpy()
    n_groups = len(set(groups))
    print(f'\n{len(df)} clips | live {int((y == 0).sum())}, replay {int((y == 1).sum())} | {n_groups} groups')
    if n_groups < 2 or len(set(y)) < 2:
        raise SystemExit('Need both classes and at least 2 groups.')
    if n_groups == len(df):
        print('WARNING: every clip is its own group (no group column, and no "_" in the file names?). If several clips share a '
              'speaker/session, cross-validation results will be optimistic.')

    n_splits = min(a.folds, n_groups)
    oof = oof_scores(X, y, groups, n_splits)
    print(f'\nGrouped {n_splits}-fold cross-validation (positive class = replay spoof)')
    print(f'{"model":<10}{"AUC":>8}{"EER %":>8}')
    for k, s in oof.items():
        print(f'{k:<10}{roc_auc_score(y, s):8.3f}{100 * eer_and_threshold(y, s)[0]:8.1f}')

    eer, thr = eer_and_threshold(y, oof['ENSEMBLE'])
    cm = confusion_matrix(y, oof['ENSEMBLE'] >= thr)
    print(f'\nEnsemble operating point (EER threshold = {thr:.3f}):')
    print(f'                    pred live   pred replay')
    print(f'  true live        {cm[0, 0]:9d}   {cm[0, 1]:11d}')
    print(f'  true replay      {cm[1, 0]:9d}   {cm[1, 1]:11d}')

    members = {k: m.fit(X, y) for k, m in make_members().items()}
    imp = sorted(zip(FEATURES, members['et'].feature_importances_), key=lambda t: -t[1])
    print('\nExtra-trees feature importance (fitted on all data):')
    for k, v in imp:
        print(f'  {k:<22}{v:6.3f}')

    joblib.dump({'members': members, 'threshold': thr, 'features': FEATURES, 'sr': SR}, a.model)
    print(f'\nSaved {a.model}')


def cmd_predict(a):
    bundle = joblib.load(a.model)
    for f in a.wav:
        y, sr = load_wav(f, target_sr=bundle['sr'])
        x = np.array([extract_features(y, sr)])
        p = float(np.mean([m.predict_proba(x)[0, 1] for m in bundle['members'].values()]))
        label = CLASSES[int(p >= bundle['threshold'])]
        print(f'{f}: {label}   P(replay)={p:.3f}  (threshold {bundle["threshold"]:.3f})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--csv', required=True, help='manifest with file and label columns')
    t.add_argument('--audio-root', default=None, help='base folder for relative file paths (default: CSV folder)')
    t.add_argument('--model', default='liveness_ensemble.joblib')
    t.add_argument('--group-sep', default='_', help='if no group column: text before this in the file name = group id')
    t.add_argument('--folds', type=int, default=5)
    t.add_argument('--jobs', type=int, default=-1, help='parallel workers for feature extraction')
    t.add_argument('--refresh', action='store_true', help='ignore the feature cache and recompute all')
    t.set_defaults(fn=cmd_train)
    p = sub.add_parser('predict')
    p.add_argument('--model', required=True)
    p.add_argument('wav', nargs='+')
    p.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    args.fn(args)