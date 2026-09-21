#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliding-window timeline of the seven acoustic liveness indices.

Each 0.5 s window (advanced by 30 ms) is scored as if it were an independent
clip, using the methods of AcousticLivenessScorer from detectliveness.py.
Two figures are produced:
  <out>_subscores.png : waveform + seven normalized sub-scores + overall spoof score
  <out>_raw.png       : waveform + the raw physical metrics behind each sub-score

Keep this file next to detectliveness.py.
"""

import argparse
import csv
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np

from detectliveness import AcousticLivenessScorer, load_wav

WIN_S = 0.5    # window length [s]
HOP_S = 0.030  # window increment [s]


# --------------------------------------------------------------------------
# Per-window feature helpers
# --------------------------------------------------------------------------
def _window_dynamic_range(seg: np.ndarray) -> float:
    """Same per-window DR as AcousticLivenessScorer._get_temporal_dynamic_range
    (95th - 5th percentile of the short-time RMS in dB)."""
    rms = librosa.feature.rms(y=seg, frame_length=512, hop_length=128)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))


def _window_peak_clipping(seg: np.ndarray, ref_peak: float):
    """Same as AcousticLivenessScorer._get_peak_clipping, but normalised by the
    peak of the WHOLE recording. Normalising by each window's own peak would
    make every quiet window look clipped (its loudest sample is always 1.0)."""
    norm = np.abs(seg) / (ref_peak + 1e-8)
    clip_ratio = float(np.mean(norm >= 0.98))
    flat = (np.abs(np.diff(seg)) < 1e-4) & (norm[:-1] >= 0.98)
    if not flat.any():
        return clip_ratio, 0.0
    edges = np.where(np.diff(np.pad(flat.astype(int), (1, 1))))[0]
    runs = edges[1::2] - edges[::2]
    return clip_ratio, float(runs.max())


def _rolling_std(x: np.ndarray, half_width: int) -> np.ndarray:
    return np.array(
        [np.std(x[max(0, i - half_width): i + half_width + 1]) for i in range(len(x))]
    )


# --------------------------------------------------------------------------
# Main timeline computation
# --------------------------------------------------------------------------
def liveness_timeline(
    y: np.ndarray,
    sr: int,
    win_s: float = WIN_S,
    hop_s: float = HOP_S,
    dr_std_context_s: float = 2.0,
    scorer: AcousticLivenessScorer = None,
    verbose: bool = True,
) -> dict:
    """Return per-window raw features, sub-scores and overall spoof score.

    dr_std_context_s: a single 0.5 s window has no 'std of DR across windows',
    so that term is taken as the rolling std of the per-window DR over a
    centred context of this length.
    """
    scorer = scorer or AcousticLivenessScorer(sample_rate=sr)
    win, hop = int(round(win_s * sr)), int(round(hop_s * sr))
    if len(y) < win:
        raise ValueError(f'Audio ({len(y) / sr:.2f} s) is shorter than one window ({win_s} s).')

    starts = np.arange(0, len(y) - win + 1, hop)
    ref_peak = float(np.max(np.abs(y)))
    keys = ['crest_factor', 'spectral_rolloff', 'tilt', 'bispectrum',
            'harmonic_high_ratio', 'temp_dr_mean', 'clip_ratio', 'max_flat_run',
            'echo_amplitude', 'echo_delay_ms']
    raw = {k: np.zeros(len(starts)) for k in keys}

    t0 = time.time()
    for i, s in enumerate(starts):
        seg = y[s: s + win]
        raw['crest_factor'][i] = scorer._get_crest_factor(seg)
        raw['spectral_rolloff'][i], raw['tilt'][i] = scorer._get_spectral_rolloff_tilt(seg)
        raw['bispectrum'][i] = scorer._get_bispectrum_magnitude(seg)
        raw['harmonic_high_ratio'][i] = scorer._get_harmonic_band_ratio(seg)
        raw['temp_dr_mean'][i] = _window_dynamic_range(seg)
        raw['clip_ratio'][i], raw['max_flat_run'][i] = _window_peak_clipping(seg, ref_peak)
        raw['echo_amplitude'][i], raw['echo_delay_ms'][i] = scorer._get_echo(seg)
        if verbose and (i + 1) % max(1, len(starts) // 10) == 0:
            print(f'  {i + 1}/{len(starts)} windows  ({time.time() - t0:.0f}s)')

    half = max(1, int(round(dr_std_context_s / hop_s)) // 2)
    raw['temp_dr_std'] = _rolling_std(raw['temp_dr_mean'], half)

    # Reuse the scorer's own normalisation, window by window
    sub_keys = list(scorer.weights)
    sub = {k: np.zeros(len(starts)) for k in sub_keys}
    for i in range(len(starts)):
        sc = scorer._normalize_to_subscores({k: v[i] for k, v in raw.items()})
        for k in sub_keys:
            sub[k][i] = sc[k]
    overall = sum(scorer.weights[k] * sub[k] for k in sub_keys)

    return {
        't': (starts + win / 2) / sr,  # window centres [s]
        'raw': raw,
        'sub': sub,
        'overall': overall,
        'weights': scorer.weights,
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _waveform(ax, y, sr):
    ax.plot(np.arange(len(y)) / sr, y, lw=0.4, color='0.35')
    ax.set_ylabel('waveform')
    ax.set_ylim(-1.05 * np.max(np.abs(y)), 1.05 * np.max(np.abs(y)))


def plot_subscores(res, y, sr, path):
    t, sub, w = res['t'], res['sub'], res['weights']
    names = list(sub)
    fig, axes = plt.subplots(len(names) + 2, 1, figsize=(11, 14), sharex=True,
                             gridspec_kw={'height_ratios': [1.2] + [1] * len(names) + [1.4]})
    _waveform(axes[0], y, sr)
    for ax, k in zip(axes[1:-1], names):
        ax.plot(t, sub[k], lw=1.2, color='tab:blue')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(f'{k}\n(w={w[k]:.2f})', fontsize=8)
        ax.grid(alpha=0.3)
    ax = axes[-1]
    ax.plot(t, res['overall'], lw=1.5, color='k')
    ax.axhline(0.5, color='tab:red', ls='--', lw=1)
    ax.fill_between(t, 0.5, res['overall'], where=res['overall'] >= 0.5,
                    color='tab:red', alpha=0.3, interpolate=True)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('overall spoof\nlikelihood', fontsize=8)
    ax.set_xlabel('time [s]  (window centre)')
    ax.grid(alpha=0.3)
    axes[0].set_title('Sub-scores per window (0 = live, 1 = spoof)')
    axes[0].set_xlim(0, len(y) / sr)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_raw(res, y, sr, path):
    t, r = res['t'], dict(res['raw'])
    # the delay is only meaningful where an echo is actually detected
    r['echo_delay_ms'] = np.where(r['echo_amplitude'] > 0.12, r['echo_delay_ms'], np.nan)
    fig, axes = plt.subplots(8, 1, figsize=(11, 14.5), sharex=True,
                             gridspec_kw={'height_ratios': [1.2] + [1] * 7})
    _waveform(axes[0], y, sr)

    def twin(ax, k1, k2, l1, l2, c1='tab:blue', c2='tab:orange'):
        ax.plot(t, r[k1], color=c1, lw=1.2)
        ax.set_ylabel(l1, color=c1, fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(t, r[k2], color=c2, lw=1.0)
        ax2.set_ylabel(l2, color=c2, fontsize=8)

    ax = axes[1]
    ax.plot(t, r['crest_factor'], lw=1.2)
    ax.set_ylabel('crest factor\n[dB]', fontsize=8)
    twin(axes[2], 'spectral_rolloff', 'tilt', 'roll-off 95%\n[Hz]', 'tilt [dB/decade]')
    axes[3].plot(t, r['bispectrum'], lw=1.2)
    axes[3].set_ylabel('bispectrum\nmagnitude', fontsize=8)
    axes[4].plot(t, r['harmonic_high_ratio'], lw=1.2)
    axes[4].set_ylabel('harmonic ratio\n3.4-8 kHz', fontsize=8)
    axes[5].plot(t, r['temp_dr_mean'], lw=1.2, label='DR (p95-p5)')
    axes[5].plot(t, r['temp_dr_std'], lw=1.0, color='tab:orange', label='rolling std')
    axes[5].set_ylabel('temporal DR\n[dB]', fontsize=8)
    axes[5].legend(fontsize=7, loc='upper right')
    twin(axes[6], 'clip_ratio', 'max_flat_run', 'clip ratio', 'max flat run\n[samples]')
    twin(axes[7], 'echo_amplitude', 'echo_delay_ms', 'echo amplitude\n(cepstral)', 'echo delay [ms]',
         c2='tab:green')
    axes[7].set_xlabel('time [s]  (window centre)')
    for ax in axes[1:]:
        ax.grid(alpha=0.3)
    axes[0].set_title('Raw acoustic metrics per window')
    axes[0].set_xlim(0, len(y) / sr)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def save_csv(res, path):
    cols = {'time_s': res['t'], **{f'raw_{k}': v for k, v in res['raw'].items()},
            **{f'sub_{k}': v for k, v in res['sub'].items()},
            'overall_spoof_likelihood': res['overall']}
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        wr.writerows(zip(*cols.values()))


# --------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('wav', nargs='?', default='test_sample.wav')
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--win', type=float, default=WIN_S, help='window length [s]')
    ap.add_argument('--hop', type=float, default=HOP_S, help='window increment [s]')
    ap.add_argument('--dr-context', type=float, default=2.0,
                    help='context [s] for the rolling std of temporal DR')
    ap.add_argument('--out', default=None, help='output prefix (default: wav name)')
    ap.add_argument('--show', action='store_true', help='open the figures interactively')
    args = ap.parse_args()

    out = args.out or args.wav.rsplit('.', 1)[0] + '_timeline'
    y, sr = load_wav(args.wav, target_sr=args.sr)
    print(f'{len(y) / sr:.2f} s @ {sr} Hz  ->  window {args.win}s / hop {args.hop}s')

    res = liveness_timeline(y, sr, args.win, args.hop, args.dr_context)
    plot_subscores(res, y, sr, f'{out}_subscores.png')
    plot_raw(res, y, sr, f'{out}_raw.png')
    save_csv(res, f'{out}.csv')
    print(f'Saved {out}_subscores.png, {out}_raw.png, {out}.csv')
    print(f'Mean overall spoof likelihood: {res["overall"].mean():.3f}; '
          f'{100 * np.mean(res["overall"] >= 0.5):.1f}% of windows >= 0.5')
    if args.show:
        plt.show()