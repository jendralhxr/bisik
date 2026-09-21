#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:26:53 2026
@author: rdx
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import time

WIN_S = 0.5    # window length [s]
HOP_S = 0.030  # window increment [s]




def load_wav(
    file_path: str, target_sr: int = 22050, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Loads a WAV file and returns the audio array and sample rate."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return y, sr

class AcousticLivenessScorer:
    """Extracts 6 acoustic liveness metrics, normalizes them into individual spoofing risk

    sub-scores (0.0 to 1.0), and computes an overall weighted spoof likelihood.
    """

    def __init__(self, weights: dict = None, sample_rate: int = 22050):
        self.sr = sample_rate

        # Default adjustable weights (Must sum to 1.0)
        self.weights = weights or {
            'crest_factor': 0.20,  # Peak-to-RMS energy compression
            'spectral_rolloff_tilt': 0.15,  # High-frequency band-limiting
            'bispectrum_distortion': 0.15,  # Non-linear phase/transducer artifacts
            'harmonic_dynamics': 0.15,  # High-frequency harmonic energy ratio
            'temporal_dynamic_range': 0.20,  # Temporal envelope variance (AGC/DRC)
            'peak_clipping': 0.15,  # Amplifier saturation / clipping
        }
        self._normalize_weights()

    def _normalize_weights(self):
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    # --- Feature Extraction Functions ---
    def _get_crest_factor(self, y):
        frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        peaks = np.max(np.abs(frames), axis=0)
        rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-8)
        return float(np.mean(20 * np.log10((peaks / rms) + 1e-8)))

    def _get_spectral_rolloff_tilt(self, y):
        rolloff = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=y, sr=self.sr, roll_percent=0.95
                )[0]
            )
        )
        S = np.abs(librosa.stft(y)) ** 2
        avg_power = np.mean(S, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        valid = freqs > 0
        tilt_slope, _ = np.polyfit(
            np.log10(freqs[valid]), 10 * np.log10(avg_power[valid] + 1e-8), 1
        )
        return rolloff, float(tilt_slope)

    def _get_bispectrum_magnitude(self, y, n_fft=256):
        stft = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2)
        max_bin = stft.shape[0] // 2
        bispec_sum, count = 0.0, 0
        for f1 in range(1, max_bin, 2):  # Strided for performance
            for f2 in range(1, max_bin - f1, 2):
                f3 = f1 + f2
                b_val = stft[f1, :] * stft[f2, :] * np.conj(stft[f3, :])
                bispec_sum += np.mean(np.abs(b_val))
                count += 1
        return float(bispec_sum / (count + 1e-8))

    def _get_harmonic_band_ratio(self, y):
        y_harm, _ = librosa.effects.hpss(y)
        S = np.abs(librosa.stft(y_harm, n_fft=2048)) ** 2
        freqs = librosa.fft_frequencies(sr=self.sr)
        high_harm_mask = (freqs >= 3400) & (freqs < 8000)
        e_high = np.sum(S[high_harm_mask, :]) + 1e-8
        e_total = np.sum(S) + 1e-8
        return float(e_high / e_total)

    def _get_temporal_dynamic_range(self, y):
        windows = librosa.util.frame(
            y, frame_length=int(0.5 * self.sr), hop_length=int(0.25 * self.sr)
        )
        drs = []
        for i in range(windows.shape[1]):
            rms = librosa.feature.rms(
                y=windows[:, i], frame_length=512, hop_length=128
            )[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            drs.append(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
        return float(np.mean(drs)), float(np.std(drs))

    def _get_peak_clipping(self, y):
        norm_y = np.abs(y) / (np.max(np.abs(y)) + 1e-8)
        clip_ratio = float(np.mean(norm_y >= 0.98))
        diffs = np.diff(y)
        flat_mask = (np.abs(diffs) < 1e-4) & (norm_y[:-1] >= 0.98)
        max_run = (
            float(
                np.max(
                    np.diff(
                        np.where(
                            np.diff(np.pad(flat_mask.astype(int), (1, 1)))
                        )[0]
                    )[::2]
                )
            )
            if np.any(flat_mask)
            else 0.0
        )
        return clip_ratio, max_run

    # --- Normalization (Acoustic Threshold Mapping to [0.0, 1.0]) ---
    def _normalize_to_subscores(self, raw_features: dict) -> dict:
        """Converts raw acoustic metrics into individual spoof likelihood sub-scores [0.0 = Live, 1.0 = Spoof]."""
        subscores = {}

        # 1. Crest Factor: Live ~ 14-20 dB; Replayed/Compressed ~ < 10 dB
        cf = raw_features['crest_factor']
        subscores['crest_factor'] = float(np.clip((16.0 - cf) / 8.0, 0.0, 1.0))

        # 2. Spectral Roll-off & Tilt: Low roll-off (< 4500 Hz) indicates speaker cutoff
        rolloff, tilt = raw_features['spectral_rolloff'], raw_features['tilt']
        s_rolloff = np.clip((6000.0 - rolloff) / 3500.0, 0.0, 1.0)
        s_tilt = np.clip((-1.5 - tilt) / 2.0, 0.0, 1.0)
        subscores['spectral_rolloff_tilt'] = float(
            0.6 * s_rolloff + 0.4 * s_tilt
        )

        # 3. Bispectrum Distortion: Higher non-linear phase coherence = transducer distortion
        bispec = raw_features['bispectrum']
        subscores['bispectrum_distortion'] = float(
            np.clip((bispec - 0.005) / 0.045, 0.0, 1.0)
        )

        # 4. Harmonic Band Ratio: Replayed audio drops upper harmonic energy (< 0.05)
        h_ratio = raw_features['harmonic_high_ratio']
        subscores['harmonic_dynamics'] = float(
            np.clip((0.08 - h_ratio) / 0.08, 0.0, 1.0)
        )

        # 5. Temporal Dynamic Range: Compressed audio exhibits squashed local DR (< 15 dB)
        dr_mean, dr_std = (
            raw_features['temp_dr_mean'],
            raw_features['temp_dr_std'],
        )
        s_dr = np.clip((22.0 - dr_mean) / 15.0, 0.0, 1.0)
        s_std = np.clip((6.0 - dr_std) / 5.0, 0.0, 1.0)
        subscores['temporal_dynamic_range'] = float(0.7 * s_dr + 0.3 * s_std)

        # 6. Peak Clipping: High clipping ratio (> 0.01) indicates amplifier overdrive
        clip_ratio, max_run = (
            raw_features['clip_ratio'],
            raw_features['max_flat_run'],
        )
        subscores['peak_clipping'] = float(
            np.clip(clip_ratio * 50.0 + (max_run / 20.0), 0.0, 1.0)
        )

        return subscores

    # --- Main Analysis Function ---
    def predict_liveness(self, y: np.ndarray) -> dict:
        """Extracts all features, computes individual sub-scores, and aggregates the overall spoofing likelihood metric."""
        # 1. Raw Feature Extraction
        rolloff, tilt = self._get_spectral_rolloff_tilt(y)
        dr_mean, dr_std = self._get_temporal_dynamic_range(y)
        clip_ratio, max_run = self._get_peak_clipping(y)

        raw_features = {
            'crest_factor': self._get_crest_factor(y),
            'spectral_rolloff': rolloff,
            'tilt': tilt,
            'bispectrum': self._get_bispectrum_magnitude(y),
            'harmonic_high_ratio': self._get_harmonic_band_ratio(y),
            'temp_dr_mean': dr_mean,
            'temp_dr_std': dr_std,
            'clip_ratio': clip_ratio,
            'max_flat_run': max_run,
        }

        # 2. Convert to normalized sub-scores [0.0 - 1.0]
        subscores = self._normalize_to_subscores(raw_features)

        # 3. Calculate Weighted Aggregated Spoofing Score
        overall_spoof_likelihood = sum(
            self.weights[k] * subscores[k] for k in self.weights
        )

        return {
            'spoof_likelihood': float(
                np.round(overall_spoof_likelihood, 4)
            ),  # 0.0 = Live, 1.0 = Spoof
            'decision': 'SPOOF' if overall_spoof_likelihood >= 0.50 else 'LIVE',
            'subscores': {k: round(v, 4) for k, v in subscores.items()},
            'applied_weights': self.weights,
            'raw_features': {k: round(v, 4) for k, v in raw_features.items()},
        }


# --- Example Usage ---
""" ingest all at once
if __name__ == '__main__':
    y, sr = load_wav('test.wav')

    #scorer = AcousticLivenessScorer(weights=custom_weights, sample_rate=sr)
    scorer = AcousticLivenessScorer(sample_rate=sr)
    result = scorer.predict_liveness(y)
    print(result)

    print(f"Overall Spoof Likelihood: {result['spoof_likelihood'] * 100:.2f}%")
    print(f"Decision: {result['decision']}")
    print('\nIndividual Sub-Scores (0 = Live, 1 = Spoof):')
    for k, v in result['subscores'].items():
        print(f"  - {k}: {v}")
"""
        
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
            'harmonic_high_ratio', 'temp_dr_mean', 'clip_ratio', 'max_flat_run']
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
        if (i + 1) % max(1, len(starts) // 10) == 0:
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
    fig, axes = plt.subplots(len(names) + 2, 1, figsize=(11, 13), sharex=True,
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
    t, r = res['t'], res['raw']
    fig, axes = plt.subplots(7, 1, figsize=(11, 13), sharex=True,
                             gridspec_kw={'height_ratios': [1.2] + [1] * 6})
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
    axes[6].set_xlabel('time [s]  (window centre)')
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
    ap.add_argument('wav', nargs='?', default='harmonika.wav')
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
        