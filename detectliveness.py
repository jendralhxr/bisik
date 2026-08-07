#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:26:53 2026
@author: rdx
"""

import librosa
import numpy as np


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
if __name__ == '__main__':
    y, sr = load_wav('test_sample.wav')

    #scorer = AcousticLivenessScorer(weights=custom_weights, sample_rate=sr)
    scorer = AcousticLivenessScorer(sample_rate=sr)
    result = scorer.predict_liveness(y)
    print(result)

    print(f"Overall Spoof Likelihood: {result['spoof_likelihood'] * 100:.2f}%")
    print(f"Decision: {result['decision']}")
    print('\nIndividual Sub-Scores (0 = Live, 1 = Spoof):')
    for k, v in result['subscores'].items():
        print(f"  - {k}: {v}")
        
        
        