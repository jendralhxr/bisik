import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# OPTIMIZED FEATURE EXTRACTION FUNCTIONS (Designed for 600ms / 30ms windowing)
# ==============================================================================

def extract_crest_factor(y_frame, frame_length=2048, hop_length=512):
    """Computes frame-level Crest Factor (Peak-to-RMS ratio in dB)."""
    frames = librosa.util.frame(y_frame, frame_length=frame_length, hop_length=hop_length)
    peaks = np.max(np.abs(frames), axis=0)
    rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-8)
    crest_factor_db = 20 * np.log10((peaks / (rms + 1e-8)) + 1e-8)
    return np.mean(crest_factor_db)

def extract_temporal_dynamic_range(y_frame):
    """Computes local Dynamic Range (95th vs 5th percentile dB) within window."""
    rms = librosa.feature.rms(y=y_frame, frame_length=512, hop_length=128)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    return np.percentile(rms_db, 95) - np.percentile(rms_db, 5)

def extract_peak_clipping_metrics(y_frame, threshold=0.98):
    """Detects hard and soft peak clipping artifacts within window."""
    abs_y = np.abs(y_frame)
    max_val = np.max(abs_y) + 1e-8
    norm_y = abs_y / max_val

    # 1. Clipping Ratio
    clipped_mask = norm_y >= threshold
    clipping_ratio = np.mean(clipped_mask)

    # 2. Mean Flat Run
    diffs = np.diff(y_frame)
    flat_mask = (np.abs(diffs) < 1e-4) & (norm_y[:-1] >= threshold)
    if np.any(flat_mask):
        run_starts = np.where(np.diff(np.pad(flat_mask.astype(int), (1, 1))))[0]
        run_lengths = run_starts[1::2] - run_starts[::2]
        mean_flat_run = np.mean(run_lengths)
    else:
        mean_flat_run = 0.0

    # 3. Top Peak Variance
    top_peaks = norm_y[norm_y >= np.percentile(norm_y, 95)]
    top_peak_variance = np.var(top_peaks)

    return clipping_ratio, mean_flat_run, top_peak_variance

def extract_temporal_harmonic_bands(y_frame, sr=22050):
    """Tracks harmonic power ratios across specific frequency bands."""
    y_harmonic, _ = librosa.effects.hpss(y_frame)
    S = np.abs(librosa.stft(y_harmonic, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)

    f0_mask = (freqs >= 50) & (freqs < 300)
    mid_harm_mask = (freqs >= 300) & (freqs < 3400)
    high_harm_mask = (freqs >= 3400) & (freqs < 8000)

    e_f0 = np.sum(S[f0_mask, :]) + 1e-8
    e_mid = np.sum(S[mid_harm_mask, :]) + 1e-8
    e_high = np.sum(S[high_harm_mask, :]) + 1e-8
    e_total = np.sum(S) + 1e-8

    return e_f0 / e_total, e_mid / e_total, e_high / e_total

def extract_spectral_tilt_and_rolloff(y_frame, sr=22050):
    """Computes 95% spectral roll-off frequency and log-spectral tilt slope."""
    rolloff = librosa.feature.spectral_rolloff(y=y_frame, sr=sr, roll_percent=0.95)[0]
    mean_rolloff = np.mean(rolloff)

    S = np.abs(librosa.stft(y_frame)) ** 2
    avg_power_spectrum = np.mean(S, axis=1)

    freqs = librosa.fft_frequencies(sr=sr)
    valid_idx = freqs > 0

    log_freqs = np.log10(freqs[valid_idx])
    log_power = 10 * np.log10(avg_power_spectrum[valid_idx] + 1e-8)

    tilt_slope, _ = np.polyfit(log_freqs, log_power, 1)
    return mean_rolloff, tilt_slope

def extract_bispectrum_magnitude_fast(y_frame, n_fft=256):
    """Vectorized Bispectral magnitude estimation for fast sliding-window execution."""
    stft = librosa.stft(y_frame, n_fft=n_fft, hop_length=n_fft // 2)
    max_bin = (n_fft // 2) // 2

    f1_idx, f2_idx = np.tril_indices(max_bin - 1)
    f1_idx += 1
    f2_idx += 1
    f3_idx = f1_idx + f2_idx

    # Vectorized computation across time frames
    b_vals = stft[f1_idx, :] * stft[f2_idx, :] * np.conj(stft[f3_idx, :])
    return np.mean(np.abs(b_vals))

# ==============================================================================
# MAIN PROCESSING & PLOTTING PIPELINE
# ==============================================================================

def process_and_plot_audio_files(wav_directory="."):
    wav_files = sorted(glob.glob(os.path.join(wav_directory, "*.wav")))
    
    if not wav_files:
        print("No .wav files found in the target directory.")
        return

    # Track metrics for group analysis
    group_summary = {"live": {}, "replayspoof": {}, "mixed": {}}

    feature_names = [
        "1. Crest Factor (dB)",
        "2. Temp Dynamic Range (dB)",
        "3. Clipping Ratio",
        "4. High-Harmonic Energy Ratio",
        "5. Spectral Rolloff (Hz)",
        "6. Spectral Tilt Slope",
        "7. Bispectrum Magnitude"
    ]

    for filepath in wav_files:
        filename = os.path.basename(filepath)
        
        # Identify group tag
        if filename.startswith("live"):
            group = "live"
        elif filename.startswith("replayspoof"):
            group = "replayspoof"
        else:
            group = "mixed"

        print(f"Processing [{group.upper()}]: {filename}...")

        y, sr = librosa.load(filepath, sr=22050)
        
        # Setup 600 ms sliding window with 30 ms step
        window_sec = 0.60
        hop_sec = 0.03
        window_len = int(window_sec * sr)
        hop_len = int(hop_sec * sr)

        # Pad signal to ensure frame coverage
        if len(y) < window_len:
            y = np.pad(y, (0, window_len - len(y)))

        frames = librosa.util.frame(y, frame_length=window_len, hop_length=hop_len)
        num_frames = frames.shape[1]
        time_axis = np.arange(num_frames) * hop_sec + (window_sec / 2.0)

        # Pre-allocate feature arrays
        features = np.zeros((7, num_frames))

        for idx in range(num_frames):
            win = frames[:, idx]

            cf = extract_crest_factor(win)
            tdr = extract_temporal_dynamic_range(win)
            clip_ratio, _, _ = extract_peak_clipping_metrics(win)
            _, _, high_harm = extract_temporal_harmonic_bands(win, sr=sr)
            rolloff, tilt = extract_spectral_tilt_and_rolloff(win, sr=sr)
            bispec = extract_bispectrum_magnitude_fast(win)

            features[:, idx] = [cf, tdr, clip_ratio, high_harm, rolloff, tilt, bispec]

        # Record mean values for post-run comparative evaluation
        group_summary[group][filename] = np.mean(features, axis=1)

        # Plot raw waveform alongside time-series feature dynamics
        fig, axes = plt.subplots(8, 1, figsize=(14, 18), sharex=True)
        fig.suptitle(f"Spoof Analysis Dynamics: {filename} ({group.upper()})", fontsize=14, fontweight='bold')

        # Raw Audio Signal
        audio_time = np.linspace(0, len(y) / sr, len(y))
        axes[0].plot(audio_time, y, color='black', alpha=0.6)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Raw Audio Signal Waveform")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Feature Subplots
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        for i in range(7):
            axes[i + 1].plot(time_axis, features[i], color=colors[i], linewidth=1.5)
            axes[i + 1].set_ylabel(feature_names[i].split(". ")[1])
            axes[i + 1].set_title(feature_names[i])
            axes[i + 1].grid(True, linestyle="--", alpha=0.5)

        axes[-1].set_xlabel("Time (seconds)")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save output plots locally
        output_plot_name = f"analysis_{os.path.splitext(filename)[0]}.png"
        plt.savefig(output_plot_name, dpi=150)
        plt.close()

    # ==========================================================================
    # PRINT GROUP COMPARISON AND METRIC SELECTION GUIDANCE
    # ==========================================================================
    print("\n" + "="*80)
    print("FEATURE COMPARISON SUMMARY Across Groups (Mean Values)")
    print("="*80)
    
    for feat_idx, name in enumerate(feature_names):
        print(f"\n--- {name} ---")
        for grp in ["live", "replayspoof", "mixed"]:
            if group_summary[grp]:
                vals = [m[feat_idx] for m in group_summary[grp].values()]
                avg = np.mean(vals)
                std = np.std(vals)
                print(f"  {grp.upper():<12} | Avg: {avg:12.4f} | Std: {std:12.4f}")

# Execute script
if __name__ == "__main__":
    process_and_plot_audio_files(".")

# summary lama
# Feature Metric,Live (μ1​±σ1​),Spoof (μ2​±σ2​),Absolute Difference (Δμ),Pooled Std (σp​),Cohen's d Effect Size,Separation Quality
# 7. Bispectrum Magnitude,0.3137±0.2871,1.2549±1.8981,0.9412,1.3610,0.69,Strongest (Non-linear)
# 5. Spectral Rolloff (Hz),5977.8±1325.3,4680.6±1419.1,1297.16 Hz,1372.98,0.94,Strongest (Frequency)
# 6. Spectral Tilt Slope,−22.84±6.17,−27.83±6.29,4.99,6.23,0.80,Strong
# 2. Temporal Dynamic Range (dB),15.06±6.50,12.18±5.57,2.88 dB,6.05,0.48,Moderate
# 4. High-Harmonic Energy Ratio,0.0336±0.0688,0.0221±0.0521,0.0115,0.0610,0.19,Weak
# 1. Crest Factor (dB),9.99±1.60,10.10±1.06,0.11 dB,1.36,0.08,Insignificant
# 3. cliiping ois not even worth telling

# Feature Metric,LIVE (μ1​±σ1​),REPLAYSPOOF (μ2​±σ2​),Difference (Δμ),Pooled Std (σp​),Cohen's d,Separability (S),Trend vs. Previous Run
# 7. Bispectrum Magnitude,0.1947±0.2610,1.2549±1.8981,1.0602,1.3548,0.78,0.491,Stronger (d:0.69→0.78)
# 5. Spectral Rolloff (Hz),5887.68±1027.37,4680.63±1419.09,1207.05 Hz,1239.54,0.97,0.493,Stronger (d:0.94→0.97)
# 6. Spectral Tilt Slope,−23.5334±4.8444,−27.8312±6.2918,4.2978,5.6146,0.77,0.386,Slight drop (d:0.80→0.77)
# 2. Temp Dynamic Range (dB),15.5945±5.4979,12.1797±5.5669,3.4148 dB,5.5325,0.62,0.309,Stronger (d:0.48→0.62)
# 1. Crest Factor (dB),9.4127±1.4532,10.1023±1.0623,0.6896 dB,1.2743,0.54,0.274,Noise / Low variance
# 4. High-Harmonic Energy Ratio,0.0202±0.0549,0.0221±0.0521,0.0019,0.0535,0.04,0.018,Unusable (Virtually identical)
# 3. Clipping Ratio,0.0002±0.0001,0.0002±0.0001,0.0000,0.0001,0.00,0.000,Unusable

# ================================================================================
# FEATURE COMPARISON SUMMARY Across Groups (Mean Values), baru
# ================================================================================
# --- 1. Crest Factor (dB) ---
#   LIVE         | Avg:       9.4127 | Std:       1.4532
#   REPLAYSPOOF  | Avg:      10.1023 | Std:       1.0623
#   MIXED        | Avg:      10.3728 | Std:       0.0000

# --- 2. Temp Dynamic Range (dB) ---
#   LIVE         | Avg:      15.5945 | Std:       5.4979
#   REPLAYSPOOF  | Avg:      12.1797 | Std:       5.5669
#   MIXED        | Avg:       9.2961 | Std:       0.0000

# --- 3. Clipping Ratio ---
#   LIVE         | Avg:       0.0002 | Std:       0.0001
#   REPLAYSPOOF  | Avg:       0.0002 | Std:       0.0001
#   MIXED        | Avg:       0.0001 | Std:       0.0000

# --- 4. High-Harmonic Energy Ratio ---
#   LIVE         | Avg:       0.0202 | Std:       0.0549
#   REPLAYSPOOF  | Avg:       0.0221 | Std:       0.0521
#   MIXED        | Avg:       0.0329 | Std:       0.0000

# --- 5. Spectral Rolloff (Hz) ---
#   LIVE         | Avg:    5887.6836 | Std:    1027.3739
#   REPLAYSPOOF  | Avg:    4680.6298 | Std:    1419.0862
#   MIXED        | Avg:    7653.7972 | Std:       0.0000

# --- 6. Spectral Tilt Slope ---
#   LIVE         | Avg:     -23.5334 | Std:       4.8444
#   REPLAYSPOOF  | Avg:     -27.8312 | Std:       6.2918
#   MIXED        | Avg:     -20.0967 | Std:       0.0000

# --- 7. Bispectrum Magnitude ---
#   LIVE         | Avg:       0.1947 | Std:       0.2610
#   REPLAYSPOOF  | Avg:       1.2549 | Std:       1.8981
#   MIXED        | Avg:       0.2555 | Std:       0.0000

def compute_updated_spoof_score(rolloff, bispectrum, tilt, temp_dr):
    """Calculates spoof probability using updated dataset bounds.
    
    Inputs:
        rolloff: Spectral Rolloff in Hz (e.g., 5887.68)
        bispectrum: Bispectrum Magnitude (e.g., 0.1947)
        tilt: Spectral Tilt Slope in dB/decade (e.g., -23.53)
        temp_dr: Temporal Dynamic Range in dB (e.g., 15.59)
    """
    # Standardized Z-scores normalized by updated live baseline and std
    z_rolloff = (5887.68 - rolloff) / 1027.37     # Drop in Hz -> Spoof
    z_bispec  = (bispectrum - 0.1947) / 0.2610    # Rise in non-linear distortion -> Spoof
    z_tilt    = (-23.5334 - tilt) / 4.8444        # Steeper negative slope -> Spoof
    z_tdr     = (15.5945 - temp_dr) / 5.4979      # Lower temporal dynamics -> Spoof

    # Feature weighting based on updated Cohen's d (0.35 Rolloff, 0.35 Bispectrum, 0.20 Tilt, 0.10 DR)
    raw_score = (0.35 * z_rolloff) + (0.35 * z_bispec) + (0.20 * z_tilt) + (0.10 * z_tdr)

    # Convert to [0, 1] probability
    spoof_probability = 1.0 / (1.0 + np.exp(-raw_score))
    return spoof_probability
