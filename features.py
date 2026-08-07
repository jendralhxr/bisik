import librosa
import numpy as np

# AMPLITUDE

# Dynamic Range Compression: Crest Factor (PAPR), berhaps on the sub-vocal pop/breath vibrations (<80 Hz)
# Measures short-time peak-to-average power ratios. Live voice exhibits sharp, uncompressed acoustic transients, whereas speaker playback or lossy dynamic range compression flattens these peaks.
def extract_crest_factor(y, frame_length=2048, hop_length=512):
    """Computes frame-level Crest Factor (Peak-to-RMS ratio in dB)."""
    # Frame the signal into overlapping windows
    frames = librosa.util.frame(
        y, frame_length=frame_length, hop_length=hop_length
    )

    # Peak amplitude per frame
    peaks = np.max(np.abs(frames), axis=0)

    # Root Mean Square (RMS) energy per frame
    rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-8)

    # Crest Factor (in dB)
    crest_factor_db = 20 * np.log10((peaks / rms) + 1e-8)

    # Return summary statistics across all frames
    return np.mean(crest_factor_db), np.std(crest_factor_db)

# temporal dynamic range (envelope modulation)
# local peak-to-floor ratios within sliding sub-windows (e.g., 500ms). 
# Uncompressed live voice fluctuates heavily between silence, plosive bursts, and sustained vowels
# producing high temporal variance in local dynamic range

def extract_temporal_dynamic_range(y, sr=22050, window_sec=0.5, hop_sec=0.25):
    """Computes local Dynamic Range (95th vs 5th percentile dB) across sliding time windows.

    Differentiates natural time-varying vocal envelope dynamics from static
    hardware/software compression artifacts.
    """
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    # Split audio into sliding time windows
    windows = librosa.util.frame(y, frame_length=window_len, hop_length=hop_len)

    local_drs = []

    for idx in range(windows.shape[1]):
        win = windows[:, idx]

        # Short-time RMS energy envelope within the window
        rms = librosa.feature.rms(y=win, frame_length=512, hop_length=128)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Local Dynamic Range = 95th percentile (peak) - 5th percentile (floor)
        # Using percentiles prevents isolated single-sample noise outliers from skewing the metric
        dr_win = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)
        local_drs.append(dr_win)

    local_drs = np.array(local_drs)

    # Return mean local dynamic range and temporal variance of the local dynamic range
    return np.mean(local_drs), np.std(local_drs)

# peak clipping
# Replay attacks and re-recording pipelines frequently introduce clipping when played back through external speakers at high volumes or captured by microphones with gain boost.
# live vocal tracks recorded at appropriate gains feature smooth, rounded peak envelopes, clipped audio exhibits flat-topped waveform regions at amplitude ceilings
def extract_peak_clipping_metrics(y, threshold=0.98):
    """Detects hard and soft peak clipping artifacts caused by playback amplifier

    saturation or transducer overload.
    """
    abs_y = np.abs(y)
    max_val = np.max(abs_y) + 1e-8

    # Normalize scale to [0, 1] range if needed
    norm_y = abs_y / max_val

    # 1. Hard Clipping Sample Ratio (Proportion of samples sitting at upper limits)
    clipped_mask = norm_y >= threshold
    clipping_ratio = np.mean(clipped_mask)

    # 2. Consecutive Flat-Top Peak Lengths
    # Measures the duration of flat waveform peaks sitting at peak thresholds
    diffs = np.diff(y)
    flat_mask = (np.abs(diffs) < 1e-4) & (norm_y[:-1] >= threshold)

    if np.any(flat_mask):
        # Find continuous runs of zero-gradient peak samples
        run_starts = np.where(np.diff(np.pad(flat_mask.astype(int), (1, 1))))[0]
        run_lengths = run_starts[1::2] - run_starts[::2]
        max_flat_run = np.max(run_lengths)
        mean_flat_run = np.mean(run_lengths)
    else:
        max_flat_run = 0
        mean_flat_run = 0.0

    # 3. High-Amplitude Variance (Peak Distribution Saturation)
    # Natural live voice has high variance near peaks; clipped audio collapses peak variance to zero
    top_peaks = norm_y[norm_y >= np.percentile(norm_y, 95)]
    top_peak_variance = np.var(top_peaks)

    return clipping_ratio, mean_flat_run, max_flat_run, top_peak_variance


#### FREQUENCY

# temporal frequency bands
# Live speech exhibits continuous $f_0$ movement and dynamic energy transfer
# between fundamental frequencies and higher-order vocal formants over time windows. 
# Replayed or lossy-compressed audio exhibits flattened or missing upper harmonics 
# due to anti-aliasing filters and non-linear transducer rolloffs across sliding windows.
def extract_temporal_harmonic_bands(y, sr=22050, window_sec=1.0, hop_sec=0.5):
    """Tracks harmonic power ratios across specific frequency bands over sliding temporal windows.

    Identifies how energy shifts between fundamental vocal frequencies (f0) and
    upper harmonic structures across time.
    """
    # 1. Separate harmonic component from percussive background
    y_harmonic, _ = librosa.effects.hpss(y)

    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    # Frame harmonic signal into sliding time windows
    windows = librosa.util.frame(
        y_harmonic, frame_length=window_len, hop_length=hop_len
    )

    band_ratios = []

    for idx in range(windows.shape[1]):
        win = windows[:, idx]
        S = np.abs(librosa.stft(win, n_fft=2048)) ** 2
        freqs = librosa.fft_frequencies(sr=sr)

        # Define key vocal frequency bands
        f0_mask = (freqs >= 50) & (freqs < 300)  # Fundamental voice range
        mid_harm_mask = (
            freqs >= 300
        ) & (
            freqs < 3400
        )  # Primary formants & low harmonics (telecom band)
        high_harm_mask = (freqs >= 3400) & (
            freqs < 8000
        )  # High-order harmonics (lossy/speaker cutoffs)

        e_f0 = np.sum(S[f0_mask, :]) + 1e-8
        e_mid = np.sum(S[mid_harm_mask, :]) + 1e-8
        e_high = np.sum(S[high_harm_mask, :]) + 1e-8
        e_total = np.sum(S) + 1e-8

        # Energy distribution ratios relative to total harmonic power
        band_ratios.append([e_f0 / e_total, e_mid / e_total, e_high / e_total])

    band_ratios = np.array(band_ratios)

    # Return mean band energy ratios and temporal volatility (std dev) across windows
    mean_ratios = np.mean(band_ratios, axis=0)
    temporal_variance = np.std(band_ratios, axis=0)

    return mean_ratios, temporal_variance


# high frequency voice compression (>12–16 kHz)
# High-Frequency Compression: Spectral Roll-off & Tilt
# Detects upper frequency attenuation and band-limiting caused by anti-aliasing filters or speaker transducer cutoffs.
def extract_spectral_tilt_and_rolloff(y, sr=22050):
    """Computes 95% spectral roll-off frequency and log-spectral tilt slope."""
    # 1. Spectral Roll-off (frequency below which 95% of spectral energy lies)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[
        0
    ]  #
    mean_rolloff = np.mean(rolloff)

    # 2. Spectral Tilt (slope of power spectrum)
    S = np.abs(librosa.stft(y)) ** 2
    avg_power_spectrum = np.mean(S, axis=1)

    freqs = librosa.fft_frequencies(sr=sr)
    valid_idx = freqs > 0  # Avoid log(0)

    log_freqs = np.log10(freqs[valid_idx])
    log_power = 10 * np.log10(avg_power_spectrum[valid_idx] + 1e-8)

    # Linear fit to estimate high-frequency attenuation slope
    tilt_slope, _ = np.polyfit(log_freqs, log_power, 1)

    return mean_rolloff, tilt_slope


# transducer distortion
# the speaker coils do speak themselves
# Non-Linear Transducer Distortion: Bispectrum Analysis
# Computes third-order phase coherence $B(f_1, f_2) = X(f_1) X(f_2) X^*(f_1 + f_2)$ to isolate phase alignment and harmonic distortions introduced when audio drives physical speaker hardware.
def extract_bispectrum_magnitude(y, n_fft=512):
    """Estimates mean Bispectral magnitude across valid frequency triplets."""
    stft = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2)
    num_bins, _ = stft.shape

    bispec_sum = 0.0
    valid_triplets = 0

    # Iterate over frequency pairs where f1 + f2 < Nyquist limit
    max_bin = num_bins // 2
    for f1 in range(1, max_bin):
        for f2 in range(1, max_bin - f1):
            f3 = f1 + f2
            # B(f1, f2) = X(f1) * X(f2) * conj(X(f1 + f2))
            b_val = stft[f1, :] * stft[f2, :] * np.conj(stft[f3, :])
            bispec_sum += np.mean(np.abs(b_val))
            valid_triplets += 1

    return bispec_sum / (valid_triplets + 1e-8)


#### SUMMARY, one index to rule them all

