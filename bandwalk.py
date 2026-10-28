import os
import numpy as np
import pandas as pd
import librosa

def get_bands(sr):
    """Dynamically generates frequency bands based on input sample rate."""
    nyquist = sr / 2.0
    return {
        "Infrasound": (0, 20),
        "SubBass": (20, 50),
        "PowerLine": (50, 120),
        "CoreVocal": (120, 1000),
        "Telephony_DTMF": (1000, 3500),
        "AudiophileHigh": (3500, min(20000, nyquist)),
        "Ultrasonic_Nyquist": (20000, nyquist) if nyquist > 20000 else None
    }

def compute_temporal_dynamic_range(stft_mag):
    """Ratio of maximum frame power to minimum frame power (dB)."""
    frame_power = np.sum(stft_mag ** 2, axis=0)
    max_p = np.max(frame_power)
    min_p = np.min(frame_power[frame_power > 0]) if np.any(frame_power > 0) else 1e-10
    return float(10 * np.log10((max_p + 1e-10) / (min_p + 1e-10)))

def compute_crest_factor(band_signal):
    """Ratio of peak absolute amplitude to RMS amplitude."""
    peak = np.max(np.abs(band_signal))
    rms = np.sqrt(np.mean(band_signal**2)) + 1e-10
    return float(peak / rms)

def compute_harmonic_power_ratio(stft_mag, freqs, f0_min=80, f0_max=300):
    """Energy ratio of upper harmonics relative to fundamental power."""
    avg_spectrum = np.mean(stft_mag, axis=1)
    f0_idx_search = np.where((freqs >= f0_min) & (freqs <= f0_max))[0]
    
    if len(f0_idx_search) == 0 or np.sum(avg_spectrum) == 0:
        return 0.0
    
    f0_bin = f0_idx_search[np.argmax(avg_spectrum[f0_idx_search])]
    f0 = freqs[f0_bin]
    
    if f0 <= 0:
        return 0.0
    
    p_f0 = avg_spectrum[f0_bin] ** 2
    p_harmonics = 0.0
    max_k = int(freqs[-1] // f0) if f0 > 0 else 0
    
    for k in range(2, max_k + 1):
        target_f = k * f0
        k_bin = np.argmin(np.abs(freqs - target_f))
        p_harmonics += avg_spectrum[k_bin] ** 2
        
    return float(p_harmonics / (p_f0 + 1e-10))

def compute_psnr(stft_mag):
    """Signal-to-noise ratio above estimated noise floor (bottom 10th percentile)."""
    frame_power = np.sum(stft_mag ** 2, axis=0)
    p_max = np.max(frame_power)
    p_noise = np.percentile(frame_power, 10) + 1e-10
    return float(10 * np.log10((p_max + 1e-10) / p_noise))

def compute_peak_freq_wander(stft_mag, band_freqs):
    """Standard deviation of the peak frequency trajectory across time frames (Hz)."""
    if stft_mag.shape[0] == 0 or stft_mag.shape[1] == 0:
        return 0.0
    peak_bins = np.argmax(stft_mag, axis=0)
    peak_freqs = band_freqs[peak_bins]
    return float(np.std(peak_freqs))

def process_audio_file(filepath):
    """Processes audio and calculates metrics for each customized band."""
    signal, sr = librosa.load(filepath, sr=None)
    
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    stft_mag, _ = librosa.magphase(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    bands = get_bands(sr)
    fname = os.path.basename(filepath)
    label = "live" if fname.startswith("live_") else ("spoof" if fname.startswith("replayspoof_") else "mixed")
    row = {"Filename": fname, "Label": label, "SampleRate": sr}
    
    for band_name, band_range in bands.items():
        if band_range is None:
            continue
            
        f_min, f_max = band_range
        band_mask = (freqs >= f_min) & (freqs < min(f_max, sr / 2))
        
        if not np.any(band_mask) or f_min >= (sr / 2):
            row[f"{band_name}_TDR"] = 0.0
            row[f"{band_name}_CrestFactor"] = 0.0
            row[f"{band_name}_HPR"] = 0.0
            row[f"{band_name}_PSNR"] = 0.0
            row[f"{band_name}_PeakWander"] = 0.0
            continue
            
        band_stft_mag = stft_mag[band_mask, :]
        band_freqs = freqs[band_mask]
        
        # Reconstruct bandpass signal for time-domain Crest Factor
        band_signal = librosa.istft(stft * band_mask[:, None], hop_length=hop_length)
        
        row[f"{band_name}_TDR"] = compute_temporal_dynamic_range(band_stft_mag)
        row[f"{band_name}_CrestFactor"] = compute_crest_factor(band_signal)
        row[f"{band_name}_HPR"] = compute_harmonic_power_ratio(band_stft_mag, band_freqs)
        row[f"{band_name}_PSNR"] = compute_psnr(band_stft_mag)
        row[f"{band_name}_PeakWander"] = compute_peak_freq_wander(band_stft_mag, band_freqs)
        
    return row

def compute_cohens_d(group_live, group_spoof):
    """Calculates Cohen's d effect size between live and spoof groups."""
    n1, n2 = len(group_live), len(group_spoof)
    if n1 < 2 or n2 < 2:
        return 0.0
        
    var1, var2 = np.var(group_live, ddof=1), np.var(group_spoof, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if s_pooled == 0:
        return 0.0
        
    return float((np.mean(group_live) - np.mean(group_spoof)) / s_pooled)

def analyze_and_evaluate(file_list):
    results = []
    for filepath in file_list:
        if os.path.exists(filepath):
            metrics = process_audio_file(filepath)
            results.append(metrics)
            
    df = pd.DataFrame(results)
    
    live_df = df[df["Label"] == "live"]
    spoof_df = df[df["Label"] == "spoof"]
    
    feature_cols = [c for c in df.columns if c not in ["Filename", "Label", "SampleRate"]]
    cohen_results = []
    
    for col in feature_cols:
        d_val = compute_cohens_d(live_df[col].values, spoof_df[col].values)
        cohen_results.append({
            "Feature Metric": col,
            "Mean (Live)": np.mean(live_df[col]),
            "Mean (Spoof)": np.mean(spoof_df[col]),
            "Cohen's d": d_val,
            "Separation Quality": "Strong" if abs(d_val) >= 0.8 else ("Moderate" if abs(d_val) >= 0.5 else "Weak")
        })
        
    cohen_df = pd.DataFrame(cohen_results)
    return df, cohen_df

# Input files
audio_files = [
    "live_andhika.wav", "live_whistle.wav", "live_bayinangis.wav", 
    "live_berawaldaritatap.wav", "mixed_sitianosora.wav", "live_halotest.wav", 
    "replayspoof_alhudacandisari.wav", "live_harmonika.wav", "replayspoof_atasnama.wav", 
    "live_kaumembuatku.wav", "replayspoof_ceklagi.wav", "live_micmaono.wav", 
    "replayspoof_charlie.wav", "live_senyumwawawa.wav", "replayspoof_srikandi.wav", 
    "live_suaraasli.wav", "replayspoof_tanahair.wav", "live_terbawa.wav", 
    "replayspoof_verifikasi.wav", "live_terkadangkurangu.wav"
]

if __name__ == "__main__":
    metrics_df, cohen_df = analyze_and_evaluate(audio_files)
    
    # Save full feature dataset
    metrics_df.to_csv("spoof_detection_custom_bands.csv", index=False)
    
    # Display top separated features sorted by absolute Cohen's d
    cohen_df["Abs_d"] = cohen_df["Cohen's d"].abs()
    ranked_cohen = cohen_df.sort_values(by="Abs_d", ascending=False).drop(columns=["Abs_d"])
    
    print("\n--- Top Features Ranked by Cohen's d Separation Quality ---")
    print(ranked_cohen.head(12).to_string(index=False))