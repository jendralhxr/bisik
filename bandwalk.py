import os
import glob
import numpy as np
import pandas as pd
import librosa

# Define frequency band boundaries (in Hz)
BANDS = {
    "Low": (0, 1000),
    "Mid": (1000, 4000),
    "High": (4000, 8000),
    "Super": (8000, 16000)
}

def compute_temporal_dynamic_range(stft_mag):
    """Ratio of max frame power to quiet frame power (in dB)."""
    frame_power = np.sum(stft_mag ** 2, axis=0)
    max_p = np.max(frame_power)
    min_p = np.min(frame_power[frame_power > 0]) if np.any(frame_power > 0) else 1e-10
    dr_db = 10 * np.log10((max_p + 1e-10) / (min_p + 1e-10))
    return float(dr_db)

def compute_crest_factor(band_signal):
    """Ratio of peak amplitude to RMS amplitude."""
    peak = np.max(np.abs(band_signal))
    rms = np.sqrt(np.mean(band_signal**2)) + 1e-10
    return float(peak / rms)

def compute_harmonic_power_ratio(stft_mag, freqs, f0_min=80, f0_max=300):
    """
    Estimates fundamental frequency (f0) from average spectrum
    and computes energy ratio of upper harmonics (2f0..K*f0) to f0.
    """
    avg_spectrum = np.mean(stft_mag, axis=1)
    f0_idx_search = np.where((freqs >= f0_min) & (freqs <= f0_max))[0]
    
    if len(f0_idx_search) == 0 or np.sum(avg_spectrum) == 0:
        return 0.0
    
    f0_bin = f0_idx_search[np.argmax(avg_spectrum[f0_idx_search])]
    f0 = freqs[f0_bin]
    
    if f0 <= 0:
        return 0.0
    
    # Power at Fundamental (f0)
    p_f0 = avg_spectrum[f0_bin] ** 2
    
    # Power at Harmonics (2*f0, 3*f0, ...)
    p_harmonics = 0.0
    max_k = int(freqs[-1] // f0)
    for k in range(2, max_k + 1):
        target_f = k * f0
        k_bin = np.argmin(np.abs(freqs - target_f))
        p_harmonics += avg_spectrum[k_bin] ** 2
        
    return float(p_harmonics / (p_f0 + 1e-10))

def compute_psnr(stft_mag):
    """Ratio of max signal power to estimated noise floor (bottom 10th percentile)."""
    frame_power = np.sum(stft_mag ** 2, axis=0)
    p_max = np.max(frame_power)
    p_noise = np.percentile(frame_power, 10) + 1e-10
    psnr_db = 10 * np.log10((p_max + 1e-10) / p_noise)
    return float(psnr_db)

def process_audio_file(filepath):
    """Loads a WAV file and computes sub-band acoustic metrics."""
    signal, sr = librosa.load(filepath, sr=None)
    
    # STFT Computation
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    stft_mag, _ = librosa.magphase(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    row = {"Filename": os.path.basename(filepath)}
    
    for band_name, (f_min, f_max) in BANDS.items():
        # Mask bins belonging to current band
        band_mask = (freqs >= f_min) & (freqs < min(f_max, sr / 2))
        if not np.any(band_mask):
            continue
            
        band_stft_mag = stft_mag[band_mask, :]
        band_freqs = freqs[band_mask]
        
        # Bandpass time-domain reconstruction for Crest Factor
        band_signal = librosa.istft(stft * band_mask[:, None], hop_length=hop_length)
        
        # Calculate metrics
        row[f"{band_name}_TDR_dB"] = compute_temporal_dynamic_range(band_stft_mag)
        row[f"{band_name}_CrestFactor"] = compute_crest_factor(band_signal)
        row[f"{band_name}_HPR"] = compute_harmonic_power_ratio(band_stft_mag, band_freqs)
        row[f"{band_name}_PSNR_dB"] = compute_psnr(band_stft_mag)
        
    return row

def batch_analyze_directory(file_list):
    results = []
    for filepath in file_list:
        if os.path.exists(filepath):
            print(f"Processing: {filepath}")
            metrics = process_audio_file(filepath)
            results.append(metrics)
        else:
            print(f"File not found: {filepath}")
            
    df = pd.DataFrame(results)
    return df

# List of files provided in your directory
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
    df = batch_analyze_directory(audio_files)
    
    # Save extracted metrics matrix
    df.to_csv("spoof_detection_metrics.csv", index=False)
    print("\nExtraction complete! Sample results:")
    print(df.head())