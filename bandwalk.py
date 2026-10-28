import os
import numpy as np
import pandas as pd
import librosa

def get_bands(sr):
    nyquist = sr / 2.0
    return {
        "Infrasound_0_20Hz": (0, 20),
        "SubBass_20_50Hz": (20, 50),
        "PowerLine_50_120Hz": (50, 120),
        "CoreVocal_120_1000Hz": (120, 1000),
        "Telephony_1000_3500Hz": (1000, 3500),
        "Audiophile_3500_20000Hz": (3500, min(20000, nyquist)),
        "Ultrasonic_20000HzPlus": (20000, nyquist) if nyquist > 20000 else None
    }

UNITS = {
    "TDR": "dB",
    "CrestFactor": "Ratio",
    "HPR": "Ratio",
    "PSNR": "dB",
    "PeakWander": "Hz"
}

def compute_temporal_dynamic_range(stft_mag):
    frame_power = np.sum(stft_mag ** 2, axis=0)
    max_p = np.max(frame_power)
    min_p = np.min(frame_power[frame_power > 0]) if np.any(frame_power > 0) else 1e-10
    return float(10 * np.log10((max_p + 1e-10) / (min_p + 1e-10)))

def compute_crest_factor(band_signal):
    peak = np.max(np.abs(band_signal))
    rms = np.sqrt(np.mean(band_signal**2)) + 1e-10
    return float(peak / rms)

def compute_harmonic_power_ratio(stft_mag, freqs, f0_min=80, f0_max=300):
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
    frame_power = np.sum(stft_mag ** 2, axis=0)
    p_max = np.max(frame_power)
    p_noise = np.percentile(frame_power, 10) + 1e-10
    return float(10 * np.log10((p_max + 1e-10) / p_noise))

def compute_peak_freq_wander(stft_mag, band_freqs):
    if stft_mag.shape[0] == 0 or stft_mag.shape[1] == 0:
        return 0.0
    peak_bins = np.argmax(stft_mag, axis=0)
    peak_freqs = band_freqs[peak_bins]
    return float(np.std(peak_freqs))

def process_audio_file(filepath):
    signal, sr = librosa.load(filepath, sr=None)
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    stft_mag, _ = librosa.magphase(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    bands = get_bands(sr)
    fname = os.path.basename(filepath)
    
    if fname.startswith("live_"):
        label = "live"
    elif fname.startswith("replayspoof_"):
        label = "replayspoof"
    else:
        label = "mixed"
        
    row = {"Filename": fname, "Group": label, "SampleRate": sr}
    
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
        band_signal = librosa.istft(stft * band_mask[:, None], hop_length=hop_length)
        
        row[f"{band_name}_TDR"] = compute_temporal_dynamic_range(band_stft_mag)
        row[f"{band_name}_CrestFactor"] = compute_crest_factor(band_signal)
        row[f"{band_name}_HPR"] = compute_harmonic_power_ratio(band_stft_mag, band_freqs)
        row[f"{band_name}_PSNR"] = compute_psnr(band_stft_mag)
        row[f"{band_name}_PeakWander"] = compute_peak_freq_wander(band_stft_mag, band_freqs)
        
    return row

def compute_absolute_cohens_d(group_live, group_spoof):
    """Calculates non-negative Cohen's d (|d|)."""
    n1, n2 = len(group_live), len(group_spoof)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group_live, ddof=1), np.var(group_spoof, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return float(abs(np.mean(group_live) - np.mean(group_spoof)) / s_pooled) if s_pooled > 0 else 0.0

def generate_comparison_summary(file_list):
    results = []
    for filepath in file_list:
        if os.path.exists(filepath):
            results.append(process_audio_file(filepath))
            
    df = pd.DataFrame(results)
    
    live_df = df[df["Group"] == "live"]
    spoof_df = df[df["Group"] == "replayspoof"]
    
    feature_cols = [c for c in df.columns if c not in ["Filename", "Group", "SampleRate"]]
    
    summary_data = []
    for col in feature_cols:
        metric_type = col.split("_")[-1]
        unit = UNITS.get(metric_type, "N/A")
        
        live_vals = live_df[col].values
        spoof_vals = spoof_df[col].values
        
        live_mean = np.mean(live_vals)
        live_std = np.std(live_vals, ddof=1)
        
        spoof_mean = np.mean(spoof_vals)
        spoof_std = np.std(spoof_vals, ddof=1)
        
        # Absolute difference and non-negative Cohen's d
        abs_diff = abs(live_mean - spoof_mean)
        abs_d_val = compute_absolute_cohens_d(live_vals, spoof_vals)
        
        summary_data.append({
            "Feature Metric": col,
            "Unit": unit,
            "Live Avg": round(live_mean, 3),
            "Live StdDev": round(live_std, 3),
            "ReplaySpoof Avg": round(spoof_mean, 3),
            "ReplaySpoof StdDev": round(spoof_std, 3),
            "Abs Diff (|Δμ|)": round(abs_diff, 3),
            "Cohen's d (|d|)": round(abs_d_val, 3),
            "Separation Strength": "Strong" if abs_d_val >= 0.8 else ("Moderate" if abs_d_val >= 0.5 else "Weak")
        })
        
    summary_df = pd.DataFrame(summary_data)
    return df, summary_df

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
    raw_df, summary_df = generate_comparison_summary(audio_files)
    
    # Sort strictly by separation power (Cohen's d)
    ranked_summary = summary_df.sort_values(by="Cohen's d (|d|)", ascending=False)
    
    print("\n=========================================================================================================")
    print("              FEATURE COMPARISON SUMMARY WITH ABSOLUTE DIFFERENCE & NON-NEGATIVE COHEN'S d")
    print("=========================================================================================================\n")
    print(ranked_summary.to_string(index=False))
    
    
#  Feature Metric  Unit  Live Avg  Live StdDev  ReplaySpoof Avg  ReplaySpoof StdDev  Abs Diff (|Δμ|)  Cohen's d (|d|) Separation Strength
#          SubBass_20_50Hz_PeakWander    Hz     8.979        2.619            4.735               1.494            4.244            1.856              Strong
#        Audiophile_3500_20000Hz_PSNR    dB    33.683        6.169           22.753               8.838           10.930            1.513              Strong
#          Telephony_1000_3500Hz_PSNR    dB    37.695        8.713           24.493               8.934           13.202            1.502              Strong
#         Audiophile_3500_20000Hz_TDR    dB    41.390        9.565           27.297               9.224           14.093            1.492              Strong
#           Telephony_1000_3500Hz_TDR    dB    52.917       21.133           29.049               8.717           23.867            1.343              Strong
#           CoreVocal_120_1000Hz_PSNR    dB    32.921       10.804           19.940               8.191           12.981            1.303              Strong
#       Infrasound_0_20Hz_CrestFactor Ratio     7.851        4.284           13.764               5.994            5.912            1.193              Strong
#       PowerLine_50_120Hz_PeakWander    Hz    16.149        3.857           12.027               2.990            4.122            1.153              Strong
#            CoreVocal_120_1000Hz_TDR    dB    50.749       30.211           24.750               9.037           25.999            1.045              Strong
#      PowerLine_50_120Hz_CrestFactor Ratio     8.286        4.607            4.489               0.977            3.797            1.012              Strong
#  Ultrasonic_20000HzPlus_CrestFactor Ratio     9.039        5.590           46.709              68.711           37.669            0.917              Strong
#     CoreVocal_120_1000Hz_PeakWander    Hz   196.223       56.279          148.368              45.621           47.856            0.907              Strong
#              PowerLine_50_120Hz_TDR    dB    47.553       28.883           27.278               8.068           20.275            0.855              Strong
#                 SubBass_20_50Hz_TDR    dB    49.802       23.412           33.786               5.188           16.016            0.839              Strong
#         Ultrasonic_20000HzPlus_PSNR    dB     7.001        3.444           11.741               8.662            4.740            0.811              Strong
#             PowerLine_50_120Hz_PSNR    dB    27.385       11.775           19.033               8.069            8.352            0.787            Moderate
#          Ultrasonic_20000HzPlus_TDR    dB     8.873        3.830           12.814               8.530            3.940            0.664            Moderate
#  Audiophile_3500_20000Hz_PeakWander    Hz  2055.413     1932.373         3527.387            3168.234         1471.974            0.603            Moderate
#                SubBass_20_50Hz_PSNR    dB    27.867       11.041           22.284               4.808            5.584            0.598            Moderate
#    Telephony_1000_3500Hz_PeakWander    Hz   451.020      126.585          527.856             161.272           76.837            0.550            Moderate
#            CoreVocal_120_1000Hz_HPR Ratio    21.862       49.518            1.229               1.251           20.633            0.518            Moderate
# Audiophile_3500_20000Hz_CrestFactor Ratio    18.570       16.794           26.740              37.742            8.170            0.312                Weak
#               Infrasound_0_20Hz_TDR    dB    81.984       19.314           77.574               8.785            4.409            0.269                Weak
#   Telephony_1000_3500Hz_CrestFactor Ratio     9.728        7.384           11.167               7.149            1.439            0.197                Weak
#    CoreVocal_120_1000Hz_CrestFactor Ratio     5.729        1.604            5.474               0.833            0.255            0.184                Weak
#              Infrasound_0_20Hz_PSNR    dB    41.747       11.325           43.314               6.393            1.567            0.159                Weak
#         SubBass_20_50Hz_CrestFactor Ratio     8.625        4.441            8.829               4.338            0.203            0.046                Weak
#   Ultrasonic_20000HzPlus_PeakWander    Hz   960.695      468.215          969.772             323.774            9.077            0.021                Weak
#               Infrasound_0_20Hz_HPR Ratio     0.000        0.000            0.000               0.000            0.000            0.000                Weak
#        Infrasound_0_20Hz_PeakWander    Hz     0.000        0.000            0.000               0.000            0.000            0.000                Weak
#                 SubBass_20_50Hz_HPR Ratio     0.000        0.000            0.000               0.000            0.000            0.000                Weak
#              PowerLine_50_120Hz_HPR Ratio     0.000        0.000            0.000               0.000            0.000            0.000                Weak
#           Telephony_1000_3500Hz_HPR Ratio     0.000        0.000            0.000               0.000            0.000            0.000                Weak
#         Audiophile_3500_20000Hz_HPR Ratio     0.000        0.000            0.000               0.000            0.000            0.000                Weak
#          Ultrasonic_20000HzPlus_HPR Ratio     0.000        0.000            0.000               0.000            0.000            0.000                Weak