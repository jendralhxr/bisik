import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def plot_audio_features(wav_path):
    # 1. Load the WAV file
    y, sr = librosa.load(wav_path, sr=None)
    duration = len(y) / sr

    # Create figure with 7 subplots stacked vertically
    fig, axes = plt.subplots(7, 1, figsize=(14, 20), sharex=False)
    plt.subplots_adjust(hspace=0.5)

    # ---------------------------------------------------------
    # 1. Raw Waveform
    # ---------------------------------------------------------
    times = np.linspace(0, duration, num=len(y))
    axes[0].plot(times, y, color="b", alpha=0.7)
    axes[0].set_title("1. Soundwave (Raw Audio Time Domain)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # ---------------------------------------------------------
    # 2. Linear Spectrogram (STFT)
    # ---------------------------------------------------------
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img2 = librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="linear", ax=axes[1], cmap="magma"
    )
    axes[1].set_title("2. Spectrogram (Linear Frequency Scale)", fontsize=11, fontweight="bold")
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    # ---------------------------------------------------------
    # 3. Constant-Q Transform (Log-Frequency Spectrogram)
    # ---------------------------------------------------------
    C = librosa.cqt(y, sr=sr)
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    img3 = librosa.display.specshow(
        C_db, sr=sr, x_axis="time", y_axis="cqt_hz", ax=axes[2], cmap="magma"
    )
    axes[2].set_title("3. Constant-Q Transform (Log-Frequency Spectrogram)", fontsize=11, fontweight="bold")
    fig.colorbar(img3, ax=axes[2], format="%+2.0f dB")

    # ---------------------------------------------------------
    # 4. Cepstrum (Real Cepstrum)
    # ---------------------------------------------------------
    # Real cepstrum = IFFT(log(|FFT(y)|))
    spectrum = np.fft.rfft(y)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum)
    
    # Plot cepstrum vs. Quefrency (seconds)
    quefrency = np.arange(len(cepstrum)) / sr
    # Zoom in to low quefrencies (e.g., up to 50ms) to highlight pitch/formant structures
    max_q_idx = int(sr * 0.05)
    axes[3].plot(quefrency[:max_q_idx] * 1000, cepstrum[:max_q_idx], color="purple")
    axes[3].set_title("4. Real Cepstrum (Time Domain / Quefrency Domain)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Quefrency (ms)")
    axes[3].set_ylabel("Log Magnitude")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    # ---------------------------------------------------------
    # 5. Mel-Frequency Cepstral Coefficients (MFCCs)
    # ---------------------------------------------------------
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img5 = librosa.display.specshow(
        mfccs, sr=sr, x_axis="time", ax=axes[4], cmap="coolwarm"
    )
    axes[4].set_title("5. Mel-Frequency Cepstral Coefficients (MFCCs)", fontsize=11, fontweight="bold")
    axes[4].set_ylabel("MFCC Index")
    fig.colorbar(img5, ax=axes[4])

    # ---------------------------------------------------------
    # 6. Delta Coefficients
    # ---------------------------------------------------------
    delta_mfcc = librosa.feature.delta(mfccs)
    img6 = librosa.display.specshow(
        delta_mfcc, sr=sr, x_axis="time", ax=axes[5], cmap="coolwarm"
    )
    axes[5].set_title("6. Delta Coefficients (First Derivative of MFCCs)", fontsize=11, fontweight="bold")
    axes[5].set_ylabel("MFCC Index")
    fig.colorbar(img6, ax=axes[5])

    # ---------------------------------------------------------
    # 7. Delta-Delta Coefficients
    # ---------------------------------------------------------
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    img7 = librosa.display.specshow(
        delta2_mfcc, sr=sr, x_axis="time", ax=axes[6], cmap="coolwarm"
    )
    axes[6].set_title("7. Delta-Delta Coefficients (Second Derivative / Acceleration)", fontsize=11, fontweight="bold")
    axes[6].set_ylabel("MFCC Index")
    fig.colorbar(img7, ax=axes[6])

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Example usage:
# Replace 'your_audio.wav' with the path to your WAV file
if __name__ == "__main__":
    # Create a dummy audio signal if no file is provided
    example_file = "your_audio.wav"
    
    try:
        plot_audio_features(example_file)
    except FileNotFoundError:
        print(f"File '{example_file}' not found. Generating a synthetic audio signal for demonstration...")
        sr = 24000
        t = np.linspace(0, 2, 2 * sr)
        # Chirp signal sweeping from 200 Hz to 2000 Hz
        y = np.sin(2 * np.pi * (200 + 900 * t) * t)
        import soundfile as sf
        sf.write("synthetic_demo.wav", y, sr)
        plot_audio_features("synthetic_demo.wav")
