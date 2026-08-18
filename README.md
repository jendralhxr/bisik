**bisik** ("whisper" in Indonesian) is a lightweight, explainable sound/voice **liveness detection** toolkit written in Python. It analyzes a short audio clip and estimates the likelihood that it came from a **live human voice** versus a **replayed/spoofed recording** (e.g. played back through a speaker and re-captured by a microphone), using purely signal-level acoustic features — no machine learning model or training data required.

## How it works

Live speech and replayed/re-recorded speech differ in measurable ways because of the physical process of transduction (speaker playback + microphone capture), lossy compression, and automatic gain control. `bisik` extracts six acoustic features that capture these differences, normalizes each into a `0.0` (live) → `1.0` (spoof) sub-score, and combines them into a single weighted **spoof likelihood** score.

| Feature | What it measures | Why it indicates spoofing |
|---|---|---|
| **Crest factor** | Peak-to-RMS energy ratio per frame | Live voice has sharp, uncompressed transients; playback/DRC flattens peaks |
| **Spectral roll-off & tilt** | Frequency below which 95% of spectral energy lies, and the slope of the power spectrum | Speaker/mic band-limiting and anti-aliasing filters attenuate high frequencies |
| **Bispectrum distortion** | Third-order phase coherence across frequency triplets | Non-linear transducer artifacts introduced by speaker coils leave a phase-coupling signature |
| **Harmonic dynamics** | Ratio of energy in high-order harmonic bands (3.4–8 kHz) vs. total | Replay/lossy compression drops upper harmonic content |
| **Temporal dynamic range** | Local (95th–5th percentile) dB range across sliding windows | Live speech has large envelope variance between silence, plosives, and vowels; compressed/replayed audio is "squashed" |
| **Peak clipping** | Ratio of clipped samples and length of flat-topped waveform runs | Replay pipelines often clip due to speaker/mic gain and re-amplification |

Each feature's raw value is mapped to a sub-score in `[0, 1]` using empirically chosen thresholds, then combined with configurable weights (defaulting to values that sum to 1.0) into an overall `spoof_likelihood` score. A score ≥ 0.50 is classified as `SPOOF`, otherwise `LIVE`.

## Repository contents

| File | Description |
|---|---|
| [`features.py`](features.py) | Standalone feature-extraction functions (crest factor, temporal dynamic range, peak clipping, harmonic band energy, spectral tilt/roll-off, bispectrum magnitude), each documented with the acoustic rationale behind it. Useful as a reference or for building custom scoring pipelines. |
| [`detectliveness.py`](detectliveness.py) | The main `AcousticLivenessScorer` class, which ties the features together, normalizes them into sub-scores, applies weights, and produces a final liveness decision. Includes a runnable example (`__main__`). |

## Requirements

- Python 3.9+
- [`librosa`](https://librosa.org/)
- [`numpy`](https://numpy.org/)

Install dependencies with:

```bash
pip install librosa numpy
```

## Usage

```python
from detectliveness import AcousticLivenessScorer, load_wav

# Load a WAV file (resampled to 22.05 kHz mono by default)
y, sr = load_wav("test_sample.wav")

# Initialize the scorer (optionally pass custom feature weights)
scorer = AcousticLivenessScorer(sample_rate=sr)

# Run liveness detection
result = scorer.predict_liveness(y)

print(f"Overall Spoof Likelihood: {result['spoof_likelihood'] * 100:.2f}%")
print(f"Decision: {result['decision']}")

print("\nIndividual Sub-Scores (0 = Live, 1 = Spoof):")
for name, score in result['subscores'].items():
    print(f" - {name}: {score}")
```

Or simply run the script directly on a sample file named `test_sample.wav` in the working directory:

```bash
python detectliveness.py
```

### Customizing weights

The six feature weights can be overridden (they are automatically re-normalized to sum to 1.0):

```python
custom_weights = {
    'crest_factor': 0.25,
    'spectral_rolloff_tilt': 0.10,
    'bispectrum_distortion': 0.20,
    'harmonic_dynamics': 0.15,
    'temporal_dynamic_range': 0.20,
    'peak_clipping': 0.10,
}

scorer = AcousticLivenessScorer(weights=custom_weights, sample_rate=sr)
```

### Output format

`predict_liveness()` returns a dictionary:

```python
{
    'spoof_likelihood': 0.34,      # float in [0.0, 1.0], 0 = live, 1 = spoof
    'decision': 'LIVE',            # 'LIVE' or 'SPOOF' (threshold at 0.50)
    'subscores': {...},            # per-feature spoof sub-scores [0.0, 1.0]
    'applied_weights': {...},      # normalized weights actually used
    'raw_features': {...},         # raw (unnormalized) extracted acoustic values
}
```

## Notes & limitations

- This is a **signal-processing / heuristic** approach, not a trained statistical or deep-learning spoof-detection model — thresholds are hand-tuned based on general acoustic principles and may need adjustment for a given microphone, recording chain, or use case.
- The bispectrum computation iterates over frequency-triplet pairs and can be relatively slow for long clips or high `n_fft` values.
- Designed for research, prototyping, and experimentation rather than production-grade anti-spoofing.

## License

No license file is currently included in this repository. Contact the repository owner ([@jendralhxr](https://github.com/jendralhxr)) regarding usage terms.
