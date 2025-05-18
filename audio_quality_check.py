#!/usr/bin/env python3
"""
quickcheck_audio_quality.py – Very small proof‑of‑concept fake‑upsample detector
Author: you 😊

*Fixed version*
- Replaced the non‑existent ``librosa.iirfilter`` / ``librosa.sosfilt`` calls with their
  equivalents from ``scipy.signal``.
- Added explicit imports from ``scipy.signal``.
- Added safe JSON serialisation for NumPy scalars (they become plain ``float``/``int``)
  so ``json.dumps`` no longer crashes with ``TypeError: Object of type float32 …``.
- No other behavioural changes.
"""

import sys
import json
from pathlib import Path

import numpy as np
import librosa          # audio loading / resampling
import soundfile as sf   # to check original bit‑depth
from scipy.stats import pearsonr
from scipy.signal import iirfilter, sosfilt  # equivalents of the missing librosa wrappers
from tqdm import tqdm    # progress bar for long files

# ---------------------------- CONFIG -------------------------------- #
FRAME_LEN     = 4096        # STFT window
HOP_LEN       = 2048
DB_THRESHOLD  = -80         # minimum level to consider “content”
HF_BAND_START = 18000       # Hz – start of the band we call “HF”
CORR_BAND     = 15000       # Hz – threshold for stereo correlation calc
ROLL_OFF_SLOPE_THRESHOLD = 50  # dB per kHz considered “brick‑wall”
# -------------------------------------------------------------------- #


# -------------------------------------------------------------------- #
# Helper – convert NumPy scalars/arrays to JSON‑serialisable Python types
# -------------------------------------------------------------------- #

def _json_safe(x):
    """Return *x* converted to a JSON‑serialisable type if necessary."""
    if isinstance(x, (np.generic,)):      # NumPy scalar types
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def load_audio(path, target_sr=None, mono=False):
    """Load audio file with librosa preserving native sr unless target_sr set."""
    y, sr = librosa.load(path, sr=target_sr, mono=mono)
    return y, sr


def average_spectrum(y, sr):
    """Return mean magnitude spectrum (in dB) and frequency axis."""
    stft = librosa.stft(y, n_fft=FRAME_LEN, hop_length=HOP_LEN, window='hann')
    mag = np.abs(stft)
    mean_mag = np.mean(mag, axis=1)  # average over time
    mean_db = librosa.amplitude_to_db(mean_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LEN)
    return freqs, mean_db


def find_cutoff(freqs, mean_db, db_threshold=DB_THRESHOLD):
    """Estimate highest freq where level > threshold and roll‑off slope."""
    # find first index from Nyquist downward where energy > threshold
    valid = np.where(mean_db > db_threshold)[0]
    if valid.size == 0:
        return 0.0, float('inf')
    idx_cut = valid[-1]
    f_cut = float(freqs[idx_cut])

    # simple slope estimate: difference of 1 kHz band around cutoff
    idx_band = np.where((freqs >= f_cut-1000) & (freqs <= f_cut+1000))[0]
    if idx_band.size < 2:
        slope = 0.0
    else:
        # linear fit
        x = freqs[idx_band]
        y = mean_db[idx_band]
        p = np.polyfit(x, y, 1)  # slope in dB/Hz
        slope = float(-p[0] * 1000)     # convert to dB per kHz (positive value)
    return f_cut, slope


def hf_noise_ratio(freqs, mean_db, hf_start=HF_BAND_START):
    """Return average HF noise level minus mid‑band level (dB)."""
    # mean in 1‑5 kHz band
    mid_band = np.where((freqs >= 1000) & (freqs <= 5000))[0]
    hf_band  = np.where(freqs >= hf_start)[0]
    if hf_band.size == 0:
        return float('inf')  # no HF content at all (very low sr)
    mid_val = np.mean(mean_db[mid_band])
    hf_val  = np.mean(mean_db[hf_band])
    return float(hf_val - mid_val)      # negative means HF quieter


def stereo_hf_correlation(y, sr, band_start=CORR_BAND):
    """Correlation of L & R in HF band (1.0 ≈ identical)."""
    if y.ndim == 1:
        return None  # mono source
    # design simple Butterworth high‑pass filter (stable SOS)
    sos = iirfilter(6, band_start, btype='high', ftype='butter', fs=sr, output='sos')
    yL = sosfilt(sos, y[0])
    yR = sosfilt(sos, y[1])
    # take short random subset to speed up
    n = min(10*sr, yL.size)
    idx = np.random.choice(yL.size, n, replace=False)
    corr, _ = pearsonr(yL[idx], yR[idx])
    return float(corr)


def bit_depth_analysis(path):
    """For 24‑bit files, estimate if lower 8 bits carry info."""
    try:
        info = sf.info(path)
        if info.subtype in ("PCM_24", "PCM_32"):
            with sf.SoundFile(path) as f:
                block = f.read(frames=min(100000, len(f)), dtype='int32')
            # examine LSB of 24‑bit data
            lsb = np.abs(block) & 0xFF
            zeros = np.mean(lsb == 0)
            return {"bitdepth": info.subtype, "zero_lsb_fraction": float(zeros)}
    except Exception:
        pass
    return {}


def classify(metrics):
    """Rule‑based decision with estimated bitrate based on cutoff frequency."""
    cut     = metrics["cutoff_hz"]
    slope   = metrics["rolloff_db_per_khz"]
    hf_dif  = metrics["hf_minus_mid_db"]
    corr    = metrics["hf_stereo_corr"]
    # ROLL_OFF_SLOPE_THRESHOLD is a global constant

    base_classification = ""
    cutoff_kHz = cut / 1000.0

    # Determine base classification from cutoff frequency table
    if cut == 0.0 and slope == float('inf'): # Handle case from find_cutoff where no valid energy found
        base_classification = "🛑 SILENCE OR NO DETECTABLE AUDIO CONTENT"
    elif cutoff_kHz < 11.5: # ~11 kHz
        base_classification = "🟥🟥🟥🟥🟥 ESTIMATED BITRATE: Very low (e.g. 64 kbps MP3). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz < 17.0: # ~16 kHz (11.5kHz to <17kHz)
        base_classification = "🚫🚫🚫🚫 ESTIMATED BITRATE: Around 128 kbps. Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz < 19.5: # ~18-19 kHz (17kHz to <19.5kHz)
        base_classification = "⚠️⚠️⚠️ ESTIMATED BITRATE: Likely mid-high (e.g. 192–256 kbps). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz < 21.0: # ~20 kHz (19.5kHz to <21kHz)
        base_classification = "👍👍👍 ESTIMATED BITRATE: Near-max for MP3 (e.g. 320 kbps CBR). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz <= 22.05: # ~22 kHz (21kHz to <=22.05kHz, Nyquist for 44.1kHz SR)
        base_classification = "💿✅💿✅ ESTIMATED SOURCE: CD-quality audio (lossless 16-bit/44.1 kHz). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz > 22.05: # >22 kHz content (ultrasonic)
        base_classification = "🔊❇️🔊❇️ ESTIMATED SOURCE: High-res audio (e.g. 48 kHz+ sample rate). Observed Cutoff: >22 kHz ({:.1f} kHz)".format(cutoff_kHz)
    else: # Fallback for any other case
        base_classification = "CLEAN / LIKELY TRUE LOSSLESS (Cutoff: {:.1f} kHz)".format(cutoff_kHz)

    # Check for specific patterns that might override or provide more detail (existing rules)
    # Obvious lossy patterns
    if cut < 18000 and slope > ROLL_OFF_SLOPE_THRESHOLD and hf_dif < -50:
        return "LOSSY UPSAMPLED – likely MP3 ~128–192 kbps (Cutoff: {:.1f} kHz, Slope: {:.0f} dB/kHz, HF Diff: {:.0f} dB)".format(cutoff_kHz, slope, hf_dif)
    
    # High‑bitrate lossy (e.g. 320 kbps AAC/MP3) pattern with suspicious stereo correlation
    if 18000 <= cut < 21000 and corr is not None and corr > 0.9 and hf_dif < -40:
        return "POSSIBLE HIGH‑BITRATE TRANSCODE (Cutoff: {:.1f} kHz, HF Corr: {:.2f}, HF Diff: {:.0f} dB)".format(cutoff_kHz, corr, hf_dif)

    # If no specific patterns matched, return the base classification from the table.
    return base_classification


def analyze_file(path):
    data, sr = load_audio(path, mono=False)
    # ensure 2‑D array [channels, samples]
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    # use summed (mono) for spectrum metrics
    freqs, mean_db = average_spectrum(np.mean(data, axis=0), sr)
    f_cut, slope   = find_cutoff(freqs, mean_db)
    hf_dif         = hf_noise_ratio(freqs, mean_db)
    corr_val       = stereo_hf_correlation(data, sr)
    bitdepth       = bit_depth_analysis(path)

    metrics = {
        "sample_rate": int(sr),
        "cutoff_hz": f_cut,
        "rolloff_db_per_khz": slope,
        "hf_minus_mid_db": hf_dif,
        "hf_stereo_corr": corr_val,
        **bitdepth
    }
    # ensure every value is JSON‑serialisable
    metrics = {k: _json_safe(v) for k, v in metrics.items()}

    verdict = classify(metrics)
    return verdict, metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: quickcheck_audio_quality.py <audiofile1> [audiofile2 ...]")
        sys.exit(1)

    results = {}
    for file in tqdm(sys.argv[1:], desc="Analyzing"):
        verdict, metrics = analyze_file(file)
        results[file] = {"verdict": verdict, "metrics": metrics}

    # pretty‑print JSON so it’s easy to parse later (NumPy scalars already handled)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
