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
import os

import numpy as np
import librosa          # audio loading / resampling
import soundfile as sf   # to check original bit‑depth
from scipy.stats import pearsonr
from scipy.signal import iirfilter, sosfilt  # equivalents of the missing librosa wrappers
from scipy.interpolate import interp1d
from tqdm import tqdm    # progress bar for long files
import subprocess
import tempfile

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

    cutoff_kHz = cut / 1000.0

    # Handle silence or no detectable content first
    if cut == 0.0 and slope == float('inf'):
        return "🛑 SILENCE OR NO DETECTABLE AUDIO CONTENT"

    # Check for specific patterns (most specific to least specific)
    # Obvious lossy patterns
    if cut < 18000 and slope > ROLL_OFF_SLOPE_THRESHOLD and hf_dif < -50:
        return "LOSSY UPSAMPLED – likely MP3 ~128–192 kbps (Cutoff: {:.1f} kHz, Slope: {:.0f} dB/kHz, HF Diff: {:.0f} dB)".format(cutoff_kHz, slope, hf_dif)

    # High‑bitrate lossy (e.g. 320 kbps AAC/MP3) pattern with suspicious stereo correlation
    if 18000 <= cut < 21000 and corr is not None and corr > 0.9 and hf_dif < -40:
        return "POSSIBLE HIGH‑BITRATE TRANSCODE (Cutoff: {:.1f} kHz, HF Corr: {:.2f}, HF Diff: {:.0f} dB)".format(cutoff_kHz, corr, hf_dif)

    # Determine base classification from cutoff frequency table
    if cutoff_kHz < 11.5: # ~11 kHz
        return "🟥🟥🟥🟥🟥 ESTIMATED BITRATE: Very low (e.g. 64 kbps MP3). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz < 17.0: # ~16 kHz (11.5kHz to <17kHz)
        return "🚫🚫🚫🚫 ESTIMATED BITRATE: Around 128 kbps. Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz < 19.5: # ~18-19 kHz (17kHz to <19.5kHz)
        return "⚠️⚠️⚠️ ESTIMATED BITRATE: Likely mid-high (e.g. 192–256 kbps). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz < 21.0: # ~20 kHz (19.5kHz to <21kHz)
        return "👍👍👍 ESTIMATED BITRATE: Near-max for MP3 (e.g. 320 kbps CBR). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    elif cutoff_kHz <= 22.05: # ~22 kHz (21kHz to <=22.05kHz, Nyquist for 44.1kHz SR)
        return "💿✅💿✅ ESTIMATED SOURCE: CD-quality audio (lossless 16-bit/44.1 kHz). Observed Cutoff: ~{:.1f} kHz".format(cutoff_kHz)
    else: # >22 kHz content (ultrasonic)
        return "🔊❇️🔊❇️ ESTIMATED SOURCE: High-res audio (e.g. 48 kHz+ sample rate). Observed Cutoff: >22 kHz ({:.1f} kHz)".format(cutoff_kHz)


def detect_silence_tail(y, sr, silence_threshold_db=-50, min_silence_duration=1.0):
    """
    Detect silence at the beginning and end of an audio file.
    Returns (has_silence_start, silence_start_duration, has_silence_end, silence_end_duration)
    """
    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Calculate RMS energy in frames
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Convert to dB
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Find frames below threshold
    silent_frames = rms_db < silence_threshold_db

    total_frames = len(silent_frames)

    # Check from the beginning forward to find continuous silence
    silence_start_frame_count = 0
    for i in range(total_frames):
        if silent_frames[i]:
            silence_start_frame_count += 1
        else:
            break

    silence_start_duration = (silence_start_frame_count * hop_length) / sr
    has_silence_start = silence_start_duration >= min_silence_duration

    # Check from the end backwards to find continuous silence
    silence_end_frame_count = 0
    for i in range(total_frames - 1, -1, -1):
        if silent_frames[i]:
            silence_end_frame_count += 1
        else:
            break

    silence_end_duration = (silence_end_frame_count * hop_length) / sr
    has_silence_end = silence_end_duration >= min_silence_duration

    return has_silence_start, silence_start_duration, has_silence_end, silence_end_duration


def spectral_correlation(freqs1, mag_db1, freqs2, mag_db2):
    """
    Calculate Pearson correlation between two magnitude spectra.
    Interpolates to match frequency bins if needed.
    Returns correlation coefficient (0-1).
    """
    # Ensure same frequency bins
    if len(freqs1) != len(freqs2) or not np.allclose(freqs1, freqs2):
        # Interpolate second spectrum to match first
        interp_func = interp1d(freqs2, mag_db2, kind='linear', fill_value='extrapolate')
        mag_db2_aligned = interp_func(freqs1)
    else:
        mag_db2_aligned = mag_db2

    corr, _ = pearsonr(mag_db1, mag_db2_aligned)
    return float(corr)


def hf_energy_ratio(freqs, mag_db_orig, mag_db_reenc, hf_threshold=15000):
    """
    Compare high-frequency energy loss between original and re-encoded.
    Returns energy ratio in dB (negative = loss).
    """
    hf_idx = freqs >= hf_threshold

    if not np.any(hf_idx):
        return 0.0  # No HF content in frequency range

    # Convert dB back to linear amplitude for energy calculation
    orig_amp = librosa.db_to_amplitude(mag_db_orig[hf_idx])
    reenc_amp = librosa.db_to_amplitude(mag_db_reenc[hf_idx])

    orig_energy = np.sum(orig_amp ** 2)
    reenc_energy = np.sum(reenc_amp ** 2)

    if orig_energy == 0:
        return 0.0  # No original HF energy

    ratio_db = 10 * np.log10(reenc_energy / orig_energy)
    return float(ratio_db)


def reencode_mp3(input_path, output_path, bitrate_kbps=320):
    """
    Re-encode audio to MP3 at specific bitrate using ffmpeg.
    Returns True if successful, False otherwise.
    """
    cmd = [
        'ffmpeg', '-y',  # Overwrite output
        '-i', str(input_path),
        '-c:a', 'libmp3lame',  # MP3 codec
        '-b:a', f'{bitrate_kbps}k',  # Bitrate
        '-q:a', '0',  # Highest quality encoding
        '-loglevel', 'error',  # Only show errors
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def verify_bitrate(original_path, test_bitrates=None):
    """
    Progressive re-encoding to detect true source bitrate.

    Re-encodes the file at progressively lower bitrates and compares
    spectral characteristics to find where significant differences appear.

    Args:
        original_path: Path to the original audio file
        test_bitrates: List of bitrates to test (kbps), high to low

    Returns:
        tuple: (threshold_bitrate, results_dict, metadata)
            - threshold_bitrate: Lowest bitrate before significant difference
              (None if all similar or all different)
            - results_dict: Dict mapping bitrate -> comparison metrics
            - metadata: Dict with original file info (cutoff, quality estimate)
    """
    if test_bitrates is None:
        test_bitrates = [320, 256, 192, 128, 96]

    # Analyze original file
    try:
        orig_y, orig_sr = load_audio(original_path, mono=False)
        if orig_y.ndim == 1:
            orig_y = np.expand_dims(orig_y, axis=0)

        orig_freqs, orig_mag_db = average_spectrum(np.mean(orig_y, axis=0), orig_sr)
        orig_cutoff, orig_slope = find_cutoff(orig_freqs, orig_mag_db)
    except Exception as e:
        return None, {'error': f'Failed to analyze original file: {str(e)}'}, {}

    # Adaptive thresholds based on original file quality
    # If original already has low cutoff (<20kHz), it's already lossy
    orig_cutoff_khz = orig_cutoff / 1000.0

    if orig_cutoff_khz < 18.0:
        # Already obviously lossy - use tighter thresholds
        cutoff_threshold = 1000  # 1 kHz tolerance
        corr_threshold = 0.98    # Very high correlation required
        hf_loss_threshold = -6   # Less HF loss tolerance
        quality_note = "already_lossy"
    elif orig_cutoff_khz < 20.0:
        # Borderline/high-bitrate lossy - moderate thresholds
        cutoff_threshold = 800
        corr_threshold = 0.96
        hf_loss_threshold = -8
        quality_note = "borderline"
    else:
        # Appears lossless - use original thresholds
        cutoff_threshold = 500
        corr_threshold = 0.95
        hf_loss_threshold = -10
        quality_note = "high_quality"

    metadata = {
        'original_cutoff_hz': float(orig_cutoff),
        'original_cutoff_khz': float(orig_cutoff_khz),
        'original_slope': float(orig_slope),
        'quality_note': quality_note,
        'thresholds_used': {
            'cutoff_drop_hz': cutoff_threshold,
            'spectral_correlation': corr_threshold,
            'hf_energy_loss_db': hf_loss_threshold
        }
    }

    results = {}
    threshold_bitrate = None
    significant_differences = 0  # Count how many metrics show difference

    with tempfile.TemporaryDirectory() as tmpdir:
        for bitrate in sorted(test_bitrates, reverse=True):  # High to low
            # Re-encode at this bitrate
            reenc_path = Path(tmpdir) / f"reenc_{bitrate}kbps.mp3"

            if not reencode_mp3(original_path, reenc_path, bitrate):
                results[bitrate] = {
                    'error': 'Encoding failed (ffmpeg not available?)',
                    'is_different': None
                }
                continue

            try:
                # Analyze re-encoded file
                reenc_y, reenc_sr = load_audio(str(reenc_path), mono=False)
                if reenc_y.ndim == 1:
                    reenc_y = np.expand_dims(reenc_y, axis=0)

                reenc_freqs, reenc_mag_db = average_spectrum(np.mean(reenc_y, axis=0), reenc_sr)
                reenc_cutoff, _ = find_cutoff(reenc_freqs, reenc_mag_db)

                # Compare metrics
                cutoff_diff = orig_cutoff - reenc_cutoff
                spec_corr = spectral_correlation(orig_freqs, orig_mag_db, reenc_freqs, reenc_mag_db)
                hf_ratio = hf_energy_ratio(orig_freqs, orig_mag_db, reenc_mag_db)

                # Count how many metrics exceed thresholds (require at least 2 of 3)
                metrics_exceeded = 0
                if cutoff_diff > cutoff_threshold:
                    metrics_exceeded += 1
                if spec_corr < corr_threshold:
                    metrics_exceeded += 1
                if hf_ratio < hf_loss_threshold:
                    metrics_exceeded += 1

                # Require at least 2 metrics to agree for "different"
                is_different = metrics_exceeded >= 2

                results[bitrate] = {
                    'cutoff_drop_hz': float(cutoff_diff),
                    'spectral_correlation': float(spec_corr),
                    'hf_energy_loss_db': float(hf_ratio),
                    'is_different': is_different,
                    'metrics_exceeded': metrics_exceeded,
                    'reencoded_cutoff_hz': float(reenc_cutoff)
                }

                # Set threshold if this is first significant difference
                if is_different and threshold_bitrate is None:
                    threshold_bitrate = bitrate

            except Exception as e:
                results[bitrate] = {
                    'error': f'Analysis failed: {str(e)}',
                    'is_different': None
                }

    return threshold_bitrate, results, metadata


def analyze_file(path, check_silence=False):
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

    # Optional silence detection
    if check_silence:
        has_silence_start, silence_start_duration, has_silence_end, silence_end_duration = detect_silence_tail(np.mean(data, axis=0), sr)
        metrics["has_silence_start"] = has_silence_start
        metrics["silence_start_duration_sec"] = silence_start_duration
        metrics["has_silence_end"] = has_silence_end
        metrics["silence_end_duration_sec"] = silence_end_duration
        # Keep old key for backward compatibility
        metrics["has_silence_tail"] = has_silence_end
        metrics["silence_duration_sec"] = silence_end_duration

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
