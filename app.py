#!/usr/bin/env python3
"""
Streamlit app for audio quality analysis
Uses the audio_quality_check.py module to analyze audio files
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import io
from pathlib import Path
import tempfile
import os
import sys
import csv
import json
from datetime import datetime
import glob

# Ensure librosa.display is available
try:
    import librosa.display
except ImportError:
    st.error("librosa.display module is required. Please install it with: pip install librosa")
    st.stop()

# Import functions from the audio_quality_check module
from audio_quality_check import analyze_file, load_audio, average_spectrum

# Configuration constants
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
SUPPORTED_FORMATS = ["mp3", "wav", "flac", "aac", "m4a", "ogg"]

st.set_page_config(
    page_title="Audio Quality Analyzer",
    page_icon="🔊",
    layout="wide"
)

def scan_folder_for_audio(folder_path, recursive=True):
    """
    Scan a folder for audio files.
    Returns list of audio file paths.
    """
    audio_files = []

    if recursive:
        # Recursively search all subdirectories
        for ext in SUPPORTED_FORMATS:
            pattern = os.path.join(folder_path, '**', f'*.{ext}')
            audio_files.extend(glob.glob(pattern, recursive=True))
    else:
        # Only search top-level directory
        for ext in SUPPORTED_FORMATS:
            pattern = os.path.join(folder_path, f'*.{ext}')
            audio_files.extend(glob.glob(pattern))

    return sorted(audio_files)

def validate_file(uploaded_file):
    """
    Validate uploaded file for size and format.
    Returns (is_valid, error_message)
    """
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"

    # Check file extension
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported file format '.{file_ext}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}"

    return True, None

def validate_file_path(file_path):
    """
    Validate a file path for size and format.
    Returns (is_valid, error_message)
    """
    # Check file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"

    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"

    # Check file extension
    file_ext = file_path.split('.')[-1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported file format '.{file_ext}'"

    return True, None

def export_to_csv(results_data):
    """
    Export analysis results to CSV format.
    results_data: list of dicts with keys: filename, verdict, metrics
    """
    output = io.StringIO()
    if not results_data:
        return None

    # Prepare CSV headers
    fieldnames = ['filename', 'verdict', 'sample_rate', 'cutoff_hz', 'rolloff_db_per_khz',
                  'hf_minus_mid_db', 'hf_stereo_corr', 'bitdepth', 'zero_lsb_fraction']

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for result in results_data:
        row = {
            'filename': result['filename'],
            'verdict': result['verdict']
        }
        # Add metrics
        row.update(result['metrics'])
        writer.writerow(row)

    return output.getvalue()

def export_to_json(results_data):
    """
    Export analysis results to JSON format.
    results_data: list of dicts with keys: filename, verdict, metrics
    """
    if not results_data:
        return None

    export_data = {
        'export_date': datetime.now().isoformat(),
        'total_files': len(results_data),
        'results': results_data
    }

    return json.dumps(export_data, indent=2)

def plot_spectrogram(y, sr, classification_cutoffs=None):
    """Generate a spectrogram plot for the given audio data."""
    fig, ax = plt.subplots(figsize=(15, 6))  # Increased figure size
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(
        D, x_axis='time', y_axis='log', sr=sr, ax=ax, cmap='magma', fmin=5000  # Updated fmin
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Spectrogram (Frequencies above 5 kHz)')  # Updated title

    # Force y-axis limits to ensure the range is 5kHz and above
    min_display_freq = 5000.0  # Updated min_display_freq
    max_display_freq = sr / 2.0

    if min_display_freq < max_display_freq:
        ax.set_ylim(min_display_freq, max_display_freq)

        # Generate and set y-axis ticks
        ticks = []
        current_tick = 5000
        while current_tick <= max_display_freq:
            ticks.append(current_tick)
            if current_tick < 10000:
                current_tick += 1000  # Finer steps below 10k if needed, e.g. 1k steps
            elif current_tick < 20000:
                current_tick += 2000  # e.g. 2k steps
            else:
                current_tick += 5000  # Coarser steps above 20k
            if current_tick == 0:  # avoid infinite loop if sr is very low
                break

        # Ensure the highest frequency (sr/2) is a tick if not already too close to the last one
        if max_display_freq not in ticks and (not ticks or max_display_freq - ticks[-1] > 1000):
            # Add sr/2 if it's reasonably spaced from the last tick
            # or if it's the only tick that would be present (e.g. sr=10000, max_display_freq=5000)
            if not ticks or max_display_freq > ticks[-1]:
                ticks = [t for t in ticks if t < max_display_freq]  # remove ticks above sr/2
                ticks.append(max_display_freq)

        valid_ticks = [t for t in ticks if t >= min_display_freq and t <= max_display_freq]
        if not valid_ticks and min_display_freq <= max_display_freq:  # Ensure at least one tick if range is valid
            valid_ticks = [min_display_freq, max_display_freq]

        # Remove duplicates and sort, especially if min_display_freq or max_display_freq were added
        valid_ticks = sorted(list(set(valid_ticks)))

        ax.set_yticks(valid_ticks)
        ax.set_yticklabels([f'{int(t/1000)}k' if t % 1000 == 0 else f'{t/1000:.1f}k' for t in valid_ticks])

    if classification_cutoffs:
        colors = ['#FF3333', '#FF9933']  # More distinct colors
        linestyles = ['--', ':']
        for i, (freq, label) in enumerate(classification_cutoffs):
            # Ensure the line is within the new forced y-axis limits
            if freq >= min_display_freq and freq < max_display_freq:
                ax.axhline(y=freq, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=1.5, alpha=0.9, label=f'{label} ({freq/1000:.1f} kHz)')
                # Position text to the right of the plot, aligned with the line
                ax.text(ax.get_xlim()[1] * 1.01, freq, f'{label} ({freq/1000:.1f} kHz)', 
                        color=colors[i % len(colors)], verticalalignment='center', horizontalalignment='left', fontsize=9)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust rect to make space for text on the right

    # Return the figure instead of showing it
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_spectrum(y, sr):
    """Generate an average spectrum plot for the given audio data."""
    freqs, mean_db = average_spectrum(y, sr)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, mean_db)
    ax.set_xscale('log')
    ax.set_xlim([20, sr/2])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Average Spectrum')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Mark potential cutoff frequency
    valid = np.where(mean_db > -80)[0]
    if valid.size > 0:
        cutoff_idx = valid[-1]
        cutoff = freqs[cutoff_idx]
        ax.axvline(x=cutoff, color='r', linestyle='--', alpha=0.7)
        ax.text(cutoff*0.9, 0, f'{cutoff:.0f} Hz', 
                color='r', rotation=90, verticalalignment='center')
    
    # Return the figure instead of showing it
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

st.title("Audio Quality Analyzer")
st.write("""
Analyze audio files to detect potential lossy upsampled files.
The analysis checks for frequency cutoffs, roll-off slopes, and other characteristics
that can indicate if a file is truly lossless or has been converted from a lossy format.
""")

# Analysis options
st.subheader("Analysis Options")
check_silence = st.checkbox("Check for silence tails (may increase processing time)", value=False)

st.markdown("---")

# Input mode selection
input_mode = st.radio(
    "Select input mode:",
    ["Upload Files", "Analyze Folder"],
    horizontal=True
)

files_to_analyze = []
source_type = None

if input_mode == "Upload Files":
    # File uploader
    uploaded_files = st.file_uploader("Upload Audio Files",
                                     type=SUPPORTED_FORMATS,
                                     accept_multiple_files=True)
    if uploaded_files:
        files_to_analyze = uploaded_files
        source_type = "upload"

else:  # Analyze Folder
    folder_path = st.text_input(
        "Enter folder path:",
        placeholder="/path/to/your/music/folder",
        help="Enter the full path to the folder containing audio files"
    )

    recursive = st.checkbox("Include subfolders", value=True)

    if folder_path and st.button("Scan Folder"):
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            with st.spinner("Scanning folder..."):
                found_files = scan_folder_for_audio(folder_path, recursive)
                if found_files:
                    st.success(f"Found {len(found_files)} audio files")
                    files_to_analyze = found_files
                    source_type = "folder"
                else:
                    st.warning(f"No audio files found in {folder_path}")
        else:
            st.error("Invalid folder path")

if files_to_analyze:
    # Store results for export
    analysis_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Validate and prepare files based on source type
    valid_files = []

    if source_type == "upload":
        # Validate uploaded files
        for uploaded_file in files_to_analyze:
            is_valid, error_msg = validate_file(uploaded_file)
            if not is_valid:
                st.error(f"❌ {uploaded_file.name}: {error_msg}")
            else:
                valid_files.append(uploaded_file)
    else:  # source_type == "folder"
        # Validate file paths
        for file_path in files_to_analyze:
            is_valid, error_msg = validate_file_path(file_path)
            if not is_valid:
                st.warning(f"⚠️ {os.path.basename(file_path)}: {error_msg}")
            else:
                valid_files.append(file_path)

    if not valid_files:
        st.warning("No valid files to analyze. Please check file sizes and formats.")
    else:
        status_text.text(f"Analyzing {len(valid_files)} files...")

        # Create a temp directory for uploaded files only
        # Store in session state to persist across reruns
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = None

        if source_type == "upload":
            if st.session_state.temp_dir is None or not os.path.exists(st.session_state.temp_dir):
                st.session_state.temp_dir = tempfile.mkdtemp()
            temp_dir = st.session_state.temp_dir
        else:
            temp_dir = None

        try:
            for i, file_item in enumerate(valid_files):
                progress = (i + 1) / len(valid_files)
                progress_bar.progress(progress)

                # Determine file path and name based on source
                if source_type == "upload":
                    file_name = file_item.name
                    file_path = os.path.join(temp_dir, file_name)
                    # Save uploaded content to temp file
                    with open(file_path, "wb") as f:
                        f.write(file_item.getbuffer())
                else:  # folder
                    file_path = file_item
                    file_name = os.path.basename(file_path)

                status_text.text(f"Analyzing {file_name}...")
                try:
                    # Analyze the file
                    verdict, metrics = analyze_file(file_path, check_silence=check_silence)

                    # Store results for export (including source type and temp path if upload)
                    analysis_results.append({
                        'filename': file_name,
                        'filepath': file_path,
                        'source_type': source_type,
                        'verdict': verdict,
                        'metrics': metrics
                    })

                except PermissionError:
                    st.error(f"❌ {file_name}: Permission denied - cannot access file")
                except OSError as e:
                    st.error(f"❌ {file_name}: File system error - {str(e)}")
                except librosa.util.exceptions.ParameterError as e:
                    st.error(f"❌ {file_name}: Invalid audio file - {str(e)}")
                except Exception as e:
                    st.error(f"❌ {file_name}: Unexpected error - {str(e)}")

            # Complete the progress bar
            progress_bar.progress(1.0)
            status_text.text(f"Analysis complete! Analyzed {len(analysis_results)} of {len(valid_files)} files.")

            # Filter and Sort controls
            if analysis_results:
                st.markdown("---")
                st.subheader("Filter and Sort Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    filter_options = ["All Files", "Lossy/Problematic Only", "Lossless Only", "High Quality Only"]
                    if check_silence:
                        filter_options.append("Files with Silence Tails")
                    filter_option = st.selectbox("Filter:", filter_options)

                with col2:
                    sort_option = st.selectbox(
                        "Sort by:",
                        ["Original Order", "Worst to Best", "Best to Worst", "Filename A-Z", "Filename Z-A", "Cutoff Frequency (Low to High)", "Cutoff Frequency (High to Low)"]
                    )

                with col3:
                    show_metrics = st.checkbox("Show detailed metrics", value=False)

                # Apply filters
                filtered_results = analysis_results.copy()

                if filter_option == "Lossy/Problematic Only":
                    filtered_results = [r for r in filtered_results if any(keyword in r['verdict'].upper() for keyword in ['LOSSY', 'TRANSCODE', 'LOW', 'ESTIMATED BITRATE'])]
                elif filter_option == "Lossless Only":
                    filtered_results = [r for r in filtered_results if any(keyword in r['verdict'] for keyword in ['CD-quality', 'High-res', 'LOSSLESS', '💿', '🔊'])]
                elif filter_option == "High Quality Only":
                    filtered_results = [r for r in filtered_results if r['metrics']['cutoff_hz'] >= 20000]
                elif filter_option == "Files with Silence Tails":
                    filtered_results = [r for r in filtered_results if r['metrics'].get('has_silence_tail', False)]

                # Apply sorting
                if sort_option == "Worst to Best":
                    filtered_results.sort(key=lambda x: x['metrics']['cutoff_hz'])
                elif sort_option == "Best to Worst":
                    filtered_results.sort(key=lambda x: x['metrics']['cutoff_hz'], reverse=True)
                elif sort_option == "Filename A-Z":
                    filtered_results.sort(key=lambda x: x['filename'])
                elif sort_option == "Filename Z-A":
                    filtered_results.sort(key=lambda x: x['filename'], reverse=True)
                elif sort_option == "Cutoff Frequency (Low to High)":
                    filtered_results.sort(key=lambda x: x['metrics']['cutoff_hz'])
                elif sort_option == "Cutoff Frequency (High to Low)":
                    filtered_results.sort(key=lambda x: x['metrics']['cutoff_hz'], reverse=True)

                st.info(f"Showing {len(filtered_results)} of {len(analysis_results)} files")

                # Display filtered and sorted results
                st.markdown("---")
                for i, result in enumerate(filtered_results):
                    file_name = result['filename']
                    verdict = result['verdict']
                    metrics = result['metrics']
                    file_path = result['filepath']
                    result_source_type = result.get('source_type', 'folder')

                    with st.expander(f"{file_name} - {verdict}", expanded=False):
                        # Show silence warning if detected
                        if metrics.get('has_silence_tail'):
                            silence_duration = metrics.get('silence_duration_sec', 0)
                            st.warning(f"⚠️ Silence tail detected: {silence_duration:.1f} seconds of silence at the end")

                        if show_metrics:
                            # Display detailed metrics
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Sample Rate", f"{metrics['sample_rate']} Hz")
                                st.metric("Cutoff Frequency", f"{metrics['cutoff_hz']:.0f} Hz ({metrics['cutoff_hz']/1000:.1f} kHz)")
                                st.metric("Roll-off Slope", f"{metrics['rolloff_db_per_khz']:.1f} dB/kHz")
                            with col_b:
                                st.metric("HF vs Mid Difference", f"{metrics['hf_minus_mid_db']:.1f} dB")
                                if metrics.get('hf_stereo_corr') is not None:
                                    st.metric("HF Stereo Correlation", f"{metrics['hf_stereo_corr']:.2f}")
                                if metrics.get('bitdepth'):
                                    st.metric("Bit Depth", metrics['bitdepth'])
                                if metrics.get('has_silence_tail'):
                                    st.metric("Silence Tail", f"{metrics['silence_duration_sec']:.1f} sec")

                        # Add audio player
                        st.subheader("Play Audio")
                        # Always use file path - works for both upload and folder modes
                        if os.path.exists(file_path):
                            st.audio(file_path)
                        else:
                            st.warning("Audio file no longer available (temporary file may have been cleaned up)")

                        if st.button("Generate Plots", key=f"filtered_plot_{i}"):
                            if os.path.exists(file_path):
                                # Load audio for visualization
                                data, sr = load_audio(file_path, mono=True)

                                # Display Spectrogram
                                st.subheader("Spectrogram")
                                cutoffs_for_plot = [
                                    (18000, "MP3/Low-Bitrate Cutoff"),
                                    (21000, "High-Bitrate Transcode Cutoff")
                                ]
                                spec_img = plot_spectrogram(data, sr, classification_cutoffs=cutoffs_for_plot)
                                st.image(spec_img)
                                st.caption("Time-frequency representation of the audio signal, with lines indicating common lossy codec frequency cutoffs.")

                                # Display Frequency Spectrum
                                st.subheader("Frequency Spectrum")
                                spec_img = plot_spectrum(data, sr)
                                st.image(spec_img)
                                st.caption("Average frequency spectrum (with cutoff highlighted)")
                            else:
                                st.error("Cannot generate plots - file no longer available")

        finally:
            # Don't clean up temp directory - it's stored in session state
            # and will be cleaned up when the session ends or on next analysis
            pass

        # Export section
        if analysis_results:
            st.markdown("---")
            st.subheader("Export Results")
            col1, col2 = st.columns(2)

            with col1:
                csv_data = export_to_csv(analysis_results)
                if csv_data:
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"audio_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            with col2:
                json_data = export_to_json(analysis_results)
                if json_data:
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"audio_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

st.markdown("---")
st.markdown("### About This Tool")
st.markdown("""
This tool analyzes audio files to detect signs of lossy-to-lossless conversion (also known as "fake lossless" files).
It examines frequency content, roll-off characteristics, and stereo correlation in high frequencies.

**Key indicators of lossy files:**
- Sharp frequency cutoff (below 20kHz)
- Steep roll-off slope at the cutoff
- High correlation between channels in high frequencies
- Low energy in high-frequency bands compared to mid-range
""")

# Add footer with version info
st.markdown("---")
st.caption(f"Audio Quality Analyzer v1.0 | Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
