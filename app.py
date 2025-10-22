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

# Ensure librosa.display is available
try:
    import librosa.display
except ImportError:
    st.error("librosa.display module is required. Please install it with: pip install librosa")
    st.stop()

# Import functions from the audio_quality_check module
from audio_quality_check import analyze_file, load_audio, average_spectrum

st.set_page_config(
    page_title="Audio Quality Analyzer",
    page_icon="🔊",
    layout="wide"
)

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
Upload audio files to analyze their quality and detect potential lossy upsampled files.
The analysis checks for frequency cutoffs, roll-off slopes, and other characteristics
that can indicate if a file is truly lossless or has been converted from a lossy format.
""")

# File uploader
uploaded_files = st.file_uploader("Upload Audio Files", 
                                 type=["mp3", "wav", "flac", "aac", "m4a", "ogg"], 
                                 accept_multiple_files=True)

if uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"Analyzing {len(uploaded_files)} files...")
    
    # Create a temp directory to save the uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, uploaded_file in enumerate(uploaded_files):
            progress = (i) / len(uploaded_files)
            progress_bar.progress(progress)
            
            # Create a temp file for the uploaded content
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save uploaded content to the temp file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Analyze the file
            status_text.text(f"Analyzing {uploaded_file.name}...")
            try:
                verdict, metrics = analyze_file(file_path)
                
                # Create an expander for each file
                with st.expander(f"{uploaded_file.name} - {verdict}", expanded=(i==0)):
                    # Add audio player
                    st.subheader("Play Audio")
                    # Read the file bytes for st.audio
                    audio_bytes = uploaded_file.getvalue() # Use getvalue() to get bytes
                    st.audio(audio_bytes, format=uploaded_file.type) # Pass file type for robustness

                    if st.button("Generate Plots", key=f"plot_{i}"):
                        # Load audio for visualization (use mono for visualization)
                        data, sr = load_audio(file_path, mono=True)
                        
                        # Display Spectrogram
                        st.subheader("Spectrogram")
                        # Define the cutoff frequencies for classification display
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
            
            except Exception as e:
                st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text(f"Analysis complete! Analyzed {len(uploaded_files)} files.")

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
