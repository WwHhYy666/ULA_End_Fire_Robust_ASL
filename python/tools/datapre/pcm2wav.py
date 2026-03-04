#!/usr/bin/env python3
# ========================================================================
# PCM2WAV: Convert multi-channel PCM audio files to WAV format
# ========================================================================
# This script converts raw PCM audio files (16-bit, 6-channel, 16kHz)
# to WAV format for easier processing and analysis.
#
# Configuration:
#   - Input: Raw PCM files from PCM_DIR
#   - Output: WAV files to WAV_DIR
#   - Format: 16-bit integer PCM, 16 kHz sample rate, 6 channels
# ========================================================================

import os
import numpy as np
from scipy.io.wavfile import write

# ===== Configuration Parameters =====
PCM_DIR = "C:\\Users\\25401\\Desktop\\acoustic\\ssl\\data_exp\\pcm_audio"
WAV_DIR = os.path.join(os.path.dirname(PCM_DIR), "wav_audio")
SAMPLE_RATE = 16000  # Sample rate in Hz
NUM_CHANNELS = 6  # Number of audio channels
DTYPE = np.int16  # Data type: 16-bit signed integer

def pcm_to_wav(pcm_file_path):
    """
    Convert a single PCM file to WAV format.
    
    Args:
        pcm_file_path (str): Full path to the input PCM file
    
    Reads raw PCM data, validates frame alignment, converts to numpy array,
    and writes the result as a WAV file.
    """
    with open(pcm_file_path, 'rb') as f:
        raw_data = f.read()

    # Compute number of samples (2 bytes per sample * 6 channels)
    bytes_per_sample = np.dtype(DTYPE).itemsize
    frame_size_bytes = bytes_per_sample * NUM_CHANNELS
    total_samples = len(raw_data) // frame_size_bytes
    remainder = len(raw_data) % frame_size_bytes

    if remainder:
        print(f"⚠️ Warning: {pcm_file_path} has incomplete data length; truncated {remainder} bytes and continuing conversion.")
        raw_data = raw_data[: total_samples * frame_size_bytes]

    # Convert raw bytes to an int16 array
    audio_array = np.frombuffer(raw_data, dtype=DTYPE)

    # Reshape to (samples, channels)
    audio_array = audio_array.reshape((total_samples, NUM_CHANNELS))

    # Output WAV file path
    wav_file_name = os.path.splitext(os.path.basename(pcm_file_path))[0] + ".wav"
    wav_file_path = os.path.join(WAV_DIR, wav_file_name)

    # Write WAV file
    write(wav_file_path, SAMPLE_RATE, audio_array)
    print(f"✅ Converted: {pcm_file_path} -> {wav_file_path}")

def main():
    """
    Main conversion routine.
    Iterates through all PCM files in PCM_DIR and converts them to WAV format.
    """
    os.makedirs(WAV_DIR, exist_ok=True)

    for filename in os.listdir(PCM_DIR):
        if filename.lower().endswith('.pcm'):
            full_path = os.path.join(PCM_DIR, filename)
            pcm_to_wav(full_path)

if __name__ == "__main__":
    main()
