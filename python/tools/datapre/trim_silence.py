#!/usr/bin/env python3
# ========================================================================
# TRIM_SILENCE: Remove leading/trailing silence from WAV files
# ========================================================================
# This script trims silent regions from audio files based on an amplitude
# threshold. Useful for preprocessing audio recordings to remove dead air
# and focus on the active speech/sound regions.
#
# Features:
#   - Configurable amplitude threshold (normalized 0-1)
#   - Optional padding before/after detected audio
#   - Multi-channel support (collapses to max magnitude)
#   - In-place override or separate output directory
# ========================================================================

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for trim_silence utility.
    
    Returns:
        argparse.Namespace: Parsed arguments including threshold, padding, and output directory
    """
    parser = argparse.ArgumentParser(
        description="Trim leading and trailing silence from WAV files by amplitude threshold."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1] / "data_exp" / "wav_audio",
        help="Directory containing source WAV files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Normalized amplitude threshold (0-1) used to detect non-silent regions.",
    )
    parser.add_argument(
        "--padding-ms",
        type=float,
        default=20.0,
        help="Padding to keep before and after detected audio in milliseconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write trimmed files. Defaults to creating a sibling 'wav_audio_trimmed'.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original files with trimmed audio.",
    )
    return parser.parse_args()


def resolve_output_dir(input_dir: Path, output_dir: Path | None, inplace: bool) -> Path:
    """
    Determine the output directory for trimmed files.
    
    Args:
        input_dir: Source directory containing original WAV files
        output_dir: User-specified output directory (optional)
        inplace: If True, overwrite original files
    
    Returns:
        Path: Resolved output directory path
    """
    if inplace:
        return input_dir
    if output_dir is not None:
        return output_dir
    return input_dir.parent / f"{input_dir.name}_trimmed"


def compute_thresholded_mask(audio: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute a boolean mask for non-silent regions.
    
    Args:
        audio: Input audio array (1D or 2D for multi-channel)
        threshold: Normalized amplitude threshold (0-1)
    
    Returns:
        np.ndarray: Boolean mask where True indicates non-silent samples
    """
    # Treat multi-channel audio by collapsing to the maximum magnitude per frame.
    if audio.ndim == 1:
        magnitude = np.abs(audio.astype(np.float64))
    else:
        magnitude = np.max(np.abs(audio.astype(np.float64)), axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
    else:
        max_val = 1.0

    if max_val == 0:
        return np.zeros_like(magnitude, dtype=bool)

    normalized = magnitude / max_val
    return normalized > threshold


def trim_indices(mask: np.ndarray, padding_samples: int, total_samples: int) -> Tuple[int, int]:
    """
    Find trimmed start and end indices from a silence mask.
    
    Args:
        mask: Boolean mask of non-silent regions
        padding_samples: Number of samples to pad before/after detected audio
        total_samples: Total number of samples in the audio
    
    Returns:
        Tuple[int, int]: (start_index, end_index) for trimmed audio
    """
    if not np.any(mask):
        return 0, total_samples

    start = int(np.argmax(mask))
    end = total_samples - int(np.argmax(mask[::-1]))
    start = max(0, start - padding_samples)
    end = min(total_samples, end + padding_samples)
    if start >= end:
        return 0, total_samples
    return start, end


def trim_file(
    file_path: Path,
    output_path: Path,
    threshold: float,
    padding_ms: float,
) -> None:
    """
    Trim silence from a single WAV file and save the result.
    
    Args:
        file_path: Path to input WAV file
        output_path: Path to output trimmed WAV file
        threshold: Amplitude threshold (0-1)
        padding_ms: Padding duration in milliseconds
    """
    sample_rate, audio = wavfile.read(file_path)

    padding_samples = int((padding_ms / 1000.0) * sample_rate)
    mask = compute_thresholded_mask(audio, threshold)
    start, end = trim_indices(mask, padding_samples, len(mask))

    trimmed = audio[start:end]
    if trimmed.shape[0] == 0:
        trimmed = audio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sample_rate, trimmed)


def main() -> None:
    """
    Main routine to process all WAV files in the input directory.
    Iterates through all .wav files and applies silence trimming.
    """
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = resolve_output_dir(input_dir, args.output_dir, args.inplace)

    for wav_path in sorted(input_dir.glob("*.wav")):
        target_path = output_dir / wav_path.name
        trim_file(wav_path, target_path, args.threshold, args.padding_ms)
        print(f"Trimmed {wav_path} -> {target_path}")


if __name__ == "__main__":
    main()
