import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile


def parse_args() -> argparse.Namespace:
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
    if inplace:
        return input_dir
    if output_dir is not None:
        return output_dir
    return input_dir.parent / f"{input_dir.name}_trimmed"


def compute_thresholded_mask(audio: np.ndarray, threshold: float) -> np.ndarray:
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
