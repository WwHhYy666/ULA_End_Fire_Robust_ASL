import argparse
import re
import wave
from pathlib import Path


FILENAME_RE = re.compile(r"^(?P<x>\d{2,3})d(?P<y>\d)m_.*\.wav$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split merged wav files into 1-second segments."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("..") / "data_exp" / "wav_audio_merged",
        help="Input directory containing merged wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("..") / "data_exp" / "wav_audio_split1s",
        help="Output directory for 1-second wav segments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing segment files.",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad the last segment with silence to 1 second if needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in sorted(input_dir.glob("*.wav")):
        match = FILENAME_RE.match(wav_path.name)
        if not match:
            print(f"Skip (name not matched): {wav_path.name}")
            continue

        x = int(match.group("x"))
        y = int(match.group("y"))
        prefix = f"{x}d{y}m"

        with wave.open(str(wav_path), "rb") as wf:
            params = wf.getparams()
            frames_per_sec = params.framerate
            total_frames = params.nframes

            if frames_per_sec <= 0:
                print(f"Skip (invalid framerate): {wav_path.name}")
                continue

            full_segments = total_frames // frames_per_sec
            remainder = total_frames % frames_per_sec
            total_segments = full_segments + (1 if args.pad and remainder > 0 else 0)

            for idx in range(total_segments):
                out_name = f"{prefix}_{idx + 1:03d}.wav"
                out_path = output_dir / out_name
                if out_path.exists() and not args.overwrite:
                    continue

                start_frame = idx * frames_per_sec
                wf.setpos(start_frame)
                frames = wf.readframes(frames_per_sec)

                if len(frames) < frames_per_sec * params.sampwidth * params.nchannels:
                    if args.pad:
                        missing_frames = frames_per_sec - len(frames) // (
                            params.sampwidth * params.nchannels
                        )
                        frames += b"\x00" * (
                            missing_frames * params.sampwidth * params.nchannels
                        )
                    else:
                        break

                with wave.open(str(out_path), "wb") as out_wf:
                    out_wf.setparams(params)
                    out_wf.writeframes(frames)

        print(
            f"Split {wav_path.name}: {total_segments} segment(s) -> {output_dir.name}"
        )


if __name__ == "__main__":
    main()
