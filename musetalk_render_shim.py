"""musetalk_render_shim — bridges the nexus-a2a renderer adapter to MuseTalk's
actual scripts.inference interface.

The adapter in shared/clinician_digital_twin/lip_sync_renderers.py invokes:

    <MUSETALK_PYTHON> -m <MUSETALK_ENTRY> \
        --source <portrait.png|.jpg|.mp4> \
        --audio  <audio.wav> \
        --output <output.mp4>

MuseTalk's scripts.inference takes --inference_config <yaml>, where the YAML
maps task_ids to {video_path, audio_path, bbox_shift}. MuseTalk wants a video,
not a still image; for stills we ffmpeg-loop to a tiny single-frame video.

Run as: MUSETALK_ENTRY=musetalk_render_shim
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _ffmpeg_bin() -> str:
    """Resolve ffmpeg: prefer imageio_ffmpeg's bundled binary, fall back to PATH."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _loop_image_to_video(image: Path, duration_s: float, fps: int, out: Path) -> None:
    """ffmpeg-loop a still image into a short video of given duration."""
    cmd = [
        _ffmpeg_bin(),
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        str(image),
        "-t",
        f"{duration_s:.3f}",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def _wav_duration_s(wav: Path) -> float:
    """Wav duration via soundfile (always present alongside MuseTalk's reqs)."""
    try:
        import soundfile as sf

        with sf.SoundFile(str(wav)) as f:
            return f.frames / float(f.samplerate)
    except Exception:
        return 5.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--version", choices=["v1", "v15"], default="v1")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    audio = Path(args.audio).resolve()
    output = Path(args.output).resolve()
    repo = Path(__file__).resolve().parent

    if not source.is_file():
        print(f"[shim] source missing: {source}", file=sys.stderr)
        return 2
    if not audio.is_file():
        print(f"[shim] audio missing: {audio}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="musetalk_shim_") as td:
        td_path = Path(td)

        # 1. If source is a still image, ffmpeg-loop to a temp MP4 matching audio length.
        if source.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            duration = _wav_duration_s(audio)
            video_for_musetalk = td_path / "source_loop.mp4"
            _loop_image_to_video(source, duration, fps=25, out=video_for_musetalk)
        else:
            video_for_musetalk = source

        # 2. Write a temp YAML config in the schema scripts.inference expects.
        task_id = f"shim_{int(time.time() * 1000)}"
        cfg_path = td_path / "shim_config.yaml"
        cfg_path.write_text(
            f"{task_id}:\n"
            f'  video_path: "{video_for_musetalk.as_posix()}"\n'
            f'  audio_path: "{audio.as_posix()}"\n'
            f"  bbox_shift: {args.bbox_shift}\n",
            encoding="utf-8",
        )

        result_dir = td_path / "results"
        result_dir.mkdir()

        # 3. Run MuseTalk's real inference.
        cmd = [
            sys.executable,
            "-m",
            "scripts.inference",
            "--inference_config",
            str(cfg_path),
            "--result_dir",
            str(result_dir),
            "--version",
            args.version,
            "--output_vid_name",
            "shim_out.mp4",
        ]
        print(f"[shim] running: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(repo))
        if proc.returncode != 0:
            print(f"[shim] scripts.inference exit {proc.returncode}", file=sys.stderr)
            return proc.returncode

        # 4. Find MuseTalk's output and copy to requested --output.
        # scripts.inference writes to result_dir/{version}/{output_vid_name}
        produced = None
        for p in result_dir.rglob("shim_out.mp4"):
            produced = p
            break
        if produced is None:
            # Fallback: take the newest .mp4 under result_dir.
            mp4s = sorted(
                result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if mp4s:
                produced = mp4s[0]
        if produced is None:
            print(f"[shim] no MP4 found under {result_dir}", file=sys.stderr)
            return 3

        shutil.copy2(produced, output)
        print(f"[shim] OK -> {output}", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
