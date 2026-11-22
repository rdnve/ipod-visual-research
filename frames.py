"""Шаг 1. Кадры-кандидаты: по кадру в секунду + контактный лист на каждый ролик.

Пример:
    python frames.py --videos videos --out out
Дальше смотришь out/sheets/*.jpg и выписываешь нужные моменты в keyframes.csv.
"""
import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def candidates(video: Path, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-vf", f"fps={fps},scale=480:-2", "-q:v", "3", str(out_dir / "%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("*.jpg"))


def contact_sheet(frames: list[Path], fps: float, out_png: Path, cols: int = 6) -> None:
    if not frames:
        return
    first = Image.open(frames[0])
    w, h = first.size
    label_h = 22
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        x, y = (i % cols) * w, (i // cols) * (h + label_h)
        sheet.paste(Image.open(f), (x, y + label_h))
        # кадр N снят примерно в момент (N - 0.5) / fps
        t = (i + 0.5) / fps
        draw.text((x + 6, y + 4), f"{int(t // 60):02d}:{t % 60:04.1f}", fill="black")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png, quality=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=Path("videos"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--fps", type=float, default=1.0, help="кадров в секунду (по умолчанию 1)")
    args = ap.parse_args()

    videos = sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"В папке {args.videos} нет видео")
    for v in videos:
        frames = candidates(v, args.out / "candidates" / v.stem, args.fps)
        contact_sheet(frames, args.fps, args.out / "sheets" / f"{v.stem}.jpg")
        print(f"{v.name}: {len(frames)} кадров -> out/sheets/{v.stem}.jpg")


if __name__ == "__main__":
    main()
