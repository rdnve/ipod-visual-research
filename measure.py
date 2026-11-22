"""Шаг 2. Замер по ключевым кадрам: сколько места занимает белое и сколько внимания забирает.

Пример:
    python measure.py --videos videos --keyframes keyframes.csv --out out
    python measure.py ... --model spectral     # запасной вариант без нейросети

На выходе:
    out/results.csv        — числа по каждому кадру
    out/frames/*.png       — исходные кадры
    out/heatmaps/*.png     — тепловая карта поверх кадра
    out/masks/*.png        — что скрипт посчитал «белым»
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image
from scipy import ndimage

from common import detect_crop, extract_frame, load_rgb, parse_time, time_tag, white_mask
from saliency import load_model


def heat_overlay(rgb: np.ndarray, density: np.ndarray) -> Image.Image:
    """Кадр приглушён, сверху тепловая карта: чем ярче, тем больше внимания."""
    norm = density / density.max()
    colors = colormaps["inferno"](norm)[..., :3]
    alpha = np.clip(norm * 1.4, 0, 0.85)[..., None]
    base = rgb.astype(np.float64) / 255.0 * 0.55
    out = base * (1 - alpha) + colors * alpha
    return Image.fromarray((out * 255).astype(np.uint8))


def mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Кадр в сером, найденное белое — красным."""
    gray = rgb.mean(axis=2, keepdims=True).repeat(3, axis=2) * 0.45
    gray[mask] = [230, 40, 40]
    return Image.fromarray(gray.astype(np.uint8))


def roi_mask(shape, roi: str) -> np.ndarray:
    """Колонка roi в keyframes.csv:
    пусто            — весь кадр;
    none             — продукта в кадре нет (маска пустая);
    x0,y0,x1,y1      — прямоугольник в долях кадра (0–1), где искать белое.
    """
    h, w = shape
    roi = (roi or "").strip().lower()
    m = np.ones((h, w), dtype=bool)
    if not roi:
        return m
    if roi == "none":
        return ~m
    x0, y0, x1, y1 = (float(v) for v in roi.split(","))
    m[:] = False
    m[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, default=Path("videos"))
    ap.add_argument("--keyframes", type=Path, default=Path("keyframes.csv"))
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--model", choices=["deepgaze", "spectral"], default="deepgaze")
    ap.add_argument("--device", default="cpu", help="cpu (по умолчанию), cuda или mps")
    ap.add_argument("--centerbias", choices=["mit1003", "uniform"], default="mit1003",
                    help="mit1003 — люди чаще смотрят в центр (как в айтрекинге); uniform — без этой поправки")
    ap.add_argument("--width", type=int, default=1024, help="ширина кадра для замера")
    ap.add_argument("--white-l", type=float, default=0.80, help="мин. светлота белого в OKLab (0–1)")
    ap.add_argument("--white-c", type=float, default=0.08, help="макс. цветность белого в OKLab")
    ap.add_argument("--max-component", type=float, default=0.15,
                    help="белое пятно больше этой доли кадра считается фоном, а не деталью")
    ap.add_argument("--max-border-component", type=float, default=0.01,
                    help="белое пятно у края кадра больше этой доли — тоже фон (светлый задник)")
    args = ap.parse_args()

    with open(args.keyframes, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("video", "").strip() and not r["video"].startswith("#")]
    if not rows:
        raise SystemExit(f"В {args.keyframes} нет строк")

    model = load_model(args.model, args.device, args.centerbias)
    crops: dict[Path, str | None] = {}
    results = []
    for r in rows:
        video = args.videos / r["video"].strip()
        if not video.exists():
            raise SystemExit(f"Нет файла {video}")
        t = parse_time(r["time"])
        name = f"{video.stem}_{time_tag(t)}"
        if video not in crops:
            crops[video] = detect_crop(video)
            if crops[video]:
                print(f"{video.name}: обрезаю чёрные поля ({crops[video]})")
        frame_png = extract_frame(video, t, args.out / "frames" / f"{name}.png", crops[video])
        rgb = load_rgb(frame_png, args.width)

        roi = r.get("roi", "")
        mask = white_mask(rgb, args.white_l, args.white_c, max_component=args.max_component,
                          max_border_component=args.max_border_component)
        mask &= roi_mask(mask.shape, roi)
        density = model(rgb)

        area = float(mask.mean())
        attention = float(density[mask].sum())
        lift = attention / area if area > 0 else float("nan")
        # пик внимания на белом (с допуском 1% ширины кадра)
        near = ndimage.binary_dilation(mask, iterations=max(1, rgb.shape[1] // 100)) if area > 0 else mask
        peak = np.unravel_index(np.argmax(density), density.shape)
        peak_on_white = bool(near[peak])

        for sub, img in (("heatmaps", heat_overlay(rgb, density)), ("masks", mask_overlay(rgb, mask))):
            (args.out / sub).mkdir(parents=True, exist_ok=True)
            img.save(args.out / sub / f"{name}.png")

        results.append({
            "video": video.name, "time_s": f"{t:.2f}", "label": r.get("label", "").strip(),
            "area_pct": f"{area * 100:.2f}", "attention_pct": f"{attention * 100:.2f}",
            "lift": f"{lift:.1f}" if area > 0 else "",
            "peak_on_white": ("да" if peak_on_white else "нет") if area > 0 else "",
            "roi": (roi or "").strip(),
            "model": model.name,
        })
        if area > 0:
            print(f"{name}: площадь белого {area * 100:.2f}% | внимание {attention * 100:.2f}% | "
                  f"×{lift:.1f} | пик на белом: {'да' if peak_on_white else 'нет'}")
        else:
            print(f"{name}: белой детали нет (roi={roi or '—'})")

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nГотово: {out_csv}")


if __name__ == "__main__":
    main()
