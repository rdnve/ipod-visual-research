"""Общие функции: кадры из видео, маска белого, цвет OKLab."""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def parse_time(value: str) -> float:
    """'7.5', '00:07.5' или '0:00:07.5' -> секунды."""
    parts = [float(p) for p in str(value).strip().split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def time_tag(seconds: float) -> str:
    return f"{seconds:07.2f}s".replace(".", "_")


def detect_crop(video: Path) -> str | None:
    """Чёрные поля по краям (ролик 4:3 внутри 16:9 и т. п.): ffmpeg cropdetect.

    Возвращает строку вида 'crop=w:h:x:y' — область, где хоть раз было изображение,
    или None, если полей нет.
    """
    import re

    cmd = ["ffmpeg", "-hide_banner", "-i", str(video), "-vf", "cropdetect=24:2:0", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", err)
    size = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    if not found or not size:
        return None
    w, h, x, y = map(int, found[-1])
    W, H = int(size.group(1)), int(size.group(2))
    if w >= W - 8 and h >= H - 8:
        return None
    return f"crop={w}:{h}:{x}:{y}"


def extract_frame(video: Path, seconds: float, out_png: Path, crop: str | None = None) -> Path:
    """Один кадр из видео в момент seconds (точный поиск ffmpeg), без чёрных полей."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-ss", f"{seconds:.3f}",
        *(["-vf", crop] if crop else []), "-frames:v", "1", str(out_png),
    ]
    subprocess.run(cmd, check=True)
    if not out_png.exists():
        raise RuntimeError(f"Не удалось взять кадр {seconds}s из {video.name} (видео короче?)")
    return out_png


def load_rgb(path: Path, width: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if width and img.width != width:
        height = round(img.height * width / img.width)
        img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """uint8 sRGB (H, W, 3) -> OKLab (H, W, 3): L в [0, 1]."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ])
    m2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ])
    lms = np.cbrt(c @ m1.T)
    return lms @ m2.T


def white_mask(rgb: np.ndarray, min_l: float = 0.80, max_c: float = 0.08,
               min_pixels: int = 4, max_component: float = 0.15,
               max_border_component: float = 0.01) -> np.ndarray:
    """Светлые нейтральные пиксели: провод, наушники, плеер.

    max_c с запасом: в сжатом видео тонкий белый провод
    подкрашивается соседним цветным фоном.
    max_component: белое пятно больше этой доли кадра — фон
    (светлая стена, белый задник), а не деталь, и в маску не идёт.
    max_border_component: пятно, которое касается края кадра и больше этой доли, —
    тоже фон: светлый задник, разрезанный фигурой на несколько кусков.
    """
    from scipy import ndimage

    lab = srgb_to_oklab(rgb)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    mask = (lab[..., 0] >= min_l) & (chroma <= max_c)
    # убрать одиночные пиксели-шум сжатия
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        touches = np.zeros(n + 1, dtype=bool)
        touches[border] = True
        touches[0] = False
        keep[1:] = ((sizes >= min_pixels) & (sizes <= max_component * mask.size)
                    & ~(touches[1:] & (sizes > max_border_component * mask.size)))
        mask = keep[labels]
    return mask
