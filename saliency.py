"""Модели внимания. Обе возвращают карту плотности: сумма по кадру = 1."""
from pathlib import Path
import urllib.request

import numpy as np
from scipy import ndimage

CENTERBIAS_URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"


class DeepGazeModel:
    """DeepGaze IIE (Linardos, Kümmerer, Press, Bethge, ICCV 2021).

    Обучена на записях айтрекинга (MIT1003 и др.) и предсказывает,
    куда смотрят люди при свободном просмотре изображения.
    """

    name = "deepgaze_iie"

    def __init__(self, device: str = "cpu", centerbias: str = "mit1003", cache: Path = Path(".cache")):
        import sys
        import types

        import torch

        # Свежий deepgaze_pytorch при импорте тянет модель MSDB, которой нужен пакет clip.
        # Мы используем только DeepGaze IIE, поэтому подставляем пустой модуль clip.
        sys.modules.setdefault("clip", types.ModuleType("clip"))
        import deepgaze_pytorch

        self.torch = torch
        self.device = device
        print("Загружаю DeepGaze IIE (первый запуск скачивает веса, ~1 ГБ)...")
        # .float(): в модели есть буферы float64, а Apple MPS умеет только float32
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).float().to(device).eval()

        if centerbias == "uniform":
            self.template = np.zeros((1024, 1024))
        else:
            cache.mkdir(parents=True, exist_ok=True)
            path = cache / "centerbias_mit1003.npy"
            if not path.exists():
                urllib.request.urlretrieve(CENTERBIAS_URL, path)
            self.template = np.load(path)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from scipy.special import logsumexp

        h, w = rgb.shape[:2]
        cb = ndimage.zoom(self.template, (h / self.template.shape[0], w / self.template.shape[1]),
                          order=0, mode="nearest")
        cb = cb[:h, :w]
        cb -= logsumexp(cb)
        torch = self.torch
        image = torch.tensor(rgb.transpose(2, 0, 1)[None].astype(np.float32), device=self.device)
        bias = torch.tensor(cb[None].astype(np.float32), device=self.device)
        with torch.no_grad():
            log_density = self.model(image, bias)[0, 0].cpu().numpy().astype(np.float64)
        density = np.exp(log_density - log_density.max())
        return density / density.sum()


class SpectralResidualModel:
    """Запасной вариант без нейросети: Hou & Zhang (2007), «Saliency detection:
    a spectral residual approach». Учитывает только контраст и редкость, не лица."""

    name = "spectral_residual"

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        gray = rgb.astype(np.float64).mean(axis=2)
        scale = 64 / max(h, w)
        small = ndimage.zoom(gray, scale, order=1)
        spectrum = np.fft.fft2(small)
        log_amp = np.log(np.abs(spectrum) + 1e-9)
        phase = np.angle(spectrum)
        residual = log_amp - ndimage.uniform_filter(log_amp, size=3, mode="nearest")
        sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
        sal = ndimage.gaussian_filter(sal, sigma=2.5)
        sal = ndimage.zoom(sal, (h / sal.shape[0], w / sal.shape[1]), order=1)[:h, :w]
        sal = np.clip(sal, 0, None)
        return sal / sal.sum()


def load_model(kind: str, device: str, centerbias: str):
    if kind == "deepgaze":
        return DeepGazeModel(device=device, centerbias=centerbias)
    if kind == "spectral":
        return SpectralResidualModel()
    raise ValueError(f"Неизвестная модель: {kind}")
