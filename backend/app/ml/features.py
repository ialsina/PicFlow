from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def image_histogram(img: Image.Image | np.ndarray, bins: np.ndarray | None = None) -> np.ndarray:
    if bins is None:
        bins = np.linspace(0, 255.0, 18)
    arr = np.asarray(img)
    return np.array([np.histogram(arr[..., i], bins=bins)[0] for i in range(arr.shape[-1])])


def extract_rgb_histogram(path: Path, histogram_bins: int = 18) -> np.ndarray | None:
    bins = np.linspace(0, 255.0, histogram_bins)
    with Image.open(path) as img:
        shape = np.asarray(img).shape
        if len(shape) != 3 or shape[-1] != 3:
            return None
        hist = image_histogram(img, bins=bins)
    hist = hist.astype(np.float64)
    hist = hist / np.maximum(hist.sum(axis=-1, keepdims=True), 1e-12)
    return hist.reshape(-1)


def is_rgb_image_path(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            shape = np.asarray(img).shape
            return len(shape) == 3 and shape[-1] == 3
    except OSError:
        return False
