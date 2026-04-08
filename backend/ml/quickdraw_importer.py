from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import requests

QUICKDRAW_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
)
QUICKDRAW_SIMPLIFIED_BASE = (
    "https://storage.googleapis.com/quickdraw_dataset/full/simplified"
)
FALLBACK_CLASSES = [
    "apple",
    "banana",
    "bicycle",
    "book",
    "cat",
    "dog",
    "fish",
    "flower",
    "house",
    "key",
    "moon",
    "star",
    "sun",
    "tree",
    "umbrella",
]


def _draw_strokes_to_image(
    drawing: list[list[list[int]]],
    source_size: int = 256,
    target_size: int = 64,
) -> np.ndarray:
    canvas = np.full((source_size, source_size), 255, dtype=np.uint8)

    for stroke in drawing:
        if len(stroke) != 2:
            continue
        xs = stroke[0]
        ys = stroke[1]
        if len(xs) < 2 or len(xs) != len(ys):
            continue

        points = np.array(list(zip(xs, ys)), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=False, color=0, thickness=5)

    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)


def get_quickdraw_classes(cache_dir: Path, max_classes: int = 120) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "quickdraw_classes.json"

    if cache_file.exists():
        try:
            classes = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(classes, list) and classes:
                return [str(item) for item in classes][:max_classes]
        except Exception:
            pass

    try:
        response = requests.get(QUICKDRAW_CATEGORIES_URL, timeout=15)
        response.raise_for_status()
        classes = [line.strip() for line in response.text.splitlines() if line.strip()]
        classes = sorted(classes)[:max_classes]
        cache_file.write_text(json.dumps(classes), encoding="utf-8")
        return classes
    except Exception:
        return FALLBACK_CLASSES[:max_classes]


def fetch_quickdraw_images_for_class(
    class_name: str,
    sample_count: int,
    cache_dir: Path,
) -> list[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = class_name.replace("/", "_")
    cached_file = cache_dir / f"quickdraw_{safe_name}.ndjson"

    if not cached_file.exists():
        url_name = requests.utils.quote(class_name, safe="")
        url = f"{QUICKDRAW_SIMPLIFIED_BASE}/{url_name}.ndjson"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        cached_file.write_text(response.text, encoding="utf-8")

    images: list[np.ndarray] = []
    with cached_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(images) >= sample_count:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if not bool(item.get("recognized", False)):
                    continue
                drawing = item.get("drawing")
                if not isinstance(drawing, list):
                    continue
                image = _draw_strokes_to_image(drawing)
                images.append(image)
            except Exception:
                continue

    return images
