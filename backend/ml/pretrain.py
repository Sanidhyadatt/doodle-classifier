from __future__ import annotations

from pathlib import Path
import pickle

import cv2
import numpy as np

from backend.ml.extractor import extract_features
from backend.ml.model_manager import NeuroModel

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
MODEL_DIR = APP_ROOT / "models"
TRAINING_DATA_PATH = DATA_DIR / "training_data.pkl"
MODEL_PATH = MODEL_DIR / "neuro_model.pkl"

IMAGE_SIZE = 64
SAMPLES_PER_CLASS = 30
RNG = np.random.default_rng(42)


def _blank_canvas() -> np.ndarray:
    return np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)


def _apply_noise(image: np.ndarray) -> np.ndarray:
    noise = RNG.integers(0, 18, size=image.shape, dtype=np.uint8)
    noisy = cv2.subtract(image, noise)
    return noisy


def _draw_circle() -> np.ndarray:
    image = _blank_canvas()
    center = (int(RNG.integers(24, 40)), int(RNG.integers(24, 40)))
    radius = int(RNG.integers(12, 20))
    cv2.circle(image, center, radius, 0, thickness=3)
    return _apply_noise(image)


def _draw_square() -> np.ndarray:
    image = _blank_canvas()
    x1 = int(RNG.integers(12, 24))
    y1 = int(RNG.integers(12, 24))
    side = int(RNG.integers(18, 28))
    x2 = min(x1 + side, IMAGE_SIZE - 10)
    y2 = min(y1 + side, IMAGE_SIZE - 10)
    cv2.rectangle(image, (x1, y1), (x2, y2), 0, thickness=3)
    return _apply_noise(image)


def _draw_triangle() -> np.ndarray:
    image = _blank_canvas()
    pts = np.array(
        [
            [int(RNG.integers(18, 30)), int(RNG.integers(14, 22))],
            [int(RNG.integers(10, 20)), int(RNG.integers(42, 52))],
            [int(RNG.integers(40, 54)), int(RNG.integers(42, 54))],
        ],
        dtype=np.int32,
    )
    cv2.polylines(image, [pts], isClosed=True, color=0, thickness=3)
    return _apply_noise(image)


def _draw_line() -> np.ndarray:
    image = _blank_canvas()
    x1, y1 = int(RNG.integers(10, 24)), int(RNG.integers(10, 24))
    x2, y2 = int(RNG.integers(40, 54)), int(RNG.integers(40, 54))
    cv2.line(image, (x1, y1), (x2, y2), 0, thickness=4)
    return _apply_noise(image)


def generate_dataset() -> tuple[list[list[float]], list[str]]:
    feature_vectors: list[list[float]] = []
    labels: list[str] = []

    generators = {
        "circle": _draw_circle,
        "square": _draw_square,
        "triangle": _draw_triangle,
        "line": _draw_line,
    }

    for label, generator in generators.items():
        for _ in range(SAMPLES_PER_CLASS):
            image = generator()
            features = extract_features(image).astype(np.float32).tolist()
            feature_vectors.append(features)
            labels.append(label)

    return feature_vectors, labels


def main() -> None:
    feature_vectors, labels = generate_dataset()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with TRAINING_DATA_PATH.open("wb") as f:
        pickle.dump({"features": feature_vectors, "labels": labels}, f)

    model = NeuroModel()
    model.train(feature_vectors, labels)
    model.save(MODEL_PATH)

    print(
        f"Pretrained SVM on {len(labels)} synthetic samples across {len(set(labels))} classes."
    )
    print(f"Saved training data to {TRAINING_DATA_PATH}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
