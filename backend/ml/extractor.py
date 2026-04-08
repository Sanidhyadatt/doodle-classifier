import cv2
import numpy as np
from skimage.feature import hog


def _normalize_doodle(gray: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Binarize, crop to foreground, pad to square, and resize for stable features."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    crop = binary[y_min : y_max + 1, x_min : x_max + 1]
    h, w = crop.shape
    side = max(h, w) + 8
    square = np.zeros((side, side), dtype=np.uint8)

    y_offset = (side - h) // 2
    x_offset = (side - w) // 2
    square[y_offset : y_offset + h, x_offset : x_offset + w] = crop

    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def extract_features(image_array: np.ndarray) -> np.ndarray:
    """Extract Harris corner count, Hu moments, and HOG features from a grayscale image."""
    if image_array.ndim != 2:
        raise ValueError("extract_features expects a 2D grayscale numpy array")

    # Ensure image values are in uint8 range for OpenCV/skimage processing.
    if image_array.dtype != np.uint8:
        gray = np.clip(image_array, 0, 255).astype(np.uint8)
    else:
        gray = image_array

    normalized = _normalize_doodle(gray, target_size=64)

    # 1) Harris corner detection: count connected corner regions above response threshold.
    gray_f32 = np.float32(normalized)
    harris_response = cv2.cornerHarris(gray_f32, blockSize=2, ksize=3, k=0.04)
    harris_response = cv2.dilate(harris_response, None)

    max_response = float(harris_response.max()) if harris_response.size else 0.0
    if max_response > 0:
        corner_mask = (harris_response > 0.01 * max_response).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(corner_mask)
        harris_corner_count = float(max(num_labels - 1, 0))
    else:
        harris_corner_count = 0.0

    # 2) Hu moments from binary silhouette.
    moments = cv2.moments(normalized)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log transform increases numerical stability of Hu moments.
    hu_moments = (-np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)).astype(
        np.float32
    )

    # 3) HOG features on standardized 64x64 image.
    hog_features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=False,
        feature_vector=True,
    ).astype(np.float32)

    combined_features = np.concatenate(
        [np.array([harris_corner_count], dtype=np.float32), hu_moments, hog_features]
    )
    return combined_features
