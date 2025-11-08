# pca3d_centered.py
from typing import List, Callable, Tuple
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

EmbedFn = Callable[[Image.Image], np.ndarray]

def fit_pca3d_on_images(
    images: List[Image.Image],
    embed_fn: EmbedFn,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA, float]:
    """
    Fit PCA (3D) on a batch of images, then scale all points into a centered cube [-1, 1]^3.
    Returns (coords_3d, pca_model, scale_factor).

    coords_3d: (N,3) float32 in [-1,1]^3
    pca_model: fitted PCA (reuse to transform new images)
    scale_factor: max absolute coordinate BEFORE scaling (reuse for consistency)
    """
    if not images:
        return np.empty((0, 3), dtype=np.float32), PCA(n_components=3), 1.0

    # 1) Embed
    embs = np.stack([embed_fn(img.convert("RGB")) for img in images]).astype(np.float32)  # (N,512)

    # 2) PCA → 3D (note: PCA centers the data before projection, so mean ~ 0)
    pca = PCA(n_components=3, random_state=random_state)
    coords = pca.fit_transform(embs).astype(np.float32)  # (N,3), mean≈0

    # 3) Uniform scale to fit inside [-1, 1]^3 (preserve aspect ratio)
    max_abs = float(np.max(np.abs(coords))) or 1.0
    coords_unit = (coords / max_abs).astype(np.float32)

    return coords_unit, pca, max_abs


def transform_images_to_pca3d(
    images: List[Image.Image],
    embed_fn: EmbedFn,
    pca: PCA,
    scale_factor: float
) -> np.ndarray:
    """
    Transform NEW images into the same PCA 3D space and scale to [-1,1]^3,
    using a previously fitted (pca, scale_factor).
    """
    if not images:
        return np.empty((0, 3), dtype=np.float32)

    embs = np.stack([embed_fn(img.convert("RGB")) for img in images]).astype(np.float32)
    coords = pca.transform(embs).astype(np.float32)
    if scale_factor == 0:
        scale_factor = 1.0
    coords_unit = (coords / scale_factor).astype(np.float32)
    return coords_unit
