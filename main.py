import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import db
from pca3d import fit_pca3d_on_images



# --- Load model & processor once globally (faster for repeated calls) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "openai/clip-vit-base-patch32"  # 512-D

_model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()  # type: ignore
_processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=True)

def image_to_clip_vector(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into a 512-D CLIP embedding (normalized).
    Works with any transformers version.

    Args:
        img: PIL.Image (RGB)

    Returns:
        np.ndarray: 512-D normalized embedding (dtype float32)
    """
    # Ensure RGB
    img = img.convert("RGB")

    # Preprocess robustly
    out = _processor(images=img)
    pixel_values = out["pixel_values"] if isinstance(out, dict) else getattr(out, "pixel_values", None)

    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.to(device=device, dtype=_model.dtype, non_blocking=True)
    else:
        if isinstance(pixel_values, list):
            pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.as_tensor(pixel_values).to(device=device, dtype=_model.dtype, non_blocking=True)

    # Forward → normalize → NumPy
    with torch.no_grad():
        feats = _model.get_image_features(pixel_values=pixel_values) # type: ignore # [1,512]
        feats = feats / feats.norm(dim=-1, keepdim=True)

    emb = feats.squeeze(0).cpu().numpy().astype(np.float32)
    return emb



# get 
def main():
    folder = Path("test_images")
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    imgs, paths = [], []
    for p in folder.iterdir():
        if p.suffix.lower() in valid:
            paths.append(p)
            imgs.append(Image.open(p))

    coords3d, pca, scale = fit_pca3d_on_images(imgs, image_to_clip_vector)
    print(coords3d)
    # use transform

    user_id = "00000000-0000-0000-0000-000000000000"
    for (p, c) in zip(paths, coords3d):
        # If you also store the 512-D embedding, compute it once here:
        emb = image_to_clip_vector(Image.open(p))
        db.upload_image(str(p), emb, c.tolist(), user_id)



# run
if __name__=="__main__":
    main()