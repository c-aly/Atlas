# db.py
import os
from pathlib import Path
from typing import Any
import numpy as np
from dotenv import load_dotenv
from supabase import create_client



# create supabase client
load_dotenv()

DB_URL = os.getenv("DB_URL") or ""
DB_SECRET_KEY = os.getenv("DB_SECRET_KEY") or ""

supabase = create_client(DB_URL, DB_SECRET_KEY)


# helpers
def _as_url_str(x: Any) -> str:
    if isinstance(x, (Path, os.PathLike)):
        return str(x)
    if isinstance(x, str):
        return x
    raise TypeError("image_url must be a string or Path-like")


def _to_vec(vec: Any, dim: int, name: str) -> list[float]:
    # Accept list/tuple/np.ndarray; coerce to 1D float32 and validate length
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"{name} must have length {dim}, got {arr.size}")
    return arr.tolist()  # JSON-serializable


# upload image data to db
def upload_image(image_url: Any, clip_vec: Any, position_vec: Any, user_id: str):
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty UUID string")

    data = {
        "user_id": user_id,
        "image_url": _as_url_str(image_url),
        "clip_vec": _to_vec(clip_vec, 512, "clip_vec"),
        "position_vec": _to_vec(position_vec, 3, "position_vec"),
    }
    res = supabase.table("images").insert(data).execute()
    return res.data
