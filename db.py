# db.py
import os
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
import numpy as np
from dotenv import load_dotenv
from supabase import create_client

# Initialize Supabase client
load_dotenv()
DB_URL = os.getenv("DB_URL") or ""
DB_SECRET_KEY = os.getenv("DB_SECRET_KEY") or ""

if not DB_URL or not DB_SECRET_KEY:
    supabase = None
else:
    supabase = create_client(DB_URL, DB_SECRET_KEY)


# Helper functions
def _as_url_str(x: Any) -> str:
    if isinstance(x, (Path, os.PathLike)):
        return str(x)
    if isinstance(x, str):
        return x
    raise TypeError("image_url must be a string or Path-like")


def _to_vec(vec: Any, dim: int, name: str) -> list[float]:
    """Convert vector to list of floats with validation"""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"{name} must have length {dim}, got {arr.size}")
    return arr.tolist()


def parse_json_array(value: Any) -> Optional[List]:
    """Parse JSON string array from Supabase"""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            try:
                return eval(value) if value.startswith("[") else None
            except:
                return None
    return None


# Database operations
def upload_image(image_url: Any, clip_vec: Any, position_vec: Any, user_id: str):
    """Upload image data to database"""
    if not supabase:
        raise ValueError("Database not configured")
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


def get_all_embeddings(user_id: str) -> List[Dict]:
    """Get all images with their embeddings for a user"""
    if not supabase:
        raise ValueError("Database not configured")
    
    result = supabase.table("images").select("id,clip_vec,position_vec").eq("user_id", user_id).execute()
    return result.data if result.data else [] # type: ignore


def update_image_position(image_id: str, position_vec: List[float]):
    """Update 3D position for an image"""
    if not supabase:
        raise ValueError("Database not configured")
    
    supabase.table("images").update({
        "position_vec": position_vec
    }).eq("id", image_id).execute()


def update_image_cluster(image_id: str, cluster_id: int):
    """Update cluster assignment for an image"""
    if not supabase:
        raise ValueError("Database not configured")
    
    try:
        result = supabase.table("images").update({
            "cluster_id": cluster_id
        }).eq("id", image_id).execute()
        
        # Verify the update was successful
        if result.data and len(result.data) > 0:
            updated_cluster = result.data[0].get("cluster_id")
            if updated_cluster == cluster_id:
                return result
            else:
                print(f"WARNING: Cluster update may have failed for {image_id}. Expected {cluster_id}, got {updated_cluster}")
                return result
        else:
            print(f"WARNING: No data returned from cluster update for {image_id}")
            return None
    except Exception as e:
        # If cluster_id column doesn't exist, log warning but don't fail completely
        error_msg = str(e).lower()
        if "column" in error_msg and ("does not exist" in error_msg or "unknown column" in error_msg):
            print(f"WARNING: cluster_id column does not exist in database. Please add it with:")
            print(f"  ALTER TABLE images ADD COLUMN cluster_id INTEGER DEFAULT 0;")
            print(f"  Skipping cluster update for {image_id}")
            return None  # Return None to indicate skip
        else:
            # Other errors should still be raised
            print(f"Error updating cluster_id for {image_id}: {e}")
            raise


def get_all_images_for_export(user_id: str) -> List[Dict]:
    """Get all images for export (with positions and metadata)"""
    if not supabase:
        raise ValueError("Database not configured")
    
    result = supabase.table("images").select("id,image_url,position_vec,cluster_id").eq("user_id", user_id).execute()
    return result.data if result.data else [] # type: ignore


def get_image_count(user_id: str) -> int:
    """Get count of images for a user"""
    if not supabase:
        return 0
    
    try:
        result = supabase.table("images").select("id", count="exact").eq("user_id", user_id).limit(1).execute()  # type: ignore
        return result.count if hasattr(result, 'count') else 0 # type: ignore
    except:
        return 0


def get_images_with_3d_count(user_id: str) -> int:
    """Get count of images with 3D positions"""
    if not supabase:
        return 0
    
    try:
        result = supabase.table("images").select("id").eq("user_id", user_id).not_.is_("position_vec", "null").execute()
        return len(result.data) if result.data else 0
    except:
        return 0


def get_all_embeddings_for_graph() -> List[Dict]:
    """Get all images with embeddings for graph building"""
    if not supabase:
        raise ValueError("Database not configured")
    
    result = supabase.table("images").select("id,clip_vec").execute()
    return result.data if result.data else [] # type: ignore
