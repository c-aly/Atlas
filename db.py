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


# Storage operations
def upload_image_to_storage(image_id: str, file_content: bytes, file_ext: str, user_id: str) -> str:
    """Upload image file to Supabase Storage and return public URL"""
    if not supabase:
        raise ValueError("Database not configured")
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty UUID string")
    
    # Storage bucket name (should match your Supabase bucket)
    bucket_name = os.getenv("SUPABASE_STORAGE_BUCKET", "images")
    
    # File path in storage: user_id/image_id.ext
    file_path = f"{user_id}/{image_id}{file_ext}"
    
    try:
        # Determine content type
        content_type = "image/jpeg"
        if file_ext.lower() in [".png"]:
            content_type = "image/png"
        elif file_ext.lower() in [".gif"]:
            content_type = "image/gif"
        elif file_ext.lower() in [".webp"]:
            content_type = "image/webp"
        
        # Upload file to Supabase Storage
        # Use upsert=True to overwrite if file already exists
        upload_result = supabase.storage.from_(bucket_name).upload(
            path=file_path,
            file=file_content,
            file_options={
                "content-type": content_type,
                "upsert": "true"  # Overwrite if exists
            }
        )
        
        # Check for errors in result (Supabase returns dict with 'data' and 'error' keys)
        if isinstance(upload_result, dict) and upload_result.get("error"):  # type: ignore
            error_info = upload_result.get("error")  # type: ignore
            raise ValueError(f"Supabase Storage error: {error_info}")
        
        # Generate signed URL for private bucket (more secure than public URLs)
        # Signed URLs expire after a set time (default 1 hour), providing secure access
        try:
            # Create signed URL that expires in 1 hour (3600 seconds)
            # For longer-lived URLs, you can increase this (e.g., 86400 for 24 hours)
            signed_url_response = supabase.storage.from_(bucket_name).create_signed_url(
                path=file_path,
                expires_in=3600  # 1 hour expiration
            )
            
            # Handle response format
            if isinstance(signed_url_response, dict):
                signed_url = signed_url_response.get("signedURL") or signed_url_response.get("url")  # type: ignore
                if not signed_url:
                    # Fallback: try public URL if bucket is public
                    try:
                        public_url_response = supabase.storage.from_(bucket_name).get_public_url(file_path)
                        if isinstance(public_url_response, dict):
                            signed_url = public_url_response.get("publicUrl")  # type: ignore
                        elif isinstance(public_url_response, str):
                            signed_url = public_url_response
                    except:
                        pass
                
                if not signed_url:
                    raise ValueError("Could not generate signed URL or public URL")
            elif isinstance(signed_url_response, str):
                signed_url = signed_url_response
            else:
                raise ValueError("Unexpected response type from create_signed_url")
                
        except Exception as url_error:
            # Fallback: try public URL (if bucket is public)
            try:
                public_url_response = supabase.storage.from_(bucket_name).get_public_url(file_path)
                if isinstance(public_url_response, dict):
                    signed_url = public_url_response.get("publicUrl")  # type: ignore
                elif isinstance(public_url_response, str):
                    signed_url = public_url_response
                else:
                    raise ValueError("Could not get public URL")
                print(f"WARNING: Using public URL instead of signed URL. Consider making bucket private for better security.")
            except Exception as fallback_error:
                # Last resort: construct URL manually (only works for public buckets)
                if DB_URL:
                    base_url = DB_URL.replace("/rest/v1", "").rstrip("/")
                    from urllib.parse import quote
                    path_parts = file_path.split("/")
                    encoded_parts = [quote(part, safe="") for part in path_parts]
                    encoded_path = "/".join(encoded_parts)
                    signed_url = f"{base_url}/storage/v1/object/public/{bucket_name}/{encoded_path}"
                    print(f"WARNING: Constructed URL manually. Bucket should be public for this to work.")
                else:
                    raise ValueError(f"Cannot generate URL: {url_error}. Fallback also failed: {fallback_error}")
        
        print(f"Generated signed URL for {file_path}: {signed_url[:100]}... (expires in 1 hour)")
        if not signed_url:
            raise ValueError("Failed to generate signed URL")
        return str(signed_url)  # Ensure it's a string
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages
        if "bucket" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
            raise ValueError(f"Storage bucket '{bucket_name}' does not exist. Please create it in your Supabase dashboard.")
        elif "permission" in error_msg.lower() or "policy" in error_msg.lower():
            raise ValueError(f"Permission denied. Please check your storage bucket policies in Supabase dashboard.")
        else:
            print(f"Error uploading image to Supabase Storage: {e}")
            raise ValueError(f"Failed to upload image to storage: {error_msg}")


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


def update_image_description(image_id: str, description: str):
    """Update description for an image"""
    if not supabase:
        raise ValueError("Database not configured")
    
    try:
        result = supabase.table("images").update({
            "description": description
        }).eq("id", image_id).execute()
        return result
    except Exception as e:
        # If description column doesn't exist, log warning but don't fail completely
        error_msg = str(e).lower()
        if "column" in error_msg and ("does not exist" in error_msg or "unknown column" in error_msg):
            print(f"WARNING: description column does not exist in database. Please add it with:")
            print(f"  ALTER TABLE images ADD COLUMN description TEXT;")
            print(f"  Skipping description update for {image_id}")
            return None  # Return None to indicate skip
        else:
            # Other errors should still be raised
            print(f"Error updating description for {image_id}: {e}")
            raise


def update_image_audio_url(image_id: str, audio_url: str):
    """Update audio URL for an image (for caching)"""
    if not supabase:
        raise ValueError("Database not configured")
    
    try:
        result = supabase.table("images").update({
            "audio_url": audio_url
        }).eq("id", image_id).execute()
        return result
    except Exception as e:
        # If audio_url column doesn't exist, log warning but don't fail completely
        error_msg = str(e).lower()
        if "column" in error_msg and ("does not exist" in error_msg or "unknown column" in error_msg):
            print(f"WARNING: audio_url column does not exist in database. Please add it with:")
            print(f"  ALTER TABLE images ADD COLUMN audio_url TEXT;")
            print(f"  Skipping audio URL update for {image_id}")
            return None  # Return None to indicate skip
        else:
            # Other errors should still be raised
            print(f"Error updating audio_url for {image_id}: {e}")
            raise


def get_image_audio_url(image_id: str, user_id: str) -> Optional[str]:
    """Get cached audio URL for an image"""
    if not supabase:
        raise ValueError("Database not configured")
    
    try:
        result = supabase.table("images").select("audio_url").eq("id", image_id).eq("user_id", user_id).limit(1).execute()
        if result.data and len(result.data) > 0:
            return result.data[0].get("audio_url")  # type: ignore
        return None
    except Exception as e:
        print(f"Error getting audio URL for {image_id}: {e}")
        return None


def get_image_by_id(image_id: str, user_id: str) -> Optional[Dict]:
    """Get a single image by ID for a user"""
    if not supabase:
        raise ValueError("Database not configured")
    
    try:
        result = supabase.table("images").select("*").eq("id", image_id).eq("user_id", user_id).limit(1).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]  # type: ignore
        return None
    except Exception as e:
        print(f"Error getting image {image_id}: {e}")
        raise


def get_signed_url_for_image(image_id: str, user_id: str, expires_in: int = 3600) -> Optional[str]:
    """Generate a fresh signed URL for an image (useful when URLs expire)"""
    if not supabase:
        raise ValueError("Database not configured")
    
    # Get image data to find the file path
    image_data = get_image_by_id(image_id, user_id)
    if not image_data:
        return None
    
    image_url = image_data.get("image_url", "")
    if not image_url:
        return None
    
    # Extract file path from URL or construct it
    bucket_name = os.getenv("SUPABASE_STORAGE_BUCKET", "images")
    
    # Try to extract path from existing URL
    # Format: .../storage/v1/object/public/{bucket}/{path} or .../storage/v1/object/sign/{bucket}/{path}?...
    if "/storage/v1/object/" in image_url:
        # Extract path after bucket name
        parts = image_url.split(f"/{bucket_name}/")
        if len(parts) > 1:
            file_path = parts[1].split("?")[0]  # Remove query params
        else:
            # Fallback: construct from user_id and image_id
            file_path = f"{user_id}/{image_id}"
    else:
        # Fallback: construct path
        file_path = f"{user_id}/{image_id}"
    
    try:
        # Generate fresh signed URL
        signed_url_response = supabase.storage.from_(bucket_name).create_signed_url(
            path=file_path,
            expires_in=expires_in
        )
        
        if isinstance(signed_url_response, dict):
            return signed_url_response.get("signedURL") or signed_url_response.get("url")  # type: ignore
        elif isinstance(signed_url_response, str):
            return signed_url_response
        else:
            return None
    except Exception as e:
        print(f"Error generating signed URL for {image_id}: {e}")
        return None


def get_all_images_for_export(user_id: str) -> List[Dict]:
    """Get all images for export (with positions and metadata)"""
    if not supabase:
        raise ValueError("Database not configured")
    
    result = supabase.table("images").select("id,image_url,position_vec,cluster_id,description").eq("user_id", user_id).execute()
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
