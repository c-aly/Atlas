from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import os
import uuid
import pickle
import numpy as np
from PIL import Image
from typing import List, Optional
from io import BytesIO
import jwt
import requests

# Import our modules
import db
from main import image_to_clip_vector

load_dotenv()

# Create FastAPI app
app = FastAPI(title="Atlas")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://goatlas.tech",
        "http://localhost:5173",  # Local development
        "http://localhost:3000",  # Alternative local port
        "*"  # Allow all for now (can restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Global state for PCA model (in production, use Redis or DB)
PCA_MODEL_PATH = MODELS_DIR / "pca_model.pkl"
PCA_SCALE_PATH = MODELS_DIR / "pca_scale.pkl"

# Supabase configuration for JWT verification
SUPABASE_URL = os.getenv("DB_URL", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# Google AI Studio (Gemini) configuration
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel (neutral narrator voice)

def get_user_id(authorization: Optional[str] = Header(None)) -> str:
    """Extract user ID from Supabase JWT token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        # Extract token from "Bearer <token>"
        token = authorization.replace("Bearer ", "")
        
        # Verify and decode JWT (Supabase uses HS256)
        if SUPABASE_JWT_SECRET:
            try:
                decoded = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
                user_id = decoded.get("sub")
                if user_id:
                    return user_id
            except jwt.InvalidTokenError:
                pass
        
        # Fallback: decode without verification (for development)
        # In production, always verify with secret
        decoded = jwt.decode(token, options={"verify_signature": False})
        user_id = decoded.get("sub")
        if user_id:
            return user_id
        
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


def get_pca_model():
    """Load PCA model if it exists"""
    if PCA_MODEL_PATH.exists() and PCA_SCALE_PATH.exists():
        with open(PCA_MODEL_PATH, "rb") as f:
            pca = pickle.load(f)
        with open(PCA_SCALE_PATH, "rb") as f:
            scale_factor = pickle.load(f)
        return pca, scale_factor
    return None, None


def save_pca_model(pca, scale_factor):
    """Save PCA model"""
    with open(PCA_MODEL_PATH, "wb") as f:
        pickle.dump(pca, f)
    with open(PCA_SCALE_PATH, "wb") as f:
        pickle.dump(scale_factor, f)


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the project's `index.html` so the browser can load the front-end modules."""
    index_file = ROOT / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.get("/health")
async def health_check():
    """Public health check endpoint (no auth required)"""
    return {
        "status": "ok",
        "message": "Backend is running"
    }


@app.get("/health/user")
async def health_check_user(user_id: str = Depends(get_user_id)):
    """User-specific health check endpoint (requires auth)"""
    try:
        total_images = db.get_image_count(user_id)
    except Exception:
        total_images = 0
    
    return {
        "status": "ok",
        "total_images": total_images
    }


@app.post("/embed/batch")
async def embed_batch(files: List[UploadFile] = File(...), user_id: str = Depends(get_user_id)):
    """
    Upload images, generate CLIP embeddings, fit/apply PCA, and store everything in database.
    Does everything in one call for simplicity.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Initialize variables before try block to avoid unbound variable errors
    new_embeddings = []
    new_image_ids = []
    
    try:
        from sklearn.decomposition import PCA
        
        # Step 1: Upload images and generate embeddings
        
        for file in files:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                continue
            
            # Generate unique ID
            image_id = str(uuid.uuid4())
            
            # Get file extension
            file_ext = Path(file.filename).suffix or ".jpg" # type: ignore
            
            # Read file content
            content = await file.read()
            
            # Upload to Supabase Storage and get public URL
            try:
                storage_url = db.upload_image_to_storage(
                    image_id=image_id,
                    file_content=content,
                    file_ext=file_ext,
                    user_id=user_id
                )
            except Exception as e:
                print(f"Error uploading image {image_id} to storage: {e}")
                continue
            
            # Generate CLIP embedding from image bytes
            img = Image.open(BytesIO(content))
            clip_emb = image_to_clip_vector(img)
            
            # Store in database with placeholder position (will update after PCA)
            try:
                db.upload_image(
                    image_url=storage_url,
                    clip_vec=clip_emb,
                    position_vec=[0.0, 0.0, 0.0],  # Placeholder
                    user_id=user_id
                )
                new_embeddings.append(clip_emb)
                new_image_ids.append(image_id)
            except Exception as e:
                print(f"Error storing image {image_id}: {e}")
                continue
        
        if not new_image_ids:
            raise HTTPException(status_code=400, detail="No valid images uploaded")
        
        # Step 2: Get ALL embeddings from database (including new ones) and fit/apply PCA
        images_data = db.get_all_embeddings(user_id)
        
        if not images_data:
            raise HTTPException(status_code=500, detail="Failed to retrieve embeddings")
        
        # Extract all embeddings
        all_embeddings = []
        all_image_ids = []
        
        for img_data in images_data:
            img_id = img_data.get("id")
            clip_vec = img_data.get("clip_vec")
            
            if clip_vec:
                # Parse JSON string from Supabase
                clip_vec_parsed = db.parse_json_array(clip_vec)
                if clip_vec_parsed is None:
                    continue
                
                emb_array = np.array(clip_vec_parsed, dtype=np.float32)
                if emb_array.shape == (512,):
                    all_embeddings.append(emb_array)
                    all_image_ids.append(img_id)
        
        if not all_embeddings:
            # If no embeddings, just return success (images are already saved)
            return {
                "success": True,
                "count": len(new_image_ids),
                "message": f"Uploaded {len(new_image_ids)} images (no PCA applied - need at least 1 image with embedding)"
            }
        
        # Stack into matrix
        embeddings_matrix = np.stack(all_embeddings).astype(np.float32)
        
        # Step 3: Fit or apply PCA
        pca, scale_factor = get_pca_model()
        
        if pca is None or scale_factor is None:
            # Fit new PCA
            pca = PCA(n_components=3, random_state=42)
            coords_3d = pca.fit_transform(embeddings_matrix).astype(np.float32)
            max_abs = float(np.max(np.abs(coords_3d))) or 1.0
            coords_3d = (coords_3d / max_abs).astype(np.float32)
            scale_factor = max_abs
            save_pca_model(pca, scale_factor)
        else:
            # Apply existing PCA
            coords_3d = pca.transform(embeddings_matrix).astype(np.float32)
            if scale_factor == 0:
                scale_factor = 1.0
            coords_3d = (coords_3d / float(scale_factor)).astype(np.float32)
        
        # Step 4: Update ALL images with 3D positions
        for img_id, pos_3d in zip(all_image_ids, coords_3d):
            try:
                db.update_image_position(img_id, pos_3d.tolist())
            except Exception as e:
                print(f"Error updating position for {img_id}: {e}")
        
        # Step 5: Perform clustering on embeddings
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            # Determine optimal number of clusters using silhouette score
            # Try different k values and pick the one with best silhouette score
            n_samples = len(all_embeddings)
            if n_samples < 4:
                n_clusters = max(1, n_samples - 1)
            else:
                # Try k from 2 to min(10, sqrt(n)/2) and pick the best
                max_k = max(2, min(10, int(np.sqrt(n_samples) / 2)))
                best_k = 2
                best_score = -1
                
                # For small datasets, just use a simple heuristic
                if n_samples < 20:
                    n_clusters = max(2, min(4, n_samples // 5))
                else:
                    # Try k values from 2 to max_k
                    for k in range(2, max_k + 1):
                        try:
                            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)  # type: ignore
                            labels_test = kmeans_test.fit_predict(embeddings_matrix)
                            # Calculate silhouette score (only if we have enough samples)
                            if len(set(labels_test)) > 1:  # Need at least 2 clusters
                                score = silhouette_score(embeddings_matrix, labels_test)
                                if score > best_score:
                                    best_score = score
                                    best_k = k
                        except Exception:
                            continue
                    
                    n_clusters = best_k
                    print(f"Optimal number of clusters determined: {n_clusters} (silhouette score: {best_score:.3f})")
            
            # Perform k-means clustering with optimal k
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # type: ignore
            cluster_labels = kmeans.fit_predict(embeddings_matrix)
            
            # Update cluster assignments in database
            updated_count = 0
            failed_count = 0
            print(f"Starting cluster assignment: {len(all_image_ids)} images, {n_clusters} clusters")
            print(f"Cluster label range: {cluster_labels.min()} to {cluster_labels.max()}")
            
            for img_id, cluster_id in zip(all_image_ids, cluster_labels):
                try:
                    result = db.update_image_cluster(img_id, int(cluster_id))
                    if result is not None:  # None means column doesn't exist, skip silently
                        updated_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Error updating cluster for {img_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_count += 1
            
            print(f"Clustered {len(all_image_ids)} images into {n_clusters} clusters")
            print(f"  - Successfully updated: {updated_count}")
            print(f"  - Failed/skipped: {failed_count}")
            
            if updated_count == 0 and failed_count > 0:
                print("WARNING: No clusters were updated! Check if cluster_id column exists in database.")
        except Exception as e:
            print(f"Error during clustering: {e}")
            import traceback
            traceback.print_exc()
            # Continue even if clustering fails
        
        return {
            "success": True,
            "count": len(new_image_ids),
            "message": f"Uploaded and processed {len(new_image_ids)} images"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Still return success if images were saved, even if PCA failed
        if new_image_ids and len(new_image_ids) > 0:
            return {
                "success": True,
                "count": len(new_image_ids),
                "message": f"Uploaded {len(new_image_ids)} images (PCA processing may have failed)"
            }
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


@app.post("/graph")
async def build_graph(k: int = 5, user_id: str = Depends(get_user_id)):
    """
    Build kNN graph based on CLIP embeddings, ensuring connectivity.
    Uses kNN for local structure and MST to ensure all nodes are connected.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.sparse import csr_matrix
        
        # Fetch all images with embeddings for this user
        images_data = db.get_all_embeddings(user_id)
        
        if not images_data or len(images_data) < 2:
            return {"success": True, "edges": [], "count": 0}
        
        # Build kNN graph
        embeddings = []
        image_ids = []
        
        for img_data in images_data:
            clip_vec = img_data.get("clip_vec")
            if clip_vec:
                clip_vec_parsed = db.parse_json_array(clip_vec)
                if clip_vec_parsed is not None:
                    embeddings.append(np.array(clip_vec_parsed, dtype=np.float32))
                    image_ids.append(img_data["id"])
        
        if len(embeddings) < 2:
            return {"success": True, "edges": [], "count": 0}
        
        n_nodes = len(embeddings)
        
        # Fit kNN (use k=5 as default, but ensure at least 1 neighbor)
        n_neighbors = min(k, n_nodes - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
        nn.fit(embeddings)
        
        # Get neighbors
        distances, indices = nn.kneighbors(embeddings)
        
        # Build kNN edges (skip self-connection)
        edge_set = set()  # Use set to avoid duplicates
        edge_weights = {}  # Store weights for each edge
        
        for i, (dists, nbrs) in enumerate(zip(distances, indices)):
            for dist, nbr_idx in zip(dists[1:], nbrs[1:]):  # Skip first (self)
                similarity = 1 - dist  # Convert cosine distance to similarity
                source_id = image_ids[i]
                target_id = image_ids[nbr_idx]
                
                # Create edge key (always use smaller ID first for consistency)
                edge_key = tuple(sorted([source_id, target_id]))
                
                # Keep the edge with highest similarity if duplicate
                if edge_key not in edge_weights or similarity > edge_weights[edge_key]:
                    edge_weights[edge_key] = similarity
                    edge_set.add(edge_key)
        
        # Build distance matrix for MST (to ensure connectivity)
        # Create a full distance matrix
        distance_matrix = np.ones((n_nodes, n_nodes), dtype=np.float32)
        
        # Fill in known distances from kNN
        id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
        for (source_id, target_id), weight in edge_weights.items():
            src_idx = id_to_idx[source_id]
            tgt_idx = id_to_idx[target_id]
            distance_matrix[src_idx, tgt_idx] = 1 - weight  # Convert similarity back to distance
            distance_matrix[tgt_idx, src_idx] = 1 - weight
        
        # Compute MST to ensure connectivity
        mst = minimum_spanning_tree(csr_matrix(distance_matrix))
        mst_dense = mst.toarray()
        
        # Add MST edges that aren't already in kNN edges
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if mst_dense[i, j] > 0 or mst_dense[j, i] > 0:
                    edge_key = tuple(sorted([image_ids[i], image_ids[j]]))
                    if edge_key not in edge_set:
                        # Use the distance from the matrix
                        dist = float(mst_dense[i, j] if mst_dense[i, j] > 0 else mst_dense[j, i])
                        similarity = max(0.0, 1 - dist)  # Ensure non-negative
                        edge_weights[edge_key] = similarity
                        edge_set.add(edge_key)
        
        # Build final edges list
        edges = []
        for (source_id, target_id) in edge_set:
            weight = edge_weights.get((source_id, target_id), 0.5)
            edges.append({
                "source": source_id,
                "target": target_id,
                "weight": float(weight)
            })
        
        return {
            "success": True,
            "edges": edges,
            "count": len(edges)
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error building graph: {str(e)}")


@app.post("/cluster/recompute")
async def recompute_clusters(user_id: str = Depends(get_user_id)):
    """
    Recompute clusters for all existing images.
    Useful when cluster_id column is missing or needs to be updated.
    """
    try:
        from sklearn.cluster import KMeans
        
        # Get all embeddings
        images_data = db.get_all_embeddings(user_id)
        
        if not images_data or len(images_data) < 2:
            return {
                "success": False,
                "message": "Need at least 2 images to cluster"
            }
        
        # Extract embeddings
        embeddings = []
        image_ids = []
        
        for img_data in images_data:
            img_id = img_data.get("id")
            clip_vec = img_data.get("clip_vec")
            
            if clip_vec:
                clip_vec_parsed = db.parse_json_array(clip_vec)
                if clip_vec_parsed is not None:
                    emb_array = np.array(clip_vec_parsed, dtype=np.float32)
                    if emb_array.shape == (512,):
                        embeddings.append(emb_array)
                        image_ids.append(img_id)
        
        if len(embeddings) < 2:
            return {
                "success": False,
                "message": "Need at least 2 valid embeddings to cluster"
            }
        
        # Perform clustering
        embeddings_matrix = np.stack(embeddings).astype(np.float32)
        
        # Determine optimal number of clusters using silhouette score
        from sklearn.metrics import silhouette_score
        
        n_samples = len(embeddings)
        if n_samples < 4:
            n_clusters = max(1, n_samples - 1)
        else:
            # Try different k values and pick the one with best silhouette score
            max_k = max(2, min(10, int(np.sqrt(n_samples) / 2)))
            best_k = 2
            best_score = -1
            
            # For small datasets, use a simple heuristic
            if n_samples < 20:
                n_clusters = max(2, min(4, n_samples // 5))
            else:
                # Try k values from 2 to max_k
                for k in range(2, max_k + 1):
                    try:
                        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)  # type: ignore
                        labels_test = kmeans_test.fit_predict(embeddings_matrix)
                        # Calculate silhouette score (only if we have enough samples)
                        if len(set(labels_test)) > 1:  # Need at least 2 clusters
                            score = silhouette_score(embeddings_matrix, labels_test)
                            if score > best_score:
                                best_score = score
                                best_k = k
                    except Exception:
                        continue
                
                n_clusters = best_k
                print(f"Optimal number of clusters determined: {n_clusters} (silhouette score: {best_score:.3f})")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # type: ignore
        cluster_labels = kmeans.fit_predict(embeddings_matrix)
        
        # Update cluster assignments
        updated_count = 0
        failed_count = 0
        print(f"Recomputing clusters: {len(image_ids)} images, {n_clusters} clusters")
        print(f"Cluster label range: {cluster_labels.min()} to {cluster_labels.max()}")
        
        for img_id, cluster_id in zip(image_ids, cluster_labels):
            try:
                result = db.update_image_cluster(img_id, int(cluster_id))
                if result is not None:  # None means column doesn't exist
                    updated_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"Error updating cluster for {img_id}: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
        
        print(f"Re-clustering complete: {updated_count} updated, {failed_count} failed/skipped")
        
        return {
            "success": True,
            "message": f"Re-clustered {updated_count} images into {n_clusters} clusters",
            "n_clusters": n_clusters,
            "n_images": updated_count,
            "failed": failed_count
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error recomputing clusters: {str(e)}")


@app.get("/data/export")
async def export_data(user_id: str = Depends(get_user_id)):
    """
    Export all images for the current user.
    Returns images with 3D positions and metadata in format frontend expects.
    """
    try:
        # Fetch all images for current user (including cluster info)
        images_data = db.get_all_images_for_export(user_id)
        
        if not images_data:
            return {
                "coords": {"points": []},
                "meta": {},
                "graph": {"edges": []}
            }
        
        # Build points array
        points = []
        meta = {}
        
        for img_data in images_data:
            if img_data is None:
                continue

            img_id = img_data["id"]
            position_vec = img_data.get("position_vec")
            
            if position_vec:
                # Parse position vector
                pos = db.parse_json_array(position_vec)
                if pos is None or len(pos) < 3:
                    pos = [0, 0, 0]
                
                if len(pos) >= 3:
                    points.append({
                        "id": img_id,
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2])
                    })
                    
                    # Extract filename from URL
                    image_url = img_data.get("image_url", "")
                    # For Supabase Storage URLs, extract filename from URL path
                    if image_url:
                        if "/" in image_url:
                            filename = image_url.split("/")[-1].split("?")[0]  # Remove query params
                        else:
                            filename = image_url
                    else:
                        filename = "Unknown"
                    
                    # Use Supabase Storage URL directly (already a full URL)
                    thumb_url = image_url if image_url else ""
                    
                    # Get cluster ID (default to 0 if not set)
                    cluster_id = img_data.get("cluster_id")
                    if cluster_id is None:
                        cluster_id = 0
                    else:
                        try:
                            cluster_id = int(cluster_id)
                        except:
                            cluster_id = 0
                    
                    meta[img_id] = {
                        "filename": filename,
                        "thumb": thumb_url,
                        "labels": [],
                        "cluster": cluster_id,
                        "description": img_data.get("description")  # Include description if available
                    }
        
        # Build graph edges automatically
        try:
            from sklearn.neighbors import NearestNeighbors
            from scipy.sparse.csgraph import minimum_spanning_tree
            from scipy.sparse import csr_matrix
            
            # Get embeddings for graph building
            embeddings_data = db.get_all_embeddings(user_id)
            if embeddings_data and len(embeddings_data) >= 2:
                embeddings = []
                emb_image_ids = []
                
                for img_data in embeddings_data:
                    clip_vec = img_data.get("clip_vec")
                    if clip_vec:
                        clip_vec_parsed = db.parse_json_array(clip_vec)
                        if clip_vec_parsed is not None:
                            embeddings.append(np.array(clip_vec_parsed, dtype=np.float32))
                            emb_image_ids.append(img_data["id"])
                
                if len(embeddings) >= 2:
                    n_nodes = len(embeddings)
                    k = min(5, n_nodes - 1)
                    
                    # Build kNN graph
                    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
                    nn.fit(embeddings)
                    distances, indices = nn.kneighbors(embeddings)
                    
                    edge_set = set()
                    edge_weights = {}
                    
                    for i, (dists, nbrs) in enumerate(zip(distances, indices)):
                        for dist, nbr_idx in zip(dists[1:], nbrs[1:]):
                            similarity = 1 - dist
                            source_id = emb_image_ids[i]
                            target_id = emb_image_ids[nbr_idx]
                            edge_key = tuple(sorted([source_id, target_id]))
                            if edge_key not in edge_weights or similarity > edge_weights[edge_key]:
                                edge_weights[edge_key] = similarity
                                edge_set.add(edge_key)
                    
                    # Ensure connectivity with MST
                    distance_matrix = np.ones((n_nodes, n_nodes), dtype=np.float32)
                    id_to_idx = {img_id: idx for idx, img_id in enumerate(emb_image_ids)}
                    for (source_id, target_id), weight in edge_weights.items():
                        src_idx = id_to_idx[source_id]
                        tgt_idx = id_to_idx[target_id]
                        distance_matrix[src_idx, tgt_idx] = 1 - weight
                        distance_matrix[tgt_idx, src_idx] = 1 - weight
                    
                    mst = minimum_spanning_tree(csr_matrix(distance_matrix))
                    mst_dense = mst.toarray()
                    
                    for i in range(n_nodes):
                        for j in range(i + 1, n_nodes):
                            if mst_dense[i, j] > 0 or mst_dense[j, i] > 0:
                                edge_key = tuple(sorted([emb_image_ids[i], emb_image_ids[j]]))
                                if edge_key not in edge_set:
                                    dist = float(mst_dense[i, j] if mst_dense[i, j] > 0 else mst_dense[j, i])
                                    similarity = max(0.0, 1 - dist)
                                    edge_weights[edge_key] = similarity
                                    edge_set.add(edge_key)
                    
                    graph_edges = []
                    for (source_id, target_id) in edge_set:
                        weight = edge_weights.get((source_id, target_id), 0.5)
                        graph_edges.append({
                            "source": source_id,
                            "target": target_id,
                            "weight": float(weight)
                        })
                else:
                    graph_edges = []
            else:
                graph_edges = []
        except Exception as e:
            print(f"Error building graph in export: {e}")
            graph_edges = []
        
        return {
            "coords": {"points": points},
            "meta": meta,
            "graph": {"edges": graph_edges}
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@app.post("/describe/image/{image_id}")
async def describe_image(image_id: str, user_id: str = Depends(get_user_id)):
    """
    Generate a 2-3 sentence description of an image using Google Gemini.
    Caches the description in the database.
    """
    try:
        # Check if description already exists
        image_data = db.get_image_by_id(image_id, user_id)
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # If description already exists, return it
        existing_description = image_data.get("description")
        if existing_description:
            return {
                "success": True,
                "description": existing_description,
                "cached": True
            }
        
        # Get fresh signed URL for the image (don't use stored URL as it may be expired or public)
        # Generate a fresh signed URL that works with private buckets
        try:
            signed_url = db.get_signed_url_for_image(image_id, user_id, expires_in=3600)
            if not signed_url:
                raise HTTPException(status_code=404, detail="Could not generate image URL")
            image_url = signed_url
            print(f"Generated fresh signed URL for description: {image_url[:100]}...")
        except Exception as e:
            print(f"Error generating signed URL for description: {e}")
            # Fallback to stored URL if signed URL generation fails
            image_url = image_data.get("image_url")
            if not image_url:
                raise HTTPException(status_code=404, detail="Image URL not found and could not generate signed URL")
            print(f"Using stored URL as fallback: {image_url[:100]}...")
        
        # Check if Google AI is configured
        if not GOOGLE_AI_API_KEY:
            raise HTTPException(
                status_code=500, 
                detail="Google AI API not configured. Please set GOOGLE_AI_API_KEY in environment variables."
            )
        
        # Fetch image from Supabase Storage URL
        try:
            # Fetch from Supabase Storage URL (now using fresh signed URL)
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_data_bytes = response.content
            print(f"Successfully fetched image for description (size: {len(image_data_bytes)} bytes)")
        except Exception as e:
            print(f"Error loading image for description: {e}")
            print(f"Failed URL: {image_url}")
            raise HTTPException(status_code=500, detail=f"Failed to load image: {str(e)}")
        
        # Call Google Gemini API
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=GOOGLE_AI_API_KEY)
            
            # List available models first to see what's accessible
            try:
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                print(f"Available models: {available_models}")
            except Exception as e:
                print(f"Could not list models: {e}")
            
            # Try vision-capable models - use models that are actually available
            # Based on the available models list, use these in order of preference
            model_names = [
                'gemini-2.0-flash',      # Latest flash model
                'gemini-2.0-flash-001',  # Specific version
                'gemini-2.5-flash',      # 2.5 flash version
                'gemini-2.5-pro',        # 2.5 pro version
                'gemini-flash-latest',   # Latest flash (alias)
                'gemini-pro-latest'      # Latest pro (alias)
            ]
            
            model = None
            last_error = None
            
            for model_name in model_names:
                try:
                    # Check if model supports vision by checking its input modalities
                    model_info = genai.get_model(model_name)
                    print(f"Trying model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    # If we get here, model was created successfully
                    break
                except Exception as e:
                    last_error = e
                    print(f"Model {model_name} not available: {e}")
                    continue
            
            if model is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"None of the Gemini models are available. Last error: {str(last_error)}. Please check your API key and model access. Make sure you're using a valid Google AI Studio API key with vision model access."
                )
            
            # Prepare the image - convert to base64 for Gemini API
            import base64
            from PIL import Image
            from io import BytesIO
            
            # Open and potentially resize image if too large
            img = Image.open(BytesIO(image_data_bytes))
            
            # Convert image to base64
            buffered = BytesIO()
            # Save as JPEG to reduce size
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Generate description with a narrative, story-telling prompt
            prompt = """Please write a brief, engaging 2-3 sentence description of this image in a narrative, story-telling style. 
Write it as if you are a narrator describing a scene or moment. Use phrases like "Here we have..." or "This image shows..." to begin.
Be descriptive and highlight interesting visual elements, but maintain a serious, thoughtful tone. 
Avoid casual phrases like "Check out this pic!" or overly enthusiastic language. 
Write as if you are telling a story about what is happening in the image. Keep it concise and clear."""
            
            # Generate content with image and prompt
            # Use the image data directly (PIL Image object works with Gemini)
            try:
                response = model.generate_content([prompt, img])
                
                # Extract the description
                if hasattr(response, 'text') and response.text:
                    description = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    # Sometimes response structure is different
                    description = response.candidates[0].content.parts[0].text.strip()
                else:
                    raise HTTPException(status_code=500, detail="Gemini API returned empty response")
            except Exception as e:
                error_msg = str(e)
                if "not supported for generative content" in error_msg or "model not found" in error_msg.lower():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Model does not support vision/image input. Error: {error_msg}. Please ensure you're using a vision-capable model like gemini-1.5-pro or gemini-1.5-flash."
                    )
                raise
            
            if description:
                # Save description to database
                db.update_image_description(image_id, description)
                
                return {
                    "success": True,
                    "description": description,
                    "cached": False
                }
            else:
                raise HTTPException(status_code=500, detail="Google Gemini did not return a description")
            
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Google Generative AI package not installed. Run: pip install google-generativeai"
            )
        except Exception as e:
            print(f"Error calling Google Gemini API: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error describing image: {str(e)}")


@app.get("/narrate/image/{image_id}")
async def narrate_image(image_id: str, user_id: str = Depends(get_user_id)):
    """
    Generate audio narration of an image description using ElevenLabs.
    Returns the audio as a streaming response.
    """
    try:
        # Get image data and description
        image_data = db.get_image_by_id(image_id, user_id)
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get description
        description = image_data.get("description")
        if not description:
            raise HTTPException(status_code=404, detail="Image description not found. Please generate a description first.")
        
        # Check for cached audio URL first
        cached_audio_url = db.get_image_audio_url(image_id, user_id)
        if cached_audio_url:
            print(f"Found cached audio URL for {image_id}, returning cached audio")
            # Return the cached audio file from Supabase Storage
            try:
                # Fetch the cached audio
                response = requests.get(cached_audio_url, timeout=10)
                if response.status_code == 200:
                    from fastapi.responses import StreamingResponse
                    from io import BytesIO
                    audio_stream = BytesIO(response.content)
                    return StreamingResponse(
                        audio_stream,
                        media_type="audio/mpeg",
                        headers={
                            "Content-Disposition": f"inline; filename=narration_{image_id}.mp3",
                            "Cache-Control": "public, max-age=86400"  # Cache for 24 hours
                        }
                    )
                else:
                    print(f"Cached audio URL returned {response.status_code}, generating fresh audio")
                    # Fall through to generate fresh audio
            except Exception as e:
                print(f"Error fetching cached audio: {e}, generating fresh audio")
                # Fall through to generate fresh audio
        
        # Check if ElevenLabs is configured
        if not ELEVENLABS_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="ElevenLabs API not configured. Please set ELEVENLABS_API_KEY in environment variables."
            )
        
        # Generate audio using ElevenLabs
        try:
            from elevenlabs import ElevenLabs
            
            # Initialize client with API key
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            
            print(f"Generating audio for description (length: {len(description)} chars)")
            print(f"Using voice ID: {ELEVENLABS_VOICE_ID}")
            
            # Generate audio from description using the new API
            # The text_to_speech.convert method returns a generator of bytes
            try:
                audio_generator = client.text_to_speech.convert(
                    voice_id=ELEVENLABS_VOICE_ID,
                    text=description,
                    model_id="eleven_monolingual_v1"  # Optional: specify model
                )
                
                # Collect all audio bytes from the generator
                audio_bytes = b"".join(audio_generator)
                print(f"Successfully generated audio (size: {len(audio_bytes)} bytes)")
                
                # Cache audio file in Supabase Storage
                try:
                    # Use db module's upload function to store audio
                    audio_storage_id = f"{image_id}_audio"
                    cached_audio_url = db.upload_image_to_storage(
                        image_id=audio_storage_id,
                        file_content=audio_bytes,
                        file_ext=".mp3",
                        user_id=user_id
                    )
                    
                    # Save audio URL to database for future use
                    if cached_audio_url:
                        db.update_image_audio_url(image_id, cached_audio_url)
                        print(f"Cached audio file at: {cached_audio_url[:100]}...")
                except Exception as cache_error:
                    print(f"Warning: Could not cache audio file: {cache_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue without caching - audio will still be returned
            except Exception as api_error:
                error_str = str(api_error)
                print(f"Error with eleven_monolingual_v1: {error_str}")
                
                # Try without specifying model (uses default)
                try:
                    print("Trying without model parameter...")
                    audio_generator = client.text_to_speech.convert(
                        voice_id=ELEVENLABS_VOICE_ID,
                        text=description
                    )
                    audio_bytes = b"".join(audio_generator)
                    print(f"Successfully generated audio with default model (size: {len(audio_bytes)} bytes)")
                except Exception as api_error2:
                    error_str2 = str(api_error2)
                    print(f"Error with default model: {error_str2}")
                    # Re-raise with more context
                    raise Exception(f"ElevenLabs API error. First attempt: {error_str}. Second attempt: {error_str2}")
            
            # Return audio as streaming response
            from fastapi.responses import StreamingResponse
            from io import BytesIO
            
            # Convert bytes to BytesIO for streaming
            audio_stream = BytesIO(audio_bytes)
            
            return StreamingResponse(
                audio_stream,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f"inline; filename=narration_{image_id}.mp3",
                    "Cache-Control": "public, max-age=3600"  # Cache for 1 hour
                }
            )
            
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="ElevenLabs package not installed. Run: pip install elevenlabs"
            )
        except Exception as e:
            error_msg = str(e)
            print(f"Error calling ElevenLabs API: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Provide more helpful error messages
            if "401" in error_msg or "Unauthorized" in error_msg or "Invalid API key" in error_msg:
                raise HTTPException(
                    status_code=500,
                    detail="ElevenLabs API key is invalid or unauthorized. Please check your ELEVENLABS_API_KEY in .env file."
                )
            elif "404" in error_msg or "voice" in error_msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail=f"ElevenLabs voice ID not found. Please check your ELEVENLABS_VOICE_ID. Error: {error_msg}"
                )
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail="ElevenLabs API quota exceeded. Please check your account limits."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate narration: {error_msg}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error narrating image: {str(e)}")


@app.get("/image/{image_id}/url")
async def get_image_url(image_id: str, expires_in: int = 3600, user_id: str = Depends(get_user_id)):
    """
    Get a fresh signed URL for an image (useful when URLs expire).
    Returns a new signed URL that expires after expires_in seconds (default 1 hour).
    """
    try:
        print(f"Generating fresh signed URL for image {image_id}, user {user_id}, expires_in={expires_in}")
        signed_url = db.get_signed_url_for_image(image_id, user_id, expires_in)
        if not signed_url:
            raise HTTPException(status_code=404, detail="Image not found or could not generate URL")
        print(f"Successfully generated URL: {signed_url[:100]}...")
        return {
            "success": True,
            "url": signed_url,
            "expires_in": expires_in
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating image URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image URL: {str(e)}")


@app.get("/stats")
async def get_stats(user_id: str = Depends(get_user_id)):
    """Get statistics about the dataset"""
    try:
        total_images = db.get_image_count(user_id)
        images_with_3d = db.get_images_with_3d_count(user_id)
        
        return {
            "total_images": total_images,
            "images_with_3d": images_with_3d,
            "pca_fitted": PCA_MODEL_PATH.exists()
        }
    
    except Exception as e:
        return {
            "total_images": 0,
            "images_with_3d": 0,
            "pca_fitted": False,
            "error": str(e)
        }


# Images are now served from Supabase Storage, no local file serving needed

# Optional: serve a /static route for additional files
if (ROOT / "static").exists():
    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    
    # On Windows, disable reload to avoid multiprocessing issues
    # You can manually restart the server when code changes
    use_reload = sys.platform != "win32"
    
    # Get PORT from environment, handle empty string case
    port_str = os.getenv("PORT", "8001")
    port = int(port_str) if port_str and port_str.strip() else 8001
    
    uvicorn.run(
        app,  # Pass app directly instead of string to avoid import issues
        host="0.0.0.0",
        port=port,
        reload=use_reload
    )
