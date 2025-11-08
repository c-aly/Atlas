# from fastapi import FastAPI
# from fastapi.responses import HTMLResponse
# from dotenv import load_dotenv



# # create FastAPI app
# app = FastAPI(title="Atlas")

# @app.get("/", response_class=HTMLResponse)
# def home():
#     return """
#     <html>
#       <head><title>Home</title></head>
#       <body style="font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;">
#         <h1>Hello from FastAPI 👋</h1>
#       </body>
#     </html>
#     """



# if __name__=="__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)



"""
Atlas of Images - Main FastAPI Application
Handles image upload, embedding, projection, and serving the neural map data
"""


from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import json
import uuid
from pathlib import Path
from PIL import Image
import io
import os
from dotenv import load_dotenv


# Import services
from clip_service import get_clip_service
from umap_service import get_umap_service
from knn_service import get_knn_service
# from claude_service import get_claude_service


# Load environment variables
load_dotenv()


# Initialize FastAPI app
app = FastAPI(
   title="Atlas of Images API",
   description="3D neural map visualization of image embeddings",
   version="1.0.0"
)


# CORS middleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:5176").split(","),
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)


# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)


# Global state
image_database = {}  # {image_id: {path, embedding, coords, metadata}}
is_projection_fitted = False


# Pydantic models
class SearchRequest(BaseModel):
   query: str
   k: int = 10


class ClusterRequest(BaseModel):
   cluster_ids: List[str]


class EmbeddingResponse(BaseModel):
   image_id: str
   embedding_dim: int
   message: str




# Initialize services on startup
@app.on_event("startup")
async def startup_event():
   """Initialize ML services"""
   global clip_service, umap_service, knn_service, claude_service
  
   print("🚀 Starting Atlas of Images API...")
  
   # Initialize CLIP
   clip_model = os.getenv("CLIP_MODEL", "ViT-B-32")
   clip_pretrained = os.getenv("CLIP_PRETRAINED", "openai")
   clip_service = get_clip_service(clip_model, clip_pretrained)
  
   # Initialize UMAP
   umap_n_components = int(os.getenv("UMAP_N_COMPONENTS", "3"))
   umap_n_neighbors = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
   umap_min_dist = float(os.getenv("UMAP_MIN_DIST", "0.1"))
   umap_metric = os.getenv("UMAP_METRIC", "cosine")
   umap_service = get_umap_service(umap_n_components, umap_n_neighbors, umap_min_dist, umap_metric)
  
   # Initialize kNN
   knn_service = get_knn_service(clip_service.embedding_dim, metric="cosine")
  
   # Initialize Claude (optional - will gracefully fail if AWS not configured)
   bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
   aws_region = os.getenv("AWS_REGION", "us-east-1")
  #  claude_service = get_claude_service(bedrock_model_id, aws_region)
  
   print("✅ All services initialized")




@app.get("/")
async def root():
   """Health check endpoint"""
   return {
       "status": "online",
       "service": "Atlas of Images API",
       "total_images": len(image_database),
       "projection_fitted": is_projection_fitted
   }




@app.post("/upload", response_model=EmbeddingResponse)
async def upload_image(file: UploadFile = File(...)):
   """
   Upload an image and generate its CLIP embedding
   """
   try:
       # Generate unique ID
       image_id = str(uuid.uuid4())
      
       # Read image
       contents = await file.read()
       image = Image.open(io.BytesIO(contents)).convert("RGB")
      
       # Save image
       image_path = IMAGES_DIR / f"{image_id}.jpg"
       image.save(image_path, "JPEG", quality=85)
      
       # Generate embedding
       embedding = clip_service.embed_image(image)
       print(f"📊 512-D Vector for {file.filename}:")
       print(embedding)
       print("\n")
       # Store in database
       image_database[image_id] = {
           "path": str(image_path),
           "filename": file.filename,
           "embedding": embedding,
           "coords": None,
           "metadata": {}
       }
      
       # Add to kNN index
       knn_service.add_embeddings(embedding.reshape(1, -1), [image_id])
      
       print(f"✅ Uploaded: {image_id} ({file.filename})")
      
       return EmbeddingResponse(
           image_id=image_id,
           embedding_dim=len(embedding),
           message=f"Successfully uploaded and embedded {file.filename}"
       )
      
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")




@app.post("/embed/batch")
async def embed_batch(files: List[UploadFile] = File(...)):
   """
   Upload and embed multiple images
   """
   results = []
  
   for file in files:
       try:
           result = await upload_image(file)
           results.append({"image_id": result.image_id, "status": "success"})
       except Exception as e:
           results.append({"filename": file.filename, "status": "error", "message": str(e)})
  
   return {"processed": len(results), "results": results}




@app.post("/project")
async def project_to_3d():
   """
   Project all embeddings to 3D coordinates using UMAP
   """
   global is_projection_fitted
  
   if len(image_database) < 2:
       raise HTTPException(status_code=400, detail="Need at least 2 images to project")
  
   try:
       # Gather all embeddings
       image_ids = list(image_database.keys())
       embeddings = np.array([image_database[img_id]["embedding"] for img_id in image_ids])
      
       print(f"🗺️  Projecting {len(image_ids)} images to 3D...")
      
       # Fit and transform
       coords = umap_service.fit_transform(embeddings)
      
       # Normalize coordinates for better visualization
       coords_normalized = umap_service.normalize_coords(coords, scale=10.0)
      
       # Store coordinates
       for i, image_id in enumerate(image_ids):
           image_database[image_id]["coords"] = coords_normalized[i].tolist()
      
       is_projection_fitted = True
      
       # Save coordinates to JSON
       coords_data = {
           "projection": "umap-3d-cosine",
           "points": [
               {"id": img_id, "x": coords_normalized[i][0], "y": coords_normalized[i][1], "z": coords_normalized[i][2]}
               for i, img_id in enumerate(image_ids)
           ]
       }
      
       coords_path = DATA_DIR / "coords.json"
       with open(coords_path, 'w') as f:
           json.dump(coords_data, f, indent=2)
      
       print(f"✅ Projection complete, saved to {coords_path}")
      
       return {
           "status": "success",
           "num_images": len(image_ids),
           "coords_file": str(coords_path)
       }
      
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Projection error: {str(e)}")




@app.post("/graph")
async def build_graph(k: int = 8):
   """
   Build k-NN similarity graph
   """
   if len(image_database) < 2:
       raise HTTPException(status_code=400, detail="Need at least 2 images to build graph")
  
   try:
       # Build graph
       graph = knn_service.build_knn_graph(k=k)
      
       # Save to file
       graph_path = DATA_DIR / "graph.json"
       knn_service.save_graph(graph, str(graph_path))
      
       return {
           "status": "success",
           "num_nodes": len(graph),
           "graph_file": str(graph_path)
       }
      
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Graph building error: {str(e)}")




@app.post("/search")
async def search_by_text(request: SearchRequest):
   """
   Search for similar images using natural language text query
   """
   try:
       # Generate text embedding
       text_embedding = clip_service.embed_text(request.query)
      
       # Search for nearest neighbors
       neighbor_ids, distances = knn_service.search(text_embedding, k=request.k)
      
       # Build results
       results = []
       for img_id, distance in zip(neighbor_ids, distances):
           if img_id in image_database:
               results.append({
                   "image_id": img_id,
                   "similarity": float(distance),
                   "coords": image_database[img_id].get("coords"),
                   "filename": image_database[img_id].get("filename")
               })
      
       return {
           "query": request.query,
           "results": results
       }
      
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")




@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), k: int = 10):
   """
   Search for similar images using an uploaded image
   """
   try:
       # Read and embed query image
       contents = await file.read()
       query_embedding = clip_service.embed_image(contents)
      
       # Search
       neighbor_ids, distances = knn_service.search(query_embedding, k=k)
      
       # Build results
       results = []
       for img_id, distance in zip(neighbor_ids, distances):
           if img_id in image_database:
               results.append({
                   "image_id": img_id,
                   "similarity": float(distance),
                   "coords": image_database[img_id].get("coords"),
                   "filename": image_database[img_id].get("filename")
               })
      
       return {"results": results}
      
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Image search error: {str(e)}")




# @app.post("/clusters/label")
# async def label_clusters(request: ClusterRequest):
#    """
#    Use Claude to generate labels for image clusters
#    """
#    try:
#        # This is a simplified version - in production, you'd analyze the actual clusters
#        results = []
      
#        for cluster_id in request.cluster_ids:
#            # Mock cluster data - replace with actual cluster analysis
#            cluster_label = claude_service.name_cluster(
#                top_objects=[
#                    {"name": "example", "confidence": 0.8}
#                ],
#                color_profile="neutral",
#                sample_captions=None
#            )
          
#            results.append({
#                "cluster_id": cluster_id,
#                "title": cluster_label["title"],
#                "description": cluster_label["description"]
#            })
      
#        return {"clusters": results}
      
#    except Exception as e:
#        raise HTTPException(status_code=500, detail=f"Cluster labeling error: {str(e)}")




@app.get("/data/export")
async def export_data():
   """
   Export all data (coords, graph, metadata) for frontend
   """
   try:
       # Build metadata
       meta = {}
       for img_id, data in image_database.items():
           # Generate thumbnail URL (in production, upload to S3)
           thumb_url = f"/images/{img_id}.jpg"
          
           meta[img_id] = {
               "thumb": thumb_url,
               "filename": data.get("filename", "unknown"),
               "coords": data.get("coords"),
               "labels": data.get("metadata", {}).get("labels", [])
           }
      
       # Save metadata
       meta_path = DATA_DIR / "meta.json"
       with open(meta_path, 'w') as f:
           json.dump(meta, f, indent=2)
      
       # Read coords and graph
       coords_data = {}
       graph_data = {}
      
       coords_path = DATA_DIR / "coords.json"
       if coords_path.exists():
           with open(coords_path, 'r') as f:
               coords_data = json.load(f)
      
       graph_path = DATA_DIR / "graph.json"
       if graph_path.exists():
           with open(graph_path, 'r') as f:
               graph_data = json.load(f)
      
       return {
           "coords": coords_data,
           "graph": graph_data,
           "meta": meta
       }
      
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")




@app.get("/stats")
async def get_stats():
   """Get statistics about the current dataset"""
   return {
       "total_images": len(image_database),
       "projection_fitted": is_projection_fitted,
       "knn_stats": knn_service.get_stats(),
       "embedding_dim": clip_service.embedding_dim
   }




if __name__ == "__main__":
   import uvicorn
  
   host = os.getenv("API_HOST", "0.0.0.0")
   port = int(os.getenv("API_PORT", "8000"))
  
   uvicorn.run(app, host=host, port=port, log_level="info")




