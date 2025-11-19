Inspiration
As curious college students, we set out to create an engaging way to explore how images relate to one another. Atlas visualizes image relationships in 3D. We map images by semantic similarity using CLIP embeddings, then project them to 3D with PCA. Similar images cluster together, making it easier to explore and understand visual relationships.

What it does
Atlas is a 3D neural map of visual space that:

Uploads images and generates CLIP embeddings (512‑D).
Projects embeddings to 3D using PCA
Clusters images with K‑means (optimal k via silhouette score).
Builds a kNN graph and augments it with an MST for global connectivity.
Visualizes the space in an interactive 3D scene (React Three Fiber).
Supports search and exploration across clusters and neighbors.
How we built it
Backend (Python/FastAPI)
FastAPI for REST endpoints.
CLIP (OpenAI CLIP‑ViT‑Base‑Patch32) for embeddings.
PCA for 512D → 3D projection.
K‑means clustering with silhouette score optimization.
kNN + MST for graph construction.
Supabase (PostgreSQL) for storage.
JWT authentication via Supabase.
Frontend (React)
React 18 with Vite.
React Three Fiber + Three.js for 3D rendering.
Zustand for state management.
Tailwind CSS for styling.
Axios for API calls.
Interactive controls: orbit, zoom, node selection.
Architecture
Batch upload endpoint processes images end‑to‑end.
Stores embeddings, positions, and cluster assignments.
Real‑time 3D visualization with color‑coded clusters.
Challenges we ran into
We started with UMAP for dimensionality reduction, but processing time was too slow for our batch workflows. Switching to PCA (Principal Component Analysis) reduced our 512‑dimension vectors down to 3 with much better performance. Finding a good API for image description generation was also tricky; we ultimately settled on Gemini.

Accomplishments that we're proud of
End‑to‑end pipeline: upload → embed → project → cluster → visualize.
Smooth, interactive 3D visualization.
Automatic cluster optimization using silhouette scores.
Secure multi‑user support with JWT authentication.
Efficient batch processing for multiple images.
Graph structure combining kNN and MST for robust connectivity.
Modern stack: FastAPI, React, Three.js, Supabase.
What we learned
CLIP embeddings for semantic image similarity.
Dimensionality reduction with PCA for visualization.
Clustering with K‑means and silhouette score evaluation.
Graph construction with kNN and MST.
3D visualization with React Three Fiber.
Full‑stack integration: Python backend + React frontend.
Authentication and authorization with Supabase JWT.
Handling large ML models (CLIP) in production.
What's next for Atlas
We plan to build a “Spotify Wrapped”‑style experience that highlights your most popular clusters, surfacing representative images, cluster statistics, and exploration paths over time.
