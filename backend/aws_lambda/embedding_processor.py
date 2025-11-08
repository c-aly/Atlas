"""
AWS Lambda embedding processor - REMOVED

This file used to fetch images from S3, run CLIP/pytorch embedding code, and
store embeddings in DynamoDB. For local development the project should embed
images using the local `clip_service.py` via the running FastAPI backend.

If you need a local test harness for embedding generation, use
`backend/clip_service.py` or call the `/upload` endpoint on the local API.
"""

def lambda_handler(event, context):
    raise RuntimeError("AWS Lambda embedding processor is disabled for local development.")

