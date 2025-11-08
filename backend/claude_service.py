"""
Local Claude-like stub service

This module provides a drop-in replacement for the original AWS Bedrock-backed
`claude_service.py`. It intentionally does NOT require `boto3` or any AWS
credentials and returns reasonable, deterministic local responses. Use this
for local development when you don't want to call Bedrock / Claude.
"""

import json
from typing import List, Dict, Optional


class ClaudeService:
    """Lightweight local stub for cluster naming and simple NL processing."""

    def __init__(self, model_id: str = "local-stub", region: str = "local"):
        self.model_id = model_id
        self.region = region
        print(f"🧠 Using local Claude stub (model={model_id})")

    def name_cluster(
        self,
        top_objects: List[Dict[str, float]],
        color_profile: Optional[str] = None,
        sample_captions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        # Build a short title from top object names
        names = [obj.get("name", "object") for obj in top_objects[:4]]
        title = " / ".join(names) if names else "Unknown Cluster"
        # Short description
        desc = f"Images featuring {', '.join(names[:3])}."
        if color_profile:
            desc += f" Color profile: {color_profile}."
        return {"title": title[:80], "description": desc}

    def explain_cluster(
        self,
        top_objects: List[Dict[str, float]],
        color_profile: Optional[str] = None,
        num_images: int = 0,
    ) -> str:
        names = [obj.get("name", "object") for obj in top_objects[:2]]
        base = f"Grouped because they share visual elements like {', '.join(names)}"
        if color_profile:
            base += f" and {color_profile} tones"
        if num_images:
            base += f" across {num_images} images"
        return base + "."

    def process_nl_query(self, query: str) -> Dict[str, any]:
        # Very small heuristic parser: return query as context and empty lists
        return {"objects": [], "colors": [], "context": query}

    def generate_guided_tour(self, clusters: List[Dict]) -> str:
        if not clusters:
            return "No clusters available for a tour."
        titles = [c.get("title", "cluster") for c in clusters[:3]]
        return " → ".join(titles) + ": explore these visual themes."


_claude_service: Optional[ClaudeService] = None


def get_claude_service(model_id: str = "local-stub", region: str = "local") -> ClaudeService:
    global _claude_service
    if _claude_service is None:
        _claude_service = ClaudeService(model_id=model_id, region=region)
    return _claude_service

