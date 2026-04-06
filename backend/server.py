"""
Sketch-Based Image Retrieval (SBIR) backend server.

Loads pre-computed DINOv2 edge embeddings + metadata from local .npz / .json
files, serves a single /api/search endpoint that accepts a sketch image and
returns the most similar artworks by cosine similarity.

Device auto-detection: MPS (Apple Silicon) → CUDA → CPU.

Usage:
  # Mac (M4):
  python backend/server.py

  # RunPod (CUDA):
  python backend/server.py --device cuda

  # Custom paths / port:
  python backend/server.py --embeddings-dir ./embeddings --port 8000
"""

import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# HEIC/HEIF support (Apple photos, iPhones). No-op if not installed.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# FastAPI
import requests as http_requests
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[device] CPU")
    return dev

# ─────────────────────────────────────────────────────────────────────────────
# DINOv2
# ─────────────────────────────────────────────────────────────────────────────

DINOV2_MODELS = {
    "small": "dinov2_vits14",   # D = 384
    "base":  "dinov2_vitb14",   # D = 768
    "large": "dinov2_vitl14",   # D = 1024
    "giant": "dinov2_vitg14",   # D = 1536
}

DINOV2_DIM_TO_SIZE = {384: "small", 768: "base", 1024: "large", 1536: "giant"}

DINOV2_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_dinov2(size: str, device: torch.device) -> torch.nn.Module:
    name = DINOV2_MODELS[size]
    print(f"[model] Loading {name} …")
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    model.eval().to(device)
    print(f"[model] Ready on {device}")
    return model


@torch.no_grad()
def embed_image(model: torch.nn.Module, img: Image.Image,
                device: torch.device) -> np.ndarray:
    """L2-normalised DINOv2 CLS embedding. Shape: (1, D)."""
    tensor = DINOV2_TRANSFORM(img).unsqueeze(0).to(device)
    feat = F.normalize(model(tensor), dim=-1)
    return feat.cpu().float().numpy()

# ─────────────────────────────────────────────────────────────────────────────
# Sketch pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_sketch(img: Image.Image) -> Image.Image:
    """
    Convert a user sketch (black strokes on white) into the same domain as
    the stored XDoG edge maps (white edges on black background).

    Steps:
      1. Convert to grayscale
      2. Invert (black↔white) so strokes become white-on-black
      3. Back to RGB (DINOv2 expects 3 channels)
    """
    gray = img.convert("L")
    inv = Image.eval(gray, lambda px: 255 - px)
    return inv.convert("RGB")

# ─────────────────────────────────────────────────────────────────────────────
# Embedding index
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingIndex:
    """
    In-memory index of pre-computed embeddings.
    Search is brute-force cosine similarity (fast enough for ~50k vectors).
    """

    def __init__(self, embeddings_dir: Path, device: torch.device):
        self.device = device

        # Load edge embeddings (these are what we compare sketches against)
        edge_path = embeddings_dir / "embeddings_edges_xdog.npz"
        print(f"[index] Loading {edge_path} …")
        edge_data = np.load(edge_path, allow_pickle=True)
        self.uids       = list(edge_data["uids"])
        self.embeddings  = torch.from_numpy(edge_data["embeddings"]).to(device)
        # Already L2-normalised from the pipeline, but ensure it
        self.embeddings  = F.normalize(self.embeddings, dim=-1)

        # Build uid → index for fast lookup
        self.uid_to_idx = {uid: i for i, uid in enumerate(self.uids)}

        # Load metadata and build uid → metadata lookup
        meta_path = embeddings_dir / "metadata.json"
        print(f"[index] Loading {meta_path} …")
        with open(meta_path) as f:
            meta_list = json.load(f)
        self.meta_list   = meta_list
        self.meta_by_uid = {m["uid"]: m for m in meta_list}

        print(f"[index] {len(self.uids)} artworks indexed "
              f"(dim={self.embeddings.shape[1]})")

    def search(self, query_emb: np.ndarray, top_k: int = 12) -> list[dict]:
        """
        query_emb: shape (1, D), L2-normalised.
        Returns list of dicts with metadata + similarity score.
        """
        q = torch.from_numpy(query_emb).to(self.device)
        # Cosine similarity (both are L2-normalised → dot product)
        sims = (self.embeddings @ q.T).squeeze(-1)
        top_k = min(top_k, len(self.uids))
        scores, indices = torch.topk(sims, top_k)
        scores  = scores.cpu().numpy()
        indices = indices.cpu().numpy()

        results = []
        for score, idx in zip(scores, indices):
            uid  = self.uids[idx]
            meta = self.meta_by_uid.get(uid, {})
            results.append({
                "uid":        uid,
                "museum":     meta.get("museum", ""),
                "title":      meta.get("title", ""),
                "artist":     meta.get("artist", ""),
                "date":       meta.get("date", ""),
                "medium":     meta.get("medium", ""),
                "image_url":  meta.get("image_url", ""),
                "object_url": meta.get("object_url", ""),
                "score":      round(float(score), 4),
            })
        return results

    def compare(self, query_emb: np.ndarray, uid: str) -> float | None:
        """Cosine similarity between a query embedding and a specific artwork."""
        idx = self.uid_to_idx.get(uid)
        if idx is None:
            return None
        q = torch.from_numpy(query_emb).to(self.device)
        sim = (self.embeddings[idx] @ q.squeeze()).item()
        return round(float(sim), 4)

    def _meta_dict(self, meta: dict) -> dict:
        return {
            "uid":        meta.get("uid", ""),
            "museum":     meta.get("museum", ""),
            "title":      meta.get("title", ""),
            "artist":     meta.get("artist", ""),
            "date":       meta.get("date", ""),
            "medium":     meta.get("medium", ""),
            "image_url":  meta.get("image_url", ""),
            "object_url": meta.get("object_url", ""),
        }

    def random_artworks(self, count: int = 12) -> list[dict]:
        sample = random.sample(self.meta_list, min(count, len(self.meta_list)))
        return [self._meta_dict(m) for m in sample if m.get("image_url")]

    def search_text(self, query: str, limit: int = 20) -> list[dict]:
        q = query.lower()
        results = []
        for m in self.meta_list:
            if (q in m.get("title", "").lower()
                    or q in m.get("artist", "").lower()):
                if m.get("image_url"):
                    results.append(self._meta_dict(m))
                    if len(results) >= limit:
                        break
        return results

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Sketch → Art SBIR API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    device = get_device(args.device)

    # Determine model size from embedding dimensionality
    emb_dir = Path(args.embeddings_dir)
    sample = np.load(emb_dir / "embeddings_edges_xdog.npz", allow_pickle=True)
    dim = sample["embeddings"].shape[1]
    model_size = DINOV2_DIM_TO_SIZE.get(dim)
    if not model_size:
        raise ValueError(f"Unknown embedding dim {dim}, expected one of {list(DINOV2_DIM_TO_SIZE.keys())}")
    del sample

    model = load_dinov2(model_size, device)
    index = EmbeddingIndex(emb_dir, device)

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "artworks": len(index.uids),
            "device": str(device),
            "model": model_size,
        }

    @app.post("/api/search")
    async def search(
        sketch: UploadFile = File(...),
        top_k: int = Form(12),
    ):
        t0 = time.time()

        # Read and decode the sketch image
        raw = await sketch.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # Pre-process: invert to match XDoG domain (white-on-black)
        processed = preprocess_sketch(img)

        # Compute DINOv2 embedding
        query_emb = embed_image(model, processed, device)

        # Search
        top_k = max(1, min(top_k, 100))
        results = index.search(query_emb, top_k)

        elapsed = time.time() - t0
        return JSONResponse({
            "results": results,
            "elapsed_ms": round(elapsed * 1000, 1),
        })

    # ── Compare sketch against a specific artwork ────────────────────────
    @app.post("/api/compare")
    async def compare(
        sketch: UploadFile = File(...),
        uid: str = Form(...),
    ):
        raw = await sketch.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        processed = preprocess_sketch(img)
        query_emb = embed_image(model, processed, device)

        similarity = index.compare(query_emb, uid)
        if similarity is None:
            return JSONResponse({"error": f"Unknown uid: {uid}"}, status_code=404)

        return JSONResponse({"uid": uid, "similarity": similarity})

    # ── Text search over metadata ────────────────────────────────────────
    @app.get("/api/artworks/search")
    async def search_artworks(
        q: str = Query("", min_length=1),
        limit: int = Query(20, ge=1, le=100),
    ):
        return JSONResponse({"results": index.search_text(q, limit)})

    # ── Random artworks ──────────────────────────────────────────────────
    @app.get("/api/artworks/random")
    async def random_artworks(
        count: int = Query(12, ge=1, le=50),
    ):
        return JSONResponse({"results": index.random_artworks(count)})

    # ── Image proxy (CORS bypass for museum CDNs) ────────────────────────
    @app.get("/api/proxy-image")
    async def proxy_image(url: str = Query(...)):
        try:
            resp = http_requests.get(url, timeout=15, stream=True,
                                     headers={"User-Agent": "SketchArt/1.0"})
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "image/jpeg")
            return Response(content=resp.content, media_type=ct)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=502)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SBIR backend server")
    p.add_argument("--embeddings-dir", default="embeddings",
                   help="Path to directory with .npz and metadata.json")
    p.add_argument("--device", default=None,
                   help="Force device: cuda, mps, cpu (default: auto-detect)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
