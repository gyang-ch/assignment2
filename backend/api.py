"""
Modal deployment for the Sketch-Based Image Retrieval (SBIR) API.

Setup (one-time):
  modal volume create sbir-embeddings
  modal volume put sbir-embeddings embeddings/embeddings_edges_xdog.npz /embeddings_edges_xdog.npz
  modal volume put sbir-embeddings embeddings/metadata.json /metadata.json

Deploy:
  modal deploy backend/api.py

The deployed URL will look like:
  https://<your-username>--sketch-art-sbir-sbirservice-web.modal.run
"""

import io
import json
import random
import time
from pathlib import Path

import modal

# ── Image ─────────────────────────────────────────────────────────────────────

def _download_dinov2():
    """Pre-download DINOv2 weights into the image layer."""
    import torch
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision",
        "numpy", "Pillow",
        "fastapi", "uvicorn[standard]", "python-multipart", "requests",
    )
    .run_function(_download_dinov2)
)

# ── Volume for embeddings ──────────────────────────────────────────────────────

volume = modal.Volume.from_name("sbir-embeddings", create_if_missing=True)
EMBEDDINGS_DIR = Path("/embeddings")

# ── Modal app ─────────────────────────────────────────────────────────────────

app = modal.App("sketch-art-sbir")

# ── Service class ─────────────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="T4",
    volumes={EMBEDDINGS_DIR: volume},
    scaledown_window=300,   # keep warm for 5 min after last request
)
class SBIRService:

    @modal.enter()
    def load(self):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        # ── Device ────────────────────────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("[device] CPU")

        # ── DINOv2 model (auto-detect size from embedding dim) ─────────────────
        edge_path = EMBEDDINGS_DIR / "embeddings_edges_xdog.npz"
        sample = np.load(edge_path, allow_pickle=True)
        dim = sample["embeddings"].shape[1]
        DIM_TO_MODEL = {
            384:  "dinov2_vits14",
            768:  "dinov2_vitb14",
            1024: "dinov2_vitl14",
            1536: "dinov2_vitg14",
        }
        model_name = DIM_TO_MODEL.get(dim)
        if not model_name:
            raise ValueError(f"Unexpected embedding dim {dim}")
        print(f"[model] Loading {model_name} (dim={dim}) …")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False
        )
        self.model.eval().to(self.device)
        print("[model] Ready")

        # ── Transform ─────────────────────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # ── Embeddings index ──────────────────────────────────────────────────
        print(f"[index] Loading {edge_path} …")
        edge_data = sample  # reuse the npz already loaded above
        self.uids = list(edge_data["uids"])
        embeddings = torch.from_numpy(edge_data["embeddings"]).to(self.device)
        self.embeddings = F.normalize(embeddings, dim=-1)
        self.uid_to_idx = {uid: i for i, uid in enumerate(self.uids)}

        meta_path = EMBEDDINGS_DIR / "metadata.json"
        print(f"[index] Loading {meta_path} …")
        with open(meta_path) as f:
            meta_list = json.load(f)
        self.meta_list   = meta_list
        self.meta_by_uid = {m["uid"]: m for m in meta_list}

        print(f"[index] {len(self.uids)} artworks indexed "
              f"(dim={self.embeddings.shape[1]})")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _embed(self, img):
        """L2-normalised DINOv2 CLS embedding. Shape: (1, D)."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = F.normalize(self.model(tensor), dim=-1)
        return feat.cpu().float().numpy()

    def _preprocess_sketch(self, img):
        """Invert sketch (black strokes on white → white edges on black) to match XDoG domain."""
        from PIL import Image
        gray = img.convert("L")
        inv  = Image.eval(gray, lambda px: 255 - px)
        return inv.convert("RGB")

    def _search(self, query_emb, top_k: int):
        import torch
        q = torch.from_numpy(query_emb).to(self.device)
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

    # ── FastAPI app ────────────────────────────────────────────────────────────

    @modal.asgi_app()
    def web(self):
        import requests as http_requests
        from fastapi import FastAPI, File, Form, Query, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, Response
        from PIL import Image

        api = FastAPI(title="Sketch → Art SBIR API")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/api/health")
        async def health():
            return {
                "status": "ok",
                "artworks": len(self.uids),
                "device": str(self.device),
            }

        @api.post("/api/search")
        async def search(
            sketch: UploadFile = File(...),
            top_k: int = Form(12),
        ):
            t0  = time.time()
            raw = await sketch.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            processed  = self._preprocess_sketch(img)
            query_emb  = self._embed(processed)
            top_k      = max(1, min(top_k, 100))
            results    = self._search(query_emb, top_k)
            elapsed    = time.time() - t0
            return JSONResponse({
                "results":    results,
                "elapsed_ms": round(elapsed * 1000, 1),
            })

        @api.post("/api/compare")
        async def compare(
            sketch: UploadFile = File(...),
            uid: str = Form(...),
        ):
            raw = await sketch.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            processed = self._preprocess_sketch(img)
            query_emb = self._embed(processed)
            idx = self.uid_to_idx.get(uid)
            if idx is None:
                return JSONResponse({"error": f"Unknown uid: {uid}"}, status_code=404)
            import torch
            q   = torch.from_numpy(query_emb).to(self.device)
            sim = (self.embeddings[idx] @ q.squeeze()).item()
            return JSONResponse({"uid": uid, "similarity": round(float(sim), 4)})

        @api.get("/api/artworks/search")
        async def search_artworks(
            q: str = Query(..., min_length=1),
            limit: int = Query(20, ge=1, le=100),
        ):
            ql = q.lower()
            results = []
            for m in self.meta_list:
                if (ql in m.get("title", "").lower()
                        or ql in m.get("artist", "").lower()):
                    if m.get("image_url"):
                        results.append(self._meta_dict(m))
                        if len(results) >= limit:
                            break
            return JSONResponse({"results": results})

        @api.get("/api/artworks/random")
        async def random_artworks(
            count: int = Query(12, ge=1, le=50),
        ):
            sample = random.sample(self.meta_list, min(count, len(self.meta_list)))
            results = [self._meta_dict(m) for m in sample if m.get("image_url")]
            return JSONResponse({"results": results})

        @api.get("/api/proxy-image")
        async def proxy_image(url: str = Query(...)):
            try:
                resp = http_requests.get(
                    url, timeout=15, stream=True,
                    headers={"User-Agent": "SketchArt/1.0"},
                )
                resp.raise_for_status()
                ct = resp.headers.get("content-type", "image/jpeg")
                return Response(content=resp.content, media_type=ct)
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=502)

        return api
