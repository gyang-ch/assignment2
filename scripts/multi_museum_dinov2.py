"""
Multi-Museum DINOv2 Embedding Pipeline  (with Edge-Map Embeddings + Azure Blob Storage)
=========================================================================================
Downloads artwork images from 6 museum open-access APIs, computes TWO sets of
DINOv2 embeddings per artwork, and stores everything on Azure Blob Storage (or
locally as a fallback).

  embeddings_rgb.npz         — full-colour DINOv2 embedding
  embeddings_edges_<m>.npz   — edge-map DINOv2 embedding  ← key SBIR trick

Edge methods:  xdog (default) | canny | sobel

Storage backends:
  Local  — default; saves to --output-dir on disk
  Azure  — enable with --azure-sas-url (or env var AZURE_BLOB_SAS_URL)
           Images and embeddings go to Azure Blob, minimal local disk use.

Azure layout  (container = artworks):
  images/{museum}/{id}.jpg
  edges/{method}/{museum}/{id}.jpg     (optional, if --save-edges)
  embeddings/embeddings_rgb.npz
  embeddings/embeddings_edges_{method}.npz
  embeddings/metadata.json
  embeddings/done_uids.json            ← resume checkpoint

Museums:
  met  rijks  cleveland  artic  vam  nga

Usage:
  # Local (default):
  python multi_museum_dinov2.py --museums met artic --limit 500

  # Azure — pass SAS URL via env var (recommended) or CLI flag:
  export AZURE_BLOB_SAS_URL="https://phytovision.blob.core.windows.net/?sv=..."
  python multi_museum_dinov2.py --museums all --no-limit --model large \\
      --batch-size 256 --device cuda --rijks-key YOUR_KEY

  # Azure — SAS URL inline (fine for RunPod one-off runs):
  python multi_museum_dinov2.py --museums all --limit 5000 \\
      --azure-sas-url "https://phytovision.blob.core.windows.net/?sv=..." \\
      --azure-container artworks

⚠  Never commit your SAS URL to git.  Use an env var or a .env file.
"""

import argparse
import csv
import io
import json
import os
import random
import time
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device(requested: Optional[str] = None) -> torch.device:
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


def load_dinov2(size: str = "base", device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    name = DINOV2_MODELS[size]
    print(f"[model] Loading {name} …")
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    model.eval().to(device)
    print(f"[model] Ready on {device}")
    return model


@torch.no_grad()
def embed_batch(model: torch.nn.Module,
                images: list[Image.Image],
                device: torch.device) -> np.ndarray:
    """L2-normalised DINOv2 CLS embeddings. Shape: (N, D)."""
    tensors = torch.stack([DINOV2_TRANSFORM(img) for img in images]).to(device)
    feats = F.normalize(model(tensors), dim=-1)
    return feats.cpu().float().numpy()


def _google_cdn_resize(url: str, size: int) -> str:
    """
    Append/replace the size hint on a Google CDN URL (lh3.googleusercontent.com).
    These URLs support =s{N} to cap the longest side at N pixels server-side.
    Non-Google URLs are returned unchanged.
    """
    if "googleusercontent.com" not in url:
        return url
    import re
    # Strip any existing =s\d+ or =w\d+-h\d+ parameter
    url = re.sub(r"=([swh]\d+[-hwc]*)+$", "", url)
    return f"{url}=s{size}"


def resize_to_max(img: Image.Image, max_side: int) -> Image.Image:
    """
    Downscale img so its longest side is at most max_side pixels,
    preserving aspect ratio.  Images already smaller are returned unchanged.

    Why 512 px (default)?
      - DINOv2 transform: resize to 256 → crop to 224.  Anything ≥ 256 is
        lossless for the model; 512 gives 2× headroom with no quality loss.
      - Edge detection (XDoG/Canny/Sobel): 512 px retains enough contour
        detail.  Working on 4000 px originals yields identical edge maps
        at the cost of ~60× more compute and memory.
      - Storage: 512 px JPEG ≈ 30–80 KB vs. 3–8 MB for full-res MET images.
        For 500 k artworks that is ~25 GB vs. ~2.5 TB in Azure.
    """
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


# ─────────────────────────────────────────────────────────────────────────────
# Edge extraction
# ─────────────────────────────────────────────────────────────────────────────

class EdgeExtractor:
    """
    Converts a colour PIL image into a sketch-like edge map for DINOv2 input.

    Output: 3-channel PIL RGB image (white edges on black background).
    Channels are identical — the tripling is required by DINOv2.

    Methods
    -------
    xdog (default, recommended for SBIR)
        eXtended Difference-of-Gaussians (Winnemöller et al. 2012).
        Produces stylised contours closest to hand-drawn sketches.
    canny
        Classic Canny detector — crisp, thin edges. Good for geometric art.
    sobel
        Gradient magnitude — soft edges. Good for watercolours / prints.
    """

    def __init__(self, method: str = "xdog", **kwargs):
        if method not in ("canny", "sobel", "xdog"):
            raise ValueError(f"Unknown edge method: {method!r}")
        self.method = method
        self.kwargs = kwargs

    def extract(self, img: Image.Image) -> Image.Image:
        gray = np.array(img.convert("L"), dtype=np.float32) / 255.0
        if self.method == "canny":
            edge = self._canny(gray)
        elif self.method == "sobel":
            edge = self._sobel(gray)
        else:
            edge = self._xdog(gray)
        edge_u8  = (np.clip(edge, 0, 1) * 255).astype(np.uint8)
        edge_rgb = np.stack([edge_u8] * 3, axis=-1)
        return Image.fromarray(edge_rgb, mode="RGB")

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        lo  = int(self.kwargs.get("low_threshold",  50))
        hi  = int(self.kwargs.get("high_threshold", 150))
        img8 = (gray * 255).astype(np.uint8)
        return cv2.Canny(img8, lo, hi).astype(np.float32) / 255.0

    def _sobel(self, gray: np.ndarray) -> np.ndarray:
        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        return mag / mag.max() if mag.max() > 0 else mag

    def _xdog(self, gray: np.ndarray) -> np.ndarray:
        """
        D(x) = G_σ(x) − τ·G_{kσ}(x)
        T(x) = 1  if D≥ε,  else  1 + tanh(φ·D)
        edge  = 1 − T   (white edges on black)
        """
        sigma = float(self.kwargs.get("sigma", 0.5))
        k     = float(self.kwargs.get("k",     1.6))
        tau   = float(self.kwargs.get("tau",   0.98))
        phi   = float(self.kwargs.get("phi",   200.0))
        eps   = float(self.kwargs.get("eps",   0.01))
        g1    = gaussian_filter(gray, sigma)
        g2    = gaussian_filter(gray, sigma * k)
        dog   = g1 - tau * g2
        T     = np.where(dog >= eps, 1.0, 1.0 + np.tanh(phi * dog))
        return np.clip(1.0 - T, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# ArtObject
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArtObject:
    museum:       str
    object_id:    str
    title:        str  = ""
    artist:       str  = ""
    date:         str  = ""
    medium:       str  = ""
    department:   str  = ""
    culture:      str  = ""
    image_url:    str  = ""
    object_url:   str  = ""
    is_highlight: bool = False
    extra:        dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.museum}:{self.object_id}"

    @property
    def safe_id(self) -> str:
        """Filesystem/blob-safe version of object_id."""
        return self.object_id.replace("/", "_").replace(":", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Storage backends
# ─────────────────────────────────────────────────────────────────────────────

class StorageBackend(ABC):
    """
    Abstract storage interface.  The pipeline calls these methods and is
    completely unaware of whether data lives on local disk or Azure Blob.
    """

    @abstractmethod
    def put_image(self, museum: str, safe_id: str, data: bytes) -> None:
        """Store a JPEG image."""

    @abstractmethod
    def put_edge(self, museum: str, safe_id: str, method: str, data: bytes) -> None:
        """Store an edge-map JPEG image."""

    @abstractmethod
    def put_bytes(self, blob_name: str, data: bytes) -> None:
        """Store arbitrary bytes at the given logical path."""

    @abstractmethod
    def get_bytes(self, blob_name: str) -> Optional[bytes]:
        """Retrieve bytes, or None if not found."""

    # ── Convenience wrappers ──────────────────────────────────────────────

    def put_npz(self, tag: str, data: bytes) -> None:
        self.put_bytes(f"embeddings/embeddings_{tag}.npz", data)

    def get_npz(self, tag: str) -> Optional[bytes]:
        return self.get_bytes(f"embeddings/embeddings_{tag}.npz")

    def put_json(self, name: str, obj: object) -> None:
        self.put_bytes(f"embeddings/{name}",
                       json.dumps(obj, indent=2, ensure_ascii=False).encode())

    def get_json(self, name: str) -> Optional[object]:
        data = self.get_bytes(f"embeddings/{name}")
        return json.loads(data) if data else None


# ── Local storage ─────────────────────────────────────────────────────────────

class LocalStorage(StorageBackend):
    """Stores everything under a local directory tree."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, blob_name: str) -> Path:
        p = self.root / blob_name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put_image(self, museum, safe_id, data):
        self._path(f"images/{museum}/{safe_id}.jpg").write_bytes(data)

    def put_edge(self, museum, safe_id, method, data):
        self._path(f"edges/{method}/{museum}/{safe_id}.jpg").write_bytes(data)

    def put_bytes(self, blob_name, data):
        self._path(blob_name).write_bytes(data)

    def get_bytes(self, blob_name) -> Optional[bytes]:
        p = self.root / blob_name
        return p.read_bytes() if p.exists() else None


# ── Azure Blob Storage ────────────────────────────────────────────────────────

class AzureStorage(StorageBackend):
    """
    Stores everything in an Azure Blob container via a SAS URL.

    The SAS URL must have read + write + list permissions on the container
    (sp=rwdlacupiytfx).  Pass it as:
      - CLI flag:  --azure-sas-url "https://account.blob.core.windows.net/?sv=..."
      - Env var:   AZURE_BLOB_SAS_URL

    A tiny local cache directory is used only for:
      - NGA bulk CSV downloads  (a few MB, reused across runs)
      - Temporary npz buffer    (current session only, then uploaded)
    """

    _MAX_RETRIES = 3

    def __init__(self, sas_url: str, container: str, local_cache: Path):
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise SystemExit(
                "azure-storage-blob not installed.\n"
                "Run: pip install azure-storage-blob"
            )
        # BlobServiceClient accepts the full SAS URL directly
        self._svc       = BlobServiceClient(account_url=sas_url)
        self._container = self._svc.get_container_client(container)
        self.local_cache = local_cache
        local_cache.mkdir(parents=True, exist_ok=True)
        print(f"[azure] Connected to container '{container}'")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _upload(self, blob_name: str, data: bytes) -> None:
        for attempt in range(self._MAX_RETRIES):
            try:
                self._container.get_blob_client(blob_name).upload_blob(
                    data, overwrite=True
                )
                return
            except Exception as exc:
                if attempt == self._MAX_RETRIES - 1:
                    tqdm.write(f"[azure] ERROR upload '{blob_name}': {exc}")
                else:
                    time.sleep(2 ** attempt)   # 1s, 2s backoff

    def _download(self, blob_name: str) -> Optional[bytes]:
        try:
            return (self._container
                    .get_blob_client(blob_name)
                    .download_blob()
                    .readall())
        except Exception:
            return None   # blob does not exist or other transient error

    # ── StorageBackend interface ──────────────────────────────────────────

    def put_image(self, museum, safe_id, data):
        self._upload(f"images/{museum}/{safe_id}.jpg", data)

    def put_edge(self, museum, safe_id, method, data):
        self._upload(f"edges/{method}/{museum}/{safe_id}.jpg", data)

    def put_bytes(self, blob_name, data):
        self._upload(blob_name, data)

    def get_bytes(self, blob_name) -> Optional[bytes]:
        return self._download(blob_name)


# ─────────────────────────────────────────────────────────────────────────────
# Storage factory
# ─────────────────────────────────────────────────────────────────────────────

def make_storage(args: argparse.Namespace) -> StorageBackend:
    sas_url = args.azure_sas_url or os.environ.get("AZURE_BLOB_SAS_URL", "")
    if sas_url:
        cache = Path(args.local_cache_dir)
        return AzureStorage(sas_url, args.azure_container, cache)
    print("[storage] Using local storage (no Azure SAS URL provided)")
    return LocalStorage(Path(args.output_dir))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (storage-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _npz_to_bytes(uids: list[str], museums: list[str],
                  arrays: list[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    combined = (np.concatenate(arrays, axis=0)
                if arrays else np.empty((0, 1), dtype=np.float32))
    np.savez_compressed(
        buf,
        uids=np.array(uids, dtype=object),
        museums=np.array(museums, dtype=object),
        embeddings=combined,
    )
    return buf.getvalue()


def _npz_from_bytes(data: bytes) -> tuple[list[str], list[str], np.ndarray]:
    ckpt    = np.load(io.BytesIO(data), allow_pickle=True)
    uids    = ckpt["uids"].tolist()
    museums = ckpt["museums"].tolist()
    embs    = ckpt["embeddings"]
    return uids, museums, embs


def load_checkpoint(storage: StorageBackend,
                    edge_tag: str) -> tuple[set[str], list[str], list[str],
                                           dict[str, list[np.ndarray]],
                                           list[dict]]:
    """
    Returns (done_uids_set, uids_list, museums_list, emb_lists, metadata_list).
    emb_lists maps tag → [np.ndarray, …]  (list so we can append new chunks).
    """
    uids:    list[str]  = []
    museums: list[str]  = []
    emb_lists: dict[str, list[np.ndarray]] = {"rgb": [], edge_tag: []}
    meta:    list[dict] = []

    # done_uids.json is the authoritative resume checkpoint
    done_raw = storage.get_json("done_uids.json")
    done_uids: set[str] = set(done_raw) if done_raw else set()

    if done_uids:
        print(f"[checkpoint] {len(done_uids):,} UIDs already processed")

        # Load existing embedding arrays so we can extend them
        rgb_bytes = storage.get_npz("rgb")
        if rgb_bytes:
            u, m, e = _npz_from_bytes(rgb_bytes)
            uids    = u
            museums = m
            emb_lists["rgb"].append(e)

        edge_bytes = storage.get_npz(edge_tag)
        if edge_bytes:
            _, _, e = _npz_from_bytes(edge_bytes)
            emb_lists[edge_tag].append(e)

        meta_raw = storage.get_json("metadata.json")
        if meta_raw:
            meta = meta_raw

    return done_uids, uids, museums, emb_lists, meta


def save_checkpoint(storage: StorageBackend,
                    uids: list[str],
                    museums: list[str],
                    emb_lists: dict[str, list[np.ndarray]],
                    meta: list[dict]) -> None:
    for tag, arrays in emb_lists.items():
        if arrays:
            storage.put_npz(tag, _npz_to_bytes(uids, museums, arrays))
    storage.put_json("metadata.json", meta)
    storage.put_json("done_uids.json", uids)   # uids list == done set


# ─────────────────────────────────────────────────────────────────────────────
# Museum collectors (storage-unaware — they only yield ArtObject metadata)
# ─────────────────────────────────────────────────────────────────────────────

class MuseumCollector(ABC):
    name:       str   = ""
    delay:      float = 0.1   # base seconds between requests
    max_retries: int  = 6     # attempts before giving up on one URL

    def __init__(self, session: requests.Session):
        self.session = session

    def get(self, url: str, **kwargs) -> requests.Response:
        """
        GET with jittered delay + exponential backoff on rate-limit responses.

        Museums return different status codes for rate limiting:
          MET       — 403 (undocumented; real rate limit, not auth)
          ARTIC     — 429
          Others    — 429 or 503

        Strategy:
          - Always sleep self.delay * jitter(0.8–1.2) before every request
          - On 403 / 429 / 503: wait backoff_base * 2^attempt seconds, then retry
          - backoff_base starts at 5 s so the first wait is 5 s, then 10, 20 …
          - After max_retries the exception propagates to the caller's WARN handler
        """
        backoff = 5.0   # seconds for first retry wait

        for attempt in range(self.max_retries):
            # Jitter prevents lock-step bursts when processing many IDs
            jitter = random.uniform(0.8, 1.2)
            time.sleep(self.delay * jitter)

            try:
                r = self.session.get(url, timeout=30, **kwargs)
            except requests.exceptions.ConnectionError as exc:
                if attempt == self.max_retries - 1:
                    raise
                wait = backoff * (2 ** attempt)
                tqdm.write(f"[{self.name}] connection error, retrying in {wait:.0f}s — {exc}")
                time.sleep(wait)
                continue

            if r.status_code in (403, 429, 503):
                if attempt == self.max_retries - 1:
                    r.raise_for_status()   # propagate on final attempt
                # Honour Retry-After header if present (ARTIC sends it)
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff * (2 ** attempt)
                tqdm.write(
                    f"[{self.name}] HTTP {r.status_code} — "
                    f"rate limited, waiting {wait:.0f}s (attempt {attempt+1}/{self.max_retries})"
                )
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r

        raise RuntimeError(f"[{self.name}] gave up after {self.max_retries} attempts: {url}")

    @abstractmethod
    def iter_objects(self, limit: int = 0) -> Iterator[ArtObject]: ...


# ── 1. MET ────────────────────────────────────────────────────────────────────

class METCollector(MuseumCollector):
    name  = "met"
    # MET advertises 80 req/s but in practice throttles aggressively after
    # short bursts and returns 403 (not 429).  1 req/s is the safe sustained
    # rate; combined with the ±20% jitter in get() this averages ~0.9 req/s.
    delay = 1.0
    BASE  = "https://collectionapi.metmuseum.org/public/collection/v1"

    def __init__(self, session, query="*", department_id=None, highlights_only=False):
        super().__init__(session)
        self.query            = query
        self.department_id    = department_id
        self.highlights_only  = highlights_only

    def iter_objects(self, limit=0) -> Iterator[ArtObject]:
        params: dict = {"q": self.query, "hasImages": "true"}
        if self.department_id:   params["departmentIds"] = self.department_id
        if self.highlights_only: params["isHighlight"]   = "true"
        ids = (self.get(f"{self.BASE}/search", params=params)
               .json().get("objectIDs") or [])
        if limit: ids = ids[:limit]
        print(f"[{self.name}] {len(ids):,} objects found")
        for oid in ids:
            time.sleep(self.delay)
            try:
                obj = self.get(f"{self.BASE}/objects/{oid}").json()
            except Exception as e:
                tqdm.write(f"[{self.name}] WARN {oid}: {e}"); continue
            if not obj.get("primaryImage"): continue
            # Prefer primaryImageSmall (~400-800 px) over the full-res CDN URL.
            # Both point to the same image; the small version avoids downloading
            # multi-MB files that get resized to 512 px anyway.
            image_url = obj.get("primaryImageSmall") or obj["primaryImage"]
            yield ArtObject(
                museum=self.name, object_id=str(oid),
                title=obj.get("title",""), artist=obj.get("artistDisplayName",""),
                date=obj.get("objectDate",""), medium=obj.get("medium",""),
                department=obj.get("department",""), culture=obj.get("culture",""),
                image_url=image_url, object_url=obj.get("objectURL",""),
                is_highlight=obj.get("isHighlight", False),
            )


# ── 2. Rijksmuseum ────────────────────────────────────────────────────────────

class RijksCollector(MuseumCollector):
    name  = "rijks"
    delay = 0.1
    BASE  = "https://www.rijksmuseum.nl/api/en/collection"

    def __init__(self, session, api_key: str, query="*"):
        super().__init__(session)
        self.api_key = api_key
        self.query   = query

    def iter_objects(self, limit=0) -> Iterator[ArtObject]:
        page, ps, yielded = 1, 100, 0
        while True:
            params = {"key": self.api_key,
                      "q": self.query if self.query != "*" else "",
                      "ps": ps, "p": page, "imgonly": "true", "s": "relevance"}
            try:
                data = self.get(self.BASE, params=params).json()
            except Exception as e:
                tqdm.write(f"[{self.name}] WARN page {page}: {e}"); break
            items = data.get("artObjects", [])
            if not items: break
            if page == 1:
                print(f"[{self.name}] {data.get('count',0):,} objects found")
            for obj in items:
                img_url = (obj.get("webImage") or {}).get("url","")
                if not img_url: continue
                # Rijksmuseum webImage URLs are served by Google (lh3.googleusercontent.com).
                # Appending =s512 caps the long side at 512 px server-side,
                # saving bandwidth before we resize locally anyway.
                img_url = _google_cdn_resize(img_url, 512)
                yield ArtObject(
                    museum=self.name, object_id=obj.get("objectNumber",""),
                    title=obj.get("title",""),
                    artist=obj.get("principalOrFirstMaker",""),
                    image_url=img_url,
                    object_url=obj.get("links",{}).get("web",""),
                )
                yielded += 1
                if limit and yielded >= limit: return
            if len(items) < ps: break
            page += 1


# ── 3. Cleveland Museum of Art ────────────────────────────────────────────────

class ClevelandCollector(MuseumCollector):
    name  = "cleveland"
    delay = 0.1
    BASE  = "https://openaccess-api.clevelandart.org/api/artworks/"

    def __init__(self, session, query=""):
        super().__init__(session)
        self.query = query

    def iter_objects(self, limit=0) -> Iterator[ArtObject]:
        skip, ps, yielded = 0, 1000, 0
        while True:
            params: dict = {"has_image": 1, "cc0": 1, "limit": ps, "skip": skip}
            if self.query: params["q"] = self.query
            try:
                data = self.get(self.BASE, params=params).json()
            except Exception as e:
                tqdm.write(f"[{self.name}] WARN skip={skip}: {e}"); break
            items = data.get("data", [])
            if not items: break
            if skip == 0:
                print(f"[{self.name}] {data.get('info',{}).get('total','?'):,} objects found")
            for obj in items:
                images  = obj.get("images") or {}
                img_url = (images.get("web") or images.get("print") or {}).get("url","")
                if not img_url: continue
                yield ArtObject(
                    museum=self.name, object_id=str(obj.get("id","")),
                    title=obj.get("title",""),
                    artist=", ".join(c.get("description","")
                                     for c in (obj.get("creators") or [])),
                    date=obj.get("creation_date",""), medium=obj.get("technique",""),
                    department=obj.get("department",""), culture=obj.get("culture",""),
                    image_url=img_url, object_url=obj.get("url",""),
                )
                yielded += 1
                if limit and yielded >= limit: return
            if len(items) < ps: break
            skip += ps


# ── 4. Art Institute of Chicago ───────────────────────────────────────────────

class ARTICCollector(MuseumCollector):
    name      = "artic"
    delay     = 1.0
    BASE      = "https://api.artic.edu/api/v1"
    IIIF_BASE = "https://www.artic.edu/iiif/2"
    FIELDS    = ("id,title,artist_display,date_display,medium_display,"
                 "department_title,place_of_origin,image_id,is_public_domain")

    def __init__(self, session, query=""):
        super().__init__(session)
        self.query = query

    def iter_objects(self, limit=0) -> Iterator[ArtObject]:
        page, ps, yielded = 1, 100, 0
        while True:
            params: dict = {"page": page, "limit": ps, "fields": self.FIELDS,
                            "query[term][is_public_domain]": "true"}
            if self.query: params["q"] = self.query
            try:
                data = self.get(f"{self.BASE}/artworks", params=params).json()
            except Exception as e:
                tqdm.write(f"[{self.name}] WARN page {page}: {e}"); break
            items = data.get("data", [])
            if not items: break
            if page == 1:
                print(f"[{self.name}] "
                      f"{data.get('pagination',{}).get('total','?'):,} objects found")
            for obj in items:
                if not obj.get("image_id"): continue
                yield ArtObject(
                    museum=self.name, object_id=str(obj.get("id","")),
                    title=obj.get("title",""), artist=obj.get("artist_display",""),
                    date=obj.get("date_display",""), medium=obj.get("medium_display",""),
                    department=obj.get("department_title",""),
                    culture=obj.get("place_of_origin",""),
                    image_url=f"{self.IIIF_BASE}/{obj['image_id']}/full/843,/0/default.jpg",
                    object_url=f"https://www.artic.edu/artworks/{obj['id']}",
                )
                yielded += 1
                if limit and yielded >= limit: return
            if page >= data.get("pagination",{}).get("total_pages", 1): break
            page += 1


# ── 5. Victoria & Albert Museum ───────────────────────────────────────────────

class VAMCollector(MuseumCollector):
    name      = "vam"
    delay     = 1.0
    BASE      = "https://api.vam.ac.uk/v2"
    IIIF_BASE = "https://framemark.vam.ac.uk/collections"

    def __init__(self, session, query=""):
        super().__init__(session)
        self.query = query

    def iter_objects(self, limit=0) -> Iterator[ArtObject]:
        page, ps, yielded = 1, 100, 0
        while True:
            params: dict = {
                "images_exist": 1, "image_restrict": 2,
                "page": page, "page_size": ps,
                "fields": ("objectid,_primaryImageId,titles,"
                           "artistMakerPerson,productionDates,"
                           "materialsAndTechniques,categories"),
            }
            if self.query: params["q"] = self.query
            try:
                data = self.get(f"{self.BASE}/objects/search", params=params).json()
            except Exception as e:
                tqdm.write(f"[{self.name}] WARN page {page}: {e}"); break
            records = data.get("records", [])
            if not records: break
            if page == 1:
                print(f"[{self.name}] "
                      f"{data.get('info',{}).get('record_count','?'):,} objects found")
            for obj in records:
                img_id = obj.get("_primaryImageId","")
                if not img_id: continue
                titles = obj.get("titles",[{}])
                makers = obj.get("artistMakerPerson",[])
                dates  = obj.get("productionDates",[{}])
                cats   = obj.get("categories",[{}])
                yield ArtObject(
                    museum=self.name, object_id=obj.get("objectid",""),
                    title=titles[0].get("title","") if titles else "",
                    artist="; ".join(m.get("name",{}).get("text","")
                                     for m in makers if m.get("name")),
                    date=dates[0].get("date",{}).get("text","") if dates else "",
                    medium=obj.get("materialsAndTechniques",""),
                    department=cats[0].get("text","") if cats else "",
                    image_url=f"{self.IIIF_BASE}/{img_id}/full/!600,600/0/default.jpg",
                    object_url=f"https://collections.vam.ac.uk/item/{obj.get('objectid','')}",
                )
                yielded += 1
                if limit and yielded >= limit: return
            if page >= data.get("info",{}).get("pages", 1): break
            page += 1


# ── 6. National Gallery of Art, Washington ────────────────────────────────────

class NGACollector(MuseumCollector):
    name  = "nga"
    delay = 0.05
    OBJECTS_CSV_URL = ("https://raw.githubusercontent.com/NationalGalleryOfArt/"
                       "opendata/main/data/objects.csv")
    IMAGES_ZIP_URL  = ("https://media.githubusercontent.com/media/NationalGalleryOfArt/"
                       "opendata/main/data/published_images.zip")
    IIIF_BASE  = "https://media.nga.gov/iiif/2"
    IMG_SUFFIX = "full/!600,600/0/default.jpg"

    def __init__(self, session, query="", cache_dir: Optional[Path] = None):
        super().__init__(session)
        self.query     = query.lower()
        self.cache_dir = cache_dir

    def _fetch(self, url: str, fname: str) -> bytes:
        if self.cache_dir:
            p = self.cache_dir / fname
            if p.exists():
                print(f"[{self.name}] Using cached {fname}")
                return p.read_bytes()
        print(f"[{self.name}] Downloading {fname} …")
        data = self.session.get(url, timeout=120).content
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / fname).write_bytes(data)
        return data

    def iter_objects(self, limit=0) -> Iterator[ArtObject]:
        obj_rows: dict[str, dict] = {}
        for row in csv.DictReader(io.StringIO(
                self._fetch(self.OBJECTS_CSV_URL, "nga_objects.csv")
                    .decode("utf-8", errors="replace"))):
            obj_rows[row.get("objectid","")] = row

        zip_bytes = self._fetch(self.IMAGES_ZIP_URL, "nga_published_images.zip")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                print(f"[{self.name}] No CSV in zip"); return
            img_csv = zf.read(csv_names[0]).decode("utf-8", errors="replace")

        img_map: dict[str, str] = {}
        for row in csv.DictReader(io.StringIO(img_csv)):
            oid  = row.get("depictstmsobjectid","") or row.get("objectid","")
            uuid = row.get("uuid","") or row.get("iiifthumburl","")
            if uuid.startswith("http"):
                parts = [p for p in uuid.split("/") if len(p)==36 and "-" in p]
                uuid  = parts[0] if parts else uuid
            if oid and uuid and oid not in img_map:
                img_map[oid] = uuid

        print(f"[{self.name}] {len(obj_rows):,} objects, {len(img_map):,} with images")

        yielded = 0
        for oid, obj in obj_rows.items():
            uuid = img_map.get(oid,"")
            if not uuid: continue
            if self.query:
                blob = " ".join([obj.get("title",""), obj.get("attribution",""),
                                 obj.get("medium",""), obj.get("classification","")]).lower()
                if self.query not in blob: continue
            yield ArtObject(
                museum=self.name, object_id=oid,
                title=obj.get("title",""), artist=obj.get("attribution",""),
                date=obj.get("displaydate",""), medium=obj.get("medium",""),
                department=obj.get("classification",""), culture=obj.get("school",""),
                image_url=f"{self.IIIF_BASE}/{uuid}/{self.IMG_SUFFIX}",
                object_url=obj.get("url",""),
            )
            yielded += 1
            if limit and yielded >= limit: return


# ─────────────────────────────────────────────────────────────────────────────
# Image fetch  (downloads from museum URL, uploads to storage backend)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_store(session: requests.Session,
                    obj: ArtObject,
                    storage: StorageBackend,
                    extractor: EdgeExtractor,
                    save_edges: bool,
                    image_size: int = 512) -> Optional[tuple[Image.Image, Image.Image]]:
    """
    Downloads the artwork image, resizes to max image_size px (long side),
    stores RGB + optional edge to the storage backend, and returns
    (rgb_pil, edge_pil) for embedding computation.
    Returns None on download failure.
    """
    try:
        resp = session.get(obj.image_url, timeout=60)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        tqdm.write(f"[img] WARN {obj.image_url[:80]} — {e}")
        return None

    # Resize before storing and before any processing.
    # This is the single most impactful change for bandwidth, storage, and speed.
    img = resize_to_max(img, image_size)

    # Store RGB image (already at reduced size — JPEG ~30–80 KB)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    storage.put_image(obj.museum, obj.safe_id, buf.getvalue())

    # Compute edge map
    edge_img = extractor.extract(img)

    # Optionally store edge map
    if save_edges:
        ebuf = io.BytesIO()
        edge_img.save(ebuf, format="JPEG", quality=90)
        storage.put_edge(obj.museum, obj.safe_id, extractor.method, ebuf.getvalue())

    return img, edge_img


# ─────────────────────────────────────────────────────────────────────────────
# Parallel image fetch
# ─────────────────────────────────────────────────────────────────────────────

def parallel_fetch(objs: list[ArtObject],
                   session: requests.Session,
                   storage: StorageBackend,
                   extractor: EdgeExtractor,
                   save_edges: bool,
                   image_size: int,
                   workers: int) -> list[tuple[ArtObject, Image.Image, Image.Image]]:
    """
    Download + store a list of ArtObjects concurrently.
    Returns (obj, rgb_pil, edge_pil) tuples for successful downloads only.

    Why threads (not asyncio)?
      fetch_and_store calls PIL and cv2, which release the GIL during I/O.
      For network-bound work (download + Azure upload) ThreadPoolExecutor
      gives near-linear speedup up to ~16 workers before Azure throughput
      becomes the ceiling.  No async refactor needed.
    """
    results: list[tuple[ArtObject, Image.Image, Image.Image]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_obj = {
            pool.submit(fetch_and_store, session, obj, storage,
                        extractor, save_edges, image_size): obj
            for obj in objs
        }
        for future in as_completed(future_to_obj):
            obj = future_to_obj[future]
            try:
                r = future.result()
                if r is not None:
                    results.append((obj, r[0], r[1]))
            except Exception as e:
                tqdm.write(f"[img] WARN {obj.uid}: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Collector builder
# ─────────────────────────────────────────────────────────────────────────────

ALL_MUSEUMS = ["met", "rijks", "cleveland", "artic", "vam", "nga"]


def build_collectors(args: argparse.Namespace,
                     session: requests.Session,
                     local_cache: Path) -> list[MuseumCollector]:
    museums = ALL_MUSEUMS if "all" in args.museums else args.museums
    cols: list[MuseumCollector] = []
    for m in museums:
        if m == "met":
            cols.append(METCollector(session, query=args.query or "*",
                                     department_id=args.met_department,
                                     highlights_only=args.highlights_only))
        elif m == "rijks":
            key = args.rijks_key or os.environ.get("RIJKS_API_KEY","")
            if not key:
                print("[rijks] WARN: no API key — skipping. "
                      "Use --rijks-key or RIJKS_API_KEY env var.")
                continue
            cols.append(RijksCollector(session, key, query=args.query or "*"))
        elif m == "cleveland":
            cols.append(ClevelandCollector(session, query=args.query or ""))
        elif m == "artic":
            cols.append(ARTICCollector(session, query=args.query or ""))
        elif m == "vam":
            cols.append(VAMCollector(session, query=args.query or ""))
        elif m == "nga":
            cols.append(NGACollector(session, query=args.query or "",
                                     cache_dir=local_cache / "nga"))
    return cols


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    local_cache = Path(args.local_cache_dir)
    local_cache.mkdir(parents=True, exist_ok=True)

    storage   = make_storage(args)
    device    = get_device(args.device)
    extractor = EdgeExtractor(method=args.edge_method, **_edge_kwargs(args))
    session   = requests.Session()
    session.headers["User-Agent"] = "museum-dinov2-sbir/1.0 (research)"

    edge_tag    = f"edges_{args.edge_method}"
    collectors  = build_collectors(args, session, local_cache)
    if not collectors:
        print("No museums selected / available. Exiting.")
        return

    # ── Resume ────────────────────────────────────────────────────────────
    done_uids, all_uids, all_museums, emb_lists, all_meta = \
        load_checkpoint(storage, edge_tag)
    if edge_tag not in emb_lists: emb_lists[edge_tag] = []
    if "rgb"    not in emb_lists: emb_lists["rgb"]    = []

    # ── Auto-detect model size from checkpoint embeddings ─────────────────
    model_size = args.model
    for tag, arrays in emb_lists.items():
        for arr in arrays:
            if arr.ndim >= 2 and arr.shape[1] > 0:
                ckpt_dim = arr.shape[1]
                ckpt_size = DINOV2_DIM_TO_SIZE.get(ckpt_dim)
                if ckpt_size and ckpt_size != model_size:
                    print(f"[checkpoint] Existing embeddings have D={ckpt_dim} "
                          f"(model={ckpt_size}), overriding --model {model_size} "
                          f"→ {ckpt_size} to stay compatible")
                    model_size = ckpt_size
                elif not ckpt_size:
                    raise ValueError(
                        f"Checkpoint has D={ckpt_dim} which doesn't match any "
                        f"known DINOv2 model. Delete checkpoint files to start fresh.")
                break
        else:
            continue
        break

    model = load_dinov2(model_size, device)

    SAVE_EVERY = max(1, args.batch_size * 10)

    batch_rgb:   list[Image.Image] = []
    batch_edges: list[Image.Image] = []
    batch_objs:  list[ArtObject]   = []

    def flush_batch() -> None:
        if not batch_rgb: return
        emb_lists["rgb"].append(embed_batch(model, batch_rgb,   device))
        emb_lists[edge_tag].append(embed_batch(model, batch_edges, device))
        for obj in batch_objs:
            all_uids.append(obj.uid)
            all_museums.append(obj.museum)
            all_meta.append({**asdict(obj), "uid": obj.uid})
        batch_rgb.clear(); batch_edges.clear(); batch_objs.clear()

    def checkpoint() -> None:
        flush_batch()
        save_checkpoint(storage, all_uids, all_museums, emb_lists, all_meta)

    # ── Per-museum loop ───────────────────────────────────────────────────
    for collector in collectors:
        m_name   = collector.name
        limit    = 0 if args.no_limit else args.limit
        step     = 0
        # How many objects to buffer before firing a parallel download round.
        # Larger = better GPU utilisation; smaller = more frequent checkpoints.
        prefetch = args.batch_size * args.workers

        print(f"\n{'─'*60}\n  Museum: {m_name.upper()}\n{'─'*60}")

        pending: list[ArtObject] = []   # objects waiting to be downloaded

        def _drain(objs: list[ArtObject]) -> int:
            """Download objs in parallel, push results into embedding batch."""
            nonlocal step
            fetched = parallel_fetch(objs, session, storage, extractor,
                                     args.save_edges, args.image_size, args.workers)
            for obj, rgb_img, edge_img in fetched:
                batch_rgb.append(rgb_img)
                batch_edges.append(edge_img)
                batch_objs.append(obj)
                done_uids.add(obj.uid)
                if len(batch_rgb) >= args.batch_size:
                    flush_batch()
                step += 1
                if step % SAVE_EVERY == 0:
                    checkpoint()
                    tqdm.write(f"[{m_name}] checkpoint — {len(all_uids):,} total")
            return len(fetched)

        for obj in tqdm(collector.iter_objects(limit=limit),
                        desc=m_name, unit="obj",
                        total=limit if limit else None):

            if obj.uid in done_uids or not obj.image_url:
                continue

            pending.append(obj)

            # Fire a parallel download round once the prefetch window is full.
            # While this round downloads, iter_objects() is paused (no API calls),
            # which naturally spaces out requests to rate-limited APIs.
            if len(pending) >= prefetch:
                _drain(pending)
                pending.clear()

        # Drain any remaining objects
        if pending:
            _drain(pending)

        flush_batch()
        checkpoint()
        print(f"[{m_name}] Done. Running total: {len(all_uids):,}")

    # ── Final summary ─────────────────────────────────────────────────────
    checkpoint()
    counts = Counter(all_museums)
    dim_str = (f"{emb_lists['rgb'][-1].shape[-1]}d"
               if emb_lists["rgb"] else "?")

    print(f"\n{'═'*60}")
    print(f"  COMPLETE")
    print(f"{'═'*60}")
    print(f"  Total artworks  : {len(all_uids):,}")
    print(f"  Embedding dim   : {dim_str}")
    print(f"  Per museum:")
    for m, n in sorted(counts.items()):
        print(f"    {m:<12} {n:>7,}")

    if isinstance(storage, AzureStorage):
        print(f"\n  Azure container  : {args.azure_container}")
        print(f"  embeddings/embeddings_rgb.npz")
        print(f"  embeddings/embeddings_{edge_tag}.npz")
        print(f"  embeddings/metadata.json")
    else:
        print(f"\n  Output dir: {Path(args.output_dir).resolve()}")

    print(f"\n  SBIR retrieval hint:")
    print(f"    1. Encode query sketch with DINOv2 (same transform)")
    print(f"    2. Cosine-search embeddings_{edge_tag}.npz   ← primary")
    print(f"    3. Optionally fuse with embeddings_rgb.npz   ← re-rank")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _edge_kwargs(args: argparse.Namespace) -> dict:
    if args.edge_method == "canny":
        return {"low_threshold": args.canny_lo, "high_threshold": args.canny_hi}
    if args.edge_method == "xdog":
        return {"sigma": args.xdog_sigma, "k": args.xdog_k,
                "tau": args.xdog_tau, "phi": args.xdog_phi, "eps": args.xdog_eps}
    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-museum DINOv2 + edge-map pipeline for SBIR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Museums
    p.add_argument("--museums", nargs="+", default=["met","artic","cleveland"],
                   choices=ALL_MUSEUMS + ["all"])
    p.add_argument("--query", default="")
    p.add_argument("--highlights-only", action="store_true")
    p.add_argument("--met-department", type=int, default=None)
    p.add_argument("--limit", type=int, default=1000,
                   help="Max objects per museum (0 = unlimited)")
    p.add_argument("--no-limit", action="store_true")

    # Edge
    p.add_argument("--edge-method", choices=["canny","sobel","xdog"], default="xdog")
    p.add_argument("--save-edges", action="store_true",
                   help="Also upload edge-map images to storage")
    p.add_argument("--canny-lo",   type=float, default=50)
    p.add_argument("--canny-hi",   type=float, default=150)
    p.add_argument("--xdog-sigma", type=float, default=0.5)
    p.add_argument("--xdog-k",     type=float, default=1.6)
    p.add_argument("--xdog-tau",   type=float, default=0.98)
    p.add_argument("--xdog-phi",   type=float, default=200.0)
    p.add_argument("--xdog-eps",   type=float, default=0.01)

    # Model / hardware
    p.add_argument("--workers",     type=int, default=8,
                   help=("Parallel image download threads. "
                         "8 is a good default; raise to 16 on fast connections. "
                         "Does not affect API metadata requests, which stay sequential "
                         "to respect each museum's rate limit."))
    p.add_argument("--model",       choices=list(DINOV2_MODELS.keys()), default="base")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--device",      default=None,
                   help="cuda / mps / cpu  (auto-detected if omitted)")
    p.add_argument("--image-size",  type=int, default=512,
                   help=("Resize images so the longest side is at most this many pixels "
                         "before storing and embedding. 512 is optimal: gives DINOv2 and "
                         "edge detection enough detail while saving ~60× storage vs. "
                         "full-res MET/Rijksmuseum images. Min useful value: 256."))

    # Auth
    p.add_argument("--rijks-key", default="",
                   help="Rijksmuseum API key (or RIJKS_API_KEY env var)")

    # Azure storage
    p.add_argument("--azure-sas-url", default="",
                   help=("Azure Blob SAS URL, e.g. "
                         "https://ACCOUNT.blob.core.windows.net/?sv=... "
                         "(or set AZURE_BLOB_SAS_URL env var)"))
    p.add_argument("--azure-container", default="artworks",
                   help="Azure Blob container name")

    # I/O
    p.add_argument("--output-dir",      default="museum_dataset",
                   help="Root dir for local storage mode")
    p.add_argument("--local-cache-dir", default="/tmp/museum_cache",
                   help="Local temp dir for NGA CSV cache + session npz buffer")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_limit:
        args.limit = 0
    run(args)
