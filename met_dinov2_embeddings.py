"""
MET Museum Image Downloader + DINOv2 Embedding Extractor
=========================================================
Downloads artwork images from the Metropolitan Museum of Art API
and computes DINOv2 embeddings for use in Sketch-Based Image Retrieval (SBIR).

Usage:
    python met_dinov2_embeddings.py [--query QUERY] [--department DEPT_ID]
                                    [--limit N] [--output-dir DIR]
                                    [--batch-size N] [--device DEVICE]
                                    [--model {small,base,large,giant}]

Outputs (saved to --output-dir):
    images/           Raw downloaded JPEGs
    embeddings.npz    {ids, embeddings, metadata} arrays
    metadata.json     Per-object metadata (title, artist, date, url, …)
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from torchvision import transforms


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA — {name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Apple MPS")
    else:
        device = torch.device("cpu")
        print("[device] CPU (no GPU found)")
    return device


# ---------------------------------------------------------------------------
# DINOv2 model loader
# ---------------------------------------------------------------------------

DINOV2_MODELS = {
    "small":  "dinov2_vits14",
    "base":   "dinov2_vitb14",
    "large":  "dinov2_vitl14",
    "giant":  "dinov2_vitg14",
}

def load_dinov2(model_size: str = "base", device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    model_name = DINOV2_MODELS[model_size]
    print(f"[model] Loading {model_name} from torch.hub (facebookresearch/dinov2) …")
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.eval()
    model.to(device)
    print(f"[model] {model_name} ready on {device}")
    return model


# DINOv2 expects 224×224 images normalized with ImageNet stats
DINOV2_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# MET Museum API helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
REQUEST_DELAY = 0.05   # 50 ms ≈ 20 req/s — well under the 80/s limit


def met_search(query: str = "*",
               has_images: bool = True,
               department_id: Optional[int] = None,
               is_highlight: bool = False) -> list[int]:
    """Return a list of object IDs matching the query."""
    params: dict = {"q": query, "hasImages": str(has_images).lower()}
    if department_id is not None:
        params["departmentIds"] = department_id
    if is_highlight:
        params["isHighlight"] = "true"

    url = f"{BASE_URL}/search"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ids = data.get("objectIDs") or []
    print(f"[api] Search '{query}' → {len(ids):,} objects")
    return ids


def met_object(object_id: int) -> Optional[dict]:
    """Fetch metadata + image URL for a single object."""
    url = f"{BASE_URL}/objects/{object_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[api] WARN: object {object_id} failed — {exc}")
        return None


def download_image(url: str, save_path: Path) -> Optional[Image.Image]:
    """Download an image and save it; return a PIL Image or None on failure."""
    if save_path.exists():
        try:
            return Image.open(save_path).convert("RGB")
        except Exception:
            save_path.unlink(missing_ok=True)

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(save_path, format="JPEG", quality=95)
        return img
    except Exception as exc:
        print(f"[img] WARN: failed to download {url} — {exc}")
        return None


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model: torch.nn.Module,
                       images: list[Image.Image],
                       device: torch.device,
                       normalize: bool = True) -> np.ndarray:
    """
    Run a batch of PIL images through DINOv2 and return L2-normalized embeddings.
    Shape: (N, D) where D = 384 / 768 / 1024 / 1536 depending on model size.
    """
    tensors = torch.stack([DINOV2_TRANSFORM(img) for img in images]).to(device)
    features = model(tensors)            # [N, D] CLS token
    if normalize:
        features = F.normalize(features, dim=-1)
    return features.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    model  = load_dinov2(args.model, device)

    # --- Step 1: collect object IDs ---
    object_ids = met_search(
        query=args.query,
        has_images=True,
        department_id=args.department,
        is_highlight=args.highlights_only,
    )

    if args.limit:
        object_ids = object_ids[:args.limit]
        print(f"[pipeline] Limiting to {args.limit} objects")

    print(f"[pipeline] Processing {len(object_ids):,} objects …\n")

    # --- Step 2: fetch metadata + images + embeddings ---
    all_embeddings: list[np.ndarray] = []
    all_ids:        list[int]        = []
    all_metadata:   list[dict]       = []

    # Checkpoint: load any previously saved embeddings so we can resume
    checkpoint_path = output_dir / "embeddings.npz"
    meta_path       = output_dir / "metadata.json"

    completed_ids: set[int] = set()
    if checkpoint_path.exists() and meta_path.exists():
        print("[checkpoint] Resuming from previous run …")
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        all_embeddings = [checkpoint["embeddings"]]
        all_ids        = checkpoint["ids"].tolist()
        completed_ids  = set(all_ids)
        with open(meta_path) as f:
            all_metadata = json.load(f)
        print(f"[checkpoint] {len(completed_ids):,} already done")

    remaining = [oid for oid in object_ids if oid not in completed_ids]
    print(f"[pipeline] {len(remaining):,} objects remaining\n")

    batch_imgs:  list[Image.Image] = []
    batch_ids:   list[int]         = []
    batch_metas: list[dict]        = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        embs = extract_embeddings(model, batch_imgs, device)
        all_embeddings.append(embs)
        all_ids.extend(batch_ids)
        all_metadata.extend(batch_metas)
        batch_imgs.clear()
        batch_ids.clear()
        batch_metas.clear()

    def save_checkpoint() -> None:
        combined = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0,))
        np.savez_compressed(
            checkpoint_path,
            ids=np.array(all_ids, dtype=np.int64),
            embeddings=combined,
        )
        with open(meta_path, "w") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    SAVE_EVERY = max(1, args.batch_size * 10)   # save checkpoint every ~10 batches

    for step, oid in enumerate(tqdm(remaining, desc="Objects", unit="obj")):
        time.sleep(REQUEST_DELAY)

        obj = met_object(oid)
        if obj is None:
            continue

        primary_url = obj.get("primaryImage", "")
        if not primary_url:
            continue    # skip objects without a usable image

        img_path = images_dir / f"{oid}.jpg"
        img = download_image(primary_url, img_path)
        if img is None:
            continue

        meta = {
            "objectID":      oid,
            "title":         obj.get("title", ""),
            "artistName":    obj.get("artistDisplayName", ""),
            "date":          obj.get("objectDate", ""),
            "medium":        obj.get("medium", ""),
            "department":    obj.get("department", ""),
            "culture":       obj.get("culture", ""),
            "period":        obj.get("period", ""),
            "imageUrl":      primary_url,
            "objectUrl":     obj.get("objectURL", ""),
            "isHighlight":   obj.get("isHighlight", False),
            "isPublicDomain": obj.get("isPublicDomain", False),
        }

        batch_imgs.append(img)
        batch_ids.append(oid)
        batch_metas.append(meta)

        if len(batch_imgs) >= args.batch_size:
            flush_batch()

        if (step + 1) % SAVE_EVERY == 0:
            flush_batch()
            save_checkpoint()
            tqdm.write(f"[checkpoint] Saved {len(all_ids):,} embeddings")

    flush_batch()
    save_checkpoint()

    total = len(all_ids)
    emb_shape = np.concatenate([e for e in all_embeddings], axis=0).shape if all_embeddings else (0,)
    print(f"\n[done] {total:,} embeddings saved")
    print(f"[done] Embedding matrix: {emb_shape}")
    print(f"[done] Output: {output_dir.resolve()}")
    print(f"         embeddings.npz — compressed numpy archive")
    print(f"         metadata.json  — per-object metadata")
    print(f"         images/        — raw JPEG files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MET artworks and compute DINOv2 embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", default="painting",
                        help="Search query sent to the MET API (use '*' for all)")
    parser.add_argument("--department", type=int, default=None,
                        help="Filter by MET department ID (see /departments endpoint)")
    parser.add_argument("--highlights-only", action="store_true",
                        help="Only download highlighted artworks")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Maximum number of objects to process (0 = no limit)")
    parser.add_argument("--output-dir", default="met_dataset",
                        help="Root directory for images + embeddings")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images per embedding batch")
    parser.add_argument("--device", default=None,
                        help="Force device: 'cuda', 'mps', 'cpu' (auto-detected if omitted)")
    parser.add_argument("--model", choices=list(DINOV2_MODELS.keys()), default="base",
                        help="DINOv2 model size")
    parser.add_argument("--no-limit", action="store_true",
                        help="Process ALL matching objects (overrides --limit)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_limit:
        args.limit = 0
    run(args)
