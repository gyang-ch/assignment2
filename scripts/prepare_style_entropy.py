#!/usr/bin/env python3
"""
Prepare data for the Style-Entropy Grid visualization.

For each of the 6,000 artworks in artworks_sonification.json:
  - Visual Complexity = L2 distance between RGB DINOv2 embedding and XDoG edge embedding
  - Dominant Color    = most visually interesting color extracted from the artwork image

Dominant color strategy (handles grey museum backgrounds robustly):
  1. Convert all pixels to HSL; filter out near-grey pixels (saturation < 0.15).
  2. If enough colorful pixels survive (>= 5% of image), run k-means on those only,
     then pick the largest cluster — this eliminates grey margins entirely.
  3. If too few colorful pixels (genuinely B&W or monochrome artwork), fall back to
     all pixels and pick the cluster with highest saturation centroid, so even
     desaturated images get the most "interesting" tone rather than plain grey.

Output: frontend/public/style_entropy.json

Run from project root:
    python scripts/prepare_style_entropy.py [--force]

  --force  Ignore the resume cache and recompute all dominant colors from scratch.

Requires: numpy, scikit-learn, Pillow, requests (pip install Pillow requests)

NOTE: Loads two ~589 MB embedding files simultaneously (~1.2 GB RAM).
      Image downloads are cached in style_entropy.cache.json so the
      script can be safely interrupted and resumed.
"""

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

ROOT         = Path(__file__).resolve().parent.parent
EMB_DIR      = ROOT / 'embeddings'
PUBLIC_DIR   = ROOT / 'frontend' / 'public'
OUT_PATH     = PUBLIC_DIR / 'style_entropy.json'
CACHE_PATH   = PUBLIC_DIR / 'style_entropy.cache.json'

IMAGE_SIZE        = (80, 80)   # downsample before k-means (fast, good enough)
N_COLORS          = 6          # k-means clusters to search for dominant color
SAT_THRESHOLD     = 0.15       # HSL saturation below which a pixel is "grey"
MIN_COLORFUL_FRAC = 0.05       # if fewer than this fraction are colorful → B&W fallback
TIMEOUT           = 20         # seconds per HTTP request
MAX_WORKERS       = 24         # parallel download threads

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; DH6018Research/1.0; educational use)',
    'Accept': 'image/*,*/*;q=0.8',
}


def rgb_to_hex(r: float, g: float, b: float) -> str:
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'


def pixel_saturation_hsl(pixels_01: np.ndarray) -> np.ndarray:
    """Vectorised HSL saturation for an (N, 3) float32 array in [0, 1]."""
    r, g, b   = pixels_01[:, 0], pixels_01[:, 1], pixels_01[:, 2]
    cmax      = np.maximum(np.maximum(r, g), b)
    cmin      = np.minimum(np.minimum(r, g), b)
    delta     = cmax - cmin
    lightness = (cmax + cmin) / 2.0
    denom     = 1.0 - np.abs(2.0 * lightness - 1.0)
    sat       = np.where(denom < 1e-6, 0.0, delta / denom)
    return sat.astype(np.float32)


def fetch_dominant_color(image_url: str) -> str:
    """
    Download image, return the most visually interesting dominant color as hex.
    Grey/white/black museum backgrounds are excluded before color selection.
    Falls back to '#888888' on any error.
    """
    try:
        resp = requests.get(image_url, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)

        pixels = np.array(img).reshape(-1, 3).astype(np.float32)  # [0, 255]
        pixels_01 = pixels / 255.0

        # ── Saturation filter ────────────────────────────────────────────────
        sat = pixel_saturation_hsl(pixels_01)
        colorful_mask = sat >= SAT_THRESHOLD
        colorful_frac = colorful_mask.sum() / len(pixels)

        if colorful_frac >= MIN_COLORFUL_FRAC:
            # Use only colorful pixels — grey background disappears
            target = pixels[colorful_mask]
            n = min(N_COLORS, len(target))
            km = MiniBatchKMeans(n_clusters=n, random_state=42, n_init=3, max_iter=100)
            labels = km.fit_predict(target)
            counts = np.bincount(labels, minlength=n)
            best_idx = int(counts.argmax())

        else:
            # B&W / monochrome: use all pixels, pick the cluster with highest saturation
            n = min(N_COLORS, len(pixels))
            km = MiniBatchKMeans(n_clusters=n, random_state=42, n_init=3, max_iter=100)
            km.fit(pixels)
            centers_01 = km.cluster_centers_ / 255.0
            center_sat = pixel_saturation_hsl(centers_01)
            best_idx = int(center_sat.argmax())

        color = km.cluster_centers_[best_idx]
        return rgb_to_hex(*color)

    except Exception:
        return '#888888'


def main() -> None:
    force = '--force' in sys.argv
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load sonification data ────────────────────────────────────────────────
    sonif_path = PUBLIC_DIR / 'artworks_sonification.json'
    print(f'Loading {sonif_path} …')
    with open(sonif_path) as f:
        sonif = json.load(f)
    artworks = sonif['artworks']
    print(f'  {len(artworks):,} artworks')

    # ── Load embeddings ───────────────────────────────────────────────────────
    print('Loading RGB embeddings (may take ~30 s) …')
    rgb_data = np.load(EMB_DIR / 'embeddings_rgb.npz', allow_pickle=True)
    uids_all  = list(rgb_data['uids'])
    emb_rgb   = rgb_data['embeddings'].astype(np.float32)
    uid_to_idx = {uid: i for i, uid in enumerate(uids_all)}
    print(f'  {len(uids_all):,} artworks × {emb_rgb.shape[1]} dims')

    print('Loading XDoG edge embeddings (may take ~30 s) …')
    edge_data = np.load(EMB_DIR / 'embeddings_edges_xdog.npz', allow_pickle=True)
    emb_edge  = edge_data['embeddings'].astype(np.float32)
    print(f'  {emb_edge.shape}')

    # ── Visual complexity: L2 distance between RGB and XDoG embeddings ────────
    print('Computing visual complexity …')
    raw_complexity = np.zeros(len(artworks), dtype=np.float32)
    for i, art in enumerate(artworks):
        idx = uid_to_idx.get(art['uid'])
        if idx is not None:
            diff = emb_rgb[idx] - emb_edge[idx]
            raw_complexity[i] = float(np.linalg.norm(diff))

    # Normalise to [0, 1]
    scaler = MinMaxScaler()
    complexity_norm = scaler.fit_transform(raw_complexity.reshape(-1, 1)).flatten()
    print(f'  Raw range [{raw_complexity.min():.1f}, {raw_complexity.max():.1f}]  '
          f'→ normalised [0, 1]')

    # Free memory — embeddings no longer needed
    del emb_rgb, emb_edge, rgb_data, edge_data

    # ── Load color cache (resume support) ─────────────────────────────────────
    uid_to_color: dict[str, str] = {}
    if CACHE_PATH.exists() and not force:
        with open(CACHE_PATH) as f:
            uid_to_color = json.load(f)
        print(f'  Color cache: {len(uid_to_color):,} already computed '
              f'(use --force to recompute all)')
    elif force:
        print('  --force: ignoring cache, recomputing all dominant colors')

    remaining = [
        (art['uid'], art['image_url'])
        for art in artworks
        if art['uid'] not in uid_to_color and art.get('image_url')
    ]
    print(f'  {len(remaining):,} images to download …')

    # ── Parallel image downloads ───────────────────────────────────────────────
    done = 0
    SAVE_EVERY = 200

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_uid = {
            executor.submit(fetch_dominant_color, url): uid
            for uid, url in remaining
        }
        for future in as_completed(future_to_uid):
            uid   = future_to_uid[future]
            color = future.result()
            uid_to_color[uid] = color
            done += 1
            if done % SAVE_EVERY == 0:
                print(f'  {done:,}/{len(remaining):,} done …')
                with open(CACHE_PATH, 'w') as f:
                    json.dump(uid_to_color, f)

    # Final cache save
    with open(CACHE_PATH, 'w') as f:
        json.dump(uid_to_color, f)
    print(f'  Color cache saved → {CACHE_PATH}')

    # ── Build output JSON ─────────────────────────────────────────────────────
    print('Building output JSON …')
    output = []
    for i, art in enumerate(artworks):
        uid = art['uid']
        output.append({
            'id':             art['id'],
            'uid':            uid,
            'dominant_color': uid_to_color.get(uid, '#888888'),
            'complexity':     round(float(complexity_norm[i]), 4),
            'cluster':        art['cluster'],
            'title':          art.get('title', ''),
            'artist':         art.get('artist', ''),
            'image_url':      art.get('image_url', ''),
        })

    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, separators=(',', ':'))

    mb = OUT_PATH.stat().st_size / 1_000_000
    print(f'  → {OUT_PATH}  ({mb:.2f} MB)')
    print('Done.')


if __name__ == '__main__':
    main()
