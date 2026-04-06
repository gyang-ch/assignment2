#!/usr/bin/env python3
"""
refresh_colorful_dataset.py

Rebuilds the 6,000-artwork dataset so every image is colourful (not B&W/grey).

Pipeline:
  1. Load existing artworks_sonification.json (current 6,000).
  2. Load the colour cache (style_entropy.cache.json).
     If an artwork has no cached colour yet, download and compute it now.
  3. Mark artworks as B&W if their dominant-colour HSL saturation < SAT_THRESHOLD.
  4. Keep the colourful ones; discard the B&W ones.
  5. Sample replacements from the full ~162K pool (excluding current 6,000).
     Download + colour-check each candidate; accept only colourful ones.
     Continue in batches until 6,000 colourful artworks are assembled.
  6. Run the full UMAP → KMeans → k-NN → energy → PCA pipeline on the new 6,000.
  7. Compute visual complexity = ||RGB emb − XDoG emb|| for each artwork.
  8. Write:
       frontend/public/artworks_sonification.json  (Echoes of Images tab)
       frontend/public/style_entropy.json          (Style-Entropy Grid tab)
       frontend/public/style_entropy.cache.json    (updated colour cache)

Run from project root:
    python scripts/refresh_colorful_dataset.py [--sat-threshold 0.12] [--workers 24]
"""

import argparse
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import umap as umap_lib

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
EMB_DIR    = ROOT / 'embeddings'
PUBLIC_DIR = ROOT / 'frontend' / 'public'
CACHE_PATH = PUBLIC_DIR / 'style_entropy.cache.json'
SONIF_PATH = PUBLIC_DIR / 'artworks_sonification.json'
SE_PATH    = PUBLIC_DIR / 'style_entropy.json'

# ── Dataset parameters ────────────────────────────────────────────────────────
SAMPLE_N     = 6000
N_CLUSTERS   = 12
N_NEIGHBORS  = 5
RANDOM_SEED  = 42

# ── Colour parameters ─────────────────────────────────────────────────────────
SAT_THRESHOLD     = 0.12   # dominant-colour HSL saturation below this → B&W
MIN_COLORFUL_FRAC = 0.05   # fraction of image pixels that must be colourful
IMAGE_SIZE        = (80, 80)
N_COLORS          = 6      # k-means clusters for colour extraction
BATCH_SIZE        = 400    # candidates to check per download batch

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; DH6018Research/1.0; educational use)',
    'Accept': 'image/*,*/*;q=0.8',
}
TIMEOUT = 20


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_hex(r, g, b):
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'


def hex_saturation(hex_color: str) -> float:
    """HSL saturation [0,1] of a '#rrggbb' string."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0
    l = (mx + mn) / 2
    d = mx - mn
    return d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)


def pixel_saturation_hsl(px01: np.ndarray) -> np.ndarray:
    """Vectorised HSL saturation for (N,3) float32 array in [0,1]."""
    r, g, b = px01[:, 0], px01[:, 1], px01[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    l  = (mx + mn) / 2
    d  = mx - mn
    denom = 1.0 - np.abs(2 * l - 1)
    return np.where(denom < 1e-6, 0.0, d / denom).astype(np.float32)


def fetch_dominant_color(image_url: str, sat_thresh: float) -> tuple[str, bool]:
    """
    Download image, return (hex_dominant_color, is_colourful).
    is_colourful = True if >= MIN_COLORFUL_FRAC of pixels exceed sat_thresh.
    """
    try:
        resp = requests.get(image_url, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        pixels    = np.array(img).reshape(-1, 3).astype(np.float32)
        pixels_01 = pixels / 255.0

        sat             = pixel_saturation_hsl(pixels_01)
        colorful_mask   = sat >= sat_thresh
        colorful_frac   = float(colorful_mask.sum()) / len(pixels)
        is_colorful     = colorful_frac >= MIN_COLORFUL_FRAC

        target = pixels[colorful_mask] if is_colorful else pixels
        n      = min(N_COLORS, len(target))
        km     = MiniBatchKMeans(n_clusters=n, random_state=42,
                                 n_init=3, max_iter=100)
        labels = km.fit_predict(target)

        if is_colorful:
            counts   = np.bincount(labels, minlength=n)
            best_idx = int(counts.argmax())
        else:
            centers_01 = km.cluster_centers_ / 255.0
            best_idx   = int(pixel_saturation_hsl(centers_01).argmax())

        return rgb_to_hex(*km.cluster_centers_[best_idx]), is_colorful

    except Exception:
        return '#888888', False


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Check / compute dominant colours for the current 6,000
# ─────────────────────────────────────────────────────────────────────────────

def ensure_colors_for_current(current_uids, meta_by_uid, cache,
                               sat_thresh, workers):
    """Download dominant colours for any current artwork not yet in cache."""
    missing = [(uid, meta_by_uid[uid]['image_url'])
               for uid in current_uids
               if uid not in cache and uid in meta_by_uid
               and meta_by_uid[uid].get('image_url')]

    if not missing:
        print('  All current artworks already have cached colours.')
        return

    print(f'  {len(missing)} current artworks missing colour — downloading …')
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_dominant_color, url, sat_thresh): uid
                for uid, url in missing}
        for fut in as_completed(futs):
            uid = futs[fut]
            color, _ = fut.result()
            cache[uid] = color
            done += 1
            if done % 200 == 0:
                print(f'    {done}/{len(missing)} …')


# ─────────────────────────────────────────────────────────────────────────────
# Step B: Sample colourful replacements from the full pool
# ─────────────────────────────────────────────────────────────────────────────

def find_colorful_replacements(needed, excluded_uids, uids_all,
                                meta_by_uid, cache, sat_thresh, workers):
    """
    Scan the full pool (excluding excluded_uids) in random batches.
    Return list of (uid, hex_color) for colourful artworks until `needed`.
    """
    accepted = []

    pool = [uid for uid in uids_all
            if uid not in excluded_uids
            and uid in meta_by_uid
            and meta_by_uid[uid].get('image_url')]
    random.shuffle(pool)
    print(f'  Candidate pool: {len(pool):,} artworks')

    ptr        = 0
    batch_num  = 0

    while len(accepted) < needed and ptr < len(pool):
        batch     = pool[ptr: ptr + BATCH_SIZE]
        ptr      += BATCH_SIZE
        batch_num += 1

        # Some candidates may already be in the cache
        to_download = [(uid, meta_by_uid[uid]['image_url'])
                       for uid in batch if uid not in cache]
        already     = [(uid, cache[uid])
                       for uid in batch if uid in cache]

        # Download missing ones
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_dominant_color, url, sat_thresh): uid
                    for uid, url in to_download}
            for fut in as_completed(futs):
                uid = futs[fut]
                color, is_colorful = fut.result()
                cache[uid] = color
                if is_colorful and len(accepted) < needed:
                    accepted.append((uid, color))

        # Accept already-cached colourful ones
        for uid, color in already:
            if hex_saturation(color) >= sat_thresh and len(accepted) < needed:
                accepted.append((uid, color))

        print(f'  Batch {batch_num}: {len(accepted)}/{needed} accepted …')

        # Save cache periodically
        if batch_num % 5 == 0:
            with open(CACHE_PATH, 'w') as f:
                json.dump(cache, f)

    return accepted


# ─────────────────────────────────────────────────────────────────────────────
# Step C: Full UMAP → KMeans → k-NN → complexity pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(selected_uids, color_map, uid_to_idx,
                 emb_rgb, emb_edge, meta_by_uid):
    n = len(selected_uids)
    print(f'\nRunning pipeline on {n:,} artworks …')

    indices  = [uid_to_idx[uid] for uid in selected_uids]
    emb      = emb_rgb[indices].astype(np.float32)

    # L2-normalise
    norms    = np.linalg.norm(emb, axis=1, keepdims=True)
    norms    = np.where(norms == 0, 1.0, norms)
    emb_norm = emb / norms

    # UMAP → 2D
    print('  UMAP (this takes a few minutes) …')
    coords = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                            n_components=2,
                            random_state=RANDOM_SEED).fit_transform(emb_norm)
    coords = MinMaxScaler().fit_transform(coords)
    print(f'    x=[{coords[:,0].min():.3f},{coords[:,0].max():.3f}]  '
          f'y=[{coords[:,1].min():.3f},{coords[:,1].max():.3f}]')

    # KMeans
    print('  KMeans …')
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED,
                    n_init=10).fit_predict(emb_norm)

    # k-NN graph
    print('  k-NN …')
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric='cosine')
    nn.fit(emb_norm)
    distances, nn_idx = nn.kneighbors(emb_norm)
    neighbor_indices  = nn_idx[:, 1:].tolist()
    neighbor_dists    = distances[:, 1:].tolist()

    # Energy
    mean_dist = distances[:, 1:].mean(axis=1)
    energy    = MinMaxScaler().fit_transform(
                    mean_dist.reshape(-1, 1)).flatten()

    # PCA features
    print('  PCA …')
    pca3 = PCA(n_components=3,
               random_state=RANDOM_SEED).fit_transform(emb_norm)
    pca3 = MinMaxScaler().fit_transform(pca3)

    # Cluster centroids (2-D, for echoes background fog)
    centroids_2d = []
    for c in range(N_CLUSTERS):
        mask = labels == c
        centroids_2d.append({
            'x':     round(float(coords[mask, 0].mean()), 4),
            'y':     round(float(coords[mask, 1].mean()), 4),
            'count': int(mask.sum()),
        })

    # Visual complexity: ||RGB emb − XDoG emb||, normalised to [0,1]
    print('  Visual complexity …')
    raw_complexity = np.array([
        float(np.linalg.norm(emb_rgb[idx].astype(np.float32)
                              - emb_edge[idx].astype(np.float32)))
        for idx in indices
    ], dtype=np.float32)
    complexity_norm = MinMaxScaler().fit_transform(
        raw_complexity.reshape(-1, 1)).flatten()

    # ── Write artworks_sonification.json ─────────────────────────────────────
    print('  Writing artworks_sonification.json …')
    artworks_out = []
    for i, uid in enumerate(selected_uids):
        meta = meta_by_uid.get(uid, {})
        artworks_out.append({
            'id':        i,
            'uid':       uid,
            'x':         round(float(coords[i, 0]), 4),
            'y':         round(float(coords[i, 1]), 4),
            'cluster':   int(labels[i]),
            'neighbors': neighbor_indices[i],
            'nbDists':   [round(float(d), 4) for d in neighbor_dists[i]],
            'energy':    round(float(energy[i]), 3),
            'features':  {
                'structure': round(float(pca3[i, 0]), 3),
                'palette':   round(float(pca3[i, 1]), 3),
                'texture':   round(float(pca3[i, 2]), 3),
            },
            'title':     meta.get('title', ''),
            'artist':    meta.get('artist', ''),
            'image_url': meta.get('image_url', ''),
        })

    with open(SONIF_PATH, 'w') as f:
        json.dump({'artworks': artworks_out, 'centroids': centroids_2d},
                  f, separators=(',', ':'))
    print(f'    → {SONIF_PATH}  ({SONIF_PATH.stat().st_size / 1e6:.1f} MB)')

    # ── Write style_entropy.json ──────────────────────────────────────────────
    print('  Writing style_entropy.json …')
    se_out = []
    for i, uid in enumerate(selected_uids):
        meta = meta_by_uid.get(uid, {})
        se_out.append({
            'id':             i,
            'uid':            uid,
            'dominant_color': color_map.get(uid, '#888888'),
            'complexity':     round(float(complexity_norm[i]), 4),
            'cluster':        int(labels[i]),
            'title':          meta.get('title', ''),
            'artist':         meta.get('artist', ''),
            'image_url':      meta.get('image_url', ''),
        })

    with open(SE_PATH, 'w') as f:
        json.dump(se_out, f, separators=(',', ':'))
    print(f'    → {SE_PATH}  ({SE_PATH.stat().st_size / 1e6:.1f} MB)')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sat-threshold', type=float, default=SAT_THRESHOLD,
                        help='Min HSL saturation to accept as colourful (default 0.12)')
    parser.add_argument('--workers', type=int, default=24,
                        help='Parallel download threads (default 24)')
    args = parser.parse_args()

    sat_thresh = args.sat_threshold
    workers    = args.workers

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Load colour cache ─────────────────────────────────────────────────────
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f'Colour cache: {len(cache):,} entries loaded')

    # ── Load current 6,000 ───────────────────────────────────────────────────
    print('Loading artworks_sonification.json …')
    with open(SONIF_PATH) as f:
        sonif = json.load(f)
    current_uids = [a['uid'] for a in sonif['artworks']]
    print(f'  {len(current_uids):,} artworks in current dataset')

    # ── Load embeddings + metadata ────────────────────────────────────────────
    print('Loading RGB embeddings (~30 s) …')
    rgb_data   = np.load(EMB_DIR / 'embeddings_rgb.npz', allow_pickle=True)
    uids_all   = list(rgb_data['uids'])
    emb_rgb    = rgb_data['embeddings'].astype(np.float32)
    uid_to_idx = {uid: i for i, uid in enumerate(uids_all)}
    print(f'  {len(uids_all):,} × {emb_rgb.shape[1]} dims')

    print('Loading XDoG edge embeddings (~30 s) …')
    emb_edge = np.load(EMB_DIR / 'embeddings_edges_xdog.npz',
                       allow_pickle=True)['embeddings'].astype(np.float32)

    print('Loading metadata …')
    with open(EMB_DIR / 'metadata.json') as f:
        meta_by_uid = {m['uid']: m for m in json.load(f)}
    print(f'  {len(meta_by_uid):,} metadata records')

    # ── Ensure we have colours for all current artworks ───────────────────────
    print('\nStep A: checking colours of current 6,000 …')
    ensure_colors_for_current(current_uids, meta_by_uid, cache,
                              sat_thresh, workers)

    # Classify current artworks
    colorful_uids = [uid for uid in current_uids
                     if uid in cache
                     and hex_saturation(cache[uid]) >= sat_thresh]
    bw_count      = len(current_uids) - len(colorful_uids)
    print(f'  Colourful: {len(colorful_uids):,}  |  B&W/grey: {bw_count}')

    # ── Find replacements ─────────────────────────────────────────────────────
    needed = SAMPLE_N - len(colorful_uids)
    new_artworks: list[tuple[str, str]] = []

    if needed > 0:
        print(f'\nStep B: finding {needed} colourful replacements …')
        excluded = set(current_uids)           # don't re-use any current uid
        new_artworks = find_colorful_replacements(
            needed, excluded, uids_all, meta_by_uid,
            cache, sat_thresh, workers)
    else:
        print('\nNo replacements needed — all current artworks are colourful.')

    # Save updated cache
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)
    print(f'Cache saved → {CACHE_PATH}  ({len(cache):,} entries)')

    # ── Assemble final 6,000 ──────────────────────────────────────────────────
    color_map: dict[str, str] = {uid: cache[uid] for uid in colorful_uids}

    selected_uids = list(colorful_uids)
    for uid, color in new_artworks:
        color_map[uid] = color
        selected_uids.append(uid)
        if len(selected_uids) >= SAMPLE_N:
            break

    # Validate: must be in embeddings + have metadata
    selected_uids = [uid for uid in selected_uids
                     if uid in uid_to_idx
                     and uid in meta_by_uid
                     and meta_by_uid[uid].get('image_url')][:SAMPLE_N]

    print(f'\nFinal dataset: {len(selected_uids):,} colourful artworks')
    if len(selected_uids) < SAMPLE_N:
        print(f'  (target was {SAMPLE_N} — proceeding with {len(selected_uids)})')

    # ── Run full pipeline ─────────────────────────────────────────────────────
    run_pipeline(selected_uids, color_map, uid_to_idx,
                 emb_rgb, emb_edge, meta_by_uid)

    print('\nDone. Both JSON files updated.')
    print('The Echoes of Images and Style-Entropy Grid tabs will use the new dataset.')


if __name__ == '__main__':
    main()
