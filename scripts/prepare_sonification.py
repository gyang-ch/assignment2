#!/usr/bin/env python3
"""
Prepare a lightweight JSON dataset for the Echoes of Images sonification sketch.

Pipeline:
  1. Load DINOv2 RGB embeddings (162k × 1024)
  2. Random-sample ~6000 artworks (browser-friendly)
  3. UMAP → 2D coordinates
  4. KMeans → 12 style clusters
  5. k-NN → 5 nearest neighbours per point (within the sample)
  6. Compute a style_energy scalar (embedding L2 norm variation)
  7. Write artworks_sonification.json → frontend/public/

Run from project root:
    python scripts/prepare_sonification.py
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import umap

ROOT    = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / 'embeddings'
OUT_DIR = ROOT / 'frontend' / 'public'

SAMPLE_N    = 6000
N_CLUSTERS  = 12
N_NEIGHBORS = 5
RANDOM_SEED = 42


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Load embeddings + metadata ────────────────────────────────────────────
    print('Loading embeddings…')
    rgb_data = np.load(EMB_DIR / 'embeddings_rgb.npz', allow_pickle=True)
    uids_all = list(rgb_data['uids'])
    emb_all  = rgb_data['embeddings'].astype(np.float32)  # (N, 1024)
    N_total  = len(uids_all)
    print(f'  {N_total:,} artworks × {emb_all.shape[1]} dims')

    print('Loading metadata…')
    with open(EMB_DIR / 'metadata.json') as f:
        meta_list = json.load(f)
    meta_by_uid = {m['uid']: m for m in meta_list}

    # ── Sample ────────────────────────────────────────────────────────────────
    # Only keep artworks that have metadata with an image_url
    valid_indices = [i for i, uid in enumerate(uids_all)
                     if uid in meta_by_uid and meta_by_uid[uid].get('image_url')]
    print(f'  {len(valid_indices):,} have valid metadata')

    sample_idx = sorted(random.sample(valid_indices, min(SAMPLE_N, len(valid_indices))))
    emb    = emb_all[sample_idx]
    uids   = [uids_all[i] for i in sample_idx]
    print(f'  Sampled {len(uids):,} artworks')

    # ── L2-normalise ──────────────────────────────────────────────────────────
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_norm = emb / norms

    # ── UMAP → 2D ────────────────────────────────────────────────────────────
    print('Running UMAP…')
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=RANDOM_SEED,
    )
    coords = reducer.fit_transform(emb_norm)

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(coords)
    print(f'  UMAP done — range x=[{coords[:,0].min():.2f},{coords[:,0].max():.2f}] '
          f'y=[{coords[:,1].min():.2f},{coords[:,1].max():.2f}]')

    # ── KMeans clustering ─────────────────────────────────────────────────────
    print(f'Clustering into {N_CLUSTERS} groups…')
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10).fit_predict(emb_norm)

    # ── k-NN graph ────────────────────────────────────────────────────────────
    print(f'Computing {N_NEIGHBORS}-NN graph…')
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric='cosine')
    nn.fit(emb_norm)
    distances, indices = nn.kneighbors(emb_norm)
    # first neighbor is self → skip
    neighbor_indices = indices[:, 1:].tolist()
    neighbor_dists   = distances[:, 1:].tolist()

    # ── Style energy (how "distinctive" an artwork is) ────────────────────────
    # Use mean distance to neighbors: low = tightly clustered, high = unique
    mean_dist = distances[:, 1:].mean(axis=1)
    energy = MinMaxScaler().fit_transform(mean_dist.reshape(-1, 1)).flatten()

    # ── Perceptual feature proxies (from embedding PCA) ──────────────────────
    # DINOv2 encodes structure, palette, and texture — top PCA axes capture these.
    print('Computing perceptual feature proxies via PCA…')
    pca3 = PCA(n_components=3, random_state=RANDOM_SEED).fit_transform(emb_norm)
    pca3 = MinMaxScaler().fit_transform(pca3)   # each col ∈ [0, 1]

    # ── Neighbor cosine distances (for ripple delay / consonance) ─────────────
    neighbor_cosine = distances[:, 1:].tolist()

    # ── Cluster centroids in 2-D (for background fog) ────────────────────────
    centroids_2d = []
    for c in range(N_CLUSTERS):
        mask = labels == c
        cx = float(coords[mask, 0].mean())
        cy = float(coords[mask, 1].mean())
        centroids_2d.append({'x': round(cx, 4), 'y': round(cy, 4),
                             'count': int(mask.sum())})

    # ── Build output JSON ─────────────────────────────────────────────────────
    print('Building JSON…')
    artworks_out = []
    for i in range(len(uids)):
        uid  = uids[i]
        meta = meta_by_uid.get(uid, {})
        artworks_out.append({
            'id':        i,
            'uid':       uid,
            'x':         round(float(coords[i, 0]), 4),
            'y':         round(float(coords[i, 1]), 4),
            'cluster':   int(labels[i]),
            'neighbors': neighbor_indices[i],
            'nbDists':   [round(float(d), 4) for d in neighbor_cosine[i]],
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

    output = {
        'artworks':  artworks_out,
        'centroids': centroids_2d,
    }

    out_path = OUT_DIR / 'artworks_sonification.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, separators=(',', ':'))
    mb = out_path.stat().st_size / 1_000_000
    print(f'  → {out_path}  ({mb:.1f} MB)')
    print('Done.')


if __name__ == '__main__':
    main()
