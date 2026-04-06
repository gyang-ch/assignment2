#!/usr/bin/env python3
"""
Export pre-computed DINOv2 embeddings to static browser assets.

Run from the project root:
    python scripts/export_embeddings.py

Writes to frontend/public/:
  embeddings.bin  — 8-byte header [N: uint32LE, D: uint32LE]
                    followed by N×D float32 values (row-major, little-endian)
  uids.json       — ordered UID list matching embedding row indices
  metadata.json   — copy of embeddings/metadata.json
"""

import json
import struct
import shutil
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / 'embeddings'
OUT_DIR = ROOT / 'frontend' / 'public'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    npz_path = EMB_DIR / 'embeddings_edges_xdog.npz'
    print(f'Loading {npz_path} …')
    data = np.load(npz_path, allow_pickle=True)
    uids = [str(u) for u in data['uids']]
    emb  = data['embeddings'].astype(np.float32)   # (N, D)
    N, D = emb.shape
    print(f'  {N:,} artworks  ×  {D} dims')

    # ── L2-normalise (should already be normalised, but guarantee it) ─────────
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb   = (emb / norms).astype(np.float32)

    # ── Write embeddings.bin ──────────────────────────────────────────────────
    bin_path = OUT_DIR / 'embeddings.bin'
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<II', N, D))   # 8-byte header
        f.write(emb.tobytes())              # little-endian float32 by default on x86/ARM
    mb = bin_path.stat().st_size / 1_000_000
    print(f'  → {bin_path}  ({mb:.0f} MB)')

    # ── Write uids.json ───────────────────────────────────────────────────────
    uids_path = OUT_DIR / 'uids.json'
    with open(uids_path, 'w') as f:
        json.dump(uids, f, separators=(',', ':'))
    print(f'  → {uids_path}')

    # ── Copy metadata.json ────────────────────────────────────────────────────
    meta_src = EMB_DIR / 'metadata.json'
    meta_dst = OUT_DIR / 'metadata.json'
    shutil.copy2(meta_src, meta_dst)
    print(f'  → {meta_dst}')

    print('\nDone.  Run: cd frontend && npm run dev')


if __name__ == '__main__':
    main()
