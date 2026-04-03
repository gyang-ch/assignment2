/**
 * API client for the sketch-based image retrieval backend.
 *
 * All backend communication is centralised here so the rest of the
 * frontend never constructs URLs or fetch calls directly.
 */

const DEFAULT_BASE = 'http://localhost:8000';

let _base = DEFAULT_BASE;

/** Set the backend API base URL (no trailing slash). */
export function setBaseURL(url) {
  _base = url.replace(/\/+$/, '');
}

export function getBaseURL() {
  return _base;
}

/**
 * Search for artworks similar to a sketch.
 * @param {Blob} imageBlob
 * @param {number} [topK=12]
 */
export async function searchBySketch(imageBlob, topK = 12) {
  const form = new FormData();
  form.append('sketch', imageBlob, 'sketch.png');
  form.append('top_k', String(topK));

  const res = await fetch(`${_base}/api/search`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Search failed (${res.status}): ${await res.text()}`);
  return res.json();
}

/**
 * Compare a sketch against a specific artwork.
 * @param {Blob} imageBlob
 * @param {string} uid — artwork UID (e.g. "met:437056")
 * @returns {Promise<{uid: string, similarity: number}>}
 */
export async function compareSketch(imageBlob, uid) {
  const form = new FormData();
  form.append('sketch', imageBlob, 'sketch.png');
  form.append('uid', uid);

  const res = await fetch(`${_base}/api/compare`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Compare failed (${res.status}): ${await res.text()}`);
  return res.json();
}

/**
 * Text search over artwork metadata (title, artist).
 * @param {string} query
 * @param {number} [limit=20]
 */
export async function searchArtworksByText(query, limit = 20) {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  const res = await fetch(`${_base}/api/artworks/search?${params}`);
  if (!res.ok) throw new Error(`Text search failed (${res.status})`);
  return res.json();
}

/**
 * Get random artworks.
 * @param {number} [count=12]
 */
export async function getRandomArtworks(count = 12) {
  const res = await fetch(`${_base}/api/artworks/random?count=${count}`);
  if (!res.ok) throw new Error(`Random artworks failed (${res.status})`);
  return res.json();
}

/**
 * Return a URL that proxies through the backend (CORS bypass).
 * @param {string} originalUrl
 */
export function proxyImageUrl(originalUrl) {
  return `${_base}/api/proxy-image?url=${encodeURIComponent(originalUrl)}`;
}

/**
 * Health-check.
 * @returns {Promise<boolean>}
 */
export async function ping() {
  try {
    const res = await fetch(`${_base}/api/health`, { method: 'GET' });
    return res.ok;
  } catch {
    return false;
  }
}
