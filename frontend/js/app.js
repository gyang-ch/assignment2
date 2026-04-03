/**
 * Main application — tabs, search (tldraw), and palimpsest (p5.js).
 */

import * as editor from './sketch-editor.js';
import { createPalimpsest } from './palimpsest.js';
import {
  searchBySketch, ping, setBaseURL, getBaseURL,
  searchArtworksByText, getRandomArtworks, proxyImageUrl,
} from './api.js';

/* ══════════════════════════════════════════════════════════════════════════
   DOM refs
   ══════════════════════════════════════════════════════════════════════════ */

// Tabs
const tabs        = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

// Search tab
const editorContainer = document.getElementById('editor-container');
const btnSearch       = document.getElementById('btn-search');
const btnClear        = document.getElementById('btn-clear');
const resultsGrid     = document.getElementById('results-grid');
const emptyState      = document.getElementById('empty-state');

// Palimpsest tab
const palSearchInput  = document.getElementById('pal-search-input');
const palSearchBtn    = document.getElementById('pal-search-btn');
const palRandomBtn    = document.getElementById('pal-random-btn');
const palArtworkGrid  = document.getElementById('pal-artwork-grid');
const palSelected     = document.getElementById('pal-selected');
const palSelectedImg  = document.getElementById('pal-selected-img');
const palSelectedTitle= document.getElementById('pal-selected-title');
const palSelectedMeta = document.getElementById('pal-selected-meta');
const palCanvas       = document.getElementById('palimpsest-canvas');
const palBrush        = document.getElementById('pal-brush');
const palBrushLabel   = document.getElementById('pal-brush-label');
const palClear        = document.getElementById('pal-clear');
const palCompute      = document.getElementById('pal-compute');
const palSimilarity   = document.getElementById('pal-similarity');

// Shared
const statusText      = document.getElementById('status-text');
const backendInput    = document.getElementById('backend-url');

/* ══════════════════════════════════════════════════════════════════════════
   Tabs
   ══════════════════════════════════════════════════════════════════════════ */

function switchTab(target) {
  tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === target));
  tabContents.forEach(tc => {
    const match = tc.id === `tab-${target}`;
    tc.classList.toggle('active', match);
    tc.hidden = !match;
  });
  // Lazy-init palimpsest the first time the tab is shown
  if (target === 'palimpsest') initPalimpsest();
}

tabs.forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

/* ══════════════════════════════════════════════════════════════════════════
   Backend URL
   ══════════════════════════════════════════════════════════════════════════ */

backendInput.value = getBaseURL();
backendInput.addEventListener('change', () => {
  setBaseURL(backendInput.value.trim());
  checkBackend();
});

/* ══════════════════════════════════════════════════════════════════════════
   Search Tab
   ══════════════════════════════════════════════════════════════════════════ */

editor.mount(editorContainer, {
  onReady: () => {
    setStatus('Editor ready.');
    checkBackend();
  },
});

let searching = false;

async function doSearch() {
  if (searching) return;
  if (editor.isEmpty()) {
    setStatus('Canvas is blank — draw something first.');
    return;
  }

  searching = true;
  btnSearch.disabled = true;
  setStatus('Exporting sketch & searching...');
  showLoading();

  try {
    const blob = await editor.exportAsBlob(512);
    if (!blob) { setStatus('Export failed.'); showEmpty(); return; }
    const { results } = await searchBySketch(blob, 12);
    renderResults(results);
    setStatus(`Found ${results.length} results.`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    showEmpty('Could not reach the backend. Is it running?');
  } finally {
    searching = false;
    btnSearch.disabled = false;
  }
}

btnSearch.addEventListener('click', doSearch);
btnClear.addEventListener('click', () => { editor.clear(); showEmpty(); });

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'Enter' && !e.metaKey && !e.ctrlKey) {
    e.preventDefault();
    doSearch();
  }
});

/* ── Search results rendering ──────────────────────────────────────────── */

function renderResults(results) {
  resultsGrid.innerHTML = '';
  emptyState.hidden = true;
  resultsGrid.hidden = false;

  for (const r of results) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const img = document.createElement('img');
    img.src = r.image_url;
    img.alt = r.title || 'Artwork';
    img.loading = 'lazy';

    const info = document.createElement('div');
    info.className = 'card-info';

    const title = document.createElement('div');
    title.className = 'card-title';
    title.textContent = r.title || 'Untitled';
    title.title = r.title || '';

    const meta = document.createElement('div');
    meta.className = 'card-meta';
    meta.textContent = [r.artist, r.date].filter(Boolean).join(' · ') || r.museum;

    const score = document.createElement('div');
    score.className = 'card-score';
    score.textContent = `similarity: ${(r.score * 100).toFixed(1)}%`;

    // "Open in Palimpsest" button
    const palBtn = document.createElement('button');
    palBtn.className = 'btn btn-mini';
    palBtn.textContent = 'Open in Palimpsest';
    palBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      openInPalimpsest(r);
    });

    info.append(title, meta, score, palBtn);
    card.append(img, info);

    if (r.object_url) {
      card.style.cursor = 'pointer';
      card.addEventListener('click', () => window.open(r.object_url, '_blank'));
    }

    resultsGrid.appendChild(card);
  }
}

function showEmpty(msg) {
  resultsGrid.hidden = true;
  resultsGrid.innerHTML = '';
  emptyState.hidden = false;
  if (msg) emptyState.querySelector('p').textContent = msg;
}

function showLoading() {
  resultsGrid.innerHTML = '';
  resultsGrid.hidden = true;
  emptyState.hidden = false;
  emptyState.querySelector('p').textContent = 'Searching for similar artworks...';
}

/** Switch to palimpsest tab and load an artwork from search results. */
function openInPalimpsest(artwork) {
  switchTab('palimpsest');
  selectPalimpsestArtwork(artwork);
}

/* ══════════════════════════════════════════════════════════════════════════
   Palimpsest Tab
   ══════════════════════════════════════════════════════════════════════════ */

let palimpsest = null;

function initPalimpsest() {
  if (palimpsest) return; // already created
  palimpsest = createPalimpsest(palCanvas, {
    onSimilarityUpdate: (sim) => {
      palSimilarity.textContent = `Similarity: ${(sim * 100).toFixed(1)}%`;
    },
  });
}

/* ── Artwork selection ─────────────────────────────────────────────────── */

function selectPalimpsestArtwork(artwork) {
  initPalimpsest();
  palimpsest.loadArtwork(artwork.image_url, artwork.uid);
  palimpsest.clear();

  // Show selected info
  palSelected.hidden = false;
  palSelectedImg.src = artwork.image_url;
  palSelectedTitle.textContent = artwork.title || 'Untitled';
  palSelectedMeta.textContent =
    [artwork.artist, artwork.date].filter(Boolean).join(' · ') || artwork.museum;

  palSimilarity.textContent = 'Similarity: --';
  setStatus(`Loaded: ${artwork.title || artwork.uid}`);
}

function renderMiniGrid(artworks) {
  palArtworkGrid.innerHTML = '';
  for (const a of artworks) {
    const thumb = document.createElement('div');
    thumb.className = 'mini-thumb';
    thumb.title = a.title || a.uid;

    const img = document.createElement('img');
    img.src = a.image_url;
    img.alt = a.title || '';
    img.loading = 'lazy';

    thumb.appendChild(img);
    thumb.addEventListener('click', () => selectPalimpsestArtwork(a));
    palArtworkGrid.appendChild(thumb);
  }
}

// Text search
async function palTextSearch() {
  const q = palSearchInput.value.trim();
  if (!q) return;
  setStatus('Searching artworks...');
  try {
    const { results } = await searchArtworksByText(q, 20);
    renderMiniGrid(results);
    setStatus(`Found ${results.length} artworks.`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

palSearchBtn.addEventListener('click', palTextSearch);
palSearchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') palTextSearch();
});

// Random artworks
palRandomBtn.addEventListener('click', async () => {
  setStatus('Loading random artworks...');
  try {
    const { results } = await getRandomArtworks(18);
    renderMiniGrid(results);
    setStatus(`Loaded ${results.length} random artworks.`);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
});

// Brush size
palBrush.addEventListener('input', () => {
  const sz = Number(palBrush.value);
  palimpsest.setBrushSize(sz);
  palBrushLabel.textContent = sz;
});

// Clear
palClear.addEventListener('click', () => {
  palimpsest.clear();
  palSimilarity.textContent = 'Similarity: --';
});

// Manual recompute
palCompute.addEventListener('click', () => palimpsest.computeSimilarity());

/* ══════════════════════════════════════════════════════════════════════════
   Status / Init
   ══════════════════════════════════════════════════════════════════════════ */

function setStatus(msg) {
  statusText.textContent = msg;
}

async function checkBackend() {
  const ok = await ping();
  setStatus(ok ? 'Backend connected.' : 'Backend not reachable — set the URL above.');
}

checkBackend();
