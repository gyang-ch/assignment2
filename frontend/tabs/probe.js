// Probe tab — freehand drawing or image upload to search similar artworks via DINOv2 embeddings.
// Search strategy: try Modal backend first; fall back to in-browser transformers.

const BACKEND_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : 'https://gyang-ch--sketch-art-sbir-sbirservice-web.modal.run';
let _backendAvailable = null; // null = unchecked, true = up, false = down

async function checkBackend() {
  if (_backendAvailable !== null) return _backendAvailable;
  try {
    const ctrl  = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 2000);
    const resp  = await fetch(`${BACKEND_URL}/api/health`, { signal: ctrl.signal });
    clearTimeout(timer);
    _backendAvailable = resp.ok;
  } catch {
    _backendAvailable = false;
  }
  return _backendAvailable;
}

async function searchViaBackend(blob, topK) {
  const fd = new FormData();
  fd.append('sketch', blob, 'sketch.png');
  fd.append('top_k', String(topK));
  const resp = await fetch(`${BACKEND_URL}/api/search`, { method: 'POST', body: fd });
  if (!resp.ok) throw new Error(`Backend HTTP ${resp.status}`);
  const { results } = await resp.json();
  return results;
}

let _editor       = null;
let _searchEngine = null;
let searching     = false;
let uploadedBlob  = null;
let _mounted      = false;

function applyGlowBtn(btn) {
  const fill = btn.querySelector('.glow-btn-fill');
  if (!fill || !window.gsap) return;
  btn.addEventListener('mouseenter', (e) => {
    const rect = btn.getBoundingClientRect();
    const fromTop = e.clientY < rect.top + rect.height / 2;
    gsap.killTweensOf(fill);
    gsap.fromTo(fill,
      { y: fromTop ? '-100%' : '100%' },
      { y: '0%', duration: 0.4, ease: 'power3.out' }
    );
  });
  btn.addEventListener('mouseleave', (e) => {
    const rect = btn.getBoundingClientRect();
    const toTop = e.clientY < rect.top + rect.height / 2;
    gsap.killTweensOf(fill);
    gsap.to(fill, { y: toTop ? '-100%' : '100%', duration: 0.4, ease: 'power3.out' });
  });
  btn.addEventListener('click', () => {
    gsap.timeline()
      .to(btn, { scale: 0.95, duration: 0.1, ease: 'power2.out' })
      .to(btn, { scale: 1, duration: 0.4, ease: 'elastic.out(1, 0.3)', clearProps: 'scale' });
  });
}

// Called once the first time the user clicks the Probe tab.
// Subsequent calls are no-ops thanks to the _mounted guard.
export function mount() {
  if (_mounted) return;
  _mounted = true;

  const editorContainer = document.getElementById('editor-container');
  const btnSearch       = document.getElementById('btn-search');
  const btnClear        = document.getElementById('btn-clear');
  const uploadDropzone  = document.getElementById('upload-dropzone');
  const uploadInput     = document.getElementById('upload-input');
  const uploadPrompt    = document.getElementById('upload-prompt');
  const uploadPreview   = document.getElementById('upload-preview');
  const uploadImg       = document.getElementById('upload-img');
  const uploadNoPreview = document.getElementById('upload-no-preview');
  const btnUploadClear  = document.getElementById('btn-upload-clear');
  const btnUploadSearch = document.getElementById('btn-upload-search');
  const resultsGrid     = document.getElementById('results-grid');
  const emptyState      = document.getElementById('empty-state');

  function setStatus(msg) {
    const p = emptyState?.querySelector('p');
    if (p) p.textContent = msg;
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

  function renderResults(results) {
    resultsGrid.innerHTML = '';
    emptyState.hidden = true;
    resultsGrid.hidden = false;
    for (const r of results) {
      const card  = document.createElement('div');
      card.className = 'result-card';
      const img   = document.createElement('img');
      img.src     = r.image_url;
      img.alt     = r.title || 'Artwork';
      img.loading = 'lazy';
      const info  = document.createElement('div');
      info.className = 'card-info';
      const title = document.createElement('div');
      title.className = 'card-title';
      title.textContent = r.title || 'Untitled';
      title.title = r.title || '';
      const meta  = document.createElement('div');
      meta.className = 'card-meta';
      meta.textContent = [r.artist, r.date].filter(Boolean).join(' · ') || r.museum;
      const score = document.createElement('div');
      score.className = 'card-score';
      score.textContent = `similarity: ${(r.score * 100).toFixed(1)}%`;
      info.append(title, meta, score);
      card.append(img, info);
      if (r.object_url) {
        card.style.cursor = 'pointer';
        card.addEventListener('click', () => window.open(r.object_url, '_blank'));
      }
      resultsGrid.appendChild(card);
    }
  }

  async function ensureEditor() {
    if (_editor) return;
    try {
      _editor = await import('../util/sketch-editor.js');
      _editor.mount(editorContainer);
    } catch (err) {
      setStatus(`Drawing canvas failed to load: ${err.message}`);
    }
  }

  async function ensureSearchEngine() {
    if (_searchEngine) return;
    _searchEngine = await import('../util/search-engine.js');
    _searchEngine.initSearchEngine(setStatus).catch(err =>
      setStatus(`Search engine error: ${err.message}`)
    );
  }

  async function doSearch() {
    if (searching) return;
    await ensureEditor();
    if (!_editor || _editor.isEmpty()) {
      setStatus('Canvas is blank — draw something first.');
      return;
    }
    searching = true;
    btnSearch.disabled = true;
    showLoading();
    try {
      const blob = await _editor.exportAsBlob(512);
      if (!blob) { setStatus('Export failed.'); showEmpty(); return; }
      const useBackend = await checkBackend();
      let results;
      if (useBackend) {
        setStatus('Searching via backend…');
        results = await searchViaBackend(blob, 12);
      } else {
        setStatus('Backend offline — searching in-browser…');
        await ensureSearchEngine();
        results = await _searchEngine.searchImage(blob, 12);
      }
      renderResults(results);
      setStatus(`Found ${results.length} results.`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      showEmpty('Search failed.');
    } finally {
      searching = false;
      btnSearch.disabled = false;
    }
  }

  function setUploadedImage(blob, previewUrl) {
    uploadedBlob = blob;
    uploadPrompt.hidden = true;
    uploadPreview.hidden = false;
    if (previewUrl) {
      uploadImg.src = previewUrl;
      uploadImg.hidden = false;
      uploadNoPreview.hidden = true;
    } else {
      uploadImg.src = '';
      uploadImg.hidden = true;
      uploadNoPreview.hidden = false;
    }
    btnUploadClear.disabled = false;
    btnUploadSearch.disabled = false;
    btnUploadClear.removeAttribute('disabled');
    btnUploadSearch.removeAttribute('disabled');
  }

  function clearUpload() {
    uploadedBlob = null;
    uploadImg.src = '';
    uploadImg.hidden = false;
    uploadNoPreview.hidden = true;
    uploadPrompt.hidden = false;
    uploadPreview.hidden = true;
    btnUploadClear.disabled = true;
    btnUploadSearch.disabled = true;
    btnUploadClear.setAttribute('disabled', '');
    btnUploadSearch.setAttribute('disabled', '');
    uploadInput.value = '';
  }

  async function handleUploadFile(file) {
    if (!file) return;
    const isImage = file.type.startsWith('image/') || /\.(heic|heif)$/i.test(file.name);
    if (!isImage) { setStatus('Please upload an image file.'); return; }

    let searchBlob = file;
    let previewUrl = null;
    try {
      // Convert to JPEG for consistent embedding input
      const bitmap = await createImageBitmap(file);
      const canvas = document.createElement('canvas');
      canvas.width  = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext('2d').drawImage(bitmap, 0, 0);
      bitmap.close();
      searchBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
      previewUrl = URL.createObjectURL(searchBlob);
    } catch { /* browser can't decode format; send raw bytes */ }

    setUploadedImage(searchBlob, previewUrl);
  }

  // Wire up glow animations and event listeners
  [btnSearch, btnClear, btnUploadSearch, btnUploadClear].forEach(applyGlowBtn);
  [btnUploadSearch, btnUploadClear].forEach((btn) => {
    btn.addEventListener('pointerdown', (e) => e.stopPropagation());
    btn.addEventListener('mousedown',   (e) => e.stopPropagation());
  });

  btnSearch.addEventListener('click', doSearch);
  btnClear.addEventListener('click', async () => {
    await ensureEditor();
    _editor?.clear();
    showEmpty();
  });

  uploadDropzone.addEventListener('click', () => uploadInput.click());
  uploadInput.addEventListener('change', () => handleUploadFile(uploadInput.files[0]));

  uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadDropzone.classList.add('drag-over');
  });
  uploadDropzone.addEventListener('dragleave', () => uploadDropzone.classList.remove('drag-over'));
  uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove('drag-over');
    handleUploadFile(e.dataTransfer.files[0]);
  });

  btnUploadClear.addEventListener('click', (e) => {
    e.stopPropagation();
    clearUpload();
    showEmpty();
  });

  btnUploadSearch.addEventListener('click', async (e) => {
    e.stopPropagation();
    if (searching || !uploadedBlob) return;
    searching = true;
    btnUploadSearch.disabled = true;
    showLoading();
    try {
      const useBackend = await checkBackend();
      let results;
      if (useBackend) {
        setStatus('Searching via backend…');
        results = await searchViaBackend(uploadedBlob, 12);
      } else {
        setStatus('Backend offline — searching in-browser…');
        await ensureSearchEngine();
        results = await _searchEngine.searchImage(uploadedBlob, 12);
      }
      renderResults(results);
      setStatus(`Found ${results.length} results.`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      showEmpty('Search failed.');
    } finally {
      searching = false;
      btnUploadSearch.disabled = false;
    }
  });

  // Enter key triggers a drawing search from anywhere on the page
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'Enter' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); doSearch(); }
  });

  ensureEditor();
}
