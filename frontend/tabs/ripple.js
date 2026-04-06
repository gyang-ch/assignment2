// Ripple: 6,000 artworks plotted as a UMAP scatter, coloured by cluster.
// Hovering plays a note; clicking triggers a ripple cascade. Uses Tone.js for audio.

import p5 from 'p5';
import { AZURE_DATA_URL, AZURE_SAS_TOKEN, azureFallback } from '../util/azure-config.js';

const CLUSTER_COLORS_HEX = [
  '#6c63ff','#ff6b8a','#50e3c2','#f5a623','#bd10e0',
  '#7ed321','#4a90d9','#e86c5d','#b8e986','#d0021b',
  '#9013fe','#00bcd4',
];

// One pentatonic scale per cluster, used to map artworks to musical notes
const SCALES = [
  ['C4','D4','E4','G4','A4'],   ['D4','E4','F#4','A4','B4'],
  ['E4','F#4','G#4','B4','C#5'],['G3','A3','B3','D4','E4'],
  ['A3','B3','C#4','E4','F#4'], ['C5','D5','E5','G5','A5'],
  ['F3','G3','A3','C4','D4'],   ['Bb3','C4','D4','F4','G4'],
  ['Eb4','F4','G4','Bb4','C5'], ['Ab3','Bb3','C4','Eb4','F4'],
  ['B3','C#4','D#4','F#4','G#4'],['F#3','G#3','A#3','C#4','D#4'],
];
const DRONE_NOTES = ['C2','D2','E2','G2','A2','C3','F2','Bb2','Eb3','Ab2','B2','F#2'];

const DATA_URL       = `${AZURE_DATA_URL}/artworks_sonification.json?${AZURE_SAS_TOKEN}`;
const FEATURE_LABELS = ['Structure','Palette','Texture'];
const FEATURE_KEYS   = ['structure','palette','texture'];

const PAN_MARGIN = 120; // minimum pixels of data kept on screen while panning

export function createEchoesSketch(containerEl) {
  let artworks = null, centroids = null, loadError = null;

  fetch(DATA_URL)
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(d => { artworks = d.artworks; centroids = d.centroids; })
    .catch(e => { loadError = e.message; });

  // HTML tooltip — using a DOM element instead of canvas drawing avoids CORS issues with external images
  const htmlTip = document.createElement('div');
  htmlTip.style.cssText =
    'position:fixed; display:none; pointer-events:none; z-index:2000;' +
    'width:230px; background:rgba(10,10,16,0.97);' +
    'border:1px solid rgba(255,255,255,0.08); border-radius:8px;' +
    'overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,0.72);' +
    'font-family:"SF Mono",ui-monospace,monospace; font-size:10px; color:#888;';

  const htImg = document.createElement('img');
  htImg.style.cssText =
    'width:100%; height:140px; object-fit:cover; display:none; background:#0a0a12;';
  htImg.alt = '';

  const htBody = document.createElement('div');
  htBody.style.cssText = 'padding:9px 11px 11px;';

  const htSwatch  = document.createElement('div');
  htSwatch.style.cssText = 'height:2px; width:100%; border-radius:1px; margin-bottom:8px;';

  const htTitle   = document.createElement('div');
  htTitle.style.cssText =
    'color:#ddd; margin-bottom:2px; font-size:11px;' +
    'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';

  const htArtist  = document.createElement('div');
  htArtist.style.cssText =
    'color:#555; margin-bottom:8px; font-size:9px;' +
    'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';

  const htMeta    = document.createElement('div');
  htMeta.style.cssText = 'display:flex; align-items:center; gap:7px; margin-bottom:7px;';

  const htCluster = document.createElement('span');
  htCluster.style.cssText =
    'display:inline-block; padding:2px 7px; border-radius:3px; font-size:9px;';

  const htEnergy  = document.createElement('span');
  htEnergy.style.cssText = 'color:#444; font-size:9px;';

  htMeta.append(htCluster, htEnergy);

  const htBars    = document.createElement('div');
  htBars.style.cssText = 'display:flex; flex-direction:column; gap:4px;';

  const htNeigh   = document.createElement('div');
  htNeigh.style.cssText =
    'border-top:1px solid rgba(255,255,255,0.05); margin-top:8px;' +
    'padding-top:8px; display:none;';

  htBody.append(htSwatch, htTitle, htArtist, htMeta, htBars, htNeigh);
  htmlTip.append(htImg, htBody);
  document.body.appendChild(htmlTip);

  let screenMX = 0, screenMY = 0;
  containerEl.addEventListener('mousemove', (e) => {
    screenMX = e.clientX; screenMY = e.clientY;
    if (htmlTip.style.display !== 'none') _posHTMLTip();
  });
  containerEl.addEventListener('mouseleave', () => { htmlTip.style.display = 'none'; });

  function _posHTMLTip() {
    const TW = 230, TH = htmlTip.offsetHeight || 220;
    let lx = screenMX + 16, ly = screenMY - Math.floor(TH / 2);
    if (lx + TW > window.innerWidth  - 8) lx = screenMX - TW - 16;
    if (ly < 8) ly = 8;
    if (ly + TH > window.innerHeight - 8) ly = window.innerHeight - TH - 8;
    htmlTip.style.left = lx + 'px';
    htmlTip.style.top  = ly + 'px';
  }

  function _showHTMLTip(art, isActive) {
    // Try Azure blob URL first (reliable), fall back to original museum CDN
    const imgSrc = azureFallback(art.image_url) || art.image_url;
    if (imgSrc && htImg.dataset.src !== imgSrc) {
      htImg.dataset.src = imgSrc;
      htImg.style.display = 'none';
      htImg.onload  = () => { htImg.style.display = 'block'; };
      htImg.onerror = () => { htImg.style.display = 'none'; };
      htImg.src = imgSrc;
    } else if (imgSrc && htImg.complete && htImg.naturalWidth > 0) {
      htImg.style.display = 'block';
    }

    const hex = CLUSTER_COLORS_HEX[art.cluster % 12];
    htSwatch.style.background = art.dominant_color || '#333';

    const t = art.title || 'Untitled';
    htTitle.textContent = t.length > 34 ? t.slice(0, 32) + '…' : t;
    htTitle.title = t;

    const a = art.artist || 'Unknown artist';
    htArtist.textContent = a.length > 38 ? a.slice(0, 36) + '…' : a;

    htCluster.style.cssText =
      `display:inline-block; padding:2px 7px; border-radius:3px; font-size:9px;` +
      `background:${hex}22; color:${hex};`;
    htCluster.textContent = `Cluster ${art.cluster}`;
    htEnergy.textContent  = `Energy ${(art.energy ?? 0).toFixed(2)}`;

    // Feature bars
    htBars.innerHTML = '';
    if (art.features) {
      FEATURE_KEYS.forEach((k, i) => {
        const val = art.features[k] ?? 0;
        const vp  = (val * 100).toFixed(0);
        const row = document.createElement('div');
        row.style.cssText = 'display:flex; align-items:center; gap:6px;';
        row.innerHTML =
          `<span style="color:#3a3a4a;font-size:8.5px;width:52px;text-align:right;flex-shrink:0;">${FEATURE_LABELS[i]}</span>` +
          `<div style="flex:1;height:5px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;">` +
            `<div style="height:100%;width:${vp}%;background:${hex};border-radius:2px;opacity:0.8;"></div></div>` +
          `<span style="color:#3a3a4a;font-size:8.5px;width:26px;">${vp}%</span>`;
        htBars.appendChild(row);
      });
    }

    // Neighbour list (only when clicked/active)
    if (isActive && art.neighbors?.length) {
      htNeigh.style.display = 'block';
      htNeigh.innerHTML = '<span style="color:#3a3a4a;font-size:8.5px;">Most similar:</span>';
      art.neighbors.slice(0, 3).forEach((ni, k) => {
        const nb = artworks[ni];
        if (!nb) return;
        const sim   = ((1 - (art.nbDists?.[k] ?? 0.5)) * 100).toFixed(0);
        const nbHex = CLUSTER_COLORS_HEX[nb.cluster % 12];
        const nbT   = nb.title || 'Untitled';
        const row   = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:6px;margin-top:5px;';
        row.innerHTML =
          `<span style="display:inline-block;width:6px;height:6px;border-radius:50%;` +
          `background:${nbHex};flex-shrink:0;"></span>` +
          `<span style="color:#555;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">` +
          `${sim}% — ${nbT.length > 22 ? nbT.slice(0, 20) + '…' : nbT}</span>`;
        htNeigh.appendChild(row);
      });
    } else {
      htNeigh.style.display = 'none';
    }

    htmlTip.style.display = 'block';
    _posHTMLTip();
  }

  return new p5((p) => {
    const CELL = 40;
    const NOTE_COOLDOWN = 120;

    let clusterColorsP5 = []; // p5 colour objects cached once after setup

    let W, H;
    let hoveredIdx = -1, activeIdx = -1;
    let ripples = [];

    let grid = new Map(); // spatial hash grid for fast nearest-neighbour lookup
    let phases = null;    // per-artwork micro-motion phase offsets

    let offsetX = 0, offsetY = 0, zoom = 1;
    let dragging = false, dragStartX = 0, dragStartY = 0, dragOffX = 0, dragOffY = 0;

    let noteSynths = null, droneSynths = null, droneFilters = null, droneGains = null;
    let audioReady = false, lastNoteTime = 0;

    let listenMode = false, listenT = 0, listenPath = [];

    /* ── Control buttons (injected HTML) ─────────────────────────────── */
    let btnZoomIn, btnZoomOut, btnReset;

    function injectControls() {
      const wrap = document.createElement('div');
      wrap.style.cssText = `
        position:absolute; bottom:32px; right:14px;
        display:flex; flex-direction:column; gap:6px; z-index:10;
      `;
      const btnStyle = `
        width:34px; height:34px; border-radius:8px;
        border:1px solid rgba(255,255,255,0.12);
        background:rgba(20,20,28,0.75); backdrop-filter:blur(8px);
        color:#ccc; font-size:1rem; cursor:pointer; line-height:1;
        display:flex; align-items:center; justify-content:center;
        transition:background 0.15s, color 0.15s;
      `;
      btnZoomIn  = document.createElement('button');
      btnZoomOut = document.createElement('button');
      btnReset   = document.createElement('button');
      btnZoomIn.innerHTML  = '+';
      btnZoomOut.innerHTML = '−';
      btnReset.innerHTML   = '⌖';
      btnZoomIn.title  = 'Zoom in';
      btnZoomOut.title = 'Zoom out';
      btnReset.title   = 'Reset view';
      [btnZoomIn, btnZoomOut, btnReset].forEach(b => {
        b.style.cssText = btnStyle;
        b.addEventListener('mouseenter', () => {
          b.style.background = 'rgba(14,165,233,0.35)';
          b.style.color = '#fff';
        });
        b.addEventListener('mouseleave', () => {
          b.style.background = 'rgba(20,20,28,0.75)';
          b.style.color = '#ccc';
        });
      });
      btnZoomIn.addEventListener('click',  () => applyZoom(1.3, W/2, H/2));
      btnZoomOut.addEventListener('click', () => applyZoom(0.77, W/2, H/2));
      btnReset.addEventListener('click',   () => { offsetX = 0; offsetY = 0; zoom = 1; });
      wrap.append(btnZoomIn, btnZoomOut, btnReset);
      containerEl.style.position = 'relative';
      containerEl.appendChild(wrap);
    }

    function applyZoom(factor, mx, my) {
      offsetX = mx - (mx - offsetX) * factor;
      offsetY = my - (my - offsetY) * factor;
      zoom = p.constrain(zoom * factor, 0.3, 8);
      clampOffset();
    }

    /* ── Helpers ─────────────────────────────────────────────────────── */

    function artX(a) { return a.x * (W - 40) + 20; }
    function artY(a) { return a.y * (H - 40) + 20; }

    function artXA(a, i) {
      return artX(a) + (phases ? Math.sin(phases[i]) * 0.5 : 0);
    }
    function artYA(a, i) {
      return artY(a) + (phases ? Math.cos(phases[i] * 1.3) * 0.5 : 0);
    }

    function screenToWorld(sx, sy) {
      return [(sx - offsetX) / zoom, (sy - offsetY) / zoom];
    }

    function dotRadius(a) { return p.map(a.energy, 0, 1, 2.5, 6); }

    /* ── Pan constraint ──────────────────────────────────────────────── */
    // Prevent panning so far that all dots go off-screen
    function clampOffset() {
      const wMinX = 20,  wMaxX = W - 20;
      const wMinY = 20,  wMaxY = H - 20;
      if (offsetX + wMaxX * zoom < PAN_MARGIN)        offsetX = PAN_MARGIN - wMaxX * zoom;
      if (offsetX + wMinX * zoom > W - PAN_MARGIN)    offsetX = W - PAN_MARGIN - wMinX * zoom;
      if (offsetY + wMaxY * zoom < PAN_MARGIN)        offsetY = PAN_MARGIN - wMaxY * zoom;
      if (offsetY + wMinY * zoom > H - PAN_MARGIN)    offsetY = H - PAN_MARGIN - wMinY * zoom;
    }

    /* ── Spatial grid ────────────────────────────────────────────────── */
    function buildGrid() {
      grid.clear();
      for (let i = 0; i < artworks.length; i++) {
        const key = `${Math.floor(artX(artworks[i]) / CELL)}_${Math.floor(artY(artworks[i]) / CELL)}`;
        if (!grid.has(key)) grid.set(key, []);
        grid.get(key).push(i);
      }
    }

    function findNearest(mx, my) {
      if (!artworks) return -1;
      const [wx, wy] = screenToWorld(mx, my);
      const cx = Math.floor(wx / CELL), cy = Math.floor(wy / CELL);
      let best = -1, bestD = 18 / zoom;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const cell = grid.get(`${cx+dx}_${cy+dy}`);
          if (!cell) continue;
          for (const i of cell) {
            const ddx = artX(artworks[i]) - wx, ddy = artY(artworks[i]) - wy;
            const d = Math.sqrt(ddx*ddx + ddy*ddy);
            if (d < bestD) { bestD = d; best = i; }
          }
        }
      }
      return best;
    }


    /* ── Audio ───────────────────────────────────────────────────────── */
    function ensureAudio() {
      const Tone = window.Tone;
      if (!Tone || audioReady) return audioReady;
      try {
        const masterReverb = new Tone.Reverb({ decay: 4, wet: 0.45 }).toDestination();
        const masterDelay  = new Tone.FeedbackDelay('8n', 0.3).connect(masterReverb);

        noteSynths = SCALES.map(() => {
          const s = new Tone.PolySynth(Tone.Synth, {
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.04, decay: 0.3, sustain: 0.08, release: 1.8 },
          }).connect(masterDelay);
          s.volume.value = -16;
          return s;
        });

        const droneReverb = new Tone.Reverb({ decay: 8, wet: 0.7 }).toDestination();
        droneSynths = []; droneFilters = []; droneGains = [];
        for (let c = 0; c < 12; c++) {
          const gain   = new Tone.Gain(0).connect(droneReverb);
          const filter = new Tone.Filter(200, 'lowpass').connect(gain);
          const synth  = new Tone.Synth({
            oscillator: { type: 'sine' },
            envelope: { attack: 2, decay: 0, sustain: 1, release: 3 },
          }).connect(filter);
          synth.volume.value = -28;
          synth.triggerAttack(DRONE_NOTES[c]);
          droneSynths.push(synth); droneFilters.push(filter); droneGains.push(gain);
        }
        audioReady = true;
      } catch {}
      return audioReady;
    }

    function playNote(art) {
      if (!audioReady || !noteSynths) return;
      const now = performance.now();
      if (now - lastNoteTime < NOTE_COOLDOWN) return;
      lastNoteTime = now;
      const scale = SCALES[art.cluster % 12];
      const note = scale[Math.floor(art.energy * (scale.length - 1))];
      const s = noteSynths[art.cluster % noteSynths.length];
      s.set({ detune: p.map(art.energy, 0, 1, 0, 18) });
      s.triggerAttackRelease(note, '8n', undefined, p.map(art.energy, 0, 1, 0.15, 0.5));
    }

    function playRipple(art) {
      if (!audioReady || !noteSynths) return;
      const scale = SCALES[art.cluster % 12];
      const notes = [scale[Math.floor(art.energy * (scale.length - 1))]];
      for (let k = 0; k < Math.min(2, art.neighbors.length); k++) {
        const nb = artworks[art.neighbors[k]];
        if (nb) {
          const ns = SCALES[nb.cluster % 12];
          notes.push(ns[Math.floor(nb.energy * (ns.length - 1))]);
        }
      }
      noteSynths[art.cluster % noteSynths.length].triggerAttackRelease(
        [...new Set(notes)], '4n', undefined, p.map(art.energy, 0, 1, 0.2, 0.55)
      );
    }

    function updateDrones() {
      if (!audioReady || !droneGains) return;
      const [wx1, wy1] = screenToWorld(0, 0);
      const [wx2, wy2] = screenToWorld(W, H);
      const counts = new Float32Array(12);
      let total = 0;
      for (const a of artworks) {
        const ax = artX(a), ay = artY(a);
        if (ax >= wx1 && ax <= wx2 && ay >= wy1 && ay <= wy2) { counts[a.cluster]++; total++; }
      }
      for (let c = 0; c < 12; c++) {
        const act = total > 0 ? counts[c] / total : 0;
        droneGains[c].gain.rampTo(act * 0.7, 0.5);
        droneFilters[c].frequency.rampTo(200 + act * 2800, 0.5);
      }
    }

    /* ── Listen mode ─────────────────────────────────────────────────── */
    function buildListenPath() {
      if (!centroids) return;
      const visited = new Set([0]);
      const path = [0];
      while (path.length < centroids.length) {
        const last = centroids[path[path.length - 1]];
        let bestC = -1, bestD = Infinity;
        for (let c = 0; c < centroids.length; c++) {
          if (visited.has(c)) continue;
          const dx = centroids[c].x - last.x, dy = centroids[c].y - last.y;
          const d = dx*dx + dy*dy;
          if (d < bestD) { bestD = d; bestC = c; }
        }
        if (bestC < 0) break;
        path.push(bestC); visited.add(bestC);
      }
      path.push(path[0]);
      listenPath = path;
    }

    function updateListenMode() {
      if (!listenMode || !centroids || listenPath.length < 2) return;
      listenT += p.deltaTime * 0.001 * 0.008;
      if (listenT >= listenPath.length - 1) listenT = 0;
      const seg = Math.floor(listenT), frac = listenT - seg;
      const c1 = centroids[listenPath[seg]];
      const c2 = centroids[listenPath[(seg + 1) % listenPath.length]];
      const tx = p.lerp(c1.x, c2.x, frac) * (W - 40) + 20;
      const ty = p.lerp(c1.y, c2.y, frac) * (H - 40) + 20;
      offsetX = p.lerp(offsetX, W/2 - tx * zoom, 0.02);
      offsetY = p.lerp(offsetY, H/2 - ty * zoom, 0.02);
      zoom = p.lerp(zoom, 2.5, 0.01);
      if (p.frameCount % 8 === 0) {
        const [wx, wy] = screenToWorld(W/2, H/2);
        const cx = Math.floor(wx / CELL), cy = Math.floor(wy / CELL);
        let closest = -1, closestD = Infinity;
        for (let dx = -2; dx <= 2; dx++) {
          for (let dy = -2; dy <= 2; dy++) {
            const cell = grid.get(`${cx+dx}_${cy+dy}`);
            if (!cell) continue;
            for (const i of cell) {
              const ddx = artX(artworks[i]) - wx, ddy = artY(artworks[i]) - wy;
              const d = ddx*ddx + ddy*ddy;
              if (d < closestD) { closestD = d; closest = i; }
            }
          }
        }
        if (closest >= 0) {
          const ca = artworks[closest];
          playNote(ca);
          ripples.push({ cx: artX(ca), cy: artY(ca), t: 0, maxR: 40, cluster: ca.cluster });
        }
      }
    }

    // Textured dot drawing helpers (noise-distorted concentric rings)

    function drawNoiseRing(radius, noiseAmt, nOffX, nOffY, numVerts) {
      p.beginShape();
      for (let j = 0; j <= numVerts; j++) {
        const angle = (j / numVerts) * p.TWO_PI;
        const n = p.noise(p.cos(angle) * 0.6 + nOffX, p.sin(angle) * 0.6 + nOffY);
        const d = radius + p.map(n, 0, 1, -noiseAmt, noiseAmt);
        p.vertex(d * p.cos(angle), d * p.sin(angle));
      }
      p.endShape(p.CLOSE);
    }

    // Each dot is drawn as layered noise-distorted rings. The noise offset is derived
    // from the artwork's UMAP position so each artwork has a unique but stable shape.
    function drawTexturedDot(ax, ay, a, r, col, cr, cg, cb, isFocused, isNeighbor) {
      const nX = a.x * 6 + a.cluster * 0.7;
      const nY = a.y * 6 + a.cluster * 0.5;
      const nStrength = a.energy || 0.5;

      p.push();
      p.translate(ax, ay);
      p.noFill();

      if (isFocused) {
        p.noStroke(); p.fill(cr, cg, cb, 55);
        p.ellipse(0, 0, r * 5, r * 5);
        for (let i = 0; i < 9; i++) {
          const ri    = p.map(i, 0, 8, r * 0.3, r * 2.2);
          const alpha = p.map(i, 0, 8, 220, 30);
          p.stroke(cr, cg, cb, alpha);
          p.strokeWeight(0.9 / zoom);
          drawNoiseRing(ri, nStrength * ri * 0.28, nX, nY, 12);
        }
        p.noStroke(); p.fill(col);
        p.ellipse(0, 0, r * 1.1, r * 1.1);

      } else if (isNeighbor) {
        p.noStroke(); p.fill(cr, cg, cb, 28);
        p.ellipse(0, 0, r * 3.2, r * 3.2);
        for (let i = 0; i < 4; i++) {
          const ri    = p.map(i, 0, 3, r * 0.3, r * 1.5);
          const alpha = p.map(i, 0, 3, 180, 35);
          p.stroke(cr, cg, cb, alpha);
          p.strokeWeight(0.7 / zoom);
          drawNoiseRing(ri, nStrength * ri * 0.25, nX, nY, 10);
        }
        p.noStroke(); p.fill(cr, cg, cb, 190);
        p.ellipse(0, 0, r * 0.7, r * 0.7);

      } else if (r * zoom < 1.8) {
        // too small to render rings — plain dot
        p.noStroke(); p.fill(cr, cg, cb, 140);
        p.ellipse(0, 0, r, r);

      } else {
        for (let i = 0; i < 2; i++) {
          const ri    = p.map(i, 0, 1, r * 0.35, r * 0.9);
          const alpha = p.map(i, 0, 1, 160, 55);
          p.stroke(cr, cg, cb, alpha);
          p.strokeWeight(0.5 / zoom);
          drawNoiseRing(ri, nStrength * ri * 0.22, nX, nY, 8);
        }
        p.noStroke(); p.fill(cr, cg, cb, 130);
        p.ellipse(0, 0, r * 0.38, r * 0.38);
      }
      p.pop();
    }

    /* ── p5 setup ────────────────────────────────────────────────────── */
    p.setup = () => {
      W = containerEl.offsetWidth  || 900;
      H = containerEl.offsetHeight || 600;
      const c = p.createCanvas(W, H);
      c.parent(containerEl);
      p.textFont('system-ui, sans-serif');
      p.frameRate(30);
      clusterColorsP5 = CLUSTER_COLORS_HEX.map(hex => p.color(hex));
      injectControls();
    };

    /* ── p5 draw ─────────────────────────────────────────────────────── */
    p.draw = () => {
      W = p.width; H = p.height;
      p.background(14, 14, 16);

      if (loadError) {
        p.fill(200); p.noStroke(); p.textAlign(p.CENTER, p.CENTER); p.textSize(14);
        p.text('Failed to load data: ' + loadError, W/2, H/2);
        return;
      }
      if (!artworks) {
        p.fill(136); p.noStroke(); p.textAlign(p.CENTER, p.CENTER); p.textSize(13);
        p.text('Loading 6 000 artworks…', W/2, H/2);
        return;
      }

      if (!phases) { // initialise once data is loaded
        phases = new Float32Array(artworks.length);
        for (let i = 0; i < artworks.length; i++) phases[i] = Math.random() * Math.PI * 2;
        buildGrid();
        buildListenPath();
      }

      const dt = p.deltaTime * 0.001;
      for (let i = 0; i < phases.length; i++) phases[i] += dt * (0.3 + artworks[i].energy * 0.5);

      if (listenMode) updateListenMode();
      if (p.frameCount % 10 === 0) updateDrones();

      p.push();
      p.translate(offsetX, offsetY);
      p.scale(zoom);

      // Soft cluster fog behind the dots
      if (centroids) {
        p.noStroke();
        for (let c = 0; c < centroids.length; c++) {
          const ct = centroids[c];
          const cx = ct.x * (W-40) + 20, cy = ct.y * (H-40) + 20;
          const col = clusterColorsP5[c % 12];
          const r = Math.sqrt(ct.count) * 8;
          p.fill(p.red(col), p.green(col), p.blue(col), 6);
          p.ellipse(cx, cy, r, r);
          p.fill(p.red(col), p.green(col), p.blue(col), 3);
          p.ellipse(cx, cy, r * 1.6, r * 1.6);
        }
      }

      const focusIdx = activeIdx >= 0 ? activeIdx : hoveredIdx;
      const neighborSet = focusIdx >= 0 ? new Set(artworks[focusIdx].neighbors) : null;

      // Lines connecting the focused artwork to its nearest neighbours
      if (focusIdx >= 0) {
        const fa = artworks[focusIdx];
        p.strokeWeight(0.8 / zoom);
        for (let k = 0; k < fa.neighbors.length; k++) {
          const nb = artworks[fa.neighbors[k]];
          if (!nb) continue;
          const dist = fa.nbDists?.[k] ?? 0.5;
          const alpha = p.map(dist, 0, 0.5, 60, 15);
          p.stroke(255, 255, 255, alpha);
          p.line(artXA(fa, focusIdx), artYA(fa, focusIdx), artXA(nb, fa.neighbors[k]), artYA(nb, fa.neighbors[k]));
        }
      }

      for (let r = ripples.length - 1; r >= 0; r--) {
        const rp = ripples[r];
        rp.t += dt;
        const prog = rp.t / 1.5;
        if (prog > 1) { ripples.splice(r, 1); continue; }
        const col = clusterColorsP5[rp.cluster % 12];
        p.noFill();
        p.stroke(p.red(col), p.green(col), p.blue(col), p.lerp(120, 0, prog));
        p.strokeWeight(1.5 / zoom);
        p.ellipse(rp.cx, rp.cy, p.lerp(0, rp.maxR*2, prog), p.lerp(0, rp.maxR*2, prog));
      }

      for (let i = 0; i < artworks.length; i++) {
        const a  = artworks[i];
        const ax = artXA(a, i), ay = artYA(a, i);
        const r  = dotRadius(a);
        const col = clusterColorsP5[a.cluster % 12];
        drawTexturedDot(ax, ay, a, r,
          col, p.red(col), p.green(col), p.blue(col),
          i === focusIdx,
          neighborSet != null && neighborSet.has(i));
      }
      p.pop();

      p.noStroke();
      p.fill(255, 255, 255, 22);
      p.textAlign(p.LEFT, p.TOP);
      p.textSize(11);
      p.textStyle(p.ITALIC);
      p.text('You are listening to how an AI sees similarity.', 16, 12);
      p.text("Each sound represents proximity inside a neural network's perception space.", 16, 28);
      p.textStyle(p.NORMAL);

      p.fill(72); p.textAlign(p.LEFT, p.BOTTOM); p.textSize(10);
      const hints = ['Scroll: zoom','Drag: pan','Hover/click: explore',
                     listenMode ? 'L: stop listen' : 'L: listen mode'];
      p.text(hints.join('  ·  '), 12, H - 8);
      if (!audioReady) {
        p.textAlign(p.RIGHT, p.BOTTOM);
        p.fill(CLUSTER_COLORS_HEX[0]);
        p.text('Click to enable sound', W - 12, H - 8);
      }
    };


    /* ── Interaction ─────────────────────────────────────────────────── */
    p.mouseMoved = () => {
      if (dragging) return;
      const prev = hoveredIdx;
      hoveredIdx = findNearest(p.mouseX, p.mouseY);
      if (hoveredIdx !== prev) {
        if (hoveredIdx >= 0) {
          _showHTMLTip(artworks[hoveredIdx], activeIdx === hoveredIdx);
          playNote(artworks[hoveredIdx]);
        } else {
          htmlTip.style.display = 'none';
        }
      }
      containerEl.style.cursor = hoveredIdx >= 0 ? 'pointer' : 'grab';
    };

    p.mousePressed = (e) => {
      if (e?.target?.tagName !== 'CANVAS') return;
      const Tone = window.Tone;
      if (Tone && Tone.context.state !== 'running') Tone.start();
      ensureAudio();

      const idx = findNearest(p.mouseX, p.mouseY);
      if (idx >= 0) {
        activeIdx = idx;
        const art = artworks[idx];
        _showHTMLTip(art, true);   // refresh tooltip with neighbour section
        ripples.push({ cx: artX(art), cy: artY(art), t: 0, maxR: 80, cluster: art.cluster });
        playRipple(art);
        art.neighbors.forEach((ni, k) => {
          const nb = artworks[ni];
          if (!nb) return;
          const dist = art.nbDists?.[k] ?? 0.3;
          const delay = p.map(dist, 0, 0.5, 80, 400);
          setTimeout(() => {
            ripples.push({ cx: artX(nb), cy: artY(nb), t: 0, maxR: 50, cluster: nb.cluster });
            if (audioReady && noteSynths) {
              const scale = SCALES[nb.cluster % 12];
              const note = scale[Math.floor(nb.energy * (scale.length - 1))];
              const s = noteSynths[nb.cluster % noteSynths.length];
              s.set({ detune: p.map(dist, 0, 0.5, 0, 25) });
              s.triggerAttackRelease(note, '8n', undefined, 0.2);
            }
          }, delay);
        });
        if (listenMode) listenMode = false;
      } else {
        activeIdx = -1;
        if (hoveredIdx >= 0) _showHTMLTip(artworks[hoveredIdx], false);
        dragging = true;
        dragStartX = p.mouseX; dragStartY = p.mouseY;
        dragOffX = offsetX;    dragOffY = offsetY;
        containerEl.style.cursor = 'grabbing';
      }
    };

    p.mouseDragged = () => {
      if (!dragging) return;
      offsetX = dragOffX + (p.mouseX - dragStartX);
      offsetY = dragOffY + (p.mouseY - dragStartY);
      clampOffset();
    };

    p.mouseReleased = () => {
      dragging = false;
      containerEl.style.cursor = hoveredIdx >= 0 ? 'pointer' : 'grab';
    };

    p.mouseWheel = (e) => {
      if (e?.target?.tagName !== 'CANVAS') return;
      e.preventDefault?.();
      applyZoom(e.delta > 0 ? 0.9 : 1.1, p.mouseX, p.mouseY);
    };

    p.keyPressed = () => {
      if (p.key === 'l' || p.key === 'L') {
        listenMode = !listenMode;
        if (listenMode) {
          listenT = 0;
          const Tone = window.Tone;
          if (Tone && Tone.context.state !== 'running') Tone.start();
          ensureAudio();
        }
      }
    };

    p.windowResized = () => {
      W = containerEl.offsetWidth || 900;
      H = containerEl.offsetHeight || 600;
      p.resizeCanvas(W, H);
      if (artworks) buildGrid();
    };

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    p.cleanup = () => {
      htmlTip.remove();
      containerEl.querySelectorAll('div').forEach(el => {
        if (el.contains(btnZoomIn)) el.remove();
      });
      noteSynths?.forEach(s => { try { s.dispose(); } catch {} });
      droneSynths?.forEach(s => { try { s.triggerRelease(); s.dispose(); } catch {} });
      droneFilters?.forEach(f => { try { f.dispose(); } catch {} });
      droneGains?.forEach(g => { try { g.dispose(); } catch {} });
      noteSynths = droneSynths = droneFilters = droneGains = null;
      audioReady = false;
    };

  }, containerEl);
}
