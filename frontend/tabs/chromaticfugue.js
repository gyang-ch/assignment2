// Chromatic Fugue: 6,000 artworks shown as colour stripes.
// Each stripe's colour = dominant colour, height = visual complexity.
// Hover plays a sound mapped to colour and complexity. Click opens a detail panel.

import { AZURE_DATA_URL, AZURE_SAS_TOKEN } from '../util/azure-config.js';
const DATA_URL    = `${AZURE_DATA_URL}/style_entropy.json?${AZURE_SAS_TOKEN}`;
const CONTROLS_H  = 44;
const COL_GAP     = 1;
const ROW_GAP     = 3;
const TARGET_ROWS = 20;

const MODES = [
  { key: 'cluster',       label: 'Cluster'       },
  { key: 'hue',           label: 'Hue'           },
  { key: 'gradient',      label: 'Gradient'      },
  { key: 'constellation', label: 'Constellation' },
  { key: 'asc',           label: 'Complexity'  },
];
const KEY_MAP = Object.fromEntries(MODES.map((m, i) => [String(i + 1), m.key]));

function hexToHsl(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l];
  const d = mx - mn;
  const s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  let h;
  switch (mx) {
    case r: h = (g - b) / d + (g < b ? 6 : 0); break;
    case g: h = (b - r) / d + 2; break;
    default: h = (r - g) / d + 4;
  }
  return [h / 6, s, l];
}

function clip(str, n) {
  return str && str.length > n ? str.slice(0, n) + '…' : (str || '');
}

export function createStyleEntropySketch(container) {

  let stripes        = [];
  let sorted         = [];
  let sortMode       = 'cluster';
  let stripeW        = 4;   // recomputed on resize
  let rowH           = 40;
  let fillerCount    = 0;   // blank divs to complete the last row
  let lastHoveredIdx = -1;

  // Audio mappings: hue→pitch, complexity→reverb, cluster→FMSynth preset
  const PENTATONIC = ['C3','D3','E3','G3','A3','C4','D4','E4','G4','A4','C5'];

  // One FMSynth timbre per cluster so each cluster sounds distinct
  const CLUSTER_PRESETS = [
    { harmonicity:3,   modulationIndex:8,  oscillator:{type:'sine'},     modulation:{type:'sine'},     envelope:{attack:0.01, decay:0.30, sustain:0,    release:0.6} }, // 0 luminous glass
    { harmonicity:2,   modulationIndex:5,  oscillator:{type:'sine'},     modulation:{type:'sine'},     envelope:{attack:0.02, decay:0.40, sustain:0,    release:0.9} }, // 1 warm bell
    { harmonicity:7,   modulationIndex:12, oscillator:{type:'triangle'}, modulation:{type:'square'},   envelope:{attack:0.005,decay:0.18, sustain:0,    release:0.3} }, // 2 digital shimmer
    { harmonicity:1,   modulationIndex:2,  oscillator:{type:'sine'},     modulation:{type:'sine'},     envelope:{attack:0.05, decay:0.40, sustain:0.1,  release:1.1} }, // 3 soft breath
    { harmonicity:4,   modulationIndex:6,  oscillator:{type:'triangle'}, modulation:{type:'sine'},     envelope:{attack:0.01, decay:0.35, sustain:0,    release:0.6} }, // 4 electric chime
    { harmonicity:5,   modulationIndex:15, oscillator:{type:'sine'},     modulation:{type:'sawtooth'}, envelope:{attack:0.005,decay:0.14, sustain:0,    release:0.4} }, // 5 metallic pluck
    { harmonicity:1,   modulationIndex:1,  oscillator:{type:'sine'},     modulation:{type:'sine'},     envelope:{attack:0.03, decay:0.50, sustain:0.2,  release:0.9} }, // 6 organ whisper
    { harmonicity:6,   modulationIndex:10, oscillator:{type:'sine'},     modulation:{type:'triangle'}, envelope:{attack:0.008,decay:0.25, sustain:0,    release:0.5} }, // 7 crystalline
    { harmonicity:2,   modulationIndex:20, oscillator:{type:'sine'},     modulation:{type:'square'},   envelope:{attack:0.001,decay:0.10, sustain:0,    release:0.2} }, // 8 percussive snap
    { harmonicity:1.5, modulationIndex:3,  oscillator:{type:'sine'},     modulation:{type:'triangle'}, envelope:{attack:0.08, decay:0.50, sustain:0.15, release:1.3} }, // 9 floating pad
    { harmonicity:8,   modulationIndex:14, oscillator:{type:'triangle'}, modulation:{type:'sine'},     envelope:{attack:0.005,decay:0.18, sustain:0,    release:0.35}}, // 10 bright zing
    { harmonicity:0.5, modulationIndex:4,  oscillator:{type:'sine'},     modulation:{type:'sine'},     envelope:{attack:0.04, decay:0.60, sustain:0.1,  release:1.0} }, // 11 deep pulse
  ];

  let _synth      = null;
  let _reverb     = null;
  let _toneOk     = false;
  let _hoverTimer = null;  // debounce rapid stripe crossings

  function _initAudio() {
    if (_toneOk || !window.Tone) return;
    try {
      _reverb = new Tone.Freeverb({ roomSize: 0.2, dampening: 3000, wet: 0 }).toDestination();
      _synth  = new Tone.PolySynth(Tone.FMSynth).connect(_reverb);
      _synth.volume.value = -6;
      _toneOk = true;
    } catch {}
  }

  // Maps visual traits to sound
  function _playStripe(stripe) {
    if (!window.Tone) return;
    Tone.start();
    if (!_toneOk) _initAudio();
    if (!_toneOk) return;

    const noteIdx = Math.round(stripe.hue * (PENTATONIC.length - 1));
    const note    = PENTATONIC[noteIdx];

    const wet      = stripe.art.complexity * 0.60;
    const roomSize = 0.1 + stripe.art.complexity * 0.85;
    _reverb.wet.rampTo(wet, 0.06);
    _reverb.roomSize.rampTo(roomSize, 0.06);

    const preset = CLUSTER_PRESETS[stripe.art.cluster % CLUSTER_PRESETS.length];
    _synth.set(preset);

    const dur = 0.12 + stripe.art.complexity * 0.38;
    _synth.triggerAttackRelease(note, dur);
  }

  container.style.cssText =
    'position:relative; width:100%; height:100%; background:#090910;' +
    'display:flex; flex-direction:column; overflow:hidden;';

  const controls = document.createElement('div');
  controls.style.cssText =
    `height:${CONTROLS_H}px; flex-shrink:0; display:flex; align-items:center;` +
    'gap:5px; padding:0 14px;' +
    'background:rgba(14,14,16,0.97); border-bottom:1px solid rgba(255,255,255,0.07);';
  container.appendChild(controls);

  const sortLabel = document.createElement('span');
  sortLabel.style.cssText =
    'font-family:"SF Mono",ui-monospace,monospace; font-size:11px;' +
    'color:#555; margin-right:4px; flex-shrink:0;';
  sortLabel.textContent = 'Sort:';
  controls.appendChild(sortLabel);

  const sortBtns = {};
  for (const m of MODES) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.dataset.mode  = m.key;
    btn.className = 'glow-btn-group probe-search-btn';
    btn.style.cssText =
      'flex-shrink:0; border-radius:7px;';

    const fill = document.createElement('div');
    fill.className = 'glow-btn-fill';
    fill.style.cssText =
      'border-radius:5px; will-change:transform;';

    const content = document.createElement('div');
    content.className = 'glow-btn-content';
    content.style.cssText =
      'padding:3px 9px; border-radius:5px;' +
      'background:#0d0d10; color:#555;' +
      'font-size:11px; font-family:"SF Mono",ui-monospace,monospace;' +
      'font-weight:500; letter-spacing:0.02em; text-transform:none;';
    content.textContent = m.label;

    btn._fill    = fill;
    btn._content = content;
    btn.append(fill, content);

    btn.addEventListener('click', () => {
      sortAndRender(m.key);
      if (window.gsap) {
        gsap.timeline()
          .to(btn, { scale: 0.95, duration: 0.1, ease: 'power2.out' })
          .to(btn, { scale: 1, duration: 0.4, ease: 'elastic.out(1, 0.3)', clearProps: 'scale' });
      }
    });

    btn.addEventListener('mouseenter', (e) => {
      const isActive = btn.dataset.mode === sortMode;
      content.style.background = 'transparent';
      content.style.color = '#fff';
      if (!isActive && window.gsap) {
        const rect = btn.getBoundingClientRect();
        const fromTop = e.clientY < rect.top + rect.height / 2;
        gsap.killTweensOf(fill);
        gsap.fromTo(fill,
          { y: fromTop ? '-100%' : '100%' },
          { y: '0%', duration: 0.4, ease: 'power3.out' }
        );
      }
    });
    btn.addEventListener('mouseleave', (e) => {
      const isActive = btn.dataset.mode === sortMode;
      if (!isActive) {
        if (window.gsap) {
          const rect = btn.getBoundingClientRect();
          const toTop = e.clientY < rect.top + rect.height / 2;
          gsap.killTweensOf(fill);
          gsap.to(fill, { y: toTop ? '-100%' : '100%', duration: 0.4, ease: 'power3.out' });
        }
        content.style.background = '#0d0d10';
        content.style.color = '#555';
      }
    });

    controls.appendChild(btn);
    sortBtns[m.key] = btn;
  }

  const legend = document.createElement('span');
  legend.style.cssText =
    'margin-left:auto; font-family:"SF Mono",ui-monospace,monospace;' +
    'font-size:9px; color:#2e2e36; letter-spacing:0.05em; flex-shrink:0;';
  legend.textContent = 'width = saturation · height = complexity · color = dominant';
  controls.appendChild(legend);

  const gridWrap = document.createElement('div');
  gridWrap.style.cssText =
    'flex:1; overflow-y:auto; overflow-x:hidden; min-height:0;';
  container.appendChild(gridWrap);

  const grid = document.createElement('div');
  grid.style.cssText =
    'display:flex; flex-wrap:wrap; align-items:flex-start;' +
    `column-gap:${COL_GAP}px; row-gap:${ROW_GAP}px; padding:14px 14px 20px;`;
  gridWrap.appendChild(grid);

  // Small tooltip that follows the cursor
  const tooltip = document.createElement('div');
  tooltip.style.cssText =
    'position:fixed; display:none; pointer-events:none; z-index:1000;' +
    'width:200px; background:rgba(10,10,16,0.97);' +
    'border:1px solid rgba(255,255,255,0.1); border-radius:7px;' +
    'overflow:hidden; box-shadow:0 8px 32px rgba(0,0,0,0.6);' +
    'font-family:"SF Mono",ui-monospace,monospace; font-size:10px; color:#888;';

  const ttImg = document.createElement('img');
  ttImg.style.cssText =
    'width:100%; height:110px; object-fit:cover; display:block; background:#111;';
  ttImg.alt = '';

  const ttBody = document.createElement('div');
  ttBody.style.cssText = 'padding:8px 10px 10px;';

  const ttSwatch = document.createElement('div');
  ttSwatch.style.cssText =
    'height:2px; width:100%; border-radius:1px; margin-bottom:8px;';

  const ttTitle = document.createElement('div');
  ttTitle.style.cssText =
    'color:#ddd; margin-bottom:3px;' +
    'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';

  const ttArtist = document.createElement('div');
  ttArtist.style.cssText =
    'color:#555; margin-bottom:7px; font-size:9px;' +
    'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;';

  const ttMeta = document.createElement('div');
  ttMeta.style.cssText = 'color:#444; display:flex; align-items:center; gap:6px;';

  const ttMetaColor = document.createElement('span');
  ttMetaColor.style.cssText =
    'display:inline-block; width:10px; height:10px; border-radius:2px; flex-shrink:0;';

  const ttMetaText = document.createElement('span');

  ttMeta.append(ttMetaColor, ttMetaText);
  ttBody.append(ttSwatch, ttTitle, ttArtist, ttMeta);
  tooltip.append(ttImg, ttBody);
  document.body.appendChild(tooltip);

  // Detail panel slides in from the right
  const panel = document.createElement('div');
  panel.style.cssText =
    'position:absolute; top:0; right:0; bottom:0; width:320px; z-index:30;' +
    'background:rgba(9,9,16,0.98); border-left:1px solid rgba(255,255,255,0.06);' +
    'transform:translateX(100%); transition:transform 0.26s cubic-bezier(0.4,0,0.2,1);' +
    'overflow-y:auto; font-family:"SF Mono",ui-monospace,monospace;' +
    'font-size:11px; color:#666;';
  container.appendChild(panel);

  const loader = document.createElement('div');
  loader.style.cssText =
    'position:absolute; inset:0; display:flex; align-items:center; justify-content:center;' +
    'font-family:"SF Mono",ui-monospace,monospace; font-size:11px;' +
    'color:rgba(255,255,255,0.15); text-align:center; line-height:2; pointer-events:none;';
  loader.innerHTML =
    'loading style&#x2011;entropy data…<br>' +
    '<span style="font-size:9px;color:rgba(255,255,255,0.06);">' +
    'run scripts/prepare_style_entropy.py if this persists</span>';
  container.appendChild(loader);

  fetch(DATA_URL)
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(raw => {
      loader.remove();
      buildStripes(raw);
      sortAndRender('cluster');
    })
    .catch(err => {
      loader.innerHTML =
        `<span style="color:#c44;">failed to load style-entropy data<br>` +
        `<span style="font-size:9px;color:#444;">run scripts/prepare_style_entropy.py</span><br>` +
        `<span style="font-size:9px;color:#333;">${err.message}</span></span>`;
    });

  // Precompute HSL values for each artwork so sorting is fast
  function buildStripes(raw) {
    stripes = raw.map(art => {
      const [h, s, l] = hexToHsl(art.dominant_color);
      return { art, hue: h, sat: s, lum: l };
    });
  }

  // Calculate ideal stripe width and row height so all rows fill the container exactly
  function computeLayout() {
    const N = stripes.length;
    if (!N) return;

    const W = (gridWrap.clientWidth || container.clientWidth) - 28;
    const H = gridWrap.clientHeight
           || (container.clientHeight - CONTROLS_H)
           || (window.innerHeight - 56 - CONTROLS_H); // 56px = header
    if (W <= 0 || H <= 0) return;

    const targetPerRow = Math.round(N / TARGET_ROWS);
    let bestPerRow = targetPerRow;
    let bestWaste  = N;

    for (let delta = 0; delta <= 400; delta++) {
      const candidates = delta === 0
        ? [targetPerRow]
        : [targetPerRow + delta, targetPerRow - delta];
      for (const p of candidates) {
        if (p < 2) continue;
        const rem   = N % p;
        const waste = rem === 0 ? 0 : p - rem;
        if (waste < bestWaste) {
          bestWaste  = waste;
          bestPerRow = p;
        }
        if (waste === 0) break;
      }
      if (bestWaste === 0) break;
    }

    stripeW     = Math.max(1, Math.floor((W - (bestPerRow - 1) * COL_GAP) / bestPerRow));
    fillerCount = bestWaste;

    const nRows  = Math.ceil(N / bestPerRow);
    const availH = H - 34 - (nRows - 1) * ROW_GAP;
    rowH = Math.max(8, Math.floor(availH / nRows));
  }

  // For sorting the stripes
  function sortArr(arr, mode) {
    switch (mode) {
      case 'cluster':
        arr.sort((a, b) => a.art.cluster !== b.art.cluster
          ? a.art.cluster - b.art.cluster : a.hue - b.hue);
        break;
      case 'hue':
        arr.sort((a, b) => a.hue - b.hue);
        break;
      case 'gradient':
        arr.sort((a, b) => {
          const ha = Math.floor(a.hue * 8), hb = Math.floor(b.hue * 8);
          if (ha !== hb) return ha - hb;
          const ca = Math.floor(a.art.complexity * 8),
                cb = Math.floor(b.art.complexity * 8);
          if (ca !== cb) return ca - cb;
          return a.lum - b.lum;
        });
        break;
      case 'constellation': {
        const byC = {};
        for (const s of arr) (byC[s.art.cluster] ??= []).push(s);
        for (const c of Object.values(byC))
          c.sort((a, b) => a.art.complexity - b.art.complexity);
        const cols = Object.values(byC);
        const out = [], maxL = Math.max(...cols.map(c => c.length));
        for (let i = 0; i < maxL; i++)
          for (const c of cols) if (i < c.length) out.push(c[i]);
        arr.splice(0, arr.length, ...out);
        break;
      }
      case 'asc':  arr.sort((a, b) => a.art.complexity - b.art.complexity); break;
      case 'desc': arr.sort((a, b) => b.art.complexity - a.art.complexity); break;
    }
  }

  function sortAndRender(mode) {
    sortMode = mode;

    for (const [key, btn] of Object.entries(sortBtns)) {
      const active  = key === mode;
      const fill    = btn._fill;
      const content = btn._content;
      btn.classList.toggle('is-active', active);
      if (fill) {
        if (window.gsap) {
          gsap.killTweensOf(fill);
          gsap.to(fill, { y: active ? '0%' : '-100%', duration: 0.25, ease: 'power2.out' });
        } else {
          fill.style.transform = active ? 'translateY(0%)' : 'translateY(-100%)';
        }
      }
      if (content) {
        content.style.background = active ? 'transparent' : '#0d0d10';
        content.style.color      = active ? '#fff'        : '#555';
      }
    }

    const arr = stripes.slice();
    sortArr(arr, mode);
    sorted = arr;

    computeLayout();

    const w  = stripeW;
    const rH = rowH;
    const parts = new Array(sorted.length);
    for (let i = 0; i < sorted.length; i++) {
      const s   = sorted[i];
      const barH = Math.max(2, Math.round(s.art.complexity * (rH - 2)));
      parts[i] =
        `<div class="se-s" data-i="${i}" style="` +
        `width:${w}px;height:${rH}px;flex-shrink:0;` +
        `position:relative;cursor:pointer;">` +
        `<div style="position:absolute;bottom:0;left:0;width:100%;` +
        `height:${barH}px;background:${s.art.dominant_color};"></div>` +
        `</div>`;
    }
    // Invisible filler divs so the last row is the same width as all others
    for (let i = 0; i < fillerCount; i++) {
      parts.push(
        `<div style="width:${w}px;height:${rH}px;flex-shrink:0;visibility:hidden;"></div>`
      );
    }
    grid.innerHTML = parts.join('');
    lastHoveredIdx = -1;
    hideTooltip();
  }

  function showTooltip(art) {
    ttImg.src             = art.image_url || '';
    ttImg.style.display   = art.image_url ? 'block' : 'none';
    ttSwatch.style.background = art.dominant_color;
    ttTitle.textContent   = clip(art.title || 'Untitled', 40);
    ttArtist.textContent  = clip(art.artist || '', 36);
    ttMetaColor.style.background = art.dominant_color;
    ttMetaText.textContent = `complexity: ${Math.round(art.complexity * 100)}%`;
    tooltip.style.display = 'block';
  }

  function hideTooltip() {
    tooltip.style.display = 'none';
  }

  let mouseX = 0, mouseY = 0;

  function positionTooltip() {
    const TW = 200, TH = tooltip.offsetHeight || 180;
    let lx = mouseX + 14;
    let ly = mouseY - TH - 10;
    if (lx + TW > window.innerWidth)  lx = mouseX - TW - 10;
    if (ly < 8)                        ly = mouseY + 14;
    tooltip.style.left = lx + 'px';
    tooltip.style.top  = ly + 'px';
  }

  grid.addEventListener('mouseover', (e) => {
    const el = e.target.closest('.se-s');
    if (!el) return;
    const i = +el.dataset.i;
    if (i === lastHoveredIdx) return;
    lastHoveredIdx = i;
    showTooltip(sorted[i].art);
    positionTooltip();

    // Debounce so rapid stripe crossings don't flood notes
    clearTimeout(_hoverTimer);
    _hoverTimer = setTimeout(() => _playStripe(sorted[i]), 40);
  });

  grid.addEventListener('mouseout', (e) => {
    if (!e.relatedTarget?.closest('.se-s')) {
      lastHoveredIdx = -1;
      hideTooltip();
    }
  });

  function onMouseMove(e) {
    mouseX = e.clientX;
    mouseY = e.clientY;
    if (tooltip.style.display !== 'none') positionTooltip();
  }
  document.addEventListener('mousemove', onMouseMove);

  grid.addEventListener('click', (e) => {
    const el = e.target.closest('.se-s');
    if (!el) { closePanel(); return; }
    openPanel(sorted[+el.dataset.i].art);
  });

  function openPanel(art) {
    const pct  = Math.round(art.complexity * 100);
    const [, s, l] = hexToHsl(art.dominant_color);
    const satPct = Math.round(s * 100);
    const lumPct = Math.round(l * 100);

    panel.innerHTML = `
      <div style="position:sticky;top:0;z-index:1;background:rgba(9,9,16,0.98);
        padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.05);
        display:flex;align-items:center;justify-content:space-between;">
        <span style="color:rgba(255,255,255,0.18);font-size:9px;letter-spacing:0.12em;text-transform:uppercase;">Artwork Detail</span>
        <button id="se-close" style="background:none;border:none;cursor:pointer;
          color:rgba(255,255,255,0.25);font-size:20px;line-height:1;padding:0 2px;">×</button>
      </div>

      ${art.image_url
        ? `<div style="position:relative;width:100%;background:#0a0a12;">
             <img src="${art.image_url}" style="width:100%;display:block;
               max-height:280px;object-fit:cover;" loading="lazy"
               onerror="this.parentElement.style.display='none'">
           </div>`
        : ''}

      <div style="padding:16px 16px 24px;">

        <div style="width:100%;height:3px;background:${art.dominant_color};
          border-radius:2px;margin-bottom:16px;opacity:0.85;"></div>

        <div style="color:rgba(255,255,255,0.82);font-size:13px;line-height:1.5;
          margin-bottom:6px;letter-spacing:0.01em;">${art.title || 'Untitled'}</div>
        <div style="color:#555;margin-bottom:${art.date || art.museum ? '4px' : '14px'};">
          ${art.artist || ''}</div>
        ${art.date
          ? `<div style="color:#3a3a4a;margin-bottom:${art.museum ? '2px' : '14px'};font-size:10px;">
               ${art.date}</div>` : ''}
        ${art.museum
          ? `<div style="color:#3a3a4a;margin-bottom:14px;font-size:10px;
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
               ${art.museum}</div>` : ''}

        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
          <div style="width:18px;height:18px;border-radius:4px;flex-shrink:0;
            background:${art.dominant_color};"></div>
          <span style="color:#3a3a4a;font-size:9px;">dominant colour</span>
          <span style="color:#555;font-size:9px;margin-left:auto;">${art.dominant_color}</span>
        </div>

        <div style="color:#3a3a4a;font-size:9px;letter-spacing:0.08em;margin-bottom:10px;
          text-transform:uppercase;">Visual metrics</div>

        ${[
          ['complexity', pct, 'rgba(255,255,255,0.3)'],
          ['saturation', satPct, art.dominant_color],
          ['luminance',  lumPct, art.dominant_color],
        ].map(([lbl, vp, col]) => `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;">
          <span style="color:#444;width:72px;flex-shrink:0;">${lbl}</span>
          <div style="flex:1;height:4px;background:rgba(255,255,255,0.05);border-radius:2px;">
            <div style="width:${vp}%;height:100%;background:${col};
              border-radius:2px;opacity:0.75;"></div>
          </div>
          <span style="color:#555;width:30px;">${vp}%</span>
        </div>`).join('')}

        <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.04);">
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="color:#444;">cluster</span>
            <span style="padding:2px 8px;border-radius:3px;font-size:9px;
              background:rgba(14,165,233,0.12);color:#0ea5e9;letter-spacing:0.05em;">
              ${art.cluster}</span>
          </div>
        </div>

        ${art.object_url
          ? `<div style="margin-top:14px;">
               <a href="${art.object_url}" target="_blank" rel="noopener"
                  style="display:inline-flex;align-items:center;gap:6px;
                    color:#0ea5e9;font-size:10px;text-decoration:none;
                    padding:6px 10px;border:1px solid rgba(14,165,233,0.25);
                    border-radius:5px;transition:background 0.15s;"
                  onmouseover="this.style.background='rgba(14,165,233,0.1)'"
                  onmouseout="this.style.background='transparent'">
                 ↗ View at museum
               </a>
             </div>` : ''}
      </div>`;
    panel.style.transform = 'translateX(0)';
    panel.querySelector('#se-close').addEventListener('click', (e) => {
      e.stopPropagation(); closePanel();
    });
  }

  function closePanel() {
    panel.style.transform = 'translateX(100%)';
  }

  // Keys 1–5 switch sort mode; Escape closes the detail panel
  function onKey(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'Escape') { closePanel(); return; }
    const mode = KEY_MAP[e.key];
    if (mode) sortAndRender(mode);
  }
  document.addEventListener('keydown', onKey);

  const ro = new ResizeObserver(() => {
    if (sorted.length) sortAndRender(sortMode);
  });
  ro.observe(gridWrap);

  return {
    cleanup() {
      ro.disconnect();
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousemove', onMouseMove);
      tooltip.remove();
      clearTimeout(_hoverTimer);
      try { _synth?.dispose();  } catch {}
      try { _reverb?.dispose(); } catch {}
      _toneOk = false;
    },
  };
}
