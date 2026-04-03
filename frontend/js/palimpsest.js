/**
 * Digital Palimpsest — p5.js WebGL + GLSL shader experience.
 *
 * The museum artwork is hidden under a "gesso" layer. The user's strokes
 * reveal the artwork underneath, distorted by DINOv2 similarity:
 *   - High similarity  → clear, vivid reveal
 *   - Low  similarity  → blurry, chromatic-aberrated, flow-distorted
 *
 * Drawing uses speed-based pressure simulation (slow = thick, fast = thin),
 * echoing perfect-freehand's approach.
 */

import p5 from 'p5';
import { compareSketch, proxyImageUrl } from './api.js';

/* ── GLSL Shaders ──────────────────────────────────────────────────────── */

const VERT_SRC = /* glsl */ `
precision mediump float;
attribute vec3 aPosition;
attribute vec2 aTexCoord;
varying vec2 vTexCoord;

void main() {
  vTexCoord = aTexCoord;
  vec4 pos = vec4(aPosition, 1.0);
  pos.xy = pos.xy * 2.0 - 1.0;
  gl_Position = pos;
}
`;

const FRAG_SRC = /* glsl */ `
precision mediump float;

varying vec2 vTexCoord;

uniform sampler2D uArtwork;
uniform sampler2D uSketch;
uniform float uSimilarity;
uniform float uTime;
uniform vec2 uResolution;

/* ---- noise ---- */
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
    mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
    f.y
  );
}

void main() {
  vec2 uv = vTexCoord;
  uv.y = 1.0 - uv.y;

  vec2 px = 1.0 / uResolution;

  /* ---- sketch mask (1.0 = drawn) ---- */
  float mask = 1.0 - texture2D(uSketch, uv).r;

  /* ---- sketch gradient → flow field ---- */
  float mL = 1.0 - texture2D(uSketch, uv - vec2(px.x * 2.0, 0.0)).r;
  float mR = 1.0 - texture2D(uSketch, uv + vec2(px.x * 2.0, 0.0)).r;
  float mU = 1.0 - texture2D(uSketch, uv - vec2(0.0, px.y * 2.0)).r;
  float mD = 1.0 - texture2D(uSketch, uv + vec2(0.0, px.y * 2.0)).r;
  vec2 grad = vec2(mR - mL, mD - mU);

  /* ---- flow distortion (stronger when dissimilar) ---- */
  float distort = mix(0.06, 0.003, uSimilarity);
  vec2 flow = grad * distort;
  /* gentle time undulation */
  flow += vec2(
    sin(uv.y * 12.0 + uTime * 0.7),
    cos(uv.x * 12.0 + uTime * 0.5)
  ) * 0.003 * (1.0 - uSimilarity);

  vec2 artUV = uv + flow;

  /* ---- sample artwork ---- */
  vec4 art = texture2D(uArtwork, artUV);

  /* ---- blur for low similarity (Gaussian-ish 7x7) ---- */
  float blurScale = (1.0 - uSimilarity) * 3.0;
  if (blurScale > 0.3) {
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int i = -3; i <= 3; i++) {
      for (int j = -3; j <= 3; j++) {
        float r2 = float(i * i + j * j);
        float w = exp(-r2 / max(2.0 * blurScale * blurScale, 0.01));
        sum += texture2D(uArtwork, artUV + vec2(float(i), float(j)) * px * blurScale) * w;
        total += w;
      }
    }
    art = sum / total;
  }

  /* ---- chromatic aberration (stronger when dissimilar) ---- */
  float shift = (1.0 - uSimilarity) * 0.008;
  art.r = texture2D(uArtwork, artUV + vec2(shift, 0.0)).r;
  art.b = texture2D(uArtwork, artUV - vec2(shift, 0.0)).b;

  /* ---- gesso / parchment layer ---- */
  float n1 = noise(uv * 25.0);
  vec3 gesso = vec3(0.92, 0.89, 0.84) + (n1 - 0.5) * 0.04;
  /* subtle cracks */
  float crack = smoothstep(0.48, 0.50, noise(uv * 50.0));
  gesso -= crack * 0.04;

  /* ---- reveal ---- */
  float reveal = smoothstep(0.02, 0.35, mask);
  /* frayed edges */
  float fray = noise(uv * 35.0 + vec2(uTime * 0.05));
  reveal = smoothstep(0.0, 1.0, reveal + (fray - 0.5) * 0.15);

  /* ---- composite ---- */
  vec3 color = mix(gesso, art.rgb, reveal);

  gl_FragColor = vec4(color, 1.0);
}
`;

/* ── Palimpsest Controller ─────────────────────────────────────────────── */

/**
 * Create and mount the palimpsest experience.
 *
 * @param {HTMLElement} container — element to mount the p5 canvas into
 * @param {object} opts
 * @param {Function} [opts.onSimilarityUpdate] — called with similarity float
 * @returns {{ loadArtwork, setSimilarity, clear, getSketchBlob, destroy,
 *             computeSimilarity }}
 */
export function createPalimpsest(container, opts = {}) {
  let artworkImg   = null;
  let artworkUid   = null;
  let sketchBuffer = null;
  let shader       = null;
  let similarity   = 0.5;
  let brushSize    = 12;
  let canvasReady  = false;
  let p5Instance   = null;

  // Debounce timer for auto-similarity
  let simTimer = null;

  const sketch = (p) => {
    p.setup = () => {
      // Container must be visible and laid out for dimensions to be non-zero
      const w = container.clientWidth  || 800;
      const h = container.clientHeight || 600;
      const cnv = p.createCanvas(w, h, p.WEBGL);
      // Prevent right-click context menu on canvas
      cnv.elt.addEventListener('contextmenu', e => e.preventDefault());
      p.pixelDensity(1);
      sketchBuffer = p.createGraphics(w, h);
      sketchBuffer.background(255);
      sketchBuffer.strokeCap(p.ROUND);
      sketchBuffer.strokeJoin(p.ROUND);
      shader = p.createShader(VERT_SRC, FRAG_SRC);
      canvasReady = true;
    };

    p.draw = () => {
      if (!artworkImg) {
        // Placeholder: parchment with text hint
        p.resetShader();
        p.background(235, 228, 215);
        p.fill(160);
        p.noStroke();
        p.textSize(16);
        p.textAlign(p.CENTER, p.CENTER);
        p.text('Select an artwork to begin', 0, 0);
        return;
      }

      p.shader(shader);
      shader.setUniform('uArtwork',    artworkImg);
      shader.setUniform('uSketch',     sketchBuffer);
      shader.setUniform('uSimilarity', similarity);
      shader.setUniform('uTime',       p.millis() / 1000.0);
      shader.setUniform('uResolution', [p.width, p.height]);
      p.rect(0, 0, p.width, p.height);
    };

    /* ---- drawing ---- */
    function inCanvas() {
      return p.mouseX >= 0 && p.mouseX <= p.width &&
             p.mouseY >= 0 && p.mouseY <= p.height;
    }

    p.mouseDragged = () => {
      if (!canvasReady || !artworkImg || !inCanvas()) return;

      // Speed-based pressure: slow = thick, fast = thin
      const speed = p.dist(p.mouseX, p.mouseY, p.pmouseX, p.pmouseY);
      const sz = p.map(speed, 0, 50, brushSize * 1.5, brushSize * 0.4, true);

      sketchBuffer.stroke(0);
      sketchBuffer.strokeWeight(sz);
      sketchBuffer.line(p.pmouseX, p.pmouseY, p.mouseX, p.mouseY);

      // Auto-compute similarity after a drawing pause
      scheduleSimilarity();
      return false; // prevent default browser drag
    };

    p.mousePressed = () => {
      if (!canvasReady || !artworkImg || !inCanvas()) return;

      // Draw a dot on click
      sketchBuffer.stroke(0);
      sketchBuffer.strokeWeight(brushSize);
      sketchBuffer.point(p.mouseX, p.mouseY);
      return false;
    };

    /* ---- touch support ---- */
    p.touchMoved = () => {
      p.mouseDragged();
      return false; // prevent scroll
    };

    p.windowResized = () => {
      const w = container.clientWidth;
      const h = container.clientHeight || 600;
      p.resizeCanvas(w, h);
      // Recreate sketch buffer at new size (preserves nothing — acceptable)
      const old = sketchBuffer;
      sketchBuffer = p.createGraphics(w, h);
      sketchBuffer.background(255);
      sketchBuffer.image(old, 0, 0, w, h);
      old.remove();
    };
  };

  p5Instance = new p5(sketch, container);

  /* ── Auto-similarity (debounced) ───────────────────────────────────── */

  function scheduleSimilarity() {
    if (simTimer) clearTimeout(simTimer);
    simTimer = setTimeout(() => computeSimilarity(), 800);
  }

  async function computeSimilarity() {
    if (!artworkUid || !canvasReady) return;
    const blob = await getSketchBlob();
    if (!blob) return;
    try {
      const res = await compareSketch(blob, artworkUid);
      similarity = res.similarity;
      opts.onSimilarityUpdate?.(similarity);
    } catch (err) {
      console.warn('[palimpsest] similarity error:', err.message);
    }
  }

  /* ── Public API ────────────────────────────────────────────────────── */

  function loadArtwork(url, uid) {
    artworkUid = uid;
    similarity = 0.5; // reset until computed
    // Load via proxy to avoid CORS issues
    const proxied = proxyImageUrl(url);
    p5Instance.loadImage(proxied, (img) => {
      artworkImg = img;
    }, () => {
      // Fallback: try direct load
      p5Instance.loadImage(url, (img) => { artworkImg = img; });
    });
  }

  function setSimilarity(s) {
    similarity = s;
  }

  function setBrushSize(s) {
    brushSize = s;
  }

  function clear() {
    if (sketchBuffer) sketchBuffer.background(255);
    similarity = 0.5;
    opts.onSimilarityUpdate?.(similarity);
  }

  async function getSketchBlob() {
    if (!sketchBuffer) return null;
    // Get the sketch buffer as a data URL, convert to Blob
    const dataUrl = sketchBuffer.elt.toDataURL('image/png');
    const resp = await fetch(dataUrl);
    return resp.blob();
  }

  function destroy() {
    if (simTimer) clearTimeout(simTimer);
    if (p5Instance) p5Instance.remove();
  }

  return {
    loadArtwork,
    setSimilarity,
    setBrushSize,
    clear,
    getSketchBlob,
    computeSimilarity,
    destroy,
  };
}
