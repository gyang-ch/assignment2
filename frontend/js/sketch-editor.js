/**
 * Thin wrapper around tldraw v4 — mounts the React-based editor into a DOM
 * container and exposes a plain-JS API for the rest of the app.
 *
 * tldraw uses perfect-freehand internally for its Draw tool,
 * giving us beautiful pressure-simulated strokes out of the box.
 */

import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { Tldraw } from 'tldraw';
import 'tldraw/tldraw.css';

let _editor = null;

/* ── Mount ─────────────────────────────────────────────────────────────── */

/**
 * Mount the tldraw editor into a container element.
 * The container must have explicit width/height via CSS.
 *
 * @param {HTMLElement} container
 * @param {object} [opts]
 * @param {Function} [opts.onReady] — called with the Editor instance
 */
export function mount(container, opts = {}) {
  const root = createRoot(container);

  const app = createElement(Tldraw, {
    onMount: (editor) => {
      _editor = editor;
      // Start in draw mode by default
      editor.setCurrentTool('draw');
      opts.onReady?.(editor);
    },
    inferDarkMode: false,
  });

  root.render(app);
}

/* ── Query ─────────────────────────────────────────────────────────────── */

/** Get the raw tldraw Editor instance (for advanced use). */
export function getEditor() {
  return _editor;
}

/** True when the canvas has no shapes at all. */
export function isEmpty() {
  if (!_editor) return true;
  return _editor.getCurrentPageShapes().length === 0;
}

/* ── Actions ───────────────────────────────────────────────────────────── */

/** Delete all shapes on the current page. */
export function clear() {
  if (!_editor) return;
  const shapes = _editor.getCurrentPageShapes();
  if (shapes.length > 0) {
    _editor.deleteShapes(shapes.map(s => s.id));
  }
}

/**
 * Export the current drawing as a PNG Blob at a controlled resolution.
 *
 * Pipeline:  tldraw shapes → SVG string → rasterise on an off-screen
 * <canvas> at exactly `size × size` pixels → PNG Blob.
 *
 * This decouples the search resolution from the display size of the editor.
 *
 * @param {number} [size=512] — output width & height in pixels
 * @returns {Promise<Blob|null>} — null when the canvas is empty
 */
export async function exportAsBlob(size = 512) {
  if (!_editor || isEmpty()) return null;

  const shapes = _editor.getCurrentPageShapes();

  // Use tldraw v4's getSvgString method
  const result = await _editor.getSvgString(shapes, {
    background: true,
    padding: 16,
  });

  if (!result) return null;

  return rasteriseSvg(result.svg, size, size);
}

/* ── Internal: SVG → PNG rasterisation ─────────────────────────────────── */

/**
 * Render an SVG string onto an off-screen canvas at exact pixel dimensions.
 * Returns a PNG Blob.
 */
function rasteriseSvg(svgString, width, height) {
  return new Promise((resolve, reject) => {
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);

      const canvas = document.createElement('canvas');
      canvas.width  = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');

      // White background (matches the sketch domain: black strokes on white)
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);

      // Draw centred, preserving aspect ratio
      const scale = Math.min(width / img.naturalWidth, height / img.naturalHeight);
      const w = img.naturalWidth  * scale;
      const h = img.naturalHeight * scale;
      const x = (width  - w) / 2;
      const y = (height - h) / 2;
      ctx.drawImage(img, x, y, w, h);

      canvas.toBlob(blob => resolve(blob), 'image/png');
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to rasterise SVG'));
    };

    img.src = url;
  });
}
