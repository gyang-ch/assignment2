import { getStroke } from 'perfect-freehand';

const OPTIONS = {
  size: 5,
  thinning: 0.5,
  smoothing: 0.5,
  streamline: 0.5,
  start: { cap: true, taper: 0 },
  end:   { cap: true, taper: 0 },
};

let _canvas  = null;
let _ctx     = null;
let _strokes = [];
let _current = null;

export function mount(container) {
  container.style.position = 'relative';

  _canvas = document.createElement('canvas');
  _canvas.style.cssText =
    'position:absolute;inset:0;width:100%;height:100%;' +
    'touch-action:none;cursor:crosshair;background:#fff;border-radius:4px;';
  container.appendChild(_canvas);

  const ro = new ResizeObserver(() => _resize());
  ro.observe(container);
  _resize();

  _canvas.addEventListener('pointerdown', _onDown);
  _canvas.addEventListener('pointermove', _onMove);
  _canvas.addEventListener('pointerup',   _onUp);
  _canvas.addEventListener('pointerleave', _onUp);

  document.addEventListener('keydown', _onKey);
}

export function isEmpty() {
  return _strokes.length === 0 && _current === null;
}

export function clear() {
  _strokes = [];
  _current = null;
  _redraw();
}

export async function exportAsBlob(size = 512) {
  if (isEmpty()) return null;

  const allStrokes = _current ? [..._strokes, _current] : _strokes;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const pts of allStrokes) {
    for (const [x, y] of pts) {
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  const PAD   = 24;
  const bW    = maxX - minX || 1;
  const bH    = maxY - minY || 1;
  const scale = Math.min((size - PAD * 2) / bW, (size - PAD * 2) / bH);
  const offX  = (size - bW * scale) / 2 - minX * scale;
  const offY  = (size - bH * scale) / 2 - minY * scale;

  const off = document.createElement('canvas');
  off.width = off.height = size;
  const ctx = off.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);
  ctx.save();
  ctx.translate(offX, offY);
  ctx.scale(scale, scale);
  _drawStrokes(ctx, allStrokes);
  ctx.restore();

  return new Promise(resolve => off.toBlob(resolve, 'image/png'));
}

function _onDown(e) {
  _canvas.setPointerCapture(e.pointerId);
  _current = [[e.offsetX, e.offsetY, e.pressure || 0.5]];
  _redraw();
}

function _onMove(e) {
  if (!_current) return;
  _current.push([e.offsetX, e.offsetY, e.pressure || 0.5]);
  _redraw();
}

function _onUp() {
  if (!_current) return;
  if (_current.length > 1) _strokes.push(_current);
  _current = null;
  _redraw();
}

function _onKey(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
    e.preventDefault();
    _strokes.pop();
    _current = null;
    _redraw();
  }
}

function _resize() {
  if (!_canvas) return;
  const r = _canvas.getBoundingClientRect();
  _canvas.width  = Math.round(r.width)  || _canvas.offsetWidth;
  _canvas.height = Math.round(r.height) || _canvas.offsetHeight;
  _ctx = _canvas.getContext('2d');
  _redraw();
}

function _redraw() {
  if (!_ctx) return;
  _ctx.clearRect(0, 0, _canvas.width, _canvas.height);
  const all = _current ? [..._strokes, _current] : _strokes;
  _drawStrokes(_ctx, all);
}

function _drawStrokes(ctx, strokes) {
  ctx.fillStyle = '#1a1a1a';
  for (const pts of strokes) {
    if (pts.length < 2) continue;
    const outline = getStroke(pts, OPTIONS);
    if (!outline.length) continue;
    const path = new Path2D();
    path.moveTo(outline[0][0], outline[0][1]);
    for (let i = 1; i < outline.length; i++) {
      path.lineTo(outline[i][0], outline[i][1]);
    }
    path.closePath();
    ctx.fill(path);
  }
}
