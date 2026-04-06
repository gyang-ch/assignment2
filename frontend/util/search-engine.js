import { AZURE_DATA_URL, AZURE_SAS_TOKEN } from './azure-config.js';

const MODEL_ID = 'Xenova/dinov2-small';

const EMBEDDINGS_URL = `${AZURE_DATA_URL}/embeddings.bin?${AZURE_SAS_TOKEN}`;
const UIDS_URL       = `${AZURE_DATA_URL}/uids.json?${AZURE_SAS_TOKEN}`;
const METADATA_URL   = `${AZURE_DATA_URL}/metadata.json?${AZURE_SAS_TOKEN}`;

let _processor  = null;
let _model      = null;
let _RawImage   = null;
let _embeddings = null;
let _uids       = null;
let _metaByUid  = null;
let _N = 0, _D = 0;

let _initPromise = null;

export function initSearchEngine(onStatus) {
  if (!_initPromise) _initPromise = _init(onStatus);
  return _initPromise;
}

export async function searchImage(imageBlob, topK = 12) {
  await initSearchEngine();

  const processed = await _applyEdgePreprocessing(imageBlob);

  const url   = URL.createObjectURL(processed);
  const image = await _RawImage.fromURL(url);
  URL.revokeObjectURL(url);

  const inputs  = await _processor(image);
  const outputs = await _model(inputs);

  const cls   = outputs.last_hidden_state.data.slice(0, _D);
  const query = _l2normalise(new Float32Array(cls));

  const scores = _dotProductBatch(query, _embeddings, _N, _D);

  return _topK(scores, Math.min(topK, _N));
}

async function _init(onStatus) {
  onStatus?.('Loading search engine…');

  const { AutoModel, AutoProcessor, RawImage: RawImageCls } =
    await import('@huggingface/transformers');
  _RawImage = RawImageCls;

  const [[processor, model], [embBuffer, uids, metaList]] = await Promise.all([
    Promise.all([
      AutoProcessor.from_pretrained(MODEL_ID),
      AutoModel.from_pretrained(MODEL_ID, { dtype: 'quantized' }),
    ]),
    Promise.all([
      fetch(EMBEDDINGS_URL).then(r => {
        if (!r.ok) throw new Error(`Failed to fetch embeddings (${r.status})`);
        return r.arrayBuffer();
      }),
      fetch(UIDS_URL).then(r => r.json()),
      fetch(METADATA_URL).then(r => r.json()),
    ]),
  ]);

  _processor = processor;
  _model     = model;

  const header = new Uint32Array(embBuffer, 0, 2);
  _N = header[0];
  _D = header[1];
  _embeddings = new Float32Array(embBuffer, 8);

  _uids = uids;
  _metaByUid = new Map(metaList.map(m => [m.uid, m]));

  onStatus?.('Ready.');
}

function _applyEdgePreprocessing(blob) {
  return new Promise((resolve, reject) => {
    const img    = new Image();
    const srcUrl = URL.createObjectURL(blob);

    img.onload = () => {
      URL.revokeObjectURL(srcUrl);

      const canvas = document.createElement('canvas');
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      const id   = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const px   = id.data;
      for (let i = 0; i < px.length; i += 4) {
        const g   = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
        const inv = 255 - g;
        px[i] = px[i + 1] = px[i + 2] = inv;
      }
      ctx.putImageData(id, 0, 0);
      canvas.toBlob(resolve, 'image/jpeg', 0.95);
    };

    img.onerror = () => { URL.revokeObjectURL(srcUrl); reject(new Error('Image decode failed')); };
    img.src = srcUrl;
  });
}

function _l2normalise(vec) {
  let sq = 0;
  for (let i = 0; i < vec.length; i++) sq += vec[i] * vec[i];
  const inv = sq > 0 ? 1 / Math.sqrt(sq) : 0;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] * inv;
  return out;
}

function _dotProductBatch(query, matrix, N, D) {
  const scores = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let dot = 0;
    const off = i * D;
    for (let j = 0; j < D; j++) dot += query[j] * matrix[off + j];
    scores[i] = dot;
  }
  return scores;
}

function _topK(scores, k) {
  const indices = Array.from({ length: _N }, (_, i) => i);
  indices.sort((a, b) => scores[b] - scores[a]);

  return indices.slice(0, k).map(idx => {
    const uid  = _uids[idx];
    const meta = _metaByUid.get(uid) ?? {};
    return {
      uid,
      museum:     meta.museum     ?? '',
      title:      meta.title      ?? '',
      artist:     meta.artist     ?? '',
      date:       meta.date       ?? '',
      medium:     meta.medium     ?? '',
      image_url:  meta.image_url  ?? '',
      object_url: meta.object_url ?? '',
      score:      Math.round(scores[idx] * 10000) / 10000,
    };
  });
}
