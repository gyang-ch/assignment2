export const AZURE_BASE_URL  = 'https://phytovision.blob.core.windows.net/artworks';
export const AZURE_DATA_URL  = 'https://phytovision.blob.core.windows.net/artworks/data';
export const AZURE_SAS_TOKEN = 'sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2030-02-15T18:31:09Z&st=2026-03-29T09:16:09Z&spr=https&sig=prvNh26kGPtCZ%2BcJWtZ1HDzihcxCYDNiIrv3hfu3VVY%3D';

const _isLocal = ['localhost', '127.0.0.1'].includes(window.location.hostname);

/** Returns the correct URL for a data file: local /data/ on localhost, Azure on production. */
export function dataUrl(filename) {
  if (_isLocal) return `/data/${filename}`;
  return `${AZURE_DATA_URL}/${filename}?${AZURE_SAS_TOKEN}`;
}

export function azureFallback(originalUrl) {
  if (!AZURE_BASE_URL || !originalUrl) return null;
  try {
    const filename = originalUrl.split('/').pop().split('?')[0];
    return `${AZURE_BASE_URL}/${filename}${AZURE_SAS_TOKEN}`;
  } catch {
    return null;
  }
}
