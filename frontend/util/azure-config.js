export const AZURE_BASE_URL  = 'https://phytovision.blob.core.windows.net/artworks';
export const AZURE_DATA_URL  = 'https://phytovision.blob.core.windows.net/artworks/data';
export const AZURE_SAS_TOKEN = 'sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2030-02-15T18:31:09Z&st=2026-03-29T09:16:09Z&spr=https&sig=prvNh26kGPtCZ%2BcJWtZ1HDzihcxCYDNiIrv3hfu3VVY%3D';

export function azureFallback(originalUrl) {
  if (!AZURE_BASE_URL || !originalUrl) return null;
  try {
    const filename = originalUrl.split('/').pop().split('?')[0];
    return `${AZURE_BASE_URL}/${filename}${AZURE_SAS_TOKEN}`;
  } catch {
    return null;
  }
}
