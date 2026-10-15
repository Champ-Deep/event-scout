import { Image, NativeModules, Platform } from 'react-native';
import { Asset } from 'expo-asset';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import * as jpeg from 'jpeg-js';
import { recognize, type OcrLine, type RgbaImage, type SessionLike } from './pipeline';
import { parseCard, type ParsedCard } from './parseCard';

/**
 * On-device card reading: PaddleOCR PP-OCRv5 mobile (det 4.8 MB + en rec 7.8 MB) running in
 * onnxruntime-react-native, then parseCard() turns the lines into a contact. Needs a dev /
 * store build — Expo Go has no ONNX Runtime, so there `isOnDeviceOcrAvailable()` is false and
 * scans use the server as before.
 */

export type { ParsedCard, CardField } from './parseCard';
export type { OcrLine } from './pipeline';

const OCR_LONG_SIDE = 1280; // plenty for card text; keeps the JPEG decode fast

let ort: any = null;
function loadOrt(): any {
  if (ort) return ort;
  // Required lazily: importing it where the native module is missing throws at import time.
  ort = require('onnxruntime-react-native');
  return ort;
}

export function isOnDeviceOcrAvailable(): boolean {
  return Platform.OS !== 'web' && !!(NativeModules as any).Onnxruntime;
}

let sessions: Promise<{ det: SessionLike; rec: SessionLike }> | null = null;

async function modelPath(mod: number): Promise<string> {
  const asset = Asset.fromModule(mod);
  await asset.downloadAsync(); // copies the bundled model to a local file once
  const uri = asset.localUri || asset.uri;
  return uri.replace(/^file:\/\//, '');
}

function getSessions() {
  if (!sessions) {
    sessions = (async () => {
      const { InferenceSession } = loadOrt();
      const [detPath, recPath] = await Promise.all([
        modelPath(require('../../../../assets/ocr/ppocrv5_det.onnx')),
        modelPath(require('../../../../assets/ocr/en_ppocrv5_rec.onnx')),
      ]);
      const opts = { executionProviders: ['cpu'], graphOptimizationLevel: 'all' };
      const [det, rec] = await Promise.all([InferenceSession.create(detPath, opts), InferenceSession.create(recPath, opts)]);
      return { det, rec };
    })();
    sessions.catch(() => { sessions = null; }); // allow a retry after a failed load
  }
  return sessions;
}

/** Load the models in the background (e.g. when the scanner opens) so the first scan is quick. */
export function warmUpOnDeviceOcr() {
  if (isOnDeviceOcrAvailable()) getSessions().catch((e) => console.warn('[ocr] model load failed', e?.message));
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_INDEX = (() => { const t = new Uint8Array(128); for (let i = 0; i < 64; i++) t[B64.charCodeAt(i)] = i; return t; })();
function base64ToBytes(b64: string): Uint8Array {
  const s = b64.replace(/[^A-Za-z0-9+/]/g, '');
  const out = new Uint8Array(Math.floor((s.length * 3) / 4));
  let o = 0;
  for (let i = 0; i < s.length; i += 4) {
    const a = B64_INDEX[s.charCodeAt(i)]!, b = B64_INDEX[s.charCodeAt(i + 1)]!;
    const c = B64_INDEX[s.charCodeAt(i + 2)] ?? 0, d = B64_INDEX[s.charCodeAt(i + 3)] ?? 0;
    out[o++] = (a << 2) | (b >> 4);
    if (i + 2 < s.length) out[o++] = ((b & 15) << 4) | (c >> 2);
    if (i + 3 < s.length) out[o++] = ((c & 3) << 6) | d;
  }
  return out.subarray(0, o);
}

function imageSize(uri: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => Image.getSize(uri, (width, height) => resolve({ width, height }), reject));
}

/** Photo → RGBA pixels, scaled so the long side is at most OCR_LONG_SIDE. */
async function readPixels(uri: string): Promise<RgbaImage> {
  const { width, height } = await imageSize(uri);
  const scale = Math.min(1, OCR_LONG_SIDE / Math.max(width, height));
  const resize = width >= height ? { width: Math.round(width * scale) } : { height: Math.round(height * scale) };
  const out = await manipulateAsync(uri, scale < 1 ? [{ resize }] : [], { format: SaveFormat.JPEG, compress: 0.95, base64: true });
  const raw = jpeg.decode(base64ToBytes(out.base64 || ''), { useTArray: true, formatAsRGBA: true, maxMemoryUsageInMB: 256 });
  return { data: raw.data, width: raw.width, height: raw.height };
}

/** Small JPEG of the card for the contact's picture (same as a server scan stores). */
export async function cardThumbnail(uri: string): Promise<string | null> {
  try {
    const out = await manipulateAsync(uri, [{ resize: { width: 640 } }], { format: SaveFormat.JPEG, compress: 0.7, base64: true });
    return out.base64 ? `data:image/jpeg;base64,${out.base64}` : null;
  } catch {
    return null;
  }
}

export interface OnDeviceReadResult {
  card: ParsedCard;
  lines: OcrLine[];
  ms: number;
}

/** Read a business-card photo on this phone. Throws if ONNX Runtime isn't in this build. */
export async function readCardOnDevice(uri: string, onProgress?: (done: number, total: number) => void): Promise<OnDeviceReadResult> {
  if (!isOnDeviceOcrAvailable()) throw new Error('On-device reading needs the EventScout app build (not Expo Go).');
  const t0 = Date.now();
  const [{ det, rec }, img] = await Promise.all([getSessions(), readPixels(uri)]);
  const { Tensor } = loadOrt();
  const { lines, timings } = await recognize(img, det, rec, (d, dims) => new Tensor('float32', d, dims), onProgress);
  const card = parseCard(lines);
  const ms = Date.now() - t0;
  if (__DEV__) console.log(`[ocr] ${img.width}x${img.height} · ${timings.boxes} boxes · det ${timings.detMs} ms · rec ${timings.recMs} ms · total ${ms} ms · confidence ${card.confidence}`);
  return { card, lines, ms };
}
