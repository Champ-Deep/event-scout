import { CHARS } from './dict';

/**
 * PaddleOCR (PP-OCRv5 mobile) text detection + recognition, written against the
 * onnxruntime-common API so the same code runs on the phone (onnxruntime-react-native)
 * and in Node tests (onnxruntime-node).
 *
 * det: PP-OCRv5_mobile_det — BGR, long side 960 (multiple of 32), ImageNet mean/std →
 *      probability map → DB post-process (thresh 0.3, box 0.6, unclip 1.5).
 * rec: en_PP-OCRv5_mobile_rec — BGR crop, height 48, (x/255 − 0.5)/0.5, right-padded to ≥320 →
 *      CTC (0 = blank, i + 1 = CHARS[i], last = space).
 */

export interface RgbaImage {
  data: Uint8Array | Uint8ClampedArray; // RGBA, row-major
  width: number;
  height: number;
}

export interface Box { x: number; y: number; w: number; h: number }

/** A text region as a rotated rectangle: centre, size along/across the text, and direction. */
export interface Quad { cx: number; cy: number; w: number; h: number; ux: number; uy: number }

export interface OcrLine {
  text: string;
  score: number; // mean character confidence, 0–1
  box: Box;      // upright bounds in source-image pixels
  /** text height (across the line), unaffected by tilt */
  lineHeight: number;
}

// Minimal slices of onnxruntime-common, so this file doesn't depend on a runtime package.
export interface TensorLike { data: unknown; dims: readonly number[] }
export interface SessionLike {
  readonly inputNames: readonly string[];
  readonly outputNames: readonly string[];
  run(feeds: Record<string, any>): Promise<Record<string, TensorLike>>;
}
export type MakeTensor = (data: Float32Array, dims: number[]) => any;

const DET_LIMIT = 960;
const DET_MEAN = [0.485, 0.456, 0.406];
const DET_STD = [0.229, 0.224, 0.225];
const DB_THRESH = 0.3;
const DB_BOX_THRESH = 0.6;
const DB_UNCLIP = 1.5;
const DB_MIN_SIZE = 3;
const REC_H = 48;
const REC_MIN_W = 320;
const REC_MAX_W = 1600;
const MIN_LINE_SCORE = 0.5;

// ---- pixels ----

/** Bilinear sample of one channel set; writes B, G, R (0–255) into out. */
function sampleBGR(img: RgbaImage, fx: number, fy: number, out: number[]) {
  const { data, width, height } = img;
  const x = Math.min(Math.max(fx, 0), width - 1);
  const y = Math.min(Math.max(fy, 0), height - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, width - 1), y1 = Math.min(y0 + 1, height - 1);
  const ax = x - x0, ay = y - y0;
  const i00 = (y0 * width + x0) * 4, i10 = (y0 * width + x1) * 4;
  const i01 = (y1 * width + x0) * 4, i11 = (y1 * width + x1) * 4;
  for (let c = 0; c < 3; c++) {
    const top = data[i00 + c]! * (1 - ax) + data[i10 + c]! * ax;
    const bot = data[i01 + c]! * (1 - ax) + data[i11 + c]! * ax;
    out[2 - c] = top * (1 - ay) + bot * ay; // RGBA → BGR
  }
}

// ---- detection ----

export function detPreprocess(img: RgbaImage) {
  const ratio = DET_LIMIT / Math.max(img.width, img.height);
  const rw = Math.max(32, Math.round((img.width * ratio) / 32) * 32);
  const rh = Math.max(32, Math.round((img.height * ratio) / 32) * 32);
  const plane = rw * rh;
  const t = new Float32Array(3 * plane);
  const sx = img.width / rw, sy = img.height / rh;
  const px = [0, 0, 0];
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      sampleBGR(img, (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5, px);
      const o = y * rw + x;
      for (let c = 0; c < 3; c++) t[c * plane + o] = (px[c]! / 255 - DET_MEAN[c]!) / DET_STD[c]!;
    }
  }
  return { tensor: t, width: rw, height: rh };
}

type Pt = [number, number];

/** Andrew's monotone chain; input needn't be sorted. */
function convexHull(pts: Pt[]): Pt[] {
  if (pts.length < 3) return pts.slice();
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, q) <= 0) lower.pop(); lower.push(q); }
  const upper: Pt[] = [];
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]!; while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, q) <= 0) upper.pop(); upper.push(q); }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

/** Minimum-area rotated rectangle around a convex hull (rotating calipers over hull edges). */
function minAreaRect(hull: Pt[]): Quad {
  let best: Quad | null = null;
  let bestArea = Infinity;
  const n = hull.length;
  for (let i = 0; i < Math.max(1, n); i++) {
    const a = hull[i]!, b = hull[(i + 1) % n] ?? a;
    let ux = b[0] - a[0], uy = b[1] - a[1];
    const len = Math.hypot(ux, uy);
    if (len < 1e-6) { ux = 1; uy = 0; } else { ux /= len; uy /= len; }
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
    for (const q of hull) {
      const u = q[0] * ux + q[1] * uy, v = -q[0] * uy + q[1] * ux;
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;
    }
    const area = (maxU - minU) * (maxV - minV);
    if (area < bestArea) {
      bestArea = area;
      const cu = (minU + maxU) / 2, cv = (minV + maxV) / 2;
      best = { cx: cu * ux - cv * uy, cy: cu * uy + cv * ux, w: maxU - minU, h: maxV - minV, ux, uy };
    }
  }
  // Text runs along the long side, left → right.
  let q = best!;
  if (q.h > q.w) q = { ...q, w: q.h, h: q.w, ux: -q.uy, uy: q.ux };
  if (q.ux < 0 || (q.ux === 0 && q.uy < 0)) q = { ...q, ux: -q.ux, uy: -q.uy };
  return q;
}

export function quadBounds(q: Quad, srcW: number, srcH: number): Box {
  const ex = Math.abs(q.ux) * q.w / 2 + Math.abs(q.uy) * q.h / 2;
  const ey = Math.abs(q.uy) * q.w / 2 + Math.abs(q.ux) * q.h / 2;
  const x0 = Math.max(0, q.cx - ex), y0 = Math.max(0, q.cy - ey);
  const x1 = Math.min(srcW, q.cx + ex), y1 = Math.min(srcH, q.cy + ey);
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}

/** DB post-process: connected regions of the probability map → rotated text rectangles. */
export function dbPostprocess(prob: Float32Array, mw: number, mh: number, srcW: number, srcH: number): Quad[] {
  const n = mw * mh;
  const labels = new Int32Array(n); // 0 = unvisited
  const stack = new Int32Array(n);
  const kx = srcW / mw, ky = srcH / mh;
  const quads: Quad[] = [];
  let label = 0;
  for (let start = 0; start < n && quads.length < 1000; start++) {
    if (labels[start] !== 0 || prob[start]! <= DB_THRESH) continue;
    label++;
    let sp = 0;
    stack[sp++] = start;
    labels[start] = label;
    let count = 0, sum = 0;
    // Row extremes are enough for the convex hull.
    const rowMin = new Map<number, number>(), rowMax = new Map<number, number>();
    while (sp > 0) {
      const p = stack[--sp]!;
      const px = p % mw, py = (p - px) / mw;
      count++; sum += prob[p]!;
      const lo = rowMin.get(py); if (lo === undefined || px < lo) rowMin.set(py, px);
      const hi = rowMax.get(py); if (hi === undefined || px > hi) rowMax.set(py, px);
      for (let dy = -1; dy <= 1; dy++) {
        const ny = py + dy;
        if (ny < 0 || ny >= mh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx;
          if ((dx === 0 && dy === 0) || nx < 0 || nx >= mw) continue;
          const q = ny * mw + nx;
          if (labels[q] === 0 && prob[q]! > DB_THRESH) { labels[q] = label; stack[sp++] = q; }
        }
      }
    }
    // Score = mean probability over the region itself (a tilted line's upright box is mostly
    // background, which used to drop tilted lines).
    if (count < DB_MIN_SIZE * DB_MIN_SIZE || sum / count < DB_BOX_THRESH) continue;
    const pts: Pt[] = [];
    for (const [y, x] of rowMin) { pts.push([x, y], [x + 1, y], [x, y + 1]); }
    for (const [y, x] of rowMax) { pts.push([x + 1, y], [x + 1, y + 1], [x, y + 1]); }
    const r = minAreaRect(convexHull(pts));
    if (Math.min(r.w, r.h) < DB_MIN_SIZE) continue;
    // Unclip: the model predicts shrunk text kernels; grow by area × ratio / perimeter.
    const d = (r.w * r.h * DB_UNCLIP) / (2 * (r.w + r.h));
    const w = r.w + 2 * d, h = r.h + 2 * d;
    if (Math.min(w, h) < DB_MIN_SIZE + 2) continue;
    // Map → source pixels (x and y scale can differ slightly because of the /32 rounding).
    const ux = r.ux * kx, uy = r.uy * ky, ul = Math.hypot(ux, uy);
    const vx = -r.uy * kx, vy = r.ux * ky, vl = Math.hypot(vx, vy);
    quads.push({ cx: r.cx * kx, cy: r.cy * ky, w: w * ul, h: h * vl, ux: ux / ul, uy: uy / ul });
  }
  return quads;
}

// ---- recognition ----

/** Straightened crop of one text line (sampled along its own direction), height 48. */
export function recPreprocess(img: RgbaImage, q: Quad) {
  const ratio = q.w / q.h;
  const resizedW = Math.max(1, Math.min(REC_MAX_W, Math.ceil(REC_H * ratio)));
  const W = Math.max(REC_MIN_W, resizedW);
  const plane = REC_H * W;
  const t = new Float32Array(3 * plane); // padding stays 0 (= mid grey after normalisation)
  const sx = q.w / resizedW, sy = q.h / REC_H;
  const vx = -q.uy, vy = q.ux; // across the line, pointing down
  const ox = q.cx - q.ux * q.w / 2 - vx * q.h / 2;
  const oy = q.cy - q.uy * q.w / 2 - vy * q.h / 2;
  const px = [0, 0, 0];
  for (let y = 0; y < REC_H; y++) {
    const fy = (y + 0.5) * sy;
    for (let x = 0; x < resizedW; x++) {
      const fx = (x + 0.5) * sx;
      sampleBGR(img, ox + q.ux * fx + vx * fy - 0.5, oy + q.uy * fx + vy * fy - 0.5, px);
      const o = y * W + x;
      for (let c = 0; c < 3; c++) t[c * plane + o] = (px[c]! / 255 - 0.5) / 0.5;
    }
  }
  return { tensor: t, width: W };
}

export function ctcDecode(logits: Float32Array, steps: number, classes: number): { text: string; score: number } {
  let text = '';
  let confSum = 0, confN = 0;
  let prev = -1;
  const row = new Float32Array(classes);
  for (let t = 0; t < steps; t++) {
    const off = t * classes;
    let best = 0, bestV = -Infinity, sum = 0, min = Infinity;
    for (let c = 0; c < classes; c++) {
      const v = logits[off + c]!;
      row[c] = v;
      sum += v;
      if (v < min) min = v;
      if (v > bestV) { bestV = v; best = c; }
    }
    // The exported model already ends in softmax; fall back to it if these are raw logits.
    let p = bestV;
    if (min < 0 || Math.abs(sum - 1) > 1e-2) {
      let z = 0;
      for (let c = 0; c < classes; c++) z += Math.exp(row[c]! - bestV);
      p = 1 / z;
    }
    if (best !== 0 && best !== prev) {
      text += best - 1 < CHARS.length ? CHARS[best - 1] : ' ';
      confSum += p; confN++;
    }
    prev = best;
  }
  return { text: text.trim(), score: confN ? confSum / confN : 0 };
}

// ---- lines ----

/** Order boxes into reading order and merge words that sit on the same line close together. */
function toLines(items: OcrLine[]): OcrLine[] {
  const sorted = [...items].sort((a, b) => (a.box.y + a.box.h / 2) - (b.box.y + b.box.h / 2));
  const rows: OcrLine[][] = [];
  for (const it of sorted) {
    const cy = it.box.y + it.box.h / 2;
    const row = rows.find((r) => {
      const ref = r[0]!;
      return Math.abs(cy - (ref.box.y + ref.box.h / 2)) < Math.min(ref.lineHeight, it.lineHeight) * 0.5;
    });
    if (row) row.push(it); else rows.push([it]);
  }
  const out: OcrLine[] = [];
  for (const row of rows) {
    row.sort((a, b) => a.box.x - b.box.x);
    let cur: OcrLine | null = null;
    for (const it of row) {
      if (cur && it.box.x - (cur.box.x + cur.box.w) < Math.max(cur.lineHeight, it.lineHeight) * 1.2) {
        const x0 = Math.min(cur.box.x, it.box.x), y0 = Math.min(cur.box.y, it.box.y);
        const x1 = Math.max(cur.box.x + cur.box.w, it.box.x + it.box.w), y1 = Math.max(cur.box.y + cur.box.h, it.box.y + it.box.h);
        const len: number = cur.text.length + it.text.length || 1;
        cur = {
          text: `${cur.text} ${it.text}`,
          score: (cur.score * cur.text.length + it.score * it.text.length) / len,
          box: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
          lineHeight: Math.max(cur.lineHeight, it.lineHeight),
        };
      } else {
        if (cur) out.push(cur);
        cur = { ...it, box: { ...it.box } };
      }
    }
    if (cur) out.push(cur);
  }
  return out;
}

export interface OcrTimings { detMs: number; recMs: number; boxes: number }

/** Full pass: image → text lines in reading order. */
export async function recognize(
  img: RgbaImage,
  det: SessionLike,
  rec: SessionLike,
  makeTensor: MakeTensor,
  onProgress?: (done: number, total: number) => void,
): Promise<{ lines: OcrLine[]; timings: OcrTimings }> {
  const t0 = Date.now();
  const d = detPreprocess(img);
  const detOut = await det.run({ [det.inputNames[0]!]: makeTensor(d.tensor, [1, 3, d.height, d.width]) });
  const map = detOut[det.outputNames[0]!]!;
  const [, , mh, mw] = map.dims as number[];
  const quads = dbPostprocess(map.data as Float32Array, mw!, mh!, img.width, img.height)
    .filter((q) => q.h >= 6 && Math.abs(q.uy) < 0.5) // tiny specks; text tilted > 30° (vertical rules)
    .sort((a, b) => a.cy - b.cy)
    .slice(0, 80);
  const t1 = Date.now();

  const items: OcrLine[] = [];
  for (let i = 0; i < quads.length; i++) {
    const q = quads[i]!;
    const r = recPreprocess(img, q);
    const out = await rec.run({ [rec.inputNames[0]!]: makeTensor(r.tensor, [1, 3, REC_H, r.width]) });
    const o = out[rec.outputNames[0]!]!;
    const [, steps, classes] = o.dims as number[];
    const { text, score } = ctcDecode(o.data as Float32Array, steps!, classes!);
    if (text && score >= MIN_LINE_SCORE) items.push({ text, score, box: quadBounds(q, img.width, img.height), lineHeight: q.h });
    onProgress?.(i + 1, quads.length);
  }
  return { lines: toLines(items), timings: { detMs: t1 - t0, recMs: Date.now() - t1, boxes: quads.length } };
}
