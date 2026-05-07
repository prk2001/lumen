// Lumen live demo — implements 5 effects in pure JS using the same
// math as the Rust core (lumen-core::color + the lumen-fx-* crates).
//
// Effects: brightness_contrast, gamma, saturation, unsharp_mask,
// channel_isolate. Recipes use the same JSON format the CLI's
// `pipeline` subcommand consumes.

(() => {
  'use strict';

  // ─── sRGB ↔ linear (matches lumen-core::color) ──────────────────────
  function srgbToLinear(c) {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  function linearToSrgb(c) {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
  }

  // Convert ImageData (sRGB-encoded u8) -> linear-light Float32Array RGBA.
  function imageDataToLinearF32(imageData) {
    const n = imageData.data.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 4) {
      out[i]     = srgbToLinear(imageData.data[i]     / 255);
      out[i + 1] = srgbToLinear(imageData.data[i + 1] / 255);
      out[i + 2] = srgbToLinear(imageData.data[i + 2] / 255);
      out[i + 3] = imageData.data[i + 3] / 255;
    }
    return out;
  }

  // Convert linear-light Float32Array back to sRGB-encoded ImageData.
  function linearF32ToImageData(buf, w, h) {
    const id = new ImageData(w, h);
    for (let i = 0; i < buf.length; i += 4) {
      id.data[i]     = Math.max(0, Math.min(255, Math.round(linearToSrgb(buf[i])     * 255)));
      id.data[i + 1] = Math.max(0, Math.min(255, Math.round(linearToSrgb(buf[i + 1]) * 255)));
      id.data[i + 2] = Math.max(0, Math.min(255, Math.round(linearToSrgb(buf[i + 2]) * 255)));
      id.data[i + 3] = Math.max(0, Math.min(255, Math.round(buf[i + 3] * 255)));
    }
    return id;
  }

  // ─── Effects ────────────────────────────────────────────────────────

  // brightness_contrast — out_rgb = (in - 0.5) * contrast + 0.5 + brightness
  // (linear-light, alpha untouched). Mirrors lumen-fx-exposure.
  function fxBrightnessContrast(buf, w, h, p) {
    const b = +p.brightness, c = +p.contrast;
    for (let i = 0; i < buf.length; i += 4) {
      buf[i]     = Math.min(1, Math.max(0, (buf[i]     - 0.5) * c + 0.5 + b));
      buf[i + 1] = Math.min(1, Math.max(0, (buf[i + 1] - 0.5) * c + 0.5 + b));
      buf[i + 2] = Math.min(1, Math.max(0, (buf[i + 2] - 0.5) * c + 0.5 + b));
    }
    return buf;
  }

  // gamma — out = in^(1/gamma) per RGB. Mirrors lumen-fx-exposure.gamma.
  function fxGamma(buf, w, h, p) {
    const inv = 1 / Math.max(0.05, +p.gamma);
    for (let i = 0; i < buf.length; i += 4) {
      buf[i]     = Math.min(1, Math.max(0, Math.pow(Math.max(0, buf[i]),     inv)));
      buf[i + 1] = Math.min(1, Math.max(0, Math.pow(Math.max(0, buf[i + 1]), inv)));
      buf[i + 2] = Math.min(1, Math.max(0, Math.pow(Math.max(0, buf[i + 2]), inv)));
    }
    return buf;
  }

  // saturation — Y = 0.2126R + 0.7152G + 0.0722B; out = mix(Y, in, amount).
  // Mirrors lumen-fx-color.saturation.
  function fxSaturation(buf, w, h, p) {
    const a = +p.amount;
    for (let i = 0; i < buf.length; i += 4) {
      const y = 0.2126 * buf[i] + 0.7152 * buf[i + 1] + 0.0722 * buf[i + 2];
      buf[i]     = Math.min(1, Math.max(0, y + (buf[i]     - y) * a));
      buf[i + 1] = Math.min(1, Math.max(0, y + (buf[i + 1] - y) * a));
      buf[i + 2] = Math.min(1, Math.max(0, y + (buf[i + 2] - y) * a));
    }
    return buf;
  }

  // Separable Gaussian helper (used by unsharp_mask).
  function gaussianKernel(sigma) {
    const half = Math.max(1, Math.min(64, Math.ceil(3 * sigma)));
    const len = 2 * half + 1;
    const k = new Float32Array(len);
    const inv2sigma2 = 1 / (2 * sigma * sigma);
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const x = i - half;
      const v = Math.exp(-x * x * inv2sigma2);
      k[i] = v;
      sum += v;
    }
    for (let i = 0; i < len; i++) k[i] /= sum;
    return { kernel: k, half };
  }

  function gaussianBlurRgba(src, w, h, sigma) {
    const { kernel, half } = gaussianKernel(sigma);
    const stride = w * 4;
    const tmp = new Float32Array(src.length);
    // Horizontal pass
    for (let y = 0; y < h; y++) {
      const rowOff = y * stride;
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0, a = 0;
        for (let i = 0; i < kernel.length; i++) {
          let xi = x + i - half;
          if (xi < 0) xi = 0;
          else if (xi >= w) xi = w - 1;
          const o = rowOff + xi * 4;
          const wt = kernel[i];
          r += src[o]     * wt;
          g += src[o + 1] * wt;
          b += src[o + 2] * wt;
          a += src[o + 3] * wt;
        }
        const o = rowOff + x * 4;
        tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b; tmp[o + 3] = a;
      }
    }
    // Vertical pass back into src
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0, a = 0;
        for (let i = 0; i < kernel.length; i++) {
          let yi = y + i - half;
          if (yi < 0) yi = 0;
          else if (yi >= h) yi = h - 1;
          const o = yi * stride + x * 4;
          const wt = kernel[i];
          r += tmp[o]     * wt;
          g += tmp[o + 1] * wt;
          b += tmp[o + 2] * wt;
          a += tmp[o + 3] * wt;
        }
        const o = y * stride + x * 4;
        src[o] = r; src[o + 1] = g; src[o + 2] = b; src[o + 3] = a;
      }
    }
    return src;
  }

  // unsharp_mask — separable Gaussian blur then add detail back.
  // Mirrors lumen-fx-sharpen.unsharp_mask.
  function fxUnsharpMask(buf, w, h, p) {
    const amount = +p.amount, radius = Math.max(0.1, +p.radius), threshold = +p.threshold;
    if (amount === 0) return buf;
    const blurred = new Float32Array(buf);
    gaussianBlurRgba(blurred, w, h, radius);
    for (let i = 0; i < buf.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const detail = buf[i + c] - blurred[i + c];
        const det = Math.abs(detail) < threshold ? 0 : detail;
        buf[i + c] = Math.min(1, Math.max(0, buf[i + c] + amount * det));
      }
    }
    return buf;
  }

  // gaussian denoise — separable Gaussian, no detail addition.
  // Mirrors lumen-fx-denoise.gaussian.
  function fxGaussianDenoise(buf, w, h, p) {
    const sigma = Math.max(0.1, +p.sigma);
    gaussianBlurRgba(buf, w, h, sigma);
    return buf;
  }

  // laplacian sharpen — DoG-based edge enhancement.
  // out = in + amount * (gauss(in, sigma) - gauss(in, sigma * sigma_ratio))
  // Mirrors lumen-fx-deblur.laplacian.
  function fxLaplacian(buf, w, h, p) {
    const amount = +p.amount;
    const sigma = Math.max(0.1, +p.sigma);
    const ratio = Math.max(1.05, +p.sigma_ratio);
    if (amount === 0) return buf;
    const inner = new Float32Array(buf);
    gaussianBlurRgba(inner, w, h, sigma);
    const outer = new Float32Array(buf);
    gaussianBlurRgba(outer, w, h, sigma * ratio);
    for (let i = 0; i < buf.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const lap = inner[i + c] - outer[i + c];
        buf[i + c] = Math.min(1, Math.max(0, buf[i + c] + amount * lap));
      }
    }
    return buf;
  }

  // deblock — 1-D Gaussian along JPEG-style block boundaries only.
  // Mirrors lumen-fx-compression.deblock.
  function fxDeblock(buf, w, h, p) {
    const blockSize = Math.max(2, Math.floor(+p.block_size || 8));
    const strength = Math.max(0, +p.strength);
    if (strength === 0) return buf;
    // Build a 1D Gaussian kernel of sigma=strength.
    const { kernel, half } = gaussianKernel(strength);
    const stride = w * 4;
    // Operate on a snapshot so passes don't smear.
    const snap = new Float32Array(buf);
    // Vertical pass on horizontal block boundaries (rows blockSize, 2*blockSize, …)
    for (let by = blockSize; by < h; by += blockSize) {
      const yStart = Math.max(0, by - half);
      const yEnd = Math.min(h - 1, by + half);
      for (let y = yStart; y <= yEnd; y++) {
        for (let x = 0; x < w; x++) {
          let r = 0, g = 0, bb = 0;
          for (let i = 0; i < kernel.length; i++) {
            let yi = y + i - half;
            if (yi < 0) yi = 0; else if (yi >= h) yi = h - 1;
            const o = yi * stride + x * 4;
            const wt = kernel[i];
            r += snap[o] * wt; g += snap[o+1] * wt; bb += snap[o+2] * wt;
          }
          const o = y * stride + x * 4;
          buf[o] = r; buf[o+1] = g; buf[o+2] = bb;
        }
      }
    }
    // Horizontal pass on vertical block boundaries.
    const snap2 = new Float32Array(buf);
    for (let bx = blockSize; bx < w; bx += blockSize) {
      const xStart = Math.max(0, bx - half);
      const xEnd = Math.min(w - 1, bx + half);
      for (let y = 0; y < h; y++) {
        const rowOff = y * stride;
        for (let x = xStart; x <= xEnd; x++) {
          let r = 0, g = 0, bb = 0;
          for (let i = 0; i < kernel.length; i++) {
            let xi = x + i - half;
            if (xi < 0) xi = 0; else if (xi >= w) xi = w - 1;
            const o = rowOff + xi * 4;
            const wt = kernel[i];
            r += snap2[o] * wt; g += snap2[o+1] * wt; bb += snap2[o+2] * wt;
          }
          const o = rowOff + x * 4;
          buf[o] = r; buf[o+1] = g; buf[o+2] = bb;
        }
      }
    }
    return buf;
  }

  // dehaze (Dark Channel Prior) — He et al. 2009.
  // Mirrors lumen-fx-weather.dehaze_dcp.
  function fxDehazeDcp(buf, w, h, p) {
    const omega = Math.max(0.0, Math.min(1.0, +p.omega));
    const t0 = Math.max(0.01, +p.t0);
    const patchR = Math.max(1, Math.floor(+p.patch_radius));
    const n = w * h;
    // 1. per-pixel min over RGB
    const minRgb = new Float32Array(n);
    for (let i = 0, j = 0; i < buf.length; i += 4, j++) {
      minRgb[j] = Math.min(buf[i], buf[i + 1], buf[i + 2]);
    }
    // 2. dark channel = min-pool of minRgb over (2*patchR+1)^2 window
    const dark = new Float32Array(n);
    for (let y = 0; y < h; y++) {
      const yLo = Math.max(0, y - patchR);
      const yHi = Math.min(h - 1, y + patchR);
      for (let x = 0; x < w; x++) {
        const xLo = Math.max(0, x - patchR);
        const xHi = Math.min(w - 1, x + patchR);
        let m = 1.0;
        for (let yy = yLo; yy <= yHi; yy++) {
          for (let xx = xLo; xx <= xHi; xx++) {
            const v = minRgb[yy * w + xx];
            if (v < m) m = v;
          }
        }
        dark[y * w + x] = m;
      }
    }
    // 3. atmospheric light A: take 0.1% brightest dark-channel pixels,
    //    pick max RGB intensity among them.
    const k = Math.max(1, Math.floor(n * 0.001));
    // simple: scan through all, keep top-k indices by darkChannel
    const indices = new Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;
    indices.sort((a, b) => dark[b] - dark[a]);
    let aR = 0, aG = 0, aB = 0;
    for (let i = 0; i < k; i++) {
      const idx = indices[i];
      const o = idx * 4;
      const sum = buf[o] + buf[o + 1] + buf[o + 2];
      const ar = buf[o], ag = buf[o + 1], ab = buf[o + 2];
      if (ar + ag + ab > aR + aG + aB) {
        aR = ar; aG = ag; aB = ab;
      }
    }
    // Guard against pure-black A (rare but possible synthetic cases).
    if (aR + aG + aB < 1e-4) { aR = aG = aB = 1.0; }
    // 4. transmission t = 1 - omega * dark(I/A) (compute dark of normalized I)
    // For speed, approximate dark(I/A) ≈ minRgb / minA (per-channel A).
    const minA = Math.min(aR, aG, aB);
    // 5. recover J = (I - A)/max(t, t0) + A
    for (let i = 0, j = 0; i < buf.length; i += 4, j++) {
      const t = Math.max(t0, 1 - omega * dark[j] / Math.max(0.05, minA));
      buf[i]     = Math.min(1, Math.max(0, (buf[i]     - aR) / t + aR));
      buf[i + 1] = Math.min(1, Math.max(0, (buf[i + 1] - aG) / t + aG));
      buf[i + 2] = Math.min(1, Math.max(0, (buf[i + 2] - aB) / t + aB));
    }
    return buf;
  }

  // CLAHE — Contrast-Limited Adaptive Histogram Equalization on luma.
  // Mirrors lumen-fx-text.clahe (chroma-preserving rescale).
  function fxClahe(buf, w, h, p) {
    const tilesX = Math.max(1, Math.min(64, Math.floor(+p.tiles_x)));
    const tilesY = Math.max(1, Math.min(64, Math.floor(+p.tiles_y)));
    const clipLimit = +p.clip_limit;
    const BINS = 256;
    const tw = Math.max(1, Math.floor(w / tilesX));
    const th = Math.max(1, Math.floor(h / tilesY));
    const numTiles = tilesX * tilesY;
    const cdfs = new Array(numTiles); // each: Float32Array(BINS) mapping bin -> [0,1]

    // Per-tile histograms + CDFs
    for (let ty = 0; ty < tilesY; ty++) {
      const y0 = ty * th;
      const y1 = (ty === tilesY - 1) ? h : Math.min(h, y0 + th);
      for (let tx = 0; tx < tilesX; tx++) {
        const x0 = tx * tw;
        const x1 = (tx === tilesX - 1) ? w : Math.min(w, x0 + tw);
        const hist = new Uint32Array(BINS);
        let count = 0;
        for (let y = y0; y < y1; y++) {
          const rowOff = y * w * 4;
          for (let x = x0; x < x1; x++) {
            const o = rowOff + x * 4;
            const yLuma = 0.2126 * buf[o] + 0.7152 * buf[o + 1] + 0.0722 * buf[o + 2];
            const bin = Math.min(BINS - 1, Math.max(0, Math.round(yLuma * (BINS - 1))));
            hist[bin]++;
            count++;
          }
        }
        // Clip histograms above clipLimit * (count / BINS), redistribute.
        if (clipLimit >= 0.01 && count > 0) {
          const limit = Math.max(1, Math.floor(clipLimit * count / BINS));
          let excess = 0;
          for (let i = 0; i < BINS; i++) {
            if (hist[i] > limit) { excess += hist[i] - limit; hist[i] = limit; }
          }
          const inc = Math.floor(excess / BINS);
          for (let i = 0; i < BINS; i++) hist[i] += inc;
          let rem = excess - inc * BINS;
          for (let i = 0; i < BINS && rem > 0; i++) { hist[i]++; rem--; }
        }
        // CDF
        const cdf = new Float32Array(BINS);
        let cum = 0;
        const denom = count > 0 ? count : 1;
        for (let i = 0; i < BINS; i++) {
          cum += hist[i];
          cdf[i] = cum / denom;
        }
        // Shift so CDF starts at 0
        const cdfMin = cdf[0];
        const span = 1 - cdfMin;
        if (span > 1e-9) {
          for (let i = 0; i < BINS; i++) {
            cdf[i] = Math.min(1, Math.max(0, (cdf[i] - cdfMin) / span));
          }
        } else {
          for (let i = 0; i < BINS; i++) cdf[i] = i / (BINS - 1);
        }
        cdfs[ty * tilesX + tx] = cdf;
      }
    }

    // Per-pixel: bilinear-interpolate the four nearest tile-CDFs of the
    // input luma bin, then re-color via Y' / Y scaling.
    for (let y = 0; y < h; y++) {
      // Find tile-Y center indices straddling this row.
      const fy = Math.max(0, Math.min(tilesY - 1, (y + 0.5) / th - 0.5));
      const ty0 = Math.max(0, Math.floor(fy));
      const ty1 = Math.min(tilesY - 1, ty0 + 1);
      const wy = fy - ty0;
      for (let x = 0; x < w; x++) {
        const fx = Math.max(0, Math.min(tilesX - 1, (x + 0.5) / tw - 0.5));
        const tx0 = Math.max(0, Math.floor(fx));
        const tx1 = Math.min(tilesX - 1, tx0 + 1);
        const wx = fx - tx0;
        const o = (y * w + x) * 4;
        const r = buf[o], g = buf[o + 1], b = buf[o + 2];
        const yOld = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const bin = Math.min(BINS - 1, Math.max(0, Math.round(yOld * (BINS - 1))));
        const v00 = cdfs[ty0 * tilesX + tx0][bin];
        const v10 = cdfs[ty0 * tilesX + tx1][bin];
        const v01 = cdfs[ty1 * tilesX + tx0][bin];
        const v11 = cdfs[ty1 * tilesX + tx1][bin];
        const top = v00 * (1 - wx) + v10 * wx;
        const bot = v01 * (1 - wx) + v11 * wx;
        const yNew = top * (1 - wy) + bot * wy;
        const k = yNew / Math.max(1e-4, yOld);
        buf[o]     = Math.min(1, Math.max(0, r * k));
        buf[o + 1] = Math.min(1, Math.max(0, g * k));
        buf[o + 2] = Math.min(1, Math.max(0, b * k));
      }
    }
    return buf;
  }

  // channel_isolate — output one channel as gray.
  // Mirrors lumen-fx-modalities.channel_isolate.
  function fxChannelIsolate(buf, w, h, p) {
    const ch = String(p.channel || 'luma');
    const inv = !!p.invert;
    for (let i = 0; i < buf.length; i += 4) {
      let v;
      switch (ch) {
        case 'r':    v = buf[i];     break;
        case 'g':    v = buf[i + 1]; break;
        case 'b':    v = buf[i + 2]; break;
        case 'a':    v = buf[i + 3]; break;
        case 'luma': // fallthrough default
        default:     v = 0.2126 * buf[i] + 0.7152 * buf[i + 1] + 0.0722 * buf[i + 2];
      }
      if (inv) v = 1 - v;
      buf[i] = buf[i + 1] = buf[i + 2] = Math.min(1, Math.max(0, v));
    }
    return buf;
  }

  // ─── Effect catalog ─────────────────────────────────────────────────
  // Mirrors the metadata in the Rust EffectMetadata + ParamSpec entries
  // so the demo's controls match `lumen list-effects` byte-for-byte.
  const EFFECTS = [
    {
      id: 'lumen-fx-exposure.brightness_contrast',
      label: 'Brightness / Contrast',
      apply: fxBrightnessContrast,
      params: [
        { id: 'brightness', label: 'Brightness', kind: 'float', min: -1, max: 1, step: 0.01, default: 0.0 },
        { id: 'contrast',   label: 'Contrast',   kind: 'float', min:  0, max: 4, step: 0.01, default: 1.0 },
      ],
    },
    {
      id: 'lumen-fx-exposure.gamma',
      label: 'Gamma',
      apply: fxGamma,
      params: [
        { id: 'gamma', label: 'Gamma', kind: 'float', min: 0.1, max: 4, step: 0.01, default: 1.0 },
      ],
    },
    {
      id: 'lumen-fx-color.saturation',
      label: 'Saturation',
      apply: fxSaturation,
      params: [
        { id: 'amount', label: 'Amount', kind: 'float', min: 0, max: 2, step: 0.01, default: 1.0 },
      ],
    },
    {
      id: 'lumen-fx-sharpen.unsharp_mask',
      label: 'Unsharp Mask',
      apply: fxUnsharpMask,
      params: [
        { id: 'amount',    label: 'Amount',    kind: 'float', min: 0, max: 4,   step: 0.01, default: 0.5 },
        { id: 'radius',    label: 'Radius',    kind: 'float', min: 0.1, max: 8, step: 0.05, default: 1.0 },
        { id: 'threshold', label: 'Threshold', kind: 'float', min: 0, max: 1,   step: 0.01, default: 0.0 },
      ],
    },
    {
      id: 'lumen-fx-modalities.channel_isolate',
      label: 'Channel Isolate',
      apply: fxChannelIsolate,
      params: [
        { id: 'channel', label: 'Channel', kind: 'choice', options: ['r','g','b','a','luma'], default: 'luma' },
        { id: 'invert',  label: 'Invert',  kind: 'bool',   default: false },
      ],
    },
    {
      id: 'lumen-fx-denoise.gaussian',
      label: 'Gaussian Denoise',
      apply: fxGaussianDenoise,
      params: [
        { id: 'sigma', label: 'Sigma', kind: 'float', min: 0.1, max: 5, step: 0.05, default: 1.0 },
      ],
    },
    {
      id: 'lumen-fx-compression.deblock',
      label: 'Deblock (JPEG)',
      apply: fxDeblock,
      params: [
        { id: 'block_size', label: 'Block size', kind: 'choice', options: ['4','8','16'], default: '8' },
        { id: 'strength',   label: 'Strength',   kind: 'float',  min: 0, max: 4, step: 0.05, default: 0.6 },
      ],
    },
    {
      id: 'lumen-fx-weather.dehaze_dcp',
      label: 'Dehaze (DCP)',
      apply: fxDehazeDcp,
      params: [
        { id: 'omega',        label: 'Strength (omega)', kind: 'float', min: 0,    max: 1,   step: 0.01, default: 0.85 },
        { id: 't0',           label: 'Floor (t0)',       kind: 'float', min: 0.01, max: 0.5, step: 0.01, default: 0.10 },
        { id: 'patch_radius', label: 'Patch radius',     kind: 'float', min: 1,    max: 15,  step: 1,    default: 5 },
      ],
    },
    {
      id: 'lumen-fx-text.clahe',
      label: 'CLAHE (plate clarify)',
      apply: fxClahe,
      params: [
        { id: 'tiles_x',    label: 'Tiles X',    kind: 'float', min: 1, max: 32, step: 1,    default: 8 },
        { id: 'tiles_y',    label: 'Tiles Y',    kind: 'float', min: 1, max: 32, step: 1,    default: 8 },
        { id: 'clip_limit', label: 'Clip limit', kind: 'float', min: 0, max: 8,  step: 0.1,  default: 2.5 },
      ],
    },
    {
      id: 'lumen-fx-deblur.laplacian',
      label: 'Laplacian Deblur',
      apply: fxLaplacian,
      params: [
        { id: 'amount',      label: 'Amount',      kind: 'float', min: 0,   max: 4,   step: 0.05, default: 0.9 },
        { id: 'sigma',       label: 'Sigma',       kind: 'float', min: 0.2, max: 5,   step: 0.05, default: 0.8 },
        { id: 'sigma_ratio', label: 'Sigma ratio', kind: 'float', min: 1.05,max: 5,   step: 0.05, default: 1.6 },
      ],
    },
  ];

  // Strength tiers for "Clarify (CCTV)" — mirrors clarify.rs in lumen-cli.
  const CLARIFY_PRESETS = {
    light: {
      nr_sigma: 0.6, deblock_strength: 0.3, dehaze_omega: 0.6,
      clahe_clip: 1.5, clahe_tiles: 8, laplacian_amount: 0.5,
      unsharp_amount: 0.5, bc_contrast: 1.05,
    },
    standard: {
      nr_sigma: 0.9, deblock_strength: 0.6, dehaze_omega: 0.8,
      clahe_clip: 2.5, clahe_tiles: 8, laplacian_amount: 0.9,
      unsharp_amount: 0.8, bc_contrast: 1.10,
    },
    aggressive: {
      nr_sigma: 1.4, deblock_strength: 0.9, dehaze_omega: 0.95,
      clahe_clip: 4.0, clahe_tiles: 12, laplacian_amount: 1.4,
      unsharp_amount: 1.3, bc_contrast: 1.20,
    },
  };

  function buildClarifyChain(strength) {
    const p = CLARIFY_PRESETS[strength] || CLARIFY_PRESETS.standard;
    return [
      { effect: 'lumen-fx-denoise.gaussian',         params: { sigma: p.nr_sigma } },
      { effect: 'lumen-fx-compression.deblock',      params: { block_size: 8, strength: p.deblock_strength } },
      { effect: 'lumen-fx-weather.dehaze_dcp',       params: { omega: p.dehaze_omega, t0: 0.1, patch_radius: 5 } },
      { effect: 'lumen-fx-text.clahe',               params: { tiles_x: p.clahe_tiles, tiles_y: p.clahe_tiles, clip_limit: p.clahe_clip } },
      { effect: 'lumen-fx-deblur.laplacian',         params: { amount: p.laplacian_amount, sigma: 0.8, sigma_ratio: 1.6 } },
      { effect: 'lumen-fx-sharpen.unsharp_mask',     params: { amount: p.unsharp_amount, radius: 1.0, threshold: 0.0 } },
      { effect: 'lumen-fx-exposure.brightness_contrast', params: { brightness: 0.0, contrast: p.bc_contrast } },
    ];
  }

  function defaultParams(eff) {
    const p = {};
    for (const spec of eff.params) p[spec.id] = spec.default;
    return p;
  }
  function effectById(id) { return EFFECTS.find(e => e.id === id); }

  // ─── Image analyzer + auto-chain builder ────────────────────────────
  // Single-pass analysis of a linear-light Float32Array: percentiles,
  // per-channel means, chroma magnitude, and a horizontal-gradient
  // edge proxy. The CLI's `lumen auto-enhance` mirrors this math.
  function analyzeLinear(linBuf, w, h) {
    const n = (linBuf.length / 4) | 0;
    const yArr = new Float32Array(n);
    let rSum = 0, gSum = 0, bSum = 0, chromaSum = 0;
    for (let i = 0, j = 0; i < linBuf.length; i += 4, j++) {
      const r = linBuf[i], g = linBuf[i + 1], b = linBuf[i + 2];
      rSum += r; gSum += g; bSum += b;
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      chromaSum += mx - mn;
      yArr[j] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }
    // Sort a copy for percentiles. JS Array sort is fine here at 480x320.
    const sorted = Array.prototype.slice.call(yArr).sort((a, b) => a - b);
    const pct = q => sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1))))];
    const p01 = pct(0.005), p10 = pct(0.10), p50 = pct(0.50), p90 = pct(0.90), p99 = pct(0.995);

    // Edge proxy via horizontal abs-difference of luma (cheap, sufficient).
    let edgeSum = 0, edgeCount = 0;
    for (let y = 0; y < h; y++) {
      const rowOff = y * w;
      for (let x = 1; x < w; x++) {
        edgeSum += Math.abs(yArr[rowOff + x] - yArr[rowOff + x - 1]);
        edgeCount++;
      }
    }
    return {
      p01, p10, p50, p90, p99,
      rMean: rSum / n, gMean: gSum / n, bMean: bSum / n,
      chromaMean: chromaSum / n,
      edgeMean: edgeCount > 0 ? edgeSum / edgeCount : 0,
      luminanceMean: (rSum + gSum + bSum) / (3 * n),
    };
  }

  // Build a 4-step (max) chain from the stats. Mirrors the Rust
  // implementation in lumen-cli's auto module.
  function buildAutoChain(stats) {
    const chain = [];
    const round2 = x => Math.round(x * 100) / 100;
    const round3 = x => Math.round(x * 1000) / 1000;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

    // 1. Brightness/contrast: stretch p01->0.05 and p99->0.95.
    //    BC math: out = (in - 0.5) * c + 0.5 + b
    //    p01 -> 0.05: (p01 - 0.5)*c + 0.5 + b = 0.05
    //    p99 -> 0.95: (p99 - 0.5)*c + 0.5 + b = 0.95
    //    Subtract: (p99 - p01)*c = 0.90  ->  c = 0.90 / (p99 - p01)
    const range = Math.max(0.02, stats.p99 - stats.p01);
    let c = clamp(0.90 / range, 0.7, 2.5);
    let b = clamp(0.05 - (stats.p01 - 0.5) * c - 0.5, -0.4, 0.4);
    if (Math.abs(c - 1) > 0.04 || Math.abs(b) > 0.02) {
      chain.push({
        effect: 'lumen-fx-exposure.brightness_contrast',
        params: { brightness: round3(b), contrast: round2(c) },
      });
    }

    // 2. Gamma: pull post-BC p50 toward 0.5.
    const newP50 = clamp((stats.p50 - 0.5) * c + 0.5 + b, 0.02, 0.98);
    if (Math.abs(newP50 - 0.5) > 0.04) {
      // newP50^(1/g) = 0.5  =>  g = log(newP50) / log(0.5)
      const g = clamp(Math.log(newP50) / Math.log(0.5), 0.5, 2.0);
      if (Math.abs(g - 1) > 0.03) {
        chain.push({
          effect: 'lumen-fx-exposure.gamma',
          params: { gamma: round2(g) },
        });
      }
    }

    // 3. Saturation: boost if chroma is low; tame if very high.
    let amount = 1.0;
    if (stats.chromaMean < 0.05)      amount = 1.15;
    else if (stats.chromaMean < 0.10) amount = 1.30;
    else if (stats.chromaMean < 0.20) amount = 1.20;
    else if (stats.chromaMean > 0.40) amount = 0.92;
    if (Math.abs(amount - 1) > 0.03) {
      chain.push({
        effect: 'lumen-fx-color.saturation',
        params: { amount: round2(amount) },
      });
    }

    // 4. Unsharp mask sized by edge density.
    let sAmount = 0.5, sRadius = 1.0;
    if (stats.edgeMean < 0.04)      { sAmount = 0.9; sRadius = 1.2; }
    else if (stats.edgeMean < 0.08) { sAmount = 0.6; sRadius = 1.1; }
    else if (stats.edgeMean > 0.18) { sAmount = 0.25; sRadius = 0.9; }
    chain.push({
      effect: 'lumen-fx-sharpen.unsharp_mask',
      params: { amount: round2(sAmount), radius: round2(sRadius), threshold: 0.0 },
    });

    return chain;
  }

  // ─── State + render ─────────────────────────────────────────────────
  let chain = [
    { effect: 'lumen-fx-color.saturation',         params: { amount: 1.25 } },
    { effect: 'lumen-fx-sharpen.unsharp_mask',     params: { amount: 0.8, radius: 1.0, threshold: 0.0 } },
    { effect: 'lumen-fx-exposure.brightness_contrast', params: { brightness: 0.05, contrast: 1.1 } },
  ];
  let activeStep = 0;

  let inputCanvas, outputCanvas, inputCtx, outputCtx;
  let baseImageData = null, baseW = 0, baseH = 0;
  let renderToken = 0;

  function applyChain() {
    if (!baseImageData) return;
    const myToken = ++renderToken;
    // Lift to linear float, run chain, return.
    const buf = imageDataToLinearF32(baseImageData);
    const t0 = performance.now();
    for (const step of chain) {
      const eff = effectById(step.effect);
      if (!eff) continue;
      const filled = Object.assign(defaultParams(eff), step.params || {});
      eff.apply(buf, baseW, baseH, filled);
      if (myToken !== renderToken) return; // user moved on
    }
    const dt = performance.now() - t0;
    const out = linearF32ToImageData(buf, baseW, baseH);
    outputCtx.putImageData(out, 0, 0);
    const stamp = document.querySelector('#demo-render-time');
    if (stamp) stamp.textContent = `Rendered ${chain.length} effect${chain.length === 1 ? '' : 's'} in ${dt.toFixed(1)} ms`;
    refreshRecipe();
  }

  // Debounced apply for slider drags.
  let applyTimer = null;
  function scheduleApply() {
    clearTimeout(applyTimer);
    applyTimer = setTimeout(applyChain, 12);
  }

  // ─── UI rendering ───────────────────────────────────────────────────
  function $(sel, root) { return (root || document).querySelector(sel); }
  function el(tag, attrs, children) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'text') n.textContent = attrs[k];
      else n.setAttribute(k, attrs[k]);
    }
    if (children) for (const c of children) n.appendChild(c);
    return n;
  }

  function refreshChainStrip() {
    const strip = $('#demo-chain');
    strip.innerHTML = '';
    chain.forEach((step, i) => {
      const eff = effectById(step.effect);
      const chip = el('button', { class: 'chip' + (i === activeStep ? ' active' : '') });
      chip.innerHTML = `<span class="n">${i + 1}</span> ${eff ? eff.label : step.effect} <span class="x" title="remove">✕</span>`;
      chip.addEventListener('click', (e) => {
        if (e.target.classList.contains('x')) {
          chain.splice(i, 1);
          if (activeStep >= chain.length) activeStep = Math.max(0, chain.length - 1);
          refreshAll();
          return;
        }
        activeStep = i;
        refreshChainStrip();
        refreshControls();
      });
      strip.appendChild(chip);
    });

    // Add-effect select
    const addWrap = el('span', { class: 'add-wrap' });
    const sel = el('select', { class: 'add-select' });
    sel.appendChild(el('option', { value: '', text: '+ add effect' }));
    EFFECTS.forEach(e => {
      const opt = el('option', { value: e.id, text: e.label });
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => {
      if (!sel.value) return;
      const eff = effectById(sel.value);
      chain.push({ effect: eff.id, params: defaultParams(eff) });
      activeStep = chain.length - 1;
      sel.value = '';
      refreshAll();
    });
    addWrap.appendChild(sel);
    strip.appendChild(addWrap);
  }

  function refreshControls() {
    const controls = $('#demo-controls');
    controls.innerHTML = '';
    if (chain.length === 0) {
      controls.appendChild(el('div', { class: 'hint', text: 'Add an effect from the menu above to get started.' }));
      return;
    }
    const step = chain[activeStep];
    const eff = effectById(step.effect);
    if (!eff) return;
    const head = el('div', { class: 'controls-head' });
    head.innerHTML = `<span class="effect-id">${eff.id}</span>`;
    controls.appendChild(head);

    for (const spec of eff.params) {
      const row = el('div', { class: 'control' });
      const label = el('label', { class: 'lbl' });
      label.innerHTML = `${spec.label} <code>${spec.id}</code>`;
      row.appendChild(label);

      const valueEl = el('span', { class: 'val' });
      let input;
      const cur = step.params[spec.id] ?? spec.default;
      if (spec.kind === 'float') {
        input = el('input', {
          type: 'range', min: spec.min, max: spec.max, step: spec.step, value: String(cur),
        });
        valueEl.textContent = (+cur).toFixed(2);
        input.addEventListener('input', () => {
          step.params[spec.id] = +input.value;
          valueEl.textContent = (+input.value).toFixed(2);
          scheduleApply();
        });
      } else if (spec.kind === 'bool') {
        input = el('input', { type: 'checkbox' });
        if (cur) input.setAttribute('checked', '');
        valueEl.textContent = cur ? 'on' : 'off';
        input.addEventListener('change', () => {
          step.params[spec.id] = input.checked;
          valueEl.textContent = input.checked ? 'on' : 'off';
          scheduleApply();
        });
      } else if (spec.kind === 'choice') {
        input = el('select');
        for (const opt of spec.options) {
          const o = el('option', { value: opt, text: opt });
          if (opt === cur) o.setAttribute('selected', '');
          input.appendChild(o);
        }
        valueEl.textContent = String(cur);
        input.addEventListener('change', () => {
          step.params[spec.id] = input.value;
          valueEl.textContent = input.value;
          scheduleApply();
        });
      }
      row.appendChild(input);
      row.appendChild(valueEl);
      controls.appendChild(row);
    }
  }

  function refreshRecipe() {
    const pre = $('#demo-recipe');
    if (!pre) return;
    const recipe = {
      input:  'sample.png',
      output: 'out.png',
      chain:  chain.map(s => ({ effect: s.effect, params: s.params })),
    };
    pre.textContent = JSON.stringify(recipe, null, 2);
  }

  function refreshAll() {
    refreshChainStrip();
    refreshControls();
    applyChain();
  }

  function copyRecipe() {
    const pre = $('#demo-recipe');
    if (!pre) return;
    navigator.clipboard.writeText(pre.textContent).then(
      () => {
        const btn = $('#demo-copy');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 1300);
      },
      () => {
        // Fallback: select the text
        const range = document.createRange();
        range.selectNode(pre);
        getSelection().removeAllRanges();
        getSelection().addRange(range);
      }
    );
  }

  function reset() {
    chain = [
      { effect: 'lumen-fx-color.saturation',         params: { amount: 1.25 } },
      { effect: 'lumen-fx-sharpen.unsharp_mask',     params: { amount: 0.8, radius: 1.0, threshold: 0.0 } },
      { effect: 'lumen-fx-exposure.brightness_contrast', params: { brightness: 0.05, contrast: 1.1 } },
    ];
    activeStep = 0;
    refreshAll();
  }

  function clearChain() {
    chain = [];
    activeStep = 0;
    refreshAll();
  }

  function autoEnhance() {
    if (!baseImageData) return;
    const linBuf = imageDataToLinearF32(baseImageData);
    const t0 = performance.now();
    const stats = analyzeLinear(linBuf, baseW, baseH);
    const dt = performance.now() - t0;
    chain = buildAutoChain(stats);
    activeStep = 0;
    showStats(stats, dt);
    refreshAll();
  }

  function clarifyCctv(forcedStrength) {
    if (!baseImageData) return;
    const sel = $('#demo-clarify-strength');
    const strength = forcedStrength || (sel ? sel.value : 'standard');
    chain = buildClarifyChain(strength);
    activeStep = 0;
    const panel = $('#demo-stats');
    if (panel) {
      panel.style.display = 'flex';
      panel.innerHTML = `
        <span><b>preset</b> clarify · ${strength}</span>
        <span><b>steps</b> ${chain.length}</span>
        <span class="dim">denoise → deblock → dehaze → CLAHE → deblur → sharpen → tone</span>
      `;
    }
    refreshAll();
  }

  // Smart Auto — analyze the input, decide whether it's "degraded enough"
  // to need Clarify, or just needs general Auto-Enhance. Mirrors
  // crates/lumen-cli/src/smart.rs's pick_strategy.
  function pickSmartStrategy(stats) {
    // Degraded markers:
    //   - low edge density (likely blurry / low-detail)
    //   - low contrast (p99-p01 small)
    //   - low chroma (washed out / hazy)
    let degraded_score = 0;
    if (stats.edgeMean    < 0.06) degraded_score++;
    if (stats.edgeMean    < 0.03) degraded_score++;
    if ((stats.p99 - stats.p01) < 0.50) degraded_score++;
    if ((stats.p99 - stats.p01) < 0.30) degraded_score++;
    if (stats.chromaMean  < 0.10) degraded_score++;
    if (stats.chromaMean  < 0.05) degraded_score++;
    if (degraded_score >= 4) return { mode: 'clarify',  strength: 'aggressive' };
    if (degraded_score >= 2) return { mode: 'clarify',  strength: 'standard'   };
    return                          { mode: 'enhance',  strength: null         };
  }

  function smartAuto() {
    if (!baseImageData) return;
    const linBuf = imageDataToLinearF32(baseImageData);
    const t0 = performance.now();
    const stats = analyzeLinear(linBuf, baseW, baseH);
    const dt = performance.now() - t0;
    const decision = pickSmartStrategy(stats);

    if (decision.mode === 'clarify') {
      chain = buildClarifyChain(decision.strength);
    } else {
      chain = buildAutoChain(stats);
    }
    activeStep = 0;
    const panel = $('#demo-stats');
    if (panel) {
      const fmt = (x, d=3) => Number.isFinite(x) ? x.toFixed(d) : '—';
      panel.style.display = 'flex';
      const verdict = decision.mode === 'clarify'
        ? `degraded → clarify · ${decision.strength}`
        : 'looks fine → auto-enhance';
      panel.innerHTML = `
        <span><b>verdict</b> ${verdict}</span>
        <span><b>p01</b> ${fmt(stats.p01)}</span>
        <span><b>p99</b> ${fmt(stats.p99)}</span>
        <span><b>chroma̅</b> ${fmt(stats.chromaMean)}</span>
        <span><b>edges̅</b> ${fmt(stats.edgeMean)}</span>
        <span class="dim">decided in ${fmt(dt, 1)} ms</span>
      `;
    }
    refreshAll();
  }

  function downloadOutput() {
    if (!outputCanvas || baseW === 0) return;
    outputCanvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `lumen-output-${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1500);
    }, 'image/png');
  }

  function showStats(stats, ms) {
    const panel = $('#demo-stats');
    if (!panel) return;
    const fmt = (x, d=3) => Number.isFinite(x) ? x.toFixed(d) : '—';
    panel.style.display = 'flex';
    panel.innerHTML = `
      <span><b>p01</b> ${fmt(stats.p01)}</span>
      <span><b>p50</b> ${fmt(stats.p50)}</span>
      <span><b>p99</b> ${fmt(stats.p99)}</span>
      <span><b>chroma̅</b> ${fmt(stats.chromaMean)}</span>
      <span><b>edges̅</b> ${fmt(stats.edgeMean)}</span>
      <span><b>luma̅</b> ${fmt(stats.luminanceMean)}</span>
      <span class="dim">analyzed in ${fmt(ms, 1)} ms</span>
    `;
  }

  function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      const stamp = $('#demo-render-time');
      if (stamp) stamp.textContent = 'Please choose an image file.';
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        // Cap to a max edge for snappy real-time edits in the demo.
        const MAX = 900;
        const scale = Math.min(1, MAX / Math.max(img.naturalWidth, img.naturalHeight));
        baseW = Math.round(img.naturalWidth  * scale);
        baseH = Math.round(img.naturalHeight * scale);
        [inputCanvas, outputCanvas].forEach(c => { c.width = baseW; c.height = baseH; });
        inputCtx.drawImage(img, 0, 0, baseW, baseH);
        baseImageData = inputCtx.getImageData(0, 0, baseW, baseH);
        const stamp = $('#demo-render-time');
        if (stamp) stamp.textContent = `Loaded ${file.name} (${baseW}×${baseH})`;
        refreshAll();
      };
      img.onerror = () => {
        const stamp = $('#demo-render-time');
        if (stamp) stamp.textContent = `Couldn't decode ${file.name}.`;
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  }

  // ─── Init ───────────────────────────────────────────────────────────
  function init() {
    inputCanvas  = $('#demo-input');
    outputCanvas = $('#demo-output');
    if (!inputCanvas || !outputCanvas) return;
    inputCtx  = inputCanvas.getContext('2d');
    outputCtx = outputCanvas.getContext('2d');

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      baseW = img.naturalWidth;
      baseH = img.naturalHeight;
      [inputCanvas, outputCanvas].forEach(c => {
        c.width = baseW;
        c.height = baseH;
      });
      inputCtx.drawImage(img, 0, 0);
      baseImageData = inputCtx.getImageData(0, 0, baseW, baseH);
      refreshAll();
    };
    img.onerror = () => {
      const stamp = $('#demo-render-time');
      if (stamp) stamp.textContent = 'Failed to load sample.png';
    };
    img.src = 'sample.png';

    $('#demo-copy')  && $('#demo-copy').addEventListener('click', copyRecipe);
    $('#demo-reset') && $('#demo-reset').addEventListener('click', reset);
    $('#demo-clear') && $('#demo-clear').addEventListener('click', clearChain);
    $('#demo-auto')     && $('#demo-auto').addEventListener('click', autoEnhance);
    $('#demo-clarify')  && $('#demo-clarify').addEventListener('click', () => clarifyCctv(null));
    $('#demo-smart')    && $('#demo-smart').addEventListener('click', smartAuto);
    $('#demo-download') && $('#demo-download').addEventListener('click', downloadOutput);

    const fileInput = $('#demo-file');
    if (fileInput) {
      fileInput.addEventListener('change', () => {
        const f = fileInput.files && fileInput.files[0];
        if (f) loadFile(f);
      });
    }

    // Drag-and-drop on the input frame.
    const dropZone = inputCanvas.parentElement;
    if (dropZone) {
      ['dragenter', 'dragover'].forEach(ev => dropZone.addEventListener(ev, e => {
        e.preventDefault(); e.stopPropagation(); dropZone.classList.add('drop-hover');
      }));
      ['dragleave', 'drop'].forEach(ev => dropZone.addEventListener(ev, e => {
        e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('drop-hover');
      }));
      dropZone.addEventListener('drop', e => {
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) loadFile(f);
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
