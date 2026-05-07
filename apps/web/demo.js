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
  ];

  function defaultParams(eff) {
    const p = {};
    for (const spec of eff.params) p[spec.id] = spec.default;
    return p;
  }
  function effectById(id) { return EFFECTS.find(e => e.id === id); }

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
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
