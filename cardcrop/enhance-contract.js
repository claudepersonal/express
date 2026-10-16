const fs = require('node:fs');
const { Buffer } = require('node:buffer');

function enhanceOptsFrom(q) {
  const out = {};
  for (const [key, fallback, min, max] of [
    ['scale', 2, 1, 4], ['descratch', 0, 0, 1], ['denoise', 0, 0, 1],
    ['sharpen', 0.25, 0, 1], ['contrast', 0, 0, 0.5], ['cleanupStrength', 0.45, 0, 1],
  ]) {
    const raw = q[key];
    if (raw !== undefined && (typeof raw !== 'string' || !raw.trim())) throw new Error(`Invalid ${key}`);
    const value = raw === undefined ? fallback : Number(raw);
    if (!Number.isFinite(value) || value < min || value > max) throw new Error(`Invalid ${key}: expected ${min}–${max}`);
    out[key] = value;
  }
  for (const key of ['conservative', 'autoRotate']) {
    if (q[key] !== undefined && q[key] !== 'true' && q[key] !== 'false') throw new Error(`Invalid ${key}`);
    out[key] = q[key] !== 'false';
  }
  for (const [key, fallback, choices] of [
    ['cleanup', 'manual', ['manual', 'classical', 'cnn']],
    ['detection', 'yolo', ['yolo', 'contour', 'none']],
  ]) {
    const value = q[key] === undefined ? fallback : q[key];
    if (typeof value !== 'string' || !choices.includes(value)) throw new Error(`Invalid ${key}`);
    out[key] = value;
  }
  if (out.conservative && out.cleanup !== 'manual') {
    throw new Error('Disable conservative mode to use a cleanup profile');
  }
  return out;
}

function readEnhancement(stdout, outPath) {
  let result;
  try { result = JSON.parse(stdout); } catch { throw new Error('Enhancement worker returned invalid JSON'); }
  if (!result || typeof result !== 'object' || result.error) throw new Error(result?.error || 'Empty enhancement result');
  if (!/^\d+x\d+$/.test(result.output_size || '') || typeof result.yolo_cropped !== 'boolean') {
    throw new Error('Incomplete enhancement result');
  }
  const fd = fs.openSync(outPath, 'r');
  try {
    const signature = Buffer.alloc(8);
    if (fs.readSync(fd, signature, 0, 8, 0) !== 8 || signature.toString('hex') !== '89504e470d0a1a0a') {
      throw new Error('Enhancement output is not a PNG');
    }
  } finally { fs.closeSync(fd); }
  return result;
}
module.exports = { enhanceOptsFrom, readEnhancement };
