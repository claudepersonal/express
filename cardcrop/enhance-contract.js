const fs = require('node:fs');
const { Buffer } = require('node:buffer');
const { validateQuad, validateWorkbench } = require('./workbench-contract');

function enhanceOptsFrom(q) {
  const out = {};
  for (const [key, fallback, min, max] of [
    ['scale', 2, 1, 4], ['descratch', 0, 0, 1], ['denoise', 0, 0, 1],
    ['sharpen', 0.25, 0, 1], ['contrast', 0, 0, 0.5],
  ]) {
    const raw = q[key];
    if (raw !== undefined && (typeof raw !== 'string' || !raw.trim())) throw new Error(`Invalid ${key}`);
    const value = raw === undefined ? fallback : Number(raw);
    if (!Number.isFinite(value) || value < min || value > max) throw new Error(`Invalid ${key}: expected ${min}–${max}`);
    out[key] = value;
  }
  for (const key of ['conservative', 'autoRotate', 'autoCrop']) {
    if (q[key] !== undefined && q[key] !== 'true' && q[key] !== 'false') throw new Error(`Invalid ${key}`);
    out[key] = q[key] !== 'false';
  }
  if (q.rotation !== undefined && !['0','90','180','270'].includes(q.rotation)) throw new Error('Invalid rotation');
  out.rotation = Number(q.rotation ?? 0);
  try { out.quad = validateQuad(q.quad === undefined ? null : JSON.parse(q.quad)); }
  catch (e) { throw new Error(`Invalid quad: ${e.message}`); }
  return out;
}

function optionsForCard(base, card) {
  if (!card.workbench) return base;
  const workbench=validateWorkbench(card.workbench);
  const custom=workbench.customSettings;
  const result={...base,rotation:workbench.rotation,quad:workbench.quad};
  for(const key of ['scale','descratch','denoise','sharpen','contrast','autoCrop','autoRotate']) {
    if(custom[key]!==undefined)result[key]=custom[key];
  }
  if(custom.chromeOnly) {
    const m=card.metadata||{};
    const foil=/(chrome|foil|refractor|prizm|holo|metallic|dazzler|parallel)/i.test([m.parallel,m.card_type,m.set_name,m.rarity,m.finish,m.foil,m.all_text,m.bio_text].filter(Boolean).join(' '));
    if(!foil)result.descratch=0;
  }
  return result;
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
module.exports = { enhanceOptsFrom, readEnhancement, optionsForCard };
