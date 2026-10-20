const express = require('express');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const vlm = require('./vlm_ocr');
const { enhanceOptsFrom, readEnhancement, optionsForCard } = require('./enhance-contract');
const { validateWorkbench } = require('./workbench-contract');
let workerReady = false;
let enhancementQueue = Promise.resolve();
const ocrQueues = [Promise.resolve(), Promise.resolve()];
let nextOCR = 0;
let detector = { available: false, reason: 'checking' };

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const STORAGE = process.env.STORAGE_DIR || path.join(__dirname, 'data');
const UPLOAD_DIR = path.join(STORAGE, 'uploads');
const ENHANCED_DIR = path.join(STORAGE, 'enhanced');
const REGISTRY = path.join(STORAGE, 'jobs.json');
const COMFYUI_URL = process.env.COMFYUI_URL || '';
const YOLO_MODEL = process.env.YOLO_MODEL || path.join(__dirname, 'models', 'card_detector.pt');

[STORAGE, UPLOAD_DIR, ENHANCED_DIR].forEach(d => {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
});

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(UPLOAD_DIR));
app.use('/enhanced', express.static(ENHANCED_DIR));

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname) || '.jpg';
    cb(null, uuidv4() + ext);
  },
});
const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype.startsWith('image/')) cb(null, true);
    else cb(new Error('Only image files accepted'));
  },
});

// ══ persistent job registry ══
// Was in-memory only, so every Railway deploy replaced the container and
// orphaned the uploaded files on /data with no index pointing at them.
const jobs = new Map();
let saveTimer = null;

function snapshot() {
  return [...jobs.values()].map(c => ({
    id: c.id, fileName: c.fileName, filePath: c.filePath, url: c.url,
    enhancedUrl: c.enhancedUrl, metadata: c.metadata,
    status: (c.status === 'processing' || c.status === 'enhancing') ? 'queued' : c.status,
    createdAt: c.createdAt, workbench: c.workbench, workbenchRevision: c.workbenchRevision || 0,
  }));
}

function persist() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { fs.writeFileSync(REGISTRY + '.tmp', JSON.stringify(snapshot())); fs.renameSync(REGISTRY + '.tmp', REGISTRY); }
    catch (e) { console.warn('registry write failed:', e.message); }
  }, 400);
}

function restore() {
  let restored = 0;
  if (fs.existsSync(REGISTRY)) {
    try {
      for (const c of JSON.parse(fs.readFileSync(REGISTRY, 'utf8'))) {
        if (c.filePath && fs.existsSync(c.filePath)) { jobs.set(c.id, c); restored++; }
      }
    } catch (e) { console.warn('registry read failed:', e.message); }
  }
  let adopted = 0;
  try {
    for (const f of fs.readdirSync(UPLOAD_DIR)) {
      const id = path.basename(f, path.extname(f));
      if (jobs.has(id)) continue;
      const fp = path.join(UPLOAD_DIR, f);
      const enhName = id + '_enhanced.png';
      jobs.set(id, {
        id, fileName: f, filePath: fp, url: `/uploads/${f}`,
        enhancedUrl: fs.existsSync(path.join(ENHANCED_DIR, enhName)) ? `/enhanced/${enhName}` : null,
        metadata: {}, status: 'queued', createdAt: fs.statSync(fp).mtimeMs,
      });
      adopted++;
    }
  } catch (e) { console.warn('upload scan failed:', e.message); }
  if (restored || adopted) persist();
  return { restored, adopted };
}

// ══ health ══ Railway probes /health; the UI polls /api/health.
app.get(['/health', '/api/health'], (_req, res) => {
  res.json({
    status: 'ok',
    category: 'wrestling',
    revision: process.env.RAILWAY_GIT_COMMIT_SHA || 'local',
    enhancement: { available: workerReady, engine: 'opencv', maxLongEdge: 2200 },
    uptime: process.uptime(),
    jobs: jobs.size,
    comfyui: COMFYUI_URL ? 'configured' : 'not configured',
    vlm: vlm.status(),
    yolo: { ...detector, path: YOLO_MODEL },
    review_gate: vlm.REVIEW_GATE,
    storage: STORAGE,
  });
});

// ══ upload ══
app.post('/api/cards/upload', upload.array('files', 200), (req, res) => {
  if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No files uploaded' });
  const results = [];
  for (const file of req.files) {
    const cardId = path.basename(file.filename, path.extname(file.filename));
    const card = {
      id: cardId, fileName: file.originalname, filePath: file.path,
      url: `/uploads/${file.filename}`, enhancedUrl: null,
      metadata: {}, status: 'queued', createdAt: Date.now(),
    };
    jobs.set(cardId, card);
    results.push(card);
  }
  persist();
  res.json({ cards: results.map(sanitize) });

  for (const card of results) {
    card.status = 'processing';
    runOCR(card.filePath)
      .then(meta => { card.metadata = meta; card.status = 'ocr_complete'; })
      .catch(e => { card.metadata = { error: e.message }; card.status = 'failed'; })
      .finally(persist);
  }
});

const optsFrom = q => ({
  provider: q.provider, model: q.model, escalate: q.escalate !== 'false',
});
// ══ VLM: one card ══
app.post('/api/cards/:id/vlm', async (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  if (['processing', 'enhancing'].includes(card.status)) return res.status(409).json({ error: 'Already processing' });
  card.status = 'processing';
  try {
    const meta = await vlm.vlmReadGated(card.filePath, optsFrom(req.query));
    card.metadata = { ...card.metadata, ...meta };
    if (!meta.error) delete card.metadata.error;
    card.status = meta.error ? 'failed' : 'vlm_complete';
    persist();
    res.json({ card: sanitize(card) });
  } catch (e) {
    card.status = 'failed';
    card.metadata = { ...card.metadata, error: e.message };
    persist();
    res.status(500).json({ error: e.message });
  }
});

// ══ VLM: batch, returns immediately and the UI polls ══
app.post('/api/cards/vlm-all', (req, res) => {
  let all = [...jobs.values()].filter(c => !['enhancing', 'processing'].includes(c.status));
  if (req.query.only === 'pending') {
    all = all.filter(c => c.status !== 'vlm_complete' || c.metadata?.review_needed);
  } else if (req.query.only === 'failed') {
    all = all.filter(c => c.status === 'failed');
  }
  all.forEach(c => { c.status = 'processing'; });
  persist();
  res.json({ started: all.length });

  vlm.vlmReadMany(all, optsFrom(req.query), (card, meta) => {
    card.metadata = { ...card.metadata, ...meta };
    if (!meta.error) delete card.metadata.error;
    card.status = meta.error ? 'failed' : 'vlm_complete';
    persist();
  }).catch(() => {});
});

// ══ enhance ══
app.patch('/api/cards/:id/workbench', (req, res) => {
  const card=jobs.get(req.params.id);
  if (!card) return res.status(404).json({error:'Card not found'});
  if (['enhancing','processing'].includes(card.status)) return res.status(409).json({error:'Card is processing; wait before editing'});
  if (!req.body || !Number.isInteger(req.body.expectedRevision) || req.body.expectedRevision !== (card.workbenchRevision||0)) return res.status(409).json({error:'Card changed; reopen the editor before saving'});
  let next;
  try { next=validateWorkbench(req.body.workbench); } catch(e) {return res.status(400).json({error:e.message});}
  card.workbench=next;
  card.workbenchRevision=(card.workbenchRevision||0)+1;
  card.enhancedUrl=null;
  card.status='queued';
  for (const key of ['enhancement','enhanceError','output_size','orientation']) delete card.metadata[key];
  persist();
  res.json({card:sanitize(card)});
});

app.post('/api/cards/:id/enhance', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  if (['enhancing', 'processing'].includes(card.status)) return res.status(409).json({ error: 'Already processing' });
  let opts;
  try { opts = optionsForCard(enhanceOptsFrom(req.query), card); } catch (e) { return res.status(400).json({ error: e.message }); }
  card.status = 'enhancing';
  card.enhancedUrl = null;
  for (const key of ['enhancement','enhanceError','output_size','orientation']) delete card.metadata[key];
  persist();
  runEnhance(card, opts)
    .then(() => { card.status = 'enhanced'; })
    .catch(e => { card.status = 'failed'; card.metadata.enhanceError = e.message; })
    .finally(persist);
  res.json({ card: sanitize(card) });
});

app.post('/api/cards/enhance-all', (req, res) => {
  let opts;
  try { opts = enhanceOptsFrom(req.query); } catch (e) { return res.status(400).json({ error: e.message }); }
  const all = [...jobs.values()].filter(c => !['enhancing', 'processing'].includes(c.status));
  all.forEach(c => {
    c.status = 'enhancing'; c.enhancedUrl = null;
    for (const key of ['enhancement','enhanceError','output_size','orientation']) delete c.metadata[key];
  });
  persist();
  res.json({ started: all.length });

  (async () => {
    for (const card of all) {
      try { await runEnhance(card, optionsForCard(opts, card)); card.status = 'enhanced'; }
      catch (e) { card.status = 'failed'; card.metadata.enhanceError = e.message; }
      persist();
    }
  })();
});

// ══ read ══
app.get('/api/cards/:id', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  res.json({ card: sanitize(card) });
});

app.get('/api/cards', (_req, res) => {
  const cards = [...jobs.values()].sort((a, b) => b.createdAt - a.createdAt).map(sanitize);
  res.json({ cards });
});

// ══ delete ══
app.delete('/api/cards/:id', (req, res) => {
  const card = jobs.get(req.params.id);
  if (!card) return res.status(404).json({ error: 'Card not found' });
  if (['processing', 'enhancing'].includes(card.status)) return res.status(409).json({ error: 'Wait for processing to finish before deleting' });
  try { fs.unlinkSync(card.filePath); } catch {}
  if (card.enhancedUrl) {
    try { fs.unlinkSync(path.join(ENHANCED_DIR, path.basename(card.enhancedUrl.split('?')[0]))); } catch {}
  }
  jobs.delete(req.params.id);
  persist();
  res.json({ ok: true });
});

// ══ export ══
app.get('/api/cards/export/json', (_req, res) => {
  const data = [...jobs.values()].map(c => ({
    fileName: c.fileName, status: c.status, ...c.metadata, enhancedUrl: c.enhancedUrl,
  }));
  res.setHeader('Content-Disposition', 'attachment; filename=cardcrop_export.json');
  res.json(data);
});

// ══ python subprocesses ══
function runOCR(imagePath) {
  const lane = nextOCR++ % ocrQueues.length;
  const pending = ocrQueues[lane].then(() => executeOCR(imagePath));
  ocrQueues[lane] = pending.catch(() => {});
  return pending;
}
function executeOCR(imagePath) {
  return new Promise((resolve, reject) => {
    execFile('python3', [path.join(__dirname, 'ocr_worker.py'), imagePath],
      { timeout: 30000 }, (err, stdout, stderr) => {
        if (err) return reject(new Error(stderr || err.message));
        try { const result = JSON.parse(stdout); if (result.error) return reject(new Error(result.error)); resolve(result); }
        catch { reject(new Error('OCR output parse error')); }
      });
  });
}

function runEnhance(card, opts = {}) {
  const pending = enhancementQueue.then(() => executeEnhance(card, opts));
  enhancementQueue = pending.catch(() => {});
  return pending;
}
function executeEnhance(card, opts = {}) {
  return new Promise((resolve, reject) => {
    const outName = card.id + '_enhanced.png';
    const outPath = path.join(ENHANCED_DIR, outName);
    execFile('python3',
      [path.join(__dirname, 'enhance_worker.py'), card.filePath, outPath, JSON.stringify(opts)],
      { timeout: 180000 }, (err, stdout, stderr) => {
        if (err) return reject(new Error(stderr || err.message));
        try {
          const result = readEnhancement(stdout, outPath);
          card.enhancedUrl = `/enhanced/${outName}?v=${Date.now()}`;
          delete card.metadata.enhanceError;
          card.metadata.enhancement = result;
          Object.assign(card.metadata, {
            rotation: result.rotation, output_size: result.output_size,
            orientation: result.orientation, yolo_cropped: result.yolo_cropped,
            yolo_available: result.yolo_available, yolo_conf: result.yolo_conf,
            detection_warning: result.detection_warning,
          });
        } catch (e) { return reject(e); }
        resolve();
      });
  });
}

function sanitize(card) {
  return {
    id: card.id, fileName: card.fileName, url: card.url,
    enhancedUrl: card.enhancedUrl, metadata: card.metadata,
    status: card.status, createdAt: card.createdAt,
    workbench: card.workbench || null, workbenchRevision: card.workbenchRevision || 0,
  };
}

// ══ SPA fallback (Express 5 wildcard syntax) ══
app.get('/{*path}', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const boot = restore();
execFile('python3', [path.join(__dirname, 'enhance_worker.py'), '--capabilities'],
  { timeout: 60000 }, (err, stdout) => {
    if (err) { detector = { available: false, reason: 'worker_probe_failed' }; return; }
    try { detector = JSON.parse(stdout); workerReady = !detector.error; }
    catch { detector = { available: false, reason: 'invalid_worker_probe' }; }
  });

const listener = app.listen(PORT, '0.0.0.0', () => {
  const s = vlm.status();
  console.log(`CardCrop AI on http://0.0.0.0:${listener.address().port}`);
  console.log(`  storage: ${STORAGE}`);
  console.log(`  jobs:    ${jobs.size} (${boot.restored} from registry, ${boot.adopted} adopted from disk)`);
  console.log(`  yolo:    ${fs.existsSync(YOLO_MODEL) ? YOLO_MODEL : 'no weights — whole-frame fallback'}`);
  console.log(`  vlm:     ${s.provider} (available: ${s.available.join(', ') || 'none'})`);
  console.log(`  comfyui: ${COMFYUI_URL || 'not configured'}`);
});

for (const sig of ['SIGTERM', 'SIGINT']) {
  process.on(sig, () => {
    clearTimeout(saveTimer);
    try { fs.writeFileSync(REGISTRY + '.tmp', JSON.stringify(snapshot())); fs.renameSync(REGISTRY + '.tmp', REGISTRY); } catch {}
    process.exit(0);
  });
}
