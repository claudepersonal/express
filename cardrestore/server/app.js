'use strict';
// Card Restoration Lab API (Express 5). Serves the Vite build from ../dist.
// One Python worker runs at a time (restoration is CPU/GPU bound); jobs queue
// behind it and the client polls GET /api/jobs/:id for progress.
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const readline = require('node:readline');
const express = require('express');
const multer = require('multer');

const MAX_FILES = Number(process.env.MAX_FILES || 12);
const MAX_FILE_MB = Number(process.env.MAX_FILE_MB || 15);
const MAX_QUEUED_JOBS = Number(process.env.MAX_QUEUED_JOBS || 10);
const JOB_TTL_MS = Number(process.env.JOB_TTL_HOURS || 24) * 3600 * 1000;
const STALL_MS = Number(process.env.STALL_MINUTES || 10) * 60 * 1000;
const IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff']);
const ID_RE = /^[0-9a-f-]{36}$/;

function createApp(opts = {}) {
  const storage = opts.storageDir || process.env.STORAGE_DIR || path.join(__dirname, '..', 'data');
  const jobsDir = path.join(storage, 'jobs');
  fs.mkdirSync(jobsDir, { recursive: true });
  const python = opts.python || process.env.PYTHON || 'python3';
  const workerScript = opts.workerScript || process.env.RESTORE_WORKER || path.join(__dirname, '..', 'worker', 'restore_worker.py');
  const distDir = opts.distDir || path.join(__dirname, '..', 'dist');

  const jobs = new Map();
  const queue = [];
  let running = null;
  let model = { status: 'unchecked' };

  const app = express();
  app.disable('x-powered-by');
  app.use((_req, res, next) => {
    res.set({
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'Referrer-Policy': 'no-referrer',
      'Content-Security-Policy': "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    });
    next();
  });

  const upload = multer({
    storage: multer.diskStorage({
      destination: (req, _file, cb) => cb(null, req.jobInDir),
      filename: (req, file, cb) => {
        req.fileSeq = (req.fileSeq || 0) + 1;
        cb(null, `${String(req.fileSeq).padStart(3, '0')}${path.extname(file.originalname).toLowerCase().replace(/[^.a-z0-9]/g, '').slice(0, 6)}`);
      },
    }),
    limits: { files: MAX_FILES, fileSize: MAX_FILE_MB * 1024 * 1024, fields: 10 },
    fileFilter: (_req, file, cb) => {
      if (IMAGE_TYPES.has(file.mimetype)) cb(null, true);
      else cb(Object.assign(new Error(`${file.originalname}: only JPEG, PNG, WebP, BMP or TIFF images are accepted`), { status: 415 }));
    },
  });

  const publicJob = job => ({
    id: job.id,
    status: job.status,
    position: job.status === 'queued' ? queue.indexOf(job) + 1 : 0,
    settings: job.settings,
    total: job.inputs.length,
    files: job.inputs.map(i => i.name),
    rows: job.rows,
    errors: job.errors,
    device: job.device,
    progress: job.progress,
    message: job.message,
    createdAt: job.createdAt,
    finishedAt: job.finishedAt,
  });

  function pump() {
    if (running || queue.length === 0) return;
    const job = queue.shift();
    running = job;
    job.status = 'running';
    const child = spawn(python, [workerScript], { stdio: ['pipe', 'pipe', 'pipe'] });
    job.child = child;
    let stderr = '';
    child.stderr.on('data', d => { stderr = (stderr + d).slice(-4000); });
    readline.createInterface({ input: child.stdout }).on('line', line => {
      let ev;
      try { ev = JSON.parse(line); } catch { return; }
      if (ev.event === 'ready') {
        job.device = ev.device;
        console.log(`job ${job.id}: worker ready on ${ev.device} (${ev.threads} threads), ${job.inputs.length} file(s)`);
      } else if (ev.event === 'progress') {
        job.progress = { index: ev.index, tile: ev.tile, tiles: ev.tiles };
        job.lastProgress = Date.now();
      }
      else if (ev.event === 'file') {
        job.rows.push({ index: ev.index, ...ev.row });
        job.lastProgress = Date.now();
        console.log(`job ${job.id}: ${ev.row.output} in ${ev.row.seconds}s`);
      }
      else if (ev.event === 'error') job.errors.push({ index: ev.index, file: ev.file, message: ev.message });
      else if (ev.event === 'done') job.zip = ev.zip;
      else if (ev.event === 'fatal') job.message = ev.message;
    });
    // Kill a worker that stops reporting progress so one bad job cannot block the queue.
    job.lastProgress = Date.now();
    const watchdog = setInterval(() => {
      if (Date.now() - job.lastProgress > STALL_MS) {
        job.message = `worker stalled (no progress for ${Math.round(STALL_MS / 60000)} min) and was stopped`;
        child.kill('SIGKILL');
      }
    }, 15000);
    child.on('error', err => { job.message = `worker failed to start: ${err.message}`; });
    child.on('close', code => {
      clearInterval(watchdog);
      job.child = null;
      if (code === 0 && job.zip && job.rows.length > 0) job.status = 'done';
      else {
        job.status = 'failed';
        if (!job.message) job.message = job.rows.length === 0 && job.errors.length ? 'No file could be restored.' : `worker exited with code ${code}`;
        if (code !== 0) console.error(`job ${job.id} worker exit ${code}: ${stderr}`);
      }
      job.finishedAt = new Date().toISOString();
      running = null;
      pump();
    });
    child.stdin.end(JSON.stringify({
      outdir: job.outDir,
      ...job.settings,
      inputs: job.inputs.map(i => ({ path: i.path, name: i.name })),
    }));
  }

  function sweep() {
    const cutoff = Date.now() - JOB_TTL_MS;
    for (const job of jobs.values()) {
      if (job.status !== 'queued' && job.status !== 'running' && Date.parse(job.createdAt) < cutoff) {
        fs.rmSync(job.dir, { recursive: true, force: true });
        jobs.delete(job.id);
      }
    }
  }
  const sweeper = setInterval(sweep, 15 * 60 * 1000);
  sweeper.unref();

  app.get('/health', (_req, res) => res.json({ ok: true }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, model, queue: queue.length, running: Boolean(running), limits: { maxFiles: MAX_FILES, maxFileMB: MAX_FILE_MB } });
  });

  app.post('/api/jobs',
    (req, res, next) => {
      if (queue.length >= MAX_QUEUED_JOBS) return res.status(503).json({ error: 'The restore queue is full. Try again in a few minutes.' });
      req.jobId = crypto.randomUUID();
      req.jobDir = path.join(jobsDir, req.jobId);
      req.jobInDir = path.join(req.jobDir, 'in');
      fs.mkdirSync(req.jobInDir, { recursive: true });
      next();
    },
    (req, res, next) => upload.array('files', MAX_FILES)(req, res, err => {
      if (!err) return next();
      fs.rmSync(req.jobDir, { recursive: true, force: true });
      const status = err.status || (err instanceof multer.MulterError ? 413 : 400);
      const message = err.code === 'LIMIT_FILE_SIZE' ? `Each file must be ${MAX_FILE_MB} MB or smaller.`
        : err.code === 'LIMIT_FILE_COUNT' || err.code === 'LIMIT_UNEXPECTED_FILE' ? `Upload at most ${MAX_FILES} images per batch.`
          : err.message;
      res.status(status).json({ error: message });
    }),
    (req, res) => {
      const files = req.files || [];
      if (files.length === 0) {
        fs.rmSync(req.jobDir, { recursive: true, force: true });
        return res.status(400).json({ error: 'Add one or more card scans first.' });
      }
      const body = req.body || {};
      const settings = {
        profile: body.profile === 'standard' ? 'standard' : 'conservative',
        scale: String(body.scale) === '4' ? 4 : 2,
        tile: Math.min(512, Math.max(64, Math.round(Number(body.tile) / 64) * 64 || 256)),
        ocr: body.ocr !== 'false',
      };
      const job = {
        id: req.jobId,
        dir: req.jobDir,
        outDir: path.join(req.jobDir, 'out'),
        settings,
        inputs: files.map(f => ({ path: f.path, name: path.basename(f.originalname).slice(0, 120) || 'card', mimetype: f.mimetype })),
        rows: [],
        errors: [],
        status: 'queued',
        createdAt: new Date().toISOString(),
      };
      jobs.set(job.id, job);
      queue.push(job);
      pump();
      res.status(202).json(publicJob(job));
    });

  function findJob(req, res) {
    const job = ID_RE.test(req.params.id) && jobs.get(req.params.id);
    if (!job) res.status(404).json({ error: 'Job not found (results are kept for 24 hours).' });
    return job;
  }

  app.get('/api/jobs/:id', (req, res) => {
    const job = findJob(req, res);
    if (job) res.set('Cache-Control', 'no-store').json(publicJob(job));
  });

  app.get('/api/jobs/:id/original/:index', (req, res) => {
    const job = findJob(req, res);
    if (!job) return;
    const input = job.inputs[Number(req.params.index)];
    if (!input || !/^\d+$/.test(req.params.index)) return res.status(404).json({ error: 'No such file.' });
    res.type(input.mimetype).sendFile(input.path);
  });

  app.get('/api/jobs/:id/output/:index', (req, res) => {
    const job = findJob(req, res);
    if (!job) return;
    const row = job.rows.find(r => String(r.index) === req.params.index);
    if (!row) return res.status(404).json({ error: 'Not restored (yet).' });
    res.type('png').sendFile(path.join(job.outDir, path.basename(row.output)));
  });

  app.get('/api/jobs/:id/zip', (req, res) => {
    const job = findJob(req, res);
    if (!job) return;
    if (job.status !== 'done') return res.status(409).json({ error: 'The batch has not finished yet.' });
    res.download(path.join(job.outDir, 'restored_cards.zip'), 'restored_cards.zip');
  });

  app.use('/api', (_req, res) => res.status(404).json({ error: 'Not found' }));

  if (fs.existsSync(path.join(distDir, 'index.html'))) {
    app.use(express.static(distDir, { index: 'index.html', maxAge: '1h' }));
  }

  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    console.error(err);
    res.status(err.status || 500).json({ error: err.expose ? err.message : 'Internal error' });
  });

  app.checkModel = () => new Promise(resolve => {
    model = { status: 'checking' };
    const child = spawn(python, [workerScript, '--check'], { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    child.stdout.on('data', d => { out += d; });
    child.stderr.on('data', d => { err = (err + d).slice(-2000); });
    child.on('error', e => { model = { status: 'unavailable', reason: e.message }; resolve(model); });
    child.on('close', code => {
      const ev = out.split('\n').filter(Boolean).map(l => { try { return JSON.parse(l); } catch { return {}; } }).pop() || {};
      model = code === 0 && ev.event === 'ready' ? { status: 'ready', device: ev.device, threads: ev.threads } : { status: 'unavailable', reason: ev.message || err.trim().split('\n').pop() || `exit ${code}` };
      resolve(model);
    });
  });
  app.jobs = jobs;
  app.close = () => {
    clearInterval(sweeper);
    for (const job of jobs.values()) if (job.child) job.child.kill();
  };
  return app;
}

module.exports = { createApp };
