// Card Restoration Lab client.
//
// Flow: pick files -> each is probed in the browser (type, bytes, pixel size)
// -> eligible files are uploaded as one job (POST /api/jobs) -> the job is
// polled (GET /api/jobs/:id) until done/failed -> results render as
// before/after comparisons with per-card downloads and a ZIP of the batch.
// The job id lives in the URL hash, so a reload resumes the same batch.

const $ = (sel) => document.querySelector(sel);
const ui = {
  engine: $('#engine'), drop: $('#drop'), input: $('#files'), tray: $('#tray'), traySummary: $('#tray-summary'),
  clear: $('#clear'), form: $('#settings'), tile: $('#tile'), tileOut: $('#tile-out'), estimate: $('#estimate'),
  run: $('#run'), formStatus: $('#form-status'), batch: $('#batch'), batchHeading: $('#batch-heading'),
  overall: $('#overall'), batchStatus: $('#batch-status'), queue: $('#queue'), gallery: $('#gallery'),
  manifest: $('#manifest'), rows: $('#rows'), zip: $('#zip'), newBatch: $('#new-batch'),
};

const ACCEPTED = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff']);
const POLL_MS = 2500;
const MAX_POLL_FAILURES = 6;

const state = {
  limits: { maxFiles: 12, maxFileMB: 15, maxInputPixels: 1600 * 1600 },
  items: [],          // { id, file, url, width, height, problem }
  job: null,
  pollTimer: 0,
  pollFailures: 0,
  shown: new Set(),   // row indexes already rendered for the current job
  startedAt: 0,
  lastStatus: '',     // previous job status, to spot the moment a batch ends
};

// ------------------------------------------------------------------ helpers
function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  for (const c of children) if (c !== null && c !== undefined && c !== false) node.append(c);
  return node;
}
const fmtBytes = (n) => (n < 1048576 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1048576).toFixed(1)} MB`);
const fmtDuration = (s) => (s < 90 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
const side = () => Math.round(Math.sqrt(state.limits.maxInputPixels));
const fitOn = () => ui.form.fit.checked;

async function api(path, options) {
  const res = await fetch(path, options);
  let body = {};
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw Object.assign(new Error(body.error || `Request failed (${res.status})`), { status: res.status });
  return body;
}

// ------------------------------------------------------------------ engine status
async function refreshEngine() {
  try {
    const h = await api('/api/health');
    state.limits = { ...state.limits, ...h.limits };
    $('#lim-files').textContent = state.limits.maxFiles;
    $('#lim-mb').textContent = state.limits.maxFileMB;
    $('#lim-px').textContent = `${side()}×${side()}`;
    const m = h.model || {};
    let text; let st;
    if (m.status === 'ready') {
      st = h.running || h.queue ? 'busy' : 'ready';
      text = `Engine ready · ${m.device === 'cuda' ? 'GPU' : `CPU, ${m.threads} threads`}${h.queue ? ` · ${h.queue} batch${h.queue > 1 ? 'es' : ''} waiting` : h.running ? ' · busy' : ''}`;
    } else if (m.status === 'unavailable') {
      st = 'unavailable'; text = `Engine unavailable: ${m.reason || 'unknown reason'}`;
    } else {
      st = 'checking'; text = 'Engine starting…';
    }
    ui.engine.dataset.state = st;
    ui.engine.textContent = text;
    reclassify();
  } catch {
    ui.engine.dataset.state = 'unavailable';
    ui.engine.textContent = 'Cannot reach the server';
  }
}

// ------------------------------------------------------------------ file intake
/** Decide whether a picked file can be sent, and why not. */
function problemFor(item) {
  const { file } = item;
  if (!ACCEPTED.has(file.type)) return { level: 'err', text: 'Not a supported image type' };
  if (file.size > state.limits.maxFileMB * 1048576) return { level: 'err', text: `Over ${state.limits.maxFileMB} MB` };
  if (item.width && item.width * item.height > state.limits.maxInputPixels) {
    return fitOn() ? { level: 'warn', text: `Will shrink to fit ${side()}×${side()}` } : { level: 'err', text: 'Too large; turn on “Shrink large photos”' };
  }
  return null;
}

function reclassify() {
  state.items.forEach((it, i) => {
    it.problem = problemFor(it);
    if (!it.problem || it.problem.level !== 'err') {
      if (i >= state.limits.maxFiles) it.problem = { level: 'err', text: `Batch limit is ${state.limits.maxFiles}` };
    }
  });
  renderTray();
}

async function addFiles(fileList) {
  const known = new Set(state.items.map((it) => `${it.file.name}|${it.file.size}|${it.file.lastModified}`));
  for (const file of fileList) {
    const key = `${file.name}|${file.size}|${file.lastModified}`;
    if (known.has(key)) continue;
    known.add(key);
    const item = { id: crypto.randomUUID(), file, url: '', width: 0, height: 0, problem: null };
    // Browsers cannot decode TIFF; those upload fine but show no preview or size.
    if (ACCEPTED.has(file.type) && file.type !== 'image/tiff') {
      try {
        const bmp = await createImageBitmap(file);
        item.width = bmp.width; item.height = bmp.height;
        bmp.close();
        item.url = URL.createObjectURL(file);
      } catch { item.problem = { level: 'err', text: 'Could not read this image' }; }
    }
    state.items.push(item);
  }
  reclassify();
}

function removeItem(id) {
  const i = state.items.findIndex((it) => it.id === id);
  if (i < 0) return;
  if (state.items[i].url) URL.revokeObjectURL(state.items[i].url);
  state.items.splice(i, 1);
  reclassify();
}

function clearItems() {
  for (const it of state.items) if (it.url) URL.revokeObjectURL(it.url);
  state.items = [];
  reclassify();
}

const eligible = () => state.items.filter((it) => !it.problem || it.problem.level !== 'err');

function renderTray() {
  ui.tray.replaceChildren(...state.items.map((it) => {
    const thumb = el('div', { className: 'thumb' },
      it.url ? el('img', { src: it.url, alt: '', decoding: 'async' }) : el('span', { textContent: it.file.type === 'image/tiff' ? 'TIFF (no preview)' : 'No preview' }));
    const chip = it.problem
      ? el('span', { className: `chip ${it.problem.level}`, textContent: it.problem.text })
      : el('span', { className: 'chip ok', textContent: 'Ready' });
    const remove = el('button', { type: 'button', className: 'icon-btn', textContent: 'Remove', ariaLabel: `Remove ${it.file.name}` });
    remove.addEventListener('click', () => removeItem(it.id));
    const li = el('li', {}, thumb, el('div', { className: 'meta' },
      el('span', { className: 'name', textContent: it.file.name, title: it.file.name }),
      el('span', { className: 'dims', textContent: `${it.width ? `${it.width}×${it.height} · ` : ''}${fmtBytes(it.file.size)}` }),
      el('div', { className: 'row' }, chip, remove)));
    li.dataset.state = it.problem && it.problem.level === 'err' ? 'blocked' : 'ok';
    return li;
  }));

  const n = eligible().length;
  const blocked = state.items.length - n;
  ui.traySummary.textContent = state.items.length
    ? `${n} of ${state.items.length} scan${state.items.length > 1 ? 's' : ''} will be restored${blocked ? `; ${blocked} can't be sent (see below)` : ''}.`
    : '';
  ui.clear.hidden = state.items.length === 0;
  const busy = state.job && (state.job.status === 'queued' || state.job.status === 'running');
  ui.run.disabled = n === 0 || busy;
  ui.run.textContent = busy ? 'Restoring…' : n ? `Restore ${n} card${n > 1 ? 's' : ''}` : 'Restore';
  ui.estimate.textContent = n ? estimateText(eligible()) : '';
}

// Real-ESRGAN cost scales with input pixels (2× and 4× cost the same model time).
// Measured on the production 8-vCPU container: ~225 s per megapixel.
const SECONDS_PER_MEGAPIXEL = 225;
function estimateText(items) {
  const megapixels = items.reduce((sum, it) => {
    // Unknown size (TIFF has no preview): assume the worst case, the input limit.
    const px = it.width ? it.width * it.height : state.limits.maxInputPixels;
    return sum + Math.min(px, state.limits.maxInputPixels) / 1e6;
  }, 0);
  const s = megapixels * SECONDS_PER_MEGAPIXEL;
  const lo = Math.max(1, Math.round((s * 0.8) / 60));
  const hi = Math.max(lo + 1, Math.round((s * 1.25) / 60));
  return `About ${lo}–${hi} minutes on the server's CPU. You can leave and come back: the link keeps this batch for 24 hours.`;
}

// Copy the FileList first: it is live, and clearing the input (so the same file
// can be picked again) would empty it while addFiles is still reading.
ui.input.addEventListener('change', () => { const files = [...ui.input.files]; ui.input.value = ''; addFiles(files); });
['dragenter', 'dragover'].forEach((t) => ui.drop.addEventListener(t, (e) => { e.preventDefault(); ui.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => ui.drop.addEventListener(t, (e) => { e.preventDefault(); ui.drop.classList.remove('over'); }));
ui.drop.addEventListener('drop', (e) => { if (e.dataTransfer?.files?.length) addFiles([...e.dataTransfer.files]); });
ui.clear.addEventListener('click', clearItems);
ui.form.fit.addEventListener('change', reclassify);
ui.tile.addEventListener('input', () => { ui.tileOut.value = ui.tile.value; });

// ------------------------------------------------------------------ submit
ui.form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const files = eligible();
  if (!files.length) return;
  const data = new FormData();
  for (const it of files) data.append('files', it.file, it.file.name);
  data.append('profile', ui.form.profile.value);
  data.append('scale', ui.form.scale.value);
  data.append('tile', ui.tile.value);
  data.append('ocr', String(ui.form.ocr.checked));
  data.append('fit', String(fitOn()));
  ui.formStatus.textContent = '';
  ui.run.disabled = true;
  ui.run.textContent = 'Uploading…';
  try {
    const job = await api('/api/jobs', { method: 'POST', body: data });
    history.replaceState(null, '', `#job=${job.id}`);
    startJob(job);
  } catch (err) {
    ui.formStatus.textContent = err.message;
    renderTray();
  }
});

// ------------------------------------------------------------------ polling
function startJob(job) {
  clearTimeout(state.pollTimer);
  state.pollFailures = 0;
  ui.batch.hidden = false;
  renderJob(job);
  ui.batch.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  schedulePoll(job);
}

function schedulePoll(job) {
  if (job.status === 'queued' || job.status === 'running') {
    state.pollTimer = setTimeout(() => poll(job.id), POLL_MS);
  }
}

async function poll(id) {
  try {
    const job = await api(`/api/jobs/${id}`, { cache: 'no-store' });
    state.pollFailures = 0;
    renderJob(job);
    schedulePoll(job);
  } catch (err) {
    if (err.status === 404) {
      setBatchStatus('This batch has expired (results are kept for 24 hours). Start a new one above.', true);
      history.replaceState(null, '', location.pathname);
      state.job = null; renderTray();
      return;
    }
    // Transient network or deploy blips: back off and keep trying for a while.
    state.pollFailures += 1;
    if (state.pollFailures > MAX_POLL_FAILURES) {
      setBatchStatus(`Lost contact with the server (${err.message}). Reload the page to check this batch again.`, true);
      return;
    }
    setBatchStatus(`Connection problem, retrying… (${err.message})`, true);
    state.pollTimer = setTimeout(() => poll(id), POLL_MS * 2 ** state.pollFailures);
  }
}

// ------------------------------------------------------------------ rendering
function setBatchStatus(text, isError = false) {
  ui.batchStatus.textContent = text;
  ui.batchStatus.classList.toggle('error', isError);
}

function renderJob(job) {
  state.job = job;
  if (ui.gallery.dataset.job !== job.id) {   // a different batch: start from a clean slate
    ui.gallery.dataset.job = job.id;
    ui.gallery.replaceChildren();
    ui.rows.replaceChildren();
    state.shown.clear();
    state.startedAt = Date.parse(job.createdAt) || Date.now();
  }
  const doneRows = new Map(job.rows.map((r) => [r.index, r]));
  const errs = new Map(job.errors.map((e) => [e.index, e]));
  const finished = doneRows.size + errs.size;
  const active = job.status === 'running' && job.progress && !doneRows.has(job.progress.index) && !errs.has(job.progress.index) ? job.progress : null;
  const partial = active ? active.tile / Math.max(1, active.tiles) : 0;
  ui.overall.value = job.status === 'done' ? 100 : Math.round(((finished + partial) / Math.max(1, job.total)) * 100);

  const endedAt = job.finishedAt ? Date.parse(job.finishedAt) : Date.now();
  const elapsed = Math.max(0, (endedAt - state.startedAt) / 1000);
  const avg = job.rows.length ? job.rows.reduce((s, r) => s + Number(r.seconds || 0), 0) / job.rows.length : 0;
  const remaining = avg ? avg * (job.total - finished - partial) : 0;
  const heading = { queued: 'Waiting in line', running: 'Restoring', done: 'Restored', failed: 'Batch failed' }[job.status];
  ui.batchHeading.textContent = heading;
  if (job.status === 'queued') setBatchStatus(`Queued behind ${job.position - 1 > 0 ? `${job.position - 1} other batch${job.position > 2 ? 'es' : ''}` : 'the current batch'}.`);
  else if (job.status === 'running') setBatchStatus(`${finished} of ${job.total} finished · ${fmtDuration(elapsed)} elapsed${remaining ? ` · about ${fmtDuration(remaining)} left` : ''}`);
  else if (job.status === 'done') setBatchStatus(`${job.rows.length} restored${job.errors.length ? `, ${job.errors.length} failed` : ''} in ${fmtDuration(elapsed)}.`);
  else setBatchStatus(`Failed: ${job.message || 'unknown error'}`, true);

  ui.queue.replaceChildren(...job.files.map((name, i) => {
    let st = 'queued'; let info = job.status === 'failed' ? 'Not processed' : 'Waiting';
    if (doneRows.has(i)) { st = 'done'; info = `${doneRows.get(i).seconds} s`; }
    else if (errs.has(i)) { st = 'failed'; info = errs.get(i).message; }
    else if (active && active.index === i) { st = 'running'; info = `Tile ${active.tile} of ${active.tiles}`; }
    const li = el('li', {}, el('span', { className: 'dot', ariaHidden: 'true' }), el('span', { className: 'qname', textContent: name, title: name }), el('span', { className: 'qinfo', textContent: info }));
    li.dataset.state = st;
    return li;
  }));

  for (const row of job.rows) {
    if (state.shown.has(row.index)) continue;
    state.shown.add(row.index);
    ui.gallery.append(resultCard(job.id, row));
    ui.rows.append(el('tr', {}, ...[row.file, row.fitted_from ? `${row.input_pixels} (from ${row.fitted_from})` : row.input_pixels, row.output_pixels,
      `${row.scale}×`, row.seconds, row.input_edge_metric, row.output_edge_metric, Number(row.output_bytes).toLocaleString(), row.sha256_prefix]
      .map((v) => el('td', { textContent: String(v) }))));
  }
  ui.manifest.hidden = job.rows.length === 0;
  ui.zip.hidden = job.status !== 'done';
  ui.zip.href = `/api/jobs/${job.id}/zip`;
  ui.newBatch.hidden = job.status === 'queued' || job.status === 'running';
  // The engine pill shows queue state; refresh it as soon as this batch ends.
  const terminal = job.status === 'done' || job.status === 'failed';
  if (terminal && state.lastStatus && state.lastStatus !== job.status) refreshEngine();
  state.lastStatus = job.status;
  renderTray();
}

/** Before/after card: the range input covers the image, so drag, tap and arrow keys all work. */
function resultCard(id, row) {
  const outUrl = `/api/jobs/${id}/output/${row.index}`;
  const after = el('img', { src: outUrl, alt: `${row.file}, restored ${row.scale}×`, loading: 'lazy', decoding: 'async' });
  const before = el('div', { className: 'before' }, el('img', { src: `/api/jobs/${id}/original/${row.index}`, alt: '', loading: 'lazy', decoding: 'async' }));
  const slider = el('input', { type: 'range', min: 0, max: 100, value: 50, step: 1 });
  slider.setAttribute('aria-label', `Compare original and restored ${row.file}`);
  slider.setAttribute('aria-valuetext', '50% original');
  const compare = el('div', { className: 'compare' }, after, before, el('div', { className: 'handle' }),
    el('span', { className: 'tag l', textContent: 'Original' }), el('span', { className: 'tag r', textContent: `Restored ${row.scale}×` }), slider);
  slider.addEventListener('input', () => {
    compare.style.setProperty('--pos', `${slider.value}%`);
    slider.setAttribute('aria-valuetext', `${slider.value}% original`);
  });

  const actions = el('div', { className: 'actions' },
    el('a', { className: 'button', href: outUrl, download: row.output, textContent: 'Download PNG' }),
    el('a', { className: 'button secondary', href: outUrl, target: '_blank', rel: 'noopener', textContent: 'Open full size' }));

  const body = el('div', { className: 'body' },
    el('span', { className: 'title', textContent: row.file }),
    el('span', { className: 'facts', textContent: `${row.input_pixels} → ${row.output_pixels} · ${row.seconds} s${row.fitted_from ? ` · shrunk from ${row.fitted_from}` : ''}` }),
    actions);

  if (row.ocr !== null && row.ocr !== undefined) {
    const pre = el('pre', { textContent: row.ocr || '(no text found)' });
    const copy = el('button', { type: 'button', className: 'icon-btn', textContent: 'Copy text' });
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(row.ocr || ''); copy.textContent = 'Copied'; }
      catch { getSelection().selectAllChildren(pre); copy.textContent = 'Selected, press Ctrl+C'; }
      setTimeout(() => { copy.textContent = 'Copy text'; }, 2000);
    });
    body.append(el('details', {}, el('summary', { textContent: 'Card text (OCR)' }), pre, copy));
  }
  return el('figure', { className: 'result' }, compare, body);
}

ui.newBatch.addEventListener('click', () => {
  clearTimeout(state.pollTimer);
  state.job = null;
  ui.batch.hidden = true;
  history.replaceState(null, '', location.pathname);
  clearItems();
  ui.drop.scrollIntoView({ block: 'center' });
  ui.input.focus();
});

// ------------------------------------------------------------------ boot
refreshEngine();
setInterval(refreshEngine, 30000);
renderTray();
const resume = location.hash.match(/job=([0-9a-f-]{36})/);
if (resume) {
  ui.batch.hidden = false;
  setBatchStatus('Loading this batch…');
  poll(resume[1]);
}
