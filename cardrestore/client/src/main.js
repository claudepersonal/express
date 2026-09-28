const $ = sel => document.querySelector(sel);
const form = $('#form');
const filesInput = $('#files');
const drop = $('#drop');
const runBtn = $('#run');
const statusEl = $('#status');
const tile = $('#tile');

let limits = { maxFiles: 12, maxFileMB: 15 };
let pollTimer = null;

fetch('/api/health').then(r => r.json()).then(h => {
  limits = h.limits || limits;
  $('#lim-files').textContent = limits.maxFiles;
  $('#lim-mb').textContent = limits.maxFileMB;
  if (h.model?.status === 'unavailable') setStatus(`Restore model unavailable: ${h.model.reason}`, true);
}).catch(() => {});

tile.addEventListener('input', () => { $('#tile-out').textContent = tile.value; });

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('error', isError);
}

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  for (const c of children) node.append(c);
  return node;
}

function renderPicked() {
  const list = $('#picked');
  list.replaceChildren(...[...filesInput.files].map(f => el('li', { textContent: `${f.name} · ${f.size < 1048576 ? `${Math.max(1, Math.round(f.size / 1024))} KB` : `${(f.size / 1048576).toFixed(1)} MB`}` })));
  runBtn.disabled = filesInput.files.length === 0;
  if (filesInput.files.length > limits.maxFiles) setStatus(`Choose at most ${limits.maxFiles} files.`, true);
  else setStatus('');
}

filesInput.addEventListener('change', renderPicked);
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => {
  if (!e.dataTransfer?.files?.length) return;
  filesInput.files = e.dataTransfer.files;
  renderPicked();
});

form.addEventListener('submit', async e => {
  e.preventDefault();
  if (!filesInput.files.length) return;
  clearTimeout(pollTimer);
  const data = new FormData();
  for (const f of filesInput.files) data.append('files', f);
  data.append('profile', form.profile.value);
  data.append('scale', form.scale.value);
  data.append('tile', tile.value);
  data.append('ocr', form.ocr.checked ? 'true' : 'false');
  runBtn.disabled = true;
  setStatus('Uploading…');
  try {
    const res = await fetch('/api/jobs', { method: 'POST', body: data });
    const job = await res.json();
    if (!res.ok) throw new Error(job.error || `Upload failed (${res.status})`);
    history.replaceState(null, '', `#job=${job.id}`);
    render(job);
    poll(job.id);
  } catch (err) {
    setStatus(err.message, true);
    runBtn.disabled = false;
  }
});

async function poll(id) {
  try {
    const res = await fetch(`/api/jobs/${id}`);
    const job = await res.json();
    if (!res.ok) throw new Error(job.error);
    render(job);
    if (job.status === 'queued' || job.status === 'running') pollTimer = setTimeout(() => poll(id), 2500);
    else runBtn.disabled = filesInput.files.length === 0;
  } catch (err) {
    setStatus(err.message, true);
    runBtn.disabled = false;
  }
}

const shown = new Set();

function render(job) {
  const done = job.rows.length + job.errors.length;
  const text = {
    queued: `Queued (position ${job.position}).`,
    running: `Restoring ${Math.min(done + 1, job.total)} of ${job.total} on ${job.device || 'the server'}${job.progress && job.progress.index === done ? ` · tile ${job.progress.tile}/${job.progress.tiles}` : ''}… about a minute or two per card on CPU.`,
    done: `Done: ${job.rows.length} restored${job.errors.length ? `, ${job.errors.length} failed` : ''}.`,
    failed: `Failed: ${job.message || 'unknown error'}`,
  }[job.status];
  setStatus(text, job.status === 'failed');

  $('#results').hidden = false;
  const gallery = $('#gallery');
  if (!gallery.dataset.job || gallery.dataset.job !== job.id) {
    gallery.replaceChildren();
    $('#rows').replaceChildren();
    shown.clear();
    gallery.dataset.job = job.id;
  }
  for (const row of job.rows) {
    if (shown.has(row.index)) continue;
    shown.add(row.index);
    gallery.append(card(job.id, row));
    $('#rows').append(el('tr', {}, ...[row.file, row.input_pixels, row.output_pixels, `${row.scale}×`, row.seconds, row.input_edge_metric, row.output_edge_metric, row.output_bytes.toLocaleString(), row.sha256_prefix].map(v => el('td', { textContent: String(v) }))));
  }
  $('#errors').replaceChildren(...job.errors.map(e => el('li', { textContent: `${e.file}: ${e.message}` })));
  const zip = $('#zip');
  zip.hidden = job.status !== 'done';
  zip.href = `/api/jobs/${job.id}/zip`;
}

function card(id, row) {
  const before = el('img', { src: `/api/jobs/${id}/original/${row.index}`, alt: `${row.file} original`, loading: 'lazy' });
  const after = el('img', { src: `/api/jobs/${id}/output/${row.index}`, alt: `${row.file} restored ${row.scale}×`, loading: 'lazy' });
  const slider = el('input', { type: 'range', min: 0, max: 100, value: 50, ariaLabel: `Compare original and restored ${row.file}` });
  const compare = el('div', { className: 'compare' }, after, el('div', { className: 'clip' }, before), slider);
  compare.style.setProperty('--pos', '50%');
  slider.addEventListener('input', () => compare.style.setProperty('--pos', `${slider.value}%`));
  const figcaption = el('figcaption', {},
    el('strong', { textContent: row.file }),
    el('span', { textContent: ` ${row.input_pixels} → ${row.output_pixels} · ${row.seconds}s` }),
    ' ',
    el('a', { href: after.src, target: '_blank', rel: 'noopener', textContent: 'Open full size' }));
  const fig = el('figure', { className: 'result' }, compare, figcaption);
  if (row.ocr !== null && row.ocr !== undefined) {
    fig.append(el('details', {}, el('summary', { textContent: 'OCR text' }), el('pre', { textContent: row.ocr || '(no text found)' })));
  }
  return fig;
}

const m = location.hash.match(/job=([0-9a-f-]{36})/);
if (m) poll(m[1]);
