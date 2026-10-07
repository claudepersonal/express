// CardCrop upscale tier UI.
// Pick cards (each with an optional comp in USD) -> POST /api/batches -> poll the batch
// until every card is done or failed (failed cards are skipped, the batch continues)
// -> per-card before/after toggle -> ZIP with original filenames. The batch id lives
// in the URL hash, so a reload resumes the same batch.

const $ = (s) => document.querySelector(s);
const ui = {
  engine: $('#engine'), drop: $('#drop'), input: $('#files'), tray: $('#tray'), clear: $('#clear'),
  bulk: $('#bulk'), compAll: $('#comp-all'), applyAll: $('#apply-all'), form: $('#opts'), run: $('#run'),
  err: $('#form-err'), warn: $('#retouch-warn'), batch: $('#batch'), batchId: $('#batch-id'),
  prog: $('#prog'), sum: $('#sum'), queue: $('#queue'), zip: $('#zip'), csv: $('#csv'), next: $('#new'),
};
const TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/tiff', 'image/bmp']);
const POLL_MS = 2000;
const state = { limits: { maxBatch: 50, maxFileMB: 40, highValueUsd: 500 }, items: [], batch: null, timer: 0, fails: 0, cards: new Map() };

const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) if (k != null && k !== false) n.append(k);
  return n;
};
const usd = (v) => (v == null ? '—' : `$${Number(v).toLocaleString()}`);
const dur = (s) => (s < 90 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);

async function api(path, opts) {
  const res = await fetch(path, opts);
  let body = {};
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw Object.assign(new Error(body.detail || `Request failed (${res.status})`), { status: res.status });
  return body;
}

// ---------------------------------------------------------------- engine
async function engine() {
  try {
    const h = await api('/api/health');
    state.limits = h.limits;
    $('#lim-batch').textContent = h.limits.maxBatch;
    $('#lim-mb').textContent = h.limits.maxFileMB;
    $('#hv').textContent = usd(h.limits.highValueUsd);
    const bad = Object.entries(h.models).filter(([, m]) => !m.ok).map(([k]) => k);
    ui.engine.dataset.state = bad.length ? 'down' : h.running ? 'busy' : 'ready';
    ui.engine.textContent = bad.length ? `Model problem: ${bad.join(', ')}`
      : `Models ready · ${h.threads} CPU threads${h.running ? ' · busy' : ''}${h.queue ? ` · ${h.queue} queued` : ''}`;
    const g = document.querySelector('input[name=damage][value=gemini]');
    g.disabled = !h.gemini;
    $('#gemini-note').textContent = h.gemini ? 'Masked, text-guarded' : 'Needs GEMINI_API_KEY on the server';
  } catch {
    ui.engine.dataset.state = 'down';
    ui.engine.textContent = 'Server unreachable';
  }
}

// ---------------------------------------------------------------- tray
function problem(it, i) {
  if (!TYPES.has(it.file.type)) return 'Not a supported image';
  if (it.file.size > state.limits.maxFileMB * 1048576) return `Over ${state.limits.maxFileMB} MB`;
  if (i >= state.limits.maxBatch) return `Batch limit is ${state.limits.maxBatch}`;
  return '';
}

function addFiles(files) {
  const known = new Set(state.items.map((it) => `${it.file.name}|${it.file.size}`));
  for (const file of files) {
    const key = `${file.name}|${file.size}`;
    if (known.has(key)) continue;
    known.add(key);
    state.items.push({ file, url: TYPES.has(file.type) && file.type !== 'image/tiff' ? URL.createObjectURL(file) : '', comp: null });
  }
  renderTray();
}

function renderTray() {
  ui.tray.replaceChildren(...state.items.map((it, i) => {
    const bad = problem(it, i);
    const comp = el('input', { type: 'number', min: 0, step: 1, value: it.comp ?? '', placeholder: 'comp' });
    comp.setAttribute('aria-label', `Comp in USD for ${it.file.name}`);
    comp.addEventListener('input', () => { it.comp = comp.value === '' ? null : Number(comp.value); badge(); });
    const hv = el('span', { className: 'chip hv', textContent: 'HIGH VALUE' });
    const badge = () => { hv.hidden = !(it.comp >= state.limits.highValueUsd); };
    badge();
    const rm = el('button', { type: 'button', className: 'x', textContent: 'Remove' });
    rm.setAttribute('aria-label', `Remove ${it.file.name}`);
    rm.addEventListener('click', () => { if (it.url) URL.revokeObjectURL(it.url); state.items.splice(i, 1); renderTray(); });
    return el('li', { className: bad ? 'bad' : '' },
      el('div', { className: 'thumb' }, it.url ? el('img', { src: it.url, alt: '' }) : el('small', { textContent: 'No preview' })),
      el('div', { className: 'meta' },
        el('span', { className: 'fname', textContent: it.file.name, title: it.file.name }),
        bad ? el('span', { className: 'chip failed', textContent: bad })
          : el('div', { className: 'row' }, el('span', { className: 'usd' }, comp), hv),
        rm));
  }));
  const ok = state.items.filter((it, i) => !problem(it, i)).length;
  ui.clear.hidden = ui.bulk.hidden = state.items.length === 0;
  const busy = state.batch && state.batch.status !== 'done';
  ui.run.disabled = ok === 0 || busy;
  ui.run.textContent = busy ? 'Working…' : ok ? `Upscale ${ok} card${ok > 1 ? 's' : ''}` : 'Upscale';
}

ui.input.addEventListener('change', () => { const f = [...ui.input.files]; ui.input.value = ''; addFiles(f); });
['dragenter', 'dragover'].forEach((t) => ui.drop.addEventListener(t, (e) => { e.preventDefault(); ui.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => ui.drop.addEventListener(t, (e) => { e.preventDefault(); ui.drop.classList.remove('over'); }));
ui.drop.addEventListener('drop', (e) => { if (e.dataTransfer?.files?.length) addFiles([...e.dataTransfer.files]); });
ui.clear.addEventListener('click', () => { state.items.forEach((it) => it.url && URL.revokeObjectURL(it.url)); state.items = []; renderTray(); });
ui.applyAll.addEventListener('click', () => {
  const v = ui.compAll.value === '' ? null : Number(ui.compAll.value);
  state.items.forEach((it) => { it.comp = v; });
  renderTray();
});
ui.form.addEventListener('change', () => { ui.warn.hidden = ui.form.damage.value === 'off'; });

// ---------------------------------------------------------------- submit
ui.form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const send = state.items.filter((it, i) => !problem(it, i));
  if (!send.length) return;
  const data = new FormData();
  const comps = {};
  send.forEach((it, i) => { data.append('files', it.file, it.file.name); if (it.comp != null) comps[i] = it.comp; });
  data.append('comps', JSON.stringify(comps));
  for (const k of ['model', 'faces', 'damage']) data.append(k, ui.form[k].value);
  ui.err.textContent = '';
  ui.run.disabled = true;
  ui.run.textContent = 'Uploading…';
  try {
    const b = await api('/api/batches', { method: 'POST', body: data });
    history.replaceState(null, '', `#batch=${b.id}`);
    start(b);
  } catch (err) {
    ui.err.textContent = err.message;
    renderTray();
  }
});

// ---------------------------------------------------------------- batch
function start(b) {
  clearTimeout(state.timer);
  state.fails = 0;
  state.cards.clear();
  ui.queue.replaceChildren();
  ui.batch.hidden = false;
  render(b);
  ui.batch.scrollIntoView({ block: 'start', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  schedule(b);
}

function schedule(b) { if (b.status !== 'done') state.timer = setTimeout(() => poll(b.id), POLL_MS); }

async function poll(id) {
  try {
    const b = await api(`/api/batches/${id}`, { cache: 'no-store' });
    state.fails = 0;
    render(b);
    schedule(b);
  } catch (err) {
    if (err.status === 404) {
      ui.sum.textContent = 'This batch has expired. Start a new one.';
      history.replaceState(null, '', location.pathname);
      return;
    }
    state.fails += 1;
    ui.sum.textContent = `Connection problem, retrying… (${err.message})`;
    if (state.fails <= 6) state.timer = setTimeout(() => poll(id), POLL_MS * 2 ** state.fails);
  }
}

function cardNode(b, c) {
  let n = state.cards.get(c.index);
  if (n) return n;
  const img = c.preview === false
    ? el('span', { className: 'facts', textContent: 'No preview: the file could not be decoded' })
    : el('img', { alt: '', loading: 'lazy', decoding: 'async', src: `/api/batches/${b.id}/cards/${c.index}/before.jpg` });
  const tag = el('span', { className: 'tag', textContent: 'Before' });
  const ba = el('button', { type: 'button', className: 'ba', textContent: 'Show before', hidden: true });
  ba.setAttribute('aria-pressed', 'false');
  ba.addEventListener('click', () => {
    const before = ba.getAttribute('aria-pressed') !== 'true';
    ba.setAttribute('aria-pressed', String(before));
    ba.textContent = before ? 'Show after' : 'Show before';
    img.src = `/api/batches/${b.id}/cards/${c.index}/${before ? 'before' : 'after'}.jpg`;
    tag.textContent = before ? 'Before' : 'After';
  });
  n = {
    root: el('li', { className: 'card' }), img, tag, ba, shownAfter: false,
    chip: el('span', { className: 'chip waiting' }), stage: el('span', { className: 'facts' }),
    bar: el('i'), flags: el('div', { className: 'flags' }), facts: el('p', { className: 'facts mono' }),
    err: el('p', { className: 'cerr', hidden: true }),
  };
  n.root.append(el('div', { className: 'view' }, img, tag, ba), el('div', { className: 'cbody' },
    el('span', { className: 'fname', textContent: c.file, title: c.file }),
    el('div', { className: 'status' }, n.chip, n.stage), el('div', { className: 'bar' }, n.bar), n.flags, n.facts, n.err));
  ui.queue.append(n.root);
  state.cards.set(c.index, n);
  return n;
}

function render(b) {
  ui.batchId.textContent = b.id.slice(0, 8);
  const total = b.cards.length;
  const { done, failed, processing } = b.counts;
  let frac = done + failed;
  for (const c of b.cards) {
    const n = cardNode(b, c);
    n.root.dataset.status = c.status;
    n.root.dataset.hv = String(Boolean(c.high_value));
    n.chip.className = `chip ${c.status}`;
    n.chip.textContent = c.status;
    n.stage.textContent = c.status === 'processing' ? c.stage : c.status === 'waiting' ? `comp ${usd(c.comp_usd)}` : '';
    if (c.status === 'processing') frac += c.progress;
    n.bar.style.width = `${Math.round((c.status === 'done' || c.status === 'failed' ? 1 : c.progress) * 100)}%`;
    n.flags.replaceChildren(...(c.high_value ? [el('span', { className: 'chip hv', textContent: 'HIGH VALUE' })] : []),
      ...(c.flags || []).filter((f) => f !== 'HIGH_VALUE').map((f) => el('span', { className: 'chip flag', textContent: f.replace('_', ' ') })));
    if (c.status === 'done') {
      n.facts.textContent = `${c.input_pixels} → ${c.output_pixels} · ${c.model || 'no AI upscale'} · ${dur(c.seconds)}`
        + (c.psnr_vs_input != null ? ` · fidelity ${c.psnr_vs_input} dB` : '') + (c.text_retention != null ? ` · text kept ${Math.round(c.text_retention * 100)}%` : '');
      n.ba.hidden = false;
      if (!n.shownAfter) {  // switch to the result the first time it is ready
        n.shownAfter = true;
        n.img.src = `/api/batches/${b.id}/cards/${c.index}/after.jpg`;
        n.tag.textContent = 'After';
      }
    }
    n.err.hidden = c.status !== 'failed';
    n.err.textContent = c.error || '';
  }
  ui.prog.value = Math.round((frac / Math.max(1, total)) * 100);
  const head = b.status === 'queued' ? `Queued (position ${b.position})` : b.status === 'running' ? 'Running' : 'Finished';
  const elapsed = ((b.finished || Date.now() / 1000) - b.created);
  ui.sum.textContent = `${head} · ${done} done · ${failed} failed · ${processing} processing · ${total - done - failed - processing} waiting · ${dur(elapsed)}`;
  const finished = b.status === 'done';
  ui.zip.hidden = !finished || done === 0;
  ui.csv.hidden = !finished;
  ui.zip.href = `/api/batches/${b.id}/zip`;
  ui.csv.href = `/api/batches/${b.id}/manifest.csv`;
  ui.next.hidden = !finished;
  state.batch = b;
  renderTray();
  if (finished) engine();
}

ui.next.addEventListener('click', () => {
  clearTimeout(state.timer);
  state.batch = null;
  ui.batch.hidden = true;
  history.replaceState(null, '', location.pathname);
  state.items.forEach((it) => it.url && URL.revokeObjectURL(it.url));
  state.items = [];
  renderTray();
});

// ---------------------------------------------------------------- boot
engine();
setInterval(engine, 30000);
renderTray();
const resume = location.hash.match(/batch=([0-9a-f]{32})/);
if (resume) { ui.batch.hidden = false; poll(resume[1]); }
