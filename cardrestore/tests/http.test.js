'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createApp } = require('../server/app');

// 1x1 PNG
const PNG = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==', 'base64');

let server;
let base;
let app;

test.before(async () => {
  const storageDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cardrestore-'));
  app = createApp({ storageDir, workerScript: path.join(__dirname, 'fake_worker.py'), distDir: path.join(storageDir, 'nodist') });
  await new Promise(r => { server = app.listen(0, '127.0.0.1', r); });
  base = `http://127.0.0.1:${server.address().port}`;
});

test.after(() => { app.close(); server.close(); });

function form(files, fields = {}) {
  const fd = new FormData();
  for (const [name, buf, type] of files) fd.append('files', new Blob([buf], { type }), name);
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  return fd;
}

async function waitDone(id) {
  for (let i = 0; i < 100; i++) {
    const job = await (await fetch(`${base}/api/jobs/${id}`)).json();
    if (job.status === 'done' || job.status === 'failed') return job;
    await new Promise(r => setTimeout(r, 50));
  }
  throw new Error('job did not finish');
}

test('health endpoints respond', async () => {
  assert.deepStrictEqual(await (await fetch(`${base}/health`)).json(), { ok: true });
  const h = await (await fetch(`${base}/api/health`)).json();
  assert.strictEqual(h.limits.maxFiles, 12);
});

test('batch restore: queue, rows, per-file error, outputs and zip', async () => {
  const res = await fetch(`${base}/api/jobs`, {
    method: 'POST',
    body: form([['a.png', PNG, 'image/png'], ['bad.png', PNG, 'image/png']], { profile: 'standard', scale: '4', tile: '999', ocr: 'false' }),
  });
  assert.strictEqual(res.status, 202);
  const created = await res.json();
  assert.match(created.id, /^[0-9a-f-]{36}$/);
  assert.deepStrictEqual(created.settings, { profile: 'standard', scale: 4, tile: 512, ocr: false });
  const job = await waitDone(created.id);
  assert.strictEqual(job.status, 'done');
  assert.strictEqual(job.rows.length, 1);
  assert.strictEqual(job.rows[0].ocr, null);
  assert.strictEqual(job.errors[0].file, 'bad.png');
  const out = await fetch(`${base}/api/jobs/${job.id}/output/0`);
  assert.strictEqual(out.status, 200);
  assert.strictEqual(out.headers.get('content-type'), 'image/png');
  assert.strictEqual((await fetch(`${base}/api/jobs/${job.id}/output/1`)).status, 404);
  assert.strictEqual((await fetch(`${base}/api/jobs/${job.id}/original/0`)).status, 200);
  const zip = await fetch(`${base}/api/jobs/${job.id}/zip`);
  assert.strictEqual(zip.status, 200);
  assert.strictEqual(Buffer.from(await zip.arrayBuffer()).subarray(0, 2).toString(), 'PK');
});

test('rejects empty uploads, non-images and unknown jobs', async () => {
  const empty = await fetch(`${base}/api/jobs`, { method: 'POST', body: form([]) });
  assert.strictEqual(empty.status, 400);
  const txt = await fetch(`${base}/api/jobs`, { method: 'POST', body: form([['x.txt', Buffer.from('hi'), 'text/plain']]) });
  assert.strictEqual(txt.status, 415);
  assert.strictEqual((await fetch(`${base}/api/jobs/not-a-job`)).status, 404);
  assert.strictEqual((await fetch(`${base}/api/jobs/00000000-0000-4000-8000-000000000000/zip`)).status, 404);
});

test('rejects too many files', async () => {
  const files = Array.from({ length: 13 }, (_, i) => [`c${i}.png`, PNG, 'image/png']);
  const res = await fetch(`${base}/api/jobs`, { method: 'POST', body: form(files) });
  assert.strictEqual(res.status, 413);
});

test('path traversal in output index is not served', async () => {
  const res = await fetch(`${base}/api/jobs/..%2F..%2Fetc/output/..%2Fpasswd`);
  assert.strictEqual(res.status, 404);
});
