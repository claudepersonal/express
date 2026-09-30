'use strict';
// HTTP tests for the Open Sea page at /sea/ (routing, CSP, vendored three.js).
const test = require('node:test');
const assert = require('node:assert');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createApp } = require('../server/app');

let server;
let base;
let app;

test.before(async () => {
  const storageDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cardrestore-sea-'));
  app = createApp({ storageDir, workerScript: path.join(__dirname, 'fake_worker.py'), distDir: path.join(storageDir, 'nodist') });
  await new Promise(r => { server = app.listen(0, '127.0.0.1', r); });
  base = `http://127.0.0.1:${server.address().port}`;
});

test.after(() => { app.close(); server.close(); });

// package.json pins three exactly; the page's import map must use the same version.
const version = require('../package.json').dependencies.three;

test('/sea redirects to /sea/ so relative module URLs resolve', async () => {
  const res = await fetch(`${base}/sea`, { redirect: 'manual' });
  assert.strictEqual(res.status, 301);
  assert.strictEqual(res.headers.get('location'), '/sea/');
});

test('/sea/ serves the page under a strict, hash-pinned CSP', async () => {
  const res = await fetch(`${base}/sea/`);
  assert.strictEqual(res.status, 200);
  assert.match(res.headers.get('content-type'), /text\/html/);
  const html = await res.text();
  const csp = res.headers.get('content-security-policy');
  assert.ok(!csp.includes('unsafe-inline') && !csp.includes('unsafe-eval'), csp);
  assert.ok(!csp.includes('http'), 'no third-party hosts are allowed');
  // Every inline script (boot guard + import map) must be allowed by its exact hash.
  const inline = [...html.matchAll(/<script(?![^>]*\ssrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
  assert.strictEqual(inline.length, 2);
  for (const body of inline) {
    const h = `'sha256-${crypto.createHash('sha256').update(body, 'utf8').digest('base64')}'`;
    assert.ok(csp.includes(h), `CSP is missing ${h}`);
  }
  assert.ok(html.includes(`./vendor/three@${version}/three.webgpu.js`));
});

test('app script and vendored three.js modules are served as JavaScript', async () => {
  const paths = ['main.js', `vendor/three@${version}/three.webgpu.js`, `vendor/three@${version}/three.core.js`, `vendor/three@${version}/three.tsl.js`,
    `vendor/three@${version}/addons/tsl/display/BloomNode.js`, `vendor/three@${version}/addons/controls/OrbitControls.js`];
  for (const p of paths) {
    const res = await fetch(`${base}/sea/${p}`, { headers: { 'accept-encoding': 'gzip' } });
    assert.strictEqual(res.status, 200, p);
    assert.match(res.headers.get('content-type'), /javascript/, p);
    if (p.startsWith('vendor/')) {
      assert.match(res.headers.get('cache-control'), /immutable/, p);
      assert.strictEqual(res.headers.get('content-encoding'), 'gzip', `${p} should be compressed`);
    }
    assert.ok((await res.text()).length > 100, p);
  }
});

test('nothing outside the allowlist is reachable', async () => {
  for (const p of [`vendor/three@${version}/package.json`, `vendor/three@${version}/../../package.json`,
    'vendor/three@0.0.0/three.webgpu.js', '../server/app.js', 'index.html.bak']) {
    const res = await fetch(`${base}/sea/${p}`);
    assert.notStrictEqual(res.status, 200, p);
  }
});

test('the Lab keeps its own CSP', async () => {
  const res = await fetch(`${base}/api/health`);
  const h = await res.json();
  assert.strictEqual(h.limits.maxInputPixels, 1600 * 1600);
  assert.match(res.headers.get('content-security-policy'), /script-src 'self';/);
});
