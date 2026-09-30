'use strict';
// Open Sea: a WebGPU ocean served at /sea/ alongside the Lab.
//
// Everything is served from this origin (three.js is vendored from the npm
// package, not a CDN), so the page runs under a strict CSP: scripts only from
// 'self' plus the exact inline boot script and import map, pinned by SHA-256.
// Only an explicit allowlist of files is reachable; nothing else in
// node_modules is exposed.
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const express = require('express');

const SEA_DIR = path.join(__dirname, '..', 'sea');

/** Resolve the installed three.js package root and version. */
function locateThree() {
  // three's "exports" map does not expose package.json, so walk up from its CJS entry.
  const entry = require.resolve('three');
  let dir = path.dirname(entry);
  while (!fs.existsSync(path.join(dir, 'package.json'))) dir = path.dirname(dir);
  const { name, version } = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
  if (name !== 'three') throw new Error(`expected the three package at ${dir}, found ${name}`);
  return { root: dir, version };
}

const sha256 = text => `'sha256-${crypto.createHash('sha256').update(text, 'utf8').digest('base64')}'`;

/** CSP source hashes for every inline <script> (no src) and <style> block. */
function inlineHashes(html, tag) {
  const re = new RegExp(`<${tag}(?![^>]*\\ssrc=)[^>]*>([\\s\\S]*?)</${tag}>`, 'g');
  return [...html.matchAll(re)].map(m => sha256(m[1]));
}

/**
 * Build the /sea router. Fails fast at startup if the page references a
 * three.js version other than the installed one, or a vendored file is missing.
 */
function createSeaRouter() {
  const three = locateThree();
  const html = fs.readFileSync(path.join(SEA_DIR, 'index.html'), 'utf8');
  const vendorPrefix = `./vendor/three@${three.version}/`;
  if (!html.includes(vendorPrefix)) {
    throw new Error(`sea/index.html does not reference the installed three@${three.version}; update its import map`);
  }

  // Public path (relative to /sea/vendor/three@<version>/) -> file inside the package.
  const vendorFiles = {
    'three.webgpu.js': 'build/three.webgpu.js',
    'three.core.js': 'build/three.core.js', // imported by three.webgpu.js as ./three.core.js
    'three.tsl.js': 'build/three.tsl.js',
    'addons/tsl/display/BloomNode.js': 'examples/jsm/tsl/display/BloomNode.js',
    'addons/controls/OrbitControls.js': 'examples/jsm/controls/OrbitControls.js',
  };
  for (const rel of Object.values(vendorFiles)) {
    if (!fs.existsSync(path.join(three.root, rel))) throw new Error(`three@${three.version} is missing ${rel}`);
  }

  const csp = [
    "default-src 'self'",
    `script-src 'self' ${inlineHashes(html, 'script').join(' ')}`,
    `style-src 'self' ${inlineHashes(html, 'style').join(' ')}`,
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
    "frame-ancestors 'none'",
  ].join('; ');

  const router = express.Router({ strict: true });
  router.use((_req, res, next) => {
    res.set({ 'Content-Security-Policy': csp, 'Cross-Origin-Opener-Policy': 'same-origin' });
    next();
  });

  // Relative URLs in the page (./main.js, ./vendor/...) only resolve under /sea/.
  router.get('/', (req, res, next) => {
    if (!req.originalUrl.split('?')[0].endsWith('/')) return res.redirect(301, '/sea/');
    return next();
  });
  router.get('/', (_req, res) => {
    res.set('Cache-Control', 'no-cache').type('html').send(html);
  });
  router.get('/main.js', (_req, res) => {
    res.set('Cache-Control', 'no-cache').type('text/javascript').sendFile(path.join(SEA_DIR, 'main.js'));
  });

  // Versioned vendor URLs never change content, so they can be cached for a year.
  const vendorBase = `/vendor/three@${three.version}/`;
  for (const [pub, rel] of Object.entries(vendorFiles)) {
    router.get(vendorBase + pub, (_req, res) => {
      res.set('Cache-Control', 'public, max-age=31536000, immutable')
        .type('text/javascript')
        .sendFile(path.join(three.root, rel));
    });
  }
  router.use((_req, res) => res.status(404).type('text/plain').send('Not found'));

  router.threeVersion = three.version;
  return router;
}

module.exports = { createSeaRouter };
