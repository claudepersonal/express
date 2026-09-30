// Open Sea: a procedural ocean with WebGPURenderer + TSL (three.js r186).
//
// Pipeline
//   sky dome (shared analytic sky) ─┐
//   water plane (5 Gerstner waves) ─┴─> pass() ─> + bloom() ─> ACES tone map + sRGB (RenderPipeline)
//
// The same `sky()` TSL function, fed by the same uniforms, shades the dome and
// every reflection on the water, so clouds, sun and palette always agree.

import * as THREE from 'three/webgpu';
import {
  Fn, If, uniform, varying, vec2, vec3, float,
  positionGeometry, positionWorld, cameraPosition,
  normalize, cross, reflect, dot, length, mix, smoothstep, clamp, saturate,
  max, pow, exp, sin, cos, sqrt, abs, floor,
  mx_noise_float, hash, pass,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

window.__oceanBoot.started = true;

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const ui = {
  stage: $('stage'), loading: $('loading'), loadingText: $('loading-text'),
  hud: $('hud'), toggleHud: $('toggle-hud'), form: $('hud-form'),
  sea: $('sea'), seaOut: $('sea-out'), tod: $('tod'), todOut: $('tod-out'),
  drift: $('drift'), pause: $('pause'), resetView: $('reset-view'),
  fps: $('fps'), quality: $('quality'), desc: $('scene-desc'),
};
const fail = (title, text) => window.__oceanBoot.fail(title, text);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

// ---------------------------------------------------------------------------
// Wave model (shared by the GPU shader and the CPU camera bob)
// ---------------------------------------------------------------------------
const G = 9.81;
const WAVE_SPEC = [
  // wavelength (m), heading (deg from wind), steepness share, amplitude factor, phase
  { L: 64.0, deg: 0, s: 0.95, a: 1.00, p: 0.0 },
  { L: 37.0, deg: 23, s: 0.85, a: 0.95, p: 1.7 },
  { L: 22.5, deg: -31, s: 0.80, a: 0.90, p: 4.1 },
  { L: 13.5, deg: 52, s: 0.70, a: 0.85, p: 2.6 },
  { L: 7.9, deg: -14, s: 0.60, a: 0.80, p: 5.3 },
];
const WIND_DEG = -20;
const WAVES = WAVE_SPEC.map((w) => {
  const rad = THREE.MathUtils.degToRad(w.deg + WIND_DEG);
  const k = (2 * Math.PI) / w.L;
  return {
    ...w, k, omega: Math.sqrt(G * k), dx: Math.cos(rad), dz: Math.sin(rad),
    baseA: w.L * 0.0105 * w.a, // amplitude (m) at amplitude scale 1
  };
});
if (WAVES.length !== 5) throw new Error('The sea is defined by exactly five Gerstner waves.');

// Sea state 0..1 → shader parameters. Chop share keeps Σ c·k·A < 1, so the
// surface never self-intersects even at the roughest setting.
function seaParams(s) {
  return {
    amp: 0.3 + 1.7 * Math.pow(s, 1.3),
    chop: 0.55 + 0.75 * s,
    detail: 0.22 + 0.85 * s,
    foam: 0.55 + 0.4 * s,        // Jacobian value below which foam starts
    coverage: 0.22 + 0.45 * s,   // cloud coverage
    glitter: 0.6 + 0.7 * s,
    wind: 0.6 + 1.8 * s,
  };
}
const SEA_LABELS = [[0.12, 'Glassy'], [0.3, 'Calm'], [0.55, 'Moderate'], [0.8, 'Rough'], [1.01, 'Very rough']];

// CPU mirror of the vertical Gerstner displacement (for the floating camera).
function waveHeight(x, z, t, amp) {
  let y = 0;
  for (const w of WAVES) {
    const theta = w.k * (w.dx * x + w.dz * z) - w.omega * t + w.p;
    y += amp * w.baseA * Math.sin(theta);
  }
  return y;
}

// ---------------------------------------------------------------------------
// Time of day palette (sRGB keys, interpolated in linear space)
// ---------------------------------------------------------------------------
const KEYS = [
  { e: -30, zenith: 0x020614, horizon: 0x0e1a2c, light: 0xffffff, lightI: 0.0, disk: 0, cloud: 0x0c1322, deep: 0x02070f, scatter: 0x05202a, stars: 1.0, glow: 0.0, exposure: 1.6, bloom: 0.32 },
  { e: -8, zenith: 0x081330, horizon: 0x2e3450, light: 0xff7a48, lightI: 0.0, disk: 0, cloud: 0x1d2135, deep: 0x020713, scatter: 0x06232b, stars: 0.3, glow: 0.4, exposure: 1.2, bloom: 0.34 },
  { e: 0, zenith: 0x1b305e, horizon: 0xe98a55, light: 0xff7a33, lightI: 0.9, disk: 26, cloud: 0x5d4659, deep: 0x03101d, scatter: 0x0b4a4a, stars: 0.0, glow: 1.0, exposure: 1.0, bloom: 0.42 },
  { e: 7, zenith: 0x2653a0, horizon: 0xf0b07c, light: 0xffae5e, lightI: 1.6, disk: 40, cloud: 0xd19f89, deep: 0x041a2b, scatter: 0x0e5f5c, stars: 0.0, glow: 0.6, exposure: 0.95, bloom: 0.38 },
  { e: 22, zenith: 0x2861bf, horizon: 0xa9c6e6, light: 0xfff0d8, lightI: 2.4, disk: 55, cloud: 0xf2f3f6, deep: 0x05203a, scatter: 0x107a70, stars: 0.0, glow: 0.15, exposure: 0.85, bloom: 0.3 },
  { e: 65, zenith: 0x1c58c8, horizon: 0x9cc0e8, light: 0xffffff, lightI: 2.8, disk: 60, cloud: 0xffffff, deep: 0x06264a, scatter: 0x128a7c, stars: 0.0, glow: 0.0, exposure: 0.8, bloom: 0.28 },
];
const COLOR_KEYS = ['zenith', 'horizon', 'light', 'cloud', 'deep', 'scatter'];
const NUM_KEYS = ['lightI', 'disk', 'stars', 'glow', 'exposure', 'bloom'];
const keyColors = KEYS.map((k) => Object.fromEntries(COLOR_KEYS.map((c) => [c, new THREE.Color(k[c])])));
const MOON = new THREE.Color(0xa9bde6);
const MAX_ELEV = THREE.MathUtils.degToRad(62);

function sunState(hour) {
  // 06:00 sunrise in the east, 12:00 highest in the south, 18:00 sunset in the west.
  const day = ((hour - 6) / 12) * Math.PI;
  const elev = Math.sin(day) * MAX_ELEV;
  const az = day;
  return { elev, az, elevDeg: THREE.MathUtils.radToDeg(elev) };
}
const dirFrom = (az, elev, out) => out.set(Math.cos(az) * Math.cos(elev), Math.sin(elev), -Math.sin(az) * Math.cos(elev));

// ---------------------------------------------------------------------------
// Uniforms shared by sky and water
// ---------------------------------------------------------------------------
const U = {
  time: uniform(0),
  lightDir: uniform(new THREE.Vector3(0, 0.2, -1).normalize()),
  lightColor: uniform(new THREE.Color()),
  lightI: uniform(1),
  diskI: uniform(40),
  zenith: uniform(new THREE.Color()),
  horizon: uniform(new THREE.Color()),
  cloud: uniform(new THREE.Color()),
  ambient: uniform(new THREE.Color()),
  deep: uniform(new THREE.Color()),
  scatter: uniform(new THREE.Color()),
  stars: uniform(0),
  glow: uniform(0),
  coverage: uniform(0.35),
  wind: uniform(1),
  // sea
  amp: uniform(1), chop: uniform(1), detail: uniform(0.5), foam: uniform(0.4), glitter: uniform(1),
  origin: uniform(new THREE.Vector2()),
  fogDensity: uniform(1 / 1500),
};

// ---------------------------------------------------------------------------
// Noise helpers (MaterialX gradient noise, roughly −1..1)
// ---------------------------------------------------------------------------
function fbm(p, octaves) {
  let sum = float(0);
  let amp = 0.5;
  let q = p;
  for (let i = 0; i < octaves; i++) {
    sum = sum.add(mx_noise_float(q).mul(amp));
    q = vec3(q.x.mul(1.6).add(q.z.mul(1.2)), q.y.mul(2.03), q.z.mul(1.6).sub(q.x.mul(1.2))).add(vec3(3.1, 1.7, 7.3));
    amp *= 0.5;
  }
  return sum;
}

// ---------------------------------------------------------------------------
// The one analytic sky
// dir: normalized view/reflection direction. diskScale: 1 on the dome, 0 on the
// water, where the sun disk and stars would alias; the water draws its own
// filtered sun highlight instead.
// ---------------------------------------------------------------------------
const sky = Fn(([dir, diskScale]) => {
  const y = dir.y;
  const up = max(y, 0.0);

  // Zenith-to-horizon gradient; the palette itself changes with the sun.
  const t = pow(float(1).sub(up), 4.0);
  const col = mix(U.zenith, U.horizon, t).toVar();

  const mu = dot(dir, U.lightDir);
  const muP = max(mu, 0.0);

  // Low-sun glow hugging the horizon on the sun's side.
  const band = exp(abs(y).mul(-9.0));
  col.addAssign(U.lightColor.mul(band.mul(pow(muP, 3.0)).mul(U.glow).mul(0.9)));

  // Halo (Mie-like forward scattering), two lobes.
  const halo = pow(muP, 8.0).mul(0.12).add(pow(muP, 90.0).mul(0.55));
  col.addAssign(U.lightColor.mul(halo).mul(U.lightI).mul(0.45));

  // Sun (or moon) disk, ~0.6°.
  const disk = smoothstep(0.99990, 0.99995, mu).mul(U.diskI).mul(diskScale).toVar();

  // Stars (night only, above horizon).
  const cell = floor(dir.mul(420.0)).add(512.0);
  const star = smoothstep(0.9975, 1.0, hash(cell.x.add(cell.y.mul(113.0)).add(cell.z.mul(12011.0))));
  // Dome only: point stars in a rippled reflection would just sparkle as noise.
  col.addAssign(vec3(star.mul(U.stars).mul(smoothstep(0.02, 0.2, y)).mul(1.6).mul(diskScale)));

  // Clouds: a stretched layer above the viewer, faded into the horizon haze.
  If(y.greaterThan(0.004), () => {
    const uvc = dir.xz.div(y.add(0.12)).mul(0.55).add(vec2(U.time.mul(0.004), U.time.mul(0.0015)).mul(U.wind));
    const n = fbm(vec3(uvc.x, U.time.mul(0.006), uvc.y), 4).mul(0.5).add(0.5);
    const edge = float(1).sub(U.coverage);
    const cover = smoothstep(edge.sub(0.05), edge.add(0.3), n).mul(smoothstep(0.004, 0.2, y));
    const silver = pow(muP, 6.0).mul(U.lightI).mul(0.35);
    const shade = mix(0.55, 1.0, smoothstep(0.0, 0.35, n.sub(edge)));
    const lit = U.cloud.mul(shade).add(U.lightColor.mul(silver)).add(U.zenith.mul(0.15));
    col.assign(mix(col, lit, cover.mul(0.92)));
    disk.mulAssign(float(1).sub(cover));
  });

  return col.add(U.lightColor.mul(disk));
});

// ---------------------------------------------------------------------------
// Gerstner waves: displacement, analytic tangents, Jacobian
// ---------------------------------------------------------------------------
function gerstner(base, dist, full) {
  let disp = vec3(0);
  let sxx = float(0), szz = float(0), sxz = float(0), hx = float(0), hz = float(0);
  for (const w of WAVES) {
    // Fade each wave out where the grid (and pixel footprint) can no longer carry it.
    const fade = float(1).sub(smoothstep(w.L * 12.0, w.L * 70.0, dist));
    const A = U.amp.mul(w.baseA).mul(fade);
    const cA = A.mul(U.chop).mul(w.s);
    const theta = base.x.mul(w.k * w.dx).add(base.y.mul(w.k * w.dz)).sub(U.time.mul(w.omega)).add(w.p);
    const S = sin(theta);
    const C = cos(theta);
    disp = disp.add(vec3(cA.mul(w.dx).mul(C), A.mul(S), cA.mul(w.dz).mul(C)));
    if (full) {
      const q = cA.mul(w.k).mul(S);
      sxx = sxx.add(q.mul(w.dx * w.dx));
      szz = szz.add(q.mul(w.dz * w.dz));
      sxz = sxz.add(q.mul(w.dx * w.dz));
      const kAC = A.mul(w.k).mul(C);
      hx = hx.add(kAC.mul(w.dx));
      hz = hz.add(kAC.mul(w.dz));
    }
  }
  if (!full) return { disp };
  // ∂P/∂x0 and ∂P/∂z0 of the Gerstner map; N = ∂P/∂z0 × ∂P/∂x0.
  const tx = vec3(float(1).sub(sxx), hx, sxz.negate());
  const tz = vec3(sxz.negate(), hz, float(1).sub(szz));
  const normal = normalize(cross(tz, tx));
  const jacobian = float(1).sub(sxx).mul(float(1).sub(szz)).sub(sxz.mul(sxz));
  return { disp, normal, jacobian };
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
const PLANE_RADIUS = 6000;

// A dense plane whose vertex spacing grows with distance: a linear core
// (constant spacing near the camera) plus a u^4 tail out to the horizon.
function createSeaGeometry(segments) {
  const geo = new THREE.PlaneGeometry(2, 2, segments, segments);
  geo.rotateX(-Math.PI / 2);
  const half = segments / 2;
  const core = 0.3 * half; // 0.3 m per cell near the centre
  const tail = PLANE_RADIUS - core;
  const pos = geo.attributes.position;
  const warp = (u) => Math.sign(u) * (core * Math.abs(u) + tail * Math.pow(Math.abs(u), 4));
  for (let i = 0; i < pos.count; i++) {
    pos.setX(i, warp(pos.getX(i)));
    pos.setZ(i, warp(pos.getZ(i)));
  }
  pos.needsUpdate = true;
  geo.computeBoundingSphere();
  return { geo, snap: core / half };
}

function createWaterMaterial() {
  const mat = new THREE.MeshBasicNodeMaterial();
  mat.fog = false;

  // Vertex: undisplaced world XZ comes from the raw geometry + the grid origin.
  const baseXZ = positionGeometry.xz.add(U.origin);
  const distV = length(baseXZ.sub(cameraPosition.xz));
  const gv = gerstner(baseXZ, distV, false);
  mat.positionNode = positionGeometry.add(gv.disp);

  const vBase = varying(baseXZ, 'vBase');
  const vWorld = varying(vec3(baseXZ.x, 0, baseXZ.y).add(gv.disp), 'vWorld');

  mat.colorNode = Fn(() => {
    const P = vWorld;
    const toCam = cameraPosition.sub(P);
    const dist = length(toCam);
    const V = toCam.div(dist);
    const L = U.lightDir;
    const t = U.time;

    // Swell normal and foam Jacobian, recomputed per pixel from the analytic derivatives.
    const g = gerstner(vBase, length(vBase.sub(cameraPosition.xz)), true);

    // Capillary detail: FBM height field, normal by forward differences.
    const detailFade = float(1).sub(smoothstep(25.0, 320.0, dist));
    const flow = vec2(t.mul(0.35), t.mul(-0.2)).mul(U.wind);
    const h = (xz) => fbm(vec3(xz.x.mul(0.42).add(flow.x), t.mul(0.22), xz.y.mul(0.42).add(flow.y)), 4);
    const eps = 0.06;
    const h0 = h(P.xz);
    const dhx = h(P.xz.add(vec2(eps, 0))).sub(h0).div(eps);
    const dhz = h(P.xz.add(vec2(0, eps))).sub(h0).div(eps);
    const detailAmp = U.detail.mul(detailFade).mul(0.22);
    const N = normalize(g.normal.add(vec3(dhx.negate(), 0, dhz.negate()).mul(detailAmp))).toVar();
    // Never let a normal face away from the viewer (grazing-angle stability).
    If(dot(N, V).lessThan(0.02), () => { N.assign(normalize(N.add(V.mul(float(0.02).sub(dot(N, V)))))); });

    const NdotV = saturate(dot(N, V));
    const fresnel = float(0.02).add(float(0.98).mul(pow(float(1).sub(NdotV), 5.0)));

    // Reflection of the shared sky (without the disk).
    const R = reflect(V.negate(), N);
    const Rup = normalize(vec3(R.x, max(R.y, 0.015), R.z));
    const reflection = sky(Rup, 0.0);

    // Water body: absorption colour lit by sky ambient and a little sun.
    const NdotL = saturate(dot(N, L));
    const body = U.deep.mul(U.ambient.mul(0.9).add(U.lightColor.mul(NdotL.mul(U.lightI).mul(0.25))));

    // Backlit crests: light transmitted through thin wave tops toward the viewer.
    const crest = saturate(g.disp.y.div(U.amp.mul(1.6)).mul(0.5).add(0.5));
    const through = pow(saturate(dot(V, normalize(L.add(N.mul(0.55))).negate())), 4.0);
    const sss = U.scatter.mul(through.mul(pow(crest, 2.0)).mul(U.lightI).mul(1.4)
      .add(pow(float(1).sub(NdotV), 2.0).mul(0.12)));

    const water = mix(body.add(sss), reflection, fresnel).toVar();

    // Sun highlight, broadening with distance to stand in for filtering.
    const RdotL = saturate(dot(R, L));
    const shininess = mix(1400.0, 160.0, smoothstep(0.0, 900.0, dist));
    const spec = pow(RdotL, shininess).mul(shininess.mul(0.06)).mul(U.lightI).mul(fresnel.mul(2.0).add(0.4));
    // Glitter: sparse facets catching the sun.
    const sparkleNoise = mx_noise_float(vec3(P.x.mul(2.3), P.z.mul(2.3), t.mul(2.2)));
    const glitter = smoothstep(0.52, 0.8, sparkleNoise).mul(pow(RdotL, 28.0)).mul(U.lightI).mul(U.glitter).mul(6.0)
      .mul(float(1).sub(smoothstep(150.0, 1400.0, dist)).mul(0.8).add(0.2));
    water.addAssign(U.lightColor.mul(spec.add(glitter)));

    // Foam where the Gerstner map compresses (low Jacobian), broken up by noise.
    const foamBreak = mx_noise_float(vec3(vBase.x.mul(0.35), t.mul(0.15), vBase.y.mul(0.35))).mul(0.5).add(0.5);
    const streak = mx_noise_float(vec3(vBase.x.mul(3.4), t.mul(0.5), vBase.y.mul(3.4))).mul(0.5).add(0.5);
    const foamRaw = float(1).sub(smoothstep(U.foam.sub(0.3), U.foam, g.jacobian));
    const foam = saturate(foamRaw.mul(smoothstep(0.3, 0.75, foamBreak.mul(0.45).add(streak.mul(0.55)).add(foamRaw.mul(0.35)))))
      .mul(float(1).sub(smoothstep(400.0, 2500.0, dist)).mul(0.7).add(0.3));
    const foamCol = U.ambient.mul(0.85).add(U.lightColor.mul(NdotL.mul(0.6).add(0.15)).mul(U.lightI).mul(0.35));
    water.assign(mix(water, foamCol, foam.mul(0.9)));

    // Horizon haze: fade into the sky exactly as the dome renders at the horizon.
    const viewDir = V.negate();
    const hazeDir = normalize(vec3(viewDir.x, 0.0, viewDir.z));
    const haze = sky(hazeDir, 0.0);
    const fog = float(1).sub(exp(dist.mul(U.fogDensity).negate()));
    return mix(water, haze, clamp(fog, 0.0, 1.0));
  })();

  return mat;
}

function createSkyMaterial() {
  const mat = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide, depthWrite: false });
  mat.fog = false;
  mat.colorNode = sky(normalize(positionWorld.sub(cameraPosition)), 1.0);
  return mat;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
const state = {
  paused: false,
  drift: !reducedMotion.matches,
  sea: 0.45,
  hour: 17.67,
  simTime: 0,
  tier: null,
  dpr: 1,
};

async function checkWebGPU() {
  if (!window.isSecureContext) {
    return 'WebGPU needs a secure context. Open the page from http://localhost or over https.';
  }
  if (!('gpu' in navigator)) {
    return 'This browser does not expose WebGPU. Use a current Chrome, Edge or Safari 26+, or Firefox 141+ on Windows. On Linux Chrome, enable chrome://flags/#enable-unsafe-webgpu.';
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return `The GPU adapter request failed: ${e.message}`;
  }
  if (!adapter) {
    return 'WebGPU is present but no compatible GPU adapter was found. It may be disabled or blocklisted for this GPU or driver.';
  }
  return null;
}

async function start() {
  const problem = await checkWebGPU();
  if (problem) {
    fail('WebGPU is not available', problem + ' This page does not fall back to WebGL.');
    return;
  }

  ui.loadingText.textContent = 'Starting the GPU…';
  const renderer = new THREE.WebGPURenderer({ antialias: false, powerPreference: 'high-performance' });
  try {
    await renderer.init();
  } catch (e) {
    fail('WebGPU failed to start', String(e && e.message ? e.message : e));
    return;
  }
  // WebGPURenderer can silently switch to WebGL2; this experience refuses to.
  if (!renderer.backend || renderer.backend.isWebGPUBackend !== true) {
    renderer.dispose();
    fail('WebGPU is not available', 'The browser offered WebGPU but the device could not be created, and this page does not fall back to WebGL.');
    return;
  }
  renderer.onDeviceLost = (info) => {
    stopLoop();
    fail('The GPU device was lost', `${info && info.message ? info.message : 'The graphics driver reset or the GPU was reclaimed.'} Reload to start again.`);
  };

  const coarse = matchMedia('(pointer: coarse)').matches;
  const small = Math.min(screen.width, screen.height) < 700;
  const DPR_CAP = Math.min(window.devicePixelRatio || 1, coarse ? 1.5 : 2);
  state.dpr = Math.min(DPR_CAP, coarse ? 1.25 : 1.5);
  state.tier = coarse || small ? 'low' : 'high';

  renderer.setPixelRatio(state.dpr);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('tabindex', '0');
  renderer.domElement.setAttribute('aria-describedby', 'canvas-help');
  ui.stage.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.5, 30000);

  const skyDome = new THREE.Mesh(new THREE.SphereGeometry(15000, 64, 32), createSkyMaterial());
  skyDome.renderOrder = -1;
  skyDome.frustumCulled = false;
  scene.add(skyDome);

  const waterMat = createWaterMaterial();
  const GEO_SEGMENTS = { high: 512, low: 256 };
  let sea = createSeaGeometry(GEO_SEGMENTS[state.tier]);
  const water = new THREE.Mesh(sea.geo, waterMat);
  water.frustumCulled = false;
  scene.add(water);

  // Post: scene → bloom → ACES + sRGB output (applied by RenderPipeline).
  const pipeline = new THREE.RenderPipeline(renderer);
  const scenePass = pass(scene, camera);
  const sceneColor = scenePass.getTextureNode('output');
  const bloomPass = bloom(sceneColor, 0.35, 0.45, 1.05);
  pipeline.outputNode = sceneColor.add(bloomPass);

  // Controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance = 8;
  controls.maxDistance = 160;
  controls.minPolarAngle = 0.35;
  controls.maxPolarAngle = 1.5;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.8;
  controls.listenToKeyEvents(renderer.domElement);
  const anchor = new THREE.Vector3(0, 0, 0); // drifting "boat" position on the sea
  const EYE_HEIGHT = 3.2;

  function resetView() {
    const { az } = sunState(state.hour);
    // Face the sun's azimuth: backlit crests and the glitter path lead the eye.
    const facing = new THREE.Vector3(Math.cos(az), 0, -Math.sin(az));
    controls.target.set(anchor.x, EYE_HEIGHT, anchor.z);
    camera.position.copy(controls.target).addScaledVector(facing, -34).setY(EYE_HEIGHT + 7.5);
    controls.update();
  }

  // --------------------------------------------------------------- uniforms from UI
  const sun = new THREE.Vector3();
  const tmp = new THREE.Color();
  function applySea() {
    const p = seaParams(state.sea);
    U.amp.value = p.amp; U.chop.value = p.chop; U.detail.value = p.detail; U.foam.value = p.foam;
    U.coverage.value = p.coverage; U.glitter.value = p.glitter; U.wind.value = p.wind;
    const label = SEA_LABELS.find(([limit]) => state.sea < limit)[1];
    ui.seaOut.value = label;
    ui.sea.setAttribute('aria-valuetext', label);
    describe();
  }

  function applyTime() {
    const { elev, az, elevDeg } = sunState(state.hour);
    // Interpolate the palette by solar elevation.
    let i = 0;
    while (i < KEYS.length - 2 && elevDeg > KEYS[i + 1].e) i++;
    const a = KEYS[i], b = KEYS[i + 1];
    const f = THREE.MathUtils.clamp((elevDeg - a.e) / (b.e - a.e), 0, 1);
    const s = f * f * (3 - 2 * f);
    const ca = keyColors[i], cb = keyColors[i + 1];
    U.zenith.value.lerpColors(ca.zenith, cb.zenith, s);
    U.horizon.value.lerpColors(ca.horizon, cb.horizon, s);
    U.cloud.value.lerpColors(ca.cloud, cb.cloud, s);
    U.deep.value.lerpColors(ca.deep, cb.deep, s);
    U.scatter.value.lerpColors(ca.scatter, cb.scatter, s);
    const num = Object.fromEntries(NUM_KEYS.map((k) => [k, a[k] + (b[k] - a[k]) * s]));
    U.stars.value = num.stars;
    U.glow.value = num.glow;
    renderer.toneMappingExposure = num.exposure;
    bloomPass.strength.value = num.bloom;

    // Light: the sun by day, the moon by night; they swap at −6° where both are dark.
    if (elevDeg > -6) {
      dirFrom(az, Math.max(elev, -0.02), sun);
      U.lightColor.value.lerpColors(ca.light, cb.light, s);
      U.lightI.value = num.lightI * THREE.MathUtils.smoothstep(elevDeg, -6, 0.5);
      U.diskI.value = Math.max(num.disk, 0);
    } else {
      const moonElev = Math.min(-elev * 0.75, THREE.MathUtils.degToRad(48));
      dirFrom(az - Math.PI, moonElev, sun);
      const m = THREE.MathUtils.smoothstep(-elevDeg, 6, 14);
      U.lightColor.value.copy(MOON);
      U.lightI.value = 0.4 * m;
      U.diskI.value = 6 * m;
    }
    U.lightDir.value.copy(sun).normalize();
    tmp.copy(U.zenith.value).lerp(U.horizon.value, 0.45);
    U.ambient.value.copy(tmp);

    const hh = Math.floor(state.hour) % 24;
    const mm = Math.floor((state.hour % 1) * 60);
    const clock = `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
    ui.todOut.value = clock;
    ui.tod.setAttribute('aria-valuetext', `${clock}, ${phaseName(elevDeg)}`);
    describe();
  }

  function phaseName(elevDeg) {
    if (elevDeg < -12) return 'night';
    if (elevDeg < -2) return 'twilight';
    if (elevDeg < 9) return state.hour < 12 ? 'sunrise' : 'golden hour';
    return 'daylight';
  }

  let describeTimer = 0;
  function describe() {
    clearTimeout(describeTimer);
    describeTimer = setTimeout(() => {
      const { elevDeg } = sunState(state.hour);
      const text = `Open sea at ${ui.todOut.value}, ${phaseName(elevDeg)}, ${String(ui.seaOut.value).toLowerCase()} sea${state.drift ? ', drifting slowly' : ''}${state.paused ? ', paused' : ''}.`;
      renderer.domElement.setAttribute('aria-label', text);
      ui.desc.textContent = text;
    }, 400);
  }

  // --------------------------------------------------------------- UI wiring
  ui.form.addEventListener('submit', (e) => e.preventDefault());
  ui.sea.value = String(Math.round(state.sea * 100));
  ui.tod.value = String(state.hour);
  ui.drift.checked = state.drift;
  ui.sea.addEventListener('input', () => { state.sea = Number(ui.sea.value) / 100; applySea(); });
  ui.tod.addEventListener('input', () => { state.hour = Number(ui.tod.value); applyTime(); });
  ui.drift.addEventListener('change', () => { state.drift = ui.drift.checked; describe(); });
  ui.resetView.addEventListener('click', resetView);
  const setPaused = (p) => {
    state.paused = p;
    ui.pause.textContent = p ? 'Play' : 'Pause';
    ui.pause.setAttribute('aria-pressed', String(p));
    describe();
  };
  ui.pause.addEventListener('click', () => setPaused(!state.paused));
  const setHud = (open) => {
    ui.hud.dataset.collapsed = String(!open);
    ui.toggleHud.textContent = open ? 'Hide' : 'Show';
    ui.toggleHud.setAttribute('aria-expanded', String(open));
  };
  ui.toggleHud.addEventListener('click', () => setHud(ui.hud.dataset.collapsed === 'true'));
  // Small or short screens start with the panel folded so the sea stays visible and touchable.
  if (window.innerHeight < 560 || window.innerWidth < 520) setHud(false);
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLButtonElement || e.altKey || e.ctrlKey || e.metaKey) return;
    if (e.code === 'Space') { e.preventDefault(); setPaused(!state.paused); }
    else if (e.key === 'h' || e.key === 'H') { setHud(ui.hud.dataset.collapsed === 'true'); }
  });
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) { state.drift = false; ui.drift.checked = false; describe(); } });

  // --------------------------------------------------------------- resize
  function resize() {
    const w = Math.max(1, window.innerWidth), h = Math.max(1, window.innerHeight);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener('resize', resize);

  // --------------------------------------------------------------- adaptive quality
  const perf = { frames: 0, acc: 0, window: 0, fpsFrames: 0, fpsTime: 0, lastChange: 0 };
  function setQualityLabel() {
    ui.quality.textContent = `WebGPU · ${state.tier === 'high' ? 'high' : 'reduced'} mesh · ${state.dpr.toFixed(2)}× resolution`;
  }
  function adapt(now) {
    if (now - perf.lastChange < 2500 || perf.frames < 30) return;
    const avg = perf.acc / perf.frames;
    perf.frames = 0; perf.acc = 0;
    if (avg > 26) {
      if (state.dpr > 0.75) {
        state.dpr = Math.max(0.75, +(state.dpr - 0.25).toFixed(2));
        renderer.setPixelRatio(state.dpr);
      } else if (state.tier === 'high') {
        state.tier = 'low';
        const next = createSeaGeometry(GEO_SEGMENTS.low);
        water.geometry.dispose();
        water.geometry = next.geo;
        sea = next;
      } else return;
      perf.lastChange = now;
      setQualityLabel();
    } else if (avg < 12 && state.dpr < DPR_CAP) {
      state.dpr = Math.min(DPR_CAP, +(state.dpr + 0.25).toFixed(2));
      renderer.setPixelRatio(state.dpr);
      perf.lastChange = now;
      setQualityLabel();
    }
  }

  // --------------------------------------------------------------- frame loop
  const timer = new THREE.Timer();
  const driftVel = new THREE.Vector3();
  let driftHeading = THREE.MathUtils.degToRad(WIND_DEG + 180);
  const driftStep = new THREE.Vector3();
  let running = false;

  function frame(timestamp) {
    timer.update(timestamp);
    const rawDt = timer.getDelta();
    const dt = Math.min(rawDt, 1 / 20); // simulation step: no jumps after stalls
    const ms = rawDt * 1000;            // FPS and quality use the real frame time

    if (!state.paused) {
      state.simTime += dt;
      U.time.value = state.simTime;

      // Drift: the viewpoint floats slowly downwind, heading wandering a little.
      if (state.drift) {
        driftHeading += Math.sin(state.simTime * 0.05) * 0.02 * dt;
        driftVel.set(Math.cos(driftHeading), 0, Math.sin(driftHeading)).multiplyScalar(1.1);
        driftStep.copy(driftVel).multiplyScalar(dt);
        anchor.add(driftStep);
        camera.position.add(driftStep);
        controls.target.add(driftStep);
      }
    }

    // Float on the swell: the target rides the analytic wave height at the anchor.
    const amp = U.amp.value;
    const bob = waveHeight(anchor.x, anchor.z, state.simTime, amp) * 0.55;
    const targetY = EYE_HEIGHT + bob;
    const dy = targetY - controls.target.y;
    controls.target.y += dy;
    camera.position.y += dy;
    // Keep the eye above the highest possible crest.
    const minEye = targetY + 1.5 + amp * 1.2;
    if (camera.position.y < minEye) camera.position.y = minEye;

    controls.update(dt);

    // Grid follows the camera, snapped to the core cell so near vertices never swim.
    const snap = sea.snap;
    const ox = Math.round(camera.position.x / snap) * snap;
    const oz = Math.round(camera.position.z / snap) * snap;
    water.position.set(ox, 0, oz);
    U.origin.value.set(ox, oz);
    skyDome.position.copy(camera.position);

    pipeline.render();

    // FPS and adaptive quality
    perf.frames++; perf.acc += ms;
    perf.fpsFrames++; perf.fpsTime += ms;
    if (perf.fpsTime >= 500) {
      ui.fps.value = String(Math.round((perf.fpsFrames * 1000) / perf.fpsTime));
      perf.fpsFrames = 0; perf.fpsTime = 0;
    }
    adapt(timestamp);
  }

  function startLoop() {
    if (running) return;
    running = true;
    timer.reset();
    perf.frames = 0; perf.acc = 0; perf.lastChange = performance.now();
    renderer.setAnimationLoop(frame);
  }
  function stopLoop() {
    running = false;
    renderer.setAnimationLoop(null);
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopLoop();
    else startLoop();
  });

  // --------------------------------------------------------------- go
  applySea();
  applyTime();
  resetView();
  setQualityLabel();

  ui.loadingText.textContent = 'Compiling shaders…';
  try {
    await renderer.compileAsync(scene, camera);
    pipeline.render(); // builds the post-processing graph and the first frame
  } catch (e) {
    fail('Shader compilation failed', String(e && e.message ? e.message : e));
    return;
  }

  ui.loading.hidden = true;
  ui.hud.hidden = false;
  if (!document.hidden) startLoop();
}

start().catch((e) => {
  fail('Something went wrong', String(e && e.message ? e.message : e));
});
