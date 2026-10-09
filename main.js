import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const STEPS = [
  {
    title: "Representation",
    caption:
      "Fig. 1. Continuous density field (left) versus explicit anisotropic Gaussians (right). Didactic fake — not a trained model.",
    html: `<p>A NeRF stores the scene as a <strong>continuous neural field</strong>. Ask “what is at <em>(x, y, z)</em>?” and the network returns density <em>σ</em> and colour <em>c</em>.</p>
<p>3D Gaussian Splatting stores the scene as a list of <strong>anisotropic ellipsoids</strong> — explicit primitives you can draw directly. Orbit either panel; both cameras stay locked together.</p>`,
  },
  {
    title: "Rendering",
    caption:
      "Fig. 2. NeRF integrates samples along a ray. 3DGS projects (“splats”) Gaussians onto an image plane and composites them.",
    html: `<p>To colour a pixel, NeRF <strong>marches a ray</strong> through the volume, querying many points and integrating density and colour (volume rendering).</p>
<p>3DGS instead <strong>rasterises</strong> each Gaussian as a soft ellipse on the image — “splatting” — then blends them front-to-back. No neural network at render time.</p>`,
  },
  {
    title: "View dependence",
    caption:
      "Fig. 3. Appearance can change with viewing direction. Move θ and φ; both panels update colour without moving the geometry.",
    html: `<p>In NeRF, viewing direction <em>(θ, φ)</em> is an input to the field, so highlights can shift as you look around.</p>
<p>In 3DGS, each Gaussian can carry a compact <strong>spherical-harmonic</strong> appearance model — a cheap way to tint colour by direction without a deep network. Geometry stays fixed; only colour changes.</p>`,
  },
  {
    title: "Speed and quality",
    caption:
      "Fig. 4. Classical NeRF optimisation is slow; once Gaussians are fitted, novel views can be rasterised in real time.",
    html: `<p>Training a classical NeRF means millions of network evaluations — the progress bar is a didactic stand-in for that slow optimisation.</p>
<p>3DGS optimises explicit Gaussians and then draws them with a graphics pipeline, so <strong>novel views update instantly</strong> as you orbit. Quality depends on how well the primitives cover the scene.</p>`,
  },
  {
    title: "Free compare",
    caption:
      "Fig. 5. Explore both representations side by side. Shared camera; optional θ / φ for view-dependent colour.",
    html: `<p>Drag either panel to orbit. Use θ and φ to see view-dependent colour on both sides at once.</p>
<p>Take-away: NeRF = continuous field queried along rays; 3DGS = explicit Gaussians drawn as splats. Both can be view-dependent; 3DGS is built for real-time novel views.</p>`,
  },
];

const OBJECT_RADIUS = 1.0;
const SAMPLE = new THREE.Vector3(0.55, 0.42, 0.48).normalize().multiplyScalar(OBJECT_RADIUS * 0.92);
const GRID_Y = -OBJECT_RADIUS - 0.02;
const CLEAR = 0xeef1f5;

const thetaInput = document.getElementById("theta");
const phiInput = document.getElementById("phi");
const thetaOut = document.getElementById("theta-out");
const phiOut = document.getElementById("phi-out");
const controlsEl = document.getElementById("controls");
const trainPanel = document.getElementById("train-panel");
const modeBadge = document.getElementById("mode-badge");

let step = 0;
let trainProgress = 0;
let trainActive = false;
let syncing = false;
const sharedPose = {
  position: new THREE.Vector3(3.2, 1.9, 3.45),
  target: new THREE.Vector3(0, 0.12, 0),
};

function viewDirection(thetaDeg, phiDeg) {
  const th = THREE.MathUtils.degToRad(thetaDeg);
  const ph = THREE.MathUtils.degToRad(phiDeg);
  return new THREE.Vector3(
    Math.sin(th) * Math.cos(ph),
    Math.cos(th),
    Math.sin(th) * Math.sin(ph)
  ).normalize();
}

/** Didactic density blob (continuous field). */
function densityAt(pos) {
  const r2 = pos.lengthSq();
  return Math.exp(-r2 / (2 * 0.68 * 0.68));
}

/** Fake NeRF colour with view-dependent specular. */
function nerfColour(pos, dir) {
  const n = pos.clone().normalize();
  const light = new THREE.Vector3(0.55, 0.82, 0.28).normalize();
  const diffuse = Math.max(0, n.dot(light));
  const reflect = light.clone().negate().reflect(n);
  const spec = Math.pow(Math.max(0, reflect.dot(dir)), 26);
  const base = [0.52, 0.3, 0.22];
  return [
    THREE.MathUtils.clamp(base[0] * (0.3 + 0.7 * diffuse) + spec * 0.95, 0, 1),
    THREE.MathUtils.clamp(base[1] * (0.3 + 0.7 * diffuse) + spec * 0.9, 0, 1),
    THREE.MathUtils.clamp(base[2] * (0.3 + 0.7 * diffuse) + spec * 0.85, 0, 1),
  ];
}

/** Fake degree-1 SH appearance for a Gaussian. */
function shColour(base, dir, coeffs) {
  const y0 = 0.282095;
  const y1 = 0.488603 * dir.y;
  const y2 = 0.488603 * dir.z;
  const y3 = 0.488603 * dir.x;
  const s = y0 * coeffs[0] + y1 * coeffs[1] + y2 * coeffs[2] + y3 * coeffs[3];
  return [
    THREE.MathUtils.clamp(base[0] * (0.55 + 0.55 * s), 0, 1),
    THREE.MathUtils.clamp(base[1] * (0.55 + 0.5 * s), 0, 1),
    THREE.MathUtils.clamp(base[2] * (0.55 + 0.45 * s), 0, 1),
  ];
}

function hexColor(c) {
  const r = Math.round(c[0] * 255);
  const g = Math.round(c[1] * 255);
  const b = Math.round(c[2] * 255);
  return { r, g, b, css: `rgb(${r}, ${g}, ${b})` };
}

function addLights(scene) {
  scene.add(new THREE.AmbientLight(0xf2efe8, 0.72));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(2.4, 3.2, 1.6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xc5d4e8, 0.35);
  fill.position.set(-2.2, 0.4, -1.4);
  scene.add(fill);
}

function addGrid(scene) {
  const grid = new THREE.GridHelper(6, 12, 0xc5ccd6, 0xdde2e9);
  grid.position.y = GRID_Y;
  scene.add(grid);
  return grid;
}

function createViewport(canvasId) {
  const canvas = document.getElementById(canvasId);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(CLEAR, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.08, 40);
  camera.position.copy(sharedPose.position);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.target.copy(sharedPose.target);
  controls.minDistance = 2.0;
  controls.maxDistance = 9;

  addLights(scene);
  addGrid(scene);

  return { canvas, renderer, scene, camera, controls };
}

let activeVp = null;

function syncFrom(source) {
  if (syncing || !nerf || !gs) return;
  syncing = true;
  sharedPose.position.copy(source.camera.position);
  sharedPose.target.copy(source.controls.target);

  for (const vp of [nerf, gs]) {
    if (vp === source) continue;
    vp.camera.position.copy(sharedPose.position);
    vp.controls.target.copy(sharedPose.target);
    vp.camera.lookAt(sharedPose.target);
  }
  syncing = false;
}

function wireSharedOrbit(vp) {
  vp.controls.addEventListener("change", () => {
    activeVp = vp;
    syncFrom(vp);
  });
  vp.canvas.addEventListener("pointerdown", () => {
    activeVp = vp;
  });
}

// ——— NeRF scene ———
const nerf = createViewport("viewport-nerf");

const nerfFieldGroup = new THREE.Group();
nerf.scene.add(nerfFieldGroup);

const fieldMat = new THREE.MeshStandardMaterial({
  color: 0x8a4a32,
  metalness: 0.35,
  roughness: 0.45,
  transparent: true,
  opacity: 0.55,
  depthWrite: false,
});
const fieldMesh = new THREE.Mesh(new THREE.SphereGeometry(OBJECT_RADIUS, 48, 36), fieldMat);
nerfFieldGroup.add(fieldMesh);

// Soft volume shells to suggest continuous density
for (const [scale, opac] of [
  [1.25, 0.12],
  [1.45, 0.07],
]) {
  const shell = new THREE.Mesh(
    new THREE.SphereGeometry(OBJECT_RADIUS * scale, 32, 24),
    new THREE.MeshBasicMaterial({
      color: 0x2c4a6e,
      transparent: true,
      opacity: opac,
      depthWrite: false,
      side: THREE.BackSide,
    })
  );
  nerfFieldGroup.add(shell);
}

const queryPoint = new THREE.Mesh(
  new THREE.SphereGeometry(0.065, 18, 14),
  new THREE.MeshStandardMaterial({
    color: 0x1d3557,
    emissive: 0x2c4a6e,
    emissiveIntensity: 0.9,
    depthTest: false,
  })
);
queryPoint.renderOrder = 5;
queryPoint.position.copy(SAMPLE);
nerf.scene.add(queryPoint);

const dirArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  SAMPLE.clone(),
  0.95,
  0xc45c26,
  0.16,
  0.1
);
dirArrow.line.material.depthTest = false;
dirArrow.cone.material.depthTest = false;
dirArrow.line.renderOrder = 6;
dirArrow.cone.renderOrder = 6;
nerf.scene.add(dirArrow);

// Ray + sample points for volume rendering step
const rayGroup = new THREE.Group();
nerf.scene.add(rayGroup);

const rayOrigin = new THREE.Vector3(2.4, 1.35, 2.55);
const rayDir = SAMPLE.clone().sub(rayOrigin).normalize();
const rayEnd = rayOrigin.clone().add(rayDir.clone().multiplyScalar(4.2));

const rayGeom = new THREE.BufferGeometry().setFromPoints([rayOrigin, rayEnd]);
const rayLine = new THREE.Line(
  rayGeom,
  new THREE.LineBasicMaterial({ color: 0xc45c26, transparent: true, opacity: 0.95 })
);
rayGroup.add(rayLine);

const samplePoints = [];
const N_SAMPLES = 12;
for (let i = 0; i < N_SAMPLES; i++) {
  const t = 0.35 + (i / (N_SAMPLES - 1)) * 2.4;
  const p = rayOrigin.clone().add(rayDir.clone().multiplyScalar(t));
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 12, 10),
    new THREE.MeshStandardMaterial({
      color: 0x2c4a6e,
      emissive: 0x2c4a6e,
      emissiveIntensity: 0.4,
      transparent: true,
      opacity: 0.9,
    })
  );
  m.position.copy(p);
  rayGroup.add(m);
  samplePoints.push({ mesh: m, pos: p, t });
}

const camGizmo = new THREE.Group();
nerf.scene.add(camGizmo);
const camBody = new THREE.Mesh(
  new THREE.BoxGeometry(0.2, 0.14, 0.26),
  new THREE.MeshStandardMaterial({ color: 0x2c4a6e, metalness: 0.2, roughness: 0.5 })
);
camGizmo.add(camBody);
camGizmo.position.copy(rayOrigin);
camGizmo.lookAt(SAMPLE);

// ——— 3DGS scene ———
const gs = createViewport("viewport-gs");
wireSharedOrbit(nerf);
wireSharedOrbit(gs);

const gsGroup = new THREE.Group();
gs.scene.add(gsGroup);

const gaussianMeshes = [];
const GAUSSIAN_SEEDS = [
  [0, 0.05, 0, 1.05, 0.85, 0.95],
  [0.55, 0.35, 0.25, 0.55, 0.35, 0.42],
  [-0.5, 0.28, 0.3, 0.48, 0.38, 0.4],
  [0.2, -0.35, 0.55, 0.42, 0.32, 0.5],
  [-0.25, -0.2, -0.55, 0.45, 0.4, 0.38],
  [0.65, -0.15, -0.2, 0.38, 0.48, 0.35],
  [-0.6, -0.1, -0.15, 0.4, 0.35, 0.45],
  [0.15, 0.65, -0.15, 0.36, 0.5, 0.34],
  [-0.1, 0.55, 0.45, 0.4, 0.32, 0.38],
  [0.4, 0.1, -0.6, 0.35, 0.4, 0.42],
  [-0.35, 0.4, -0.4, 0.38, 0.36, 0.4],
  [0.0, -0.55, 0.15, 0.5, 0.34, 0.4],
  [0.48, 0.48, 0.05, 0.32, 0.38, 0.3],
  [-0.45, -0.45, 0.25, 0.34, 0.36, 0.4],
  [0.25, -0.05, 0.7, 0.36, 0.34, 0.32],
  [-0.15, 0.15, -0.72, 0.34, 0.36, 0.3],
  [0.7, 0.2, 0.35, 0.3, 0.34, 0.36],
  [-0.68, 0.25, 0.2, 0.32, 0.3, 0.34],
  [0.35, -0.55, -0.35, 0.34, 0.3, 0.36],
  [-0.3, -0.5, -0.4, 0.32, 0.34, 0.3],
  [0.1, 0.75, 0.25, 0.28, 0.36, 0.28],
  [0.55, -0.4, 0.4, 0.3, 0.32, 0.34],
  [-0.55, 0.05, 0.55, 0.32, 0.3, 0.3],
  [0.0, 0.0, 0.0, 0.7, 0.55, 0.6],
];

const baseRgb = [0.52, 0.3, 0.22];
for (const [x, y, z, sx, sy, sz] of GAUSSIAN_SEEDS) {
  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(baseRgb[0], baseRgb[1], baseRgb[2]),
    metalness: 0.15,
    roughness: 0.55,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 16), mat);
  mesh.position.set(x, y, z);
  mesh.scale.set(sx, sy, sz);
  mesh.rotation.set(x * 0.8, y * 1.1, z * 0.6);
  const coeffs = [
    0.85 + Math.abs(x) * 0.2,
    0.35 * Math.sign(y || 1),
    0.25 * Math.sign(z || 1),
    0.2 * Math.sign(x || 1),
  ];
  gsGroup.add(mesh);
  gaussianMeshes.push({ mesh, mat, coeffs, base: [...baseRgb] });
}

// Image plane + projected soft discs for splat step
const splatGroup = new THREE.Group();
gs.scene.add(splatGroup);

const imagePlane = new THREE.Mesh(
  new THREE.PlaneGeometry(1.6, 1.15),
  new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.92,
    side: THREE.DoubleSide,
  })
);
imagePlane.position.set(2.15, 0.55, 2.35);
imagePlane.lookAt(0, 0.1, 0);
splatGroup.add(imagePlane);

const planeFrame = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.PlaneGeometry(1.6, 1.15)),
  new THREE.LineBasicMaterial({ color: 0x2c4a6e })
);
planeFrame.position.copy(imagePlane.position);
planeFrame.quaternion.copy(imagePlane.quaternion);
splatGroup.add(planeFrame);

const splatDiscs = [];
const discOffsets = [
  [0, 0, 0.55, 0.4],
  [-0.35, 0.15, 0.4, 0.32],
  [0.3, -0.2, 0.38, 0.45],
  [0.15, 0.28, 0.42, 0.3],
  [-0.2, -0.25, 0.35, 0.38],
  [0.4, 0.1, 0.3, 0.28],
  [-0.4, 0.0, 0.32, 0.35],
  [0.05, -0.35, 0.36, 0.3],
];

for (const [ox, oy, rx, ry] of discOffsets) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const grd = ctx.createRadialGradient(64, 64, 4, 64, 64, 60);
  grd.addColorStop(0, "rgba(138, 74, 50, 0.85)");
  grd.addColorStop(0.45, "rgba(138, 74, 50, 0.35)");
  grd.addColorStop(1, "rgba(138, 74, 50, 0)");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const disc = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
  disc.scale.set(rx * 1.1, ry * 1.1, 1);
  const local = new THREE.Vector3(ox, oy, 0.02);
  disc.userData.local = local;
  splatGroup.add(disc);
  splatDiscs.push(disc);
}

function placeDiscsOnPlane() {
  for (const disc of splatDiscs) {
    const world = disc.userData.local.clone().applyQuaternion(imagePlane.quaternion);
    disc.position.copy(imagePlane.position).add(world);
    disc.quaternion.copy(imagePlane.quaternion);
  }
}
placeDiscsOnPlane();

const gsCamGizmo = camGizmo.clone(true);
gs.scene.add(gsCamGizmo);
gsCamGizmo.position.copy(rayOrigin);
gsCamGizmo.lookAt(SAMPLE);

// ——— Step visibility ———
function applyStepVisibility() {
  const showRay = step === 1 || step === 4;
  const showQuery = step === 0 || step === 2 || step === 4;
  const showDir = step === 2 || step === 4;
  const showSplats = step === 1 || step === 4;
  const showTrain = step === 3;

  rayGroup.visible = showRay;
  camGizmo.visible = showRay || step === 3;
  queryPoint.visible = showQuery;
  dirArrow.visible = showDir;

  fieldMat.opacity = step === 0 ? 0.5 : step === 1 ? 0.35 : 0.55;
  nerfFieldGroup.visible = true;

  // During speed step, dim NeRF until “trained”
  const trainOpacity = showTrain ? 0.15 + trainProgress * 0.5 : fieldMat.opacity;
  if (showTrain) fieldMat.opacity = trainOpacity;

  splatGroup.visible = showSplats;
  gsCamGizmo.visible = showRay || step === 3;

  // Emphasise Gaussians on representation / free compare
  for (const g of gaussianMeshes) {
    g.mat.opacity = showSplats ? 0.35 : 0.55;
    if (showTrain) g.mat.opacity = 0.55;
  }

  // Slightly ghost Gaussians when showing image-plane splats
  gsGroup.visible = true;

  controlsEl.classList.toggle("is-hidden", !(step === 2 || step === 4));
  trainPanel.classList.toggle("is-hidden", !showTrain);

  const isCompare = step === 4;
  modeBadge.textContent = isCompare ? "Compare" : "Guided";
  modeBadge.classList.toggle("is-compare", isCompare);

  trainActive = showTrain;
  if (!showTrain) {
    trainProgress = 0;
    document.getElementById("train-fill").style.width = "0%";
    document.getElementById("train-label").textContent = "0%";
  }
}

function integrateRay(dir) {
  let T = 1;
  let r = 0;
  let g = 0;
  let b = 0;
  let alpha = 0;

  for (const s of samplePoints) {
    const sigma = densityAt(s.pos);
    const c = nerfColour(s.pos, dir);
    const a = 1 - Math.exp(-sigma * 0.35);
    const w = T * a;
    r += w * c[0];
    g += w * c[1];
    b += w * c[2];
    alpha += w;
    T *= 1 - a;

    const pulse = 0.55 + 0.45 * a;
    s.mesh.material.emissiveIntensity = pulse;
    s.mesh.scale.setScalar(0.7 + a * 0.9);
    s.mesh.material.color.setRGB(c[0], c[1], c[2]);
  }

  return {
    c: [THREE.MathUtils.clamp(r, 0, 1), THREE.MathUtils.clamp(g, 0, 1), THREE.MathUtils.clamp(b, 0, 1)],
    alpha: THREE.MathUtils.clamp(alpha, 0, 1),
  };
}

function updateAppearance() {
  const theta = Number(thetaInput.value);
  const phi = Number(phiInput.value);
  thetaOut.textContent = `${theta}°`;
  phiOut.textContent = `${phi}°`;

  const dir = viewDirection(theta, phi);
  dirArrow.setDirection(dir);
  dirArrow.position.copy(SAMPLE);

  const { c, alpha } = integrateRay(dir);
  const nerfRgb = hexColor(c);
  document.getElementById("alpha-fill").style.width = `${Math.round(alpha * 100)}%`;
  document.getElementById("hud-alpha").textContent = alpha.toFixed(2);
  document.getElementById("swatch-nerf").style.background = nerfRgb.css;
  document.getElementById("hud-nerf-rgb").textContent = `rgb ${nerfRgb.r}, ${nerfRgb.g}, ${nerfRgb.b}`;

  // Keep the blob readable; blend a warm base with the integrated view colour
  fieldMat.color.setRGB(
    0.35 * 0.55 + 0.65 * c[0],
    0.22 * 0.55 + 0.65 * c[1],
    0.16 * 0.55 + 0.65 * c[2]
  );
  fieldMat.emissive.setRGB(c[0] * 0.08, c[1] * 0.05, c[2] * 0.03);

  // Average GS appearance (first few Gaussians for HUD)
  let gr = 0;
  let gg = 0;
  let gb = 0;
  for (const g of gaussianMeshes) {
    const col = shColour(g.base, dir, g.coeffs);
    g.mat.color.setRGB(col[0], col[1], col[2]);
    gr += col[0];
    gg += col[1];
    gb += col[2];
  }
  const n = gaussianMeshes.length;
  const gsRgb = hexColor([gr / n, gg / n, gb / n]);
  document.getElementById("swatch-gs").style.background = gsRgb.css;
  document.getElementById("hud-gs-rgb").textContent = `rgb ${gsRgb.r}, ${gsRgb.g}, ${gsRgb.b}`;

  // Tint splat discs with GS colour
  for (const disc of splatDiscs) {
    disc.material.color.setRGB(gsRgb.r / 255, gsRgb.g / 255, gsRgb.b / 255);
  }

  const note =
    step === 1
      ? "Ray samples accumulate opacity α; Gaussians project as soft discs."
      : step === 2
        ? "Direction changes colour; geometry stays put."
        : step === 3
          ? "Watch the slow NeRF “train” bar while 3DGS stays interactive."
          : "Shared camera — orbit either side.";
  document.getElementById("hud-note").textContent = note;
}

function renderCopy() {
  document.getElementById("step-index").textContent = `Step ${step + 1} of ${STEPS.length}`;
  document.getElementById("step-title").textContent = STEPS[step].title;
  document.getElementById("step-body").innerHTML = STEPS[step].html;
  document.getElementById("fig-caption").textContent = STEPS[step].caption;
  document.getElementById("btn-back").disabled = step === 0;
  document.getElementById("btn-next").textContent = step === STEPS.length - 1 ? "Done" : "Next";
  document.getElementById("btn-next").disabled = step === STEPS.length - 1;

  document.querySelectorAll(".dot").forEach((dot, i) => {
    const on = i === step;
    dot.classList.toggle("is-active", on);
    if (on) dot.setAttribute("aria-current", "true");
    else dot.removeAttribute("aria-current");
  });

  applyStepVisibility();
  updateAppearance();
}

function setStep(next) {
  step = THREE.MathUtils.clamp(next, 0, STEPS.length - 1);
  renderCopy();
}

function resizeViewport(vp) {
  const w = vp.canvas.clientWidth;
  const h = vp.canvas.clientHeight;
  if (vp.canvas.width !== w || vp.canvas.height !== h) {
    vp.renderer.setSize(w, h, false);
    vp.camera.aspect = w / Math.max(h, 1);
    vp.camera.updateProjectionMatrix();
  }
}

let lastTrainTick = 0;

function tick(now) {
  resizeViewport(nerf);
  resizeViewport(gs);

  nerf.controls.update();
  gs.controls.update();

  // Keep both viewports locked while damping settles after a drag
  if (activeVp && !syncing) syncFrom(activeVp);

  if (trainActive) {
    if (!lastTrainTick) lastTrainTick = now;
    const dt = Math.min(0.05, (now - lastTrainTick) / 1000);
    lastTrainTick = now;
    // Slow crawl to ~100% over ~12s
    trainProgress = Math.min(1, trainProgress + dt * 0.08);
    const pct = Math.round(trainProgress * 100);
    document.getElementById("train-fill").style.width = `${pct}%`;
    document.getElementById("train-label").textContent = `${pct}%`;
    fieldMat.opacity = 0.15 + trainProgress * 0.5;
    // Subtle sample shimmer as “training”
    for (const s of samplePoints) {
      s.mesh.material.emissiveIntensity = 0.2 + trainProgress * 0.8;
    }
  } else {
    lastTrainTick = 0;
  }

  // Animate sample pulse along ray when rendering step visible
  if (rayGroup.visible && !trainActive) {
    const t = now * 0.001;
    for (let i = 0; i < samplePoints.length; i++) {
      const pulse = 0.85 + 0.15 * Math.sin(t * 3 + i * 0.55);
      samplePoints[i].mesh.scale.setScalar(pulse);
    }
  }

  nerf.renderer.render(nerf.scene, nerf.camera);
  gs.renderer.render(gs.scene, gs.camera);
  requestAnimationFrame(tick);
}

document.getElementById("btn-back").addEventListener("click", () => setStep(step - 1));
document.getElementById("btn-next").addEventListener("click", () => {
  if (step < STEPS.length - 1) setStep(step + 1);
});
document.querySelectorAll(".dot").forEach((dot) => {
  dot.addEventListener("click", () => setStep(Number(dot.dataset.step)));
});
thetaInput.addEventListener("input", updateAppearance);
phiInput.addEventListener("input", updateAppearance);

window.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const onSlider = e.target instanceof HTMLInputElement && e.target.type === "range";
  if (onSlider) return;
  e.preventDefault();
  setStep(step + (e.key === "ArrowRight" ? 1 : -1));
});

setStep(0);
requestAnimationFrame(tick);
