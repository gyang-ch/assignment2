import * as THREE from 'three';
import { AZURE_DATA_URL, AZURE_SAS_TOKEN } from '../util/azure-config.js';

const DATA_URL = `${AZURE_DATA_URL}/style_entropy.json?${AZURE_SAS_TOKEN}`;

function isValidHexColor(value) {
  return typeof value === 'string' && /^#([0-9a-f]{6})$/i.test(value);
}

export function createStyleEntropyThreeFlowFieldSketch(containerElement) {
  containerElement.style.cssText =
    'position:relative; width:100%; height:100%; min-height:400px; overflow:hidden; background:#0a0a0a;';

  const loader = document.createElement('div');
  loader.style.cssText =
    'position:absolute; inset:0; display:flex; align-items:center; justify-content:center;' +
    'font-family:"SF Mono",ui-monospace,monospace; font-size:11px;' +
    'color:rgba(255,255,255,0.18); text-align:center; line-height:1.9; pointer-events:none;';
  loader.innerHTML =
    'loading style&#x2011;entropy flow field 3d…<br>' +
    '<span style="font-size:9px;color:rgba(255,255,255,0.08);">using the same dataset as the grid view</span>';
  containerElement.appendChild(loader);

  let cleanupFn = null;
  let aborted = false;

  fetch(DATA_URL)
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (aborted) return;
      loader.remove();
      cleanupFn = createThreeFlowField(containerElement, data);
    })
    .catch((error) => {
      loader.innerHTML =
        `<span style="color:#c44;">failed to load style-entropy data<br>` +
        `<span style="font-size:9px;color:#444;">${error.message}</span></span>`;
    });

  return {
    cleanup() {
      aborted = true;
      loader.remove();
      cleanupFn?.();
    },
    remove() {
      aborted = true;
      loader.remove();
      cleanupFn?.();
    },
  };
}

// Builds a 3D curl-noise flow field where each artwork is one coloured streamline
function createThreeFlowField(containerElement, data) {
  const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  const width = Math.max(containerElement.clientWidth, 320);
  const height = Math.max(containerElement.clientHeight, 400);

  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
  camera.position.z = 80;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  containerElement.appendChild(renderer.domElement);

  const group = new THREE.Group();
  scene.add(group);

  function getCurl(x, y, z) {
    return new THREE.Vector3(
      Math.sin(y * 0.1) - Math.cos(z * 0.1),
      Math.sin(z * 0.1) - Math.cos(x * 0.1),
      Math.sin(x * 0.1) - Math.cos(y * 0.1)
    ).normalize();
  }

  const materialCache = {};
  const geometries = [];
  const lines = [];

  data.forEach((art) => {
    const cluster = art.cluster ?? 0;
    const complexity = art.complexity || 0.5;
    const colorKey = isValidHexColor(art.dominant_color) ? art.dominant_color : '#ffffff';

    const startX = (Math.random() - 0.5) * 80 + (cluster * 1.5);
    const startY = (Math.random() - 0.5) * 80;
    const startZ = (Math.random() - 0.5) * 80;

    const currentPos = new THREE.Vector3(startX, startY, startZ);
    const points = [currentPos.clone()];

    const steps = Math.floor(20 + complexity * 80);
    const stepSize = 1.0;

    for (let i = 0; i < steps; i++) {
      const dir = getCurl(currentPos.x, currentPos.y, currentPos.z);
      currentPos.addScaledVector(dir, stepSize);
      points.push(currentPos.clone());
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    geometries.push(geometry);

    if (!materialCache[colorKey]) {
      materialCache[colorKey] = new THREE.LineBasicMaterial({
        color: colorKey,
        transparent: true,
        opacity: 0.2,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
    }

    const line = new THREE.Line(geometry, materialCache[colorKey]);
    group.add(line);
    lines.push(line);
  });

  let animationFrameId = null;

  const animate = () => {
    animationFrameId = requestAnimationFrame(animate);
    group.rotation.y += prefersReducedMotion ? 0.0006 : 0.0015;
    group.rotation.x += prefersReducedMotion ? 0.0002 : 0.0005;
    renderer.render(scene, camera);
  };
  animate();

  const handleResize = () => {
    const nextWidth = Math.max(containerElement.clientWidth, 320);
    const nextHeight = Math.max(containerElement.clientHeight, 400);
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(nextWidth, nextHeight);
  };
  window.addEventListener('resize', handleResize);

  return () => {
    cancelAnimationFrame(animationFrameId);
    window.removeEventListener('resize', handleResize);

    lines.forEach((line) => group.remove(line));
    geometries.forEach((geometry) => geometry.dispose());
    Object.values(materialCache).forEach((material) => material.dispose());

    renderer.dispose();
    renderer.domElement.remove();
  };
}
