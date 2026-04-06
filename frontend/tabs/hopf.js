// Hopf: Hopf fibration rendered via Three.js.
// Move cursor to rotate. Click to cycle colour palette.

import * as THREE from 'three';

const hexToRgb = (hex) => {
  const c = hex.replace('#', '');
  return [
    parseInt(c.slice(0, 2), 16),
    parseInt(c.slice(2, 4), 16),
    parseInt(c.slice(4, 6), 16),
  ];
};

const rawColorSchemes = [
  { colors: ['#4B3A51', '#A77A4B', '#ECC6A2', '#A43020', '#722D24'] },
  { colors: ['#204035', '#4A7169', '#BEB59C', '#735231', '#49271B'] },
  { colors: ['#293757', '#568D4B', '#D5BB56', '#D26A1B', '#A41D1A'] },
  { colors: ['#27403D', '#48725C', '#212412', '#F3E4C2', '#D88F2E'] },
  { colors: ['#1E1D20', '#B66636', '#547A56', '#BDAE5B', '#515A7C'] },
  { colors: ['#202221', '#661E2A', '#AB381B', '#EAD4A3', '#E3DED8'] },
  { colors: ['#1F284C', '#2D4472', '#6E6352', '#D9CCAC', '#ECE2C6'] },
  { colors: ['#023059', '#459DBF', '#87BF60', '#D9D16A', '#F2F2F2'] },
  { colors: ['#121510', '#6D8325', '#D6CFB7', '#E5AD4F', '#BD5630'] },
  { colors: ['#333333', '#D1B817', '#2A2996', '#B34325', '#C8CCC6'] },
  { colors: ['#151817', '#001A56', '#197C3F', '#D4A821', '#C74C25'] },
  { colors: ['#0E2523', '#324028', '#C26B61', '#5A788D', '#DE7944'] },
  { colors: ['#1D2025', '#45312A', '#7E2F28', '#202938', '#D58E40'] },
  { colors: ['#184430', '#548150', '#DEB738', '#734321', '#852419'] },
  { colors: ['#442327', '#C0BC98', '#A6885D', '#8A3230', '#973B2B'] },
  { colors: ['#0E122D', '#182044', '#51628E', '#91A1BA', '#AFD0C9'] },
  { colors: ['#1A3431', '#2B41A7', '#6283C8', '#CCC776', '#C7AD24'] },
  { colors: ['#0C0B10', '#707DA6', '#CCAD9D', '#B08E4A', '#863B34'] },
  { colors: ['#0d0d0d', '#c73131', '#1b8be7', '#b7bcc4', '#f8e744'] }
];

const PALETTES = rawColorSchemes
  .filter((scheme) => scheme.colors.length >= 4)
  .map((scheme) => ({
    bg: [5, 5, 5],
    colors: [hexToRgb(scheme.colors[1]), hexToRgb(scheme.colors[2]), hexToRgb(scheme.colors[3])],
  }));

function normalizeColor(rgbArr) {
  return new THREE.Vector3(rgbArr[0] / 255, rgbArr[1] / 255, rgbArr[2] / 255);
}

const VERTEX_SHADER = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = `
precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_color1;
uniform vec3 u_color2;
uniform vec3 u_color3;
uniform vec3 u_bg;
uniform vec2 u_rotation;

#define MAX_STEPS 100
#define MAX_DIST 25.0
#define SURF_DIST 0.005

mat2 rot(float a) {
  float s = sin(a), c = cos(a);
  return mat2(c, -s, s, c);
}

float getDistance(vec3 p) {
  float bound = length(p) - 2.8;
  if (bound > 1.0) return bound;

  p.yz *= rot(u_rotation.x);
  p.xz *= rot(u_rotation.y);

  float r2 = dot(p, p);
  vec4 s3 = vec4(2.0 * p, r2 - 1.0) / (r2 + 1.0);

  vec3 s2 = vec3(
    2.0 * (s3.x * s3.z + s3.y * s3.w),
    2.0 * (s3.y * s3.z - s3.x * s3.w),
    s3.x * s3.x + s3.y * s3.y - s3.z * s3.z - s3.w * s3.w
  );

  float t = u_time * 0.5;
  float wave1 = cos(s2.z * 6.0 + atan(s2.y, s2.x) * 3.0 + t);
  float wave2 = cos(s2.x * 5.0 - atan(s2.y, s2.z) * 2.0 - t * 1.5);

  float d_s2 = min(abs(wave1), abs(wave2)) - 0.15;
  float d_r3 = d_s2 * (r2 + 1.0) * 0.2;

  return max(d_r3, bound);
}

vec3 getNormal(vec3 p) {
  float d = getDistance(p);
  vec2 e = vec2(0.002, 0.0);
  vec3 n = d - vec3(
    getDistance(p - e.xyy),
    getDistance(p - e.yxy),
    getDistance(p - e.yyx)
  );
  return normalize(n);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

  // Pulled camera back slightly to -7.5 to give a comfortable margin
  vec3 ro = vec3(0.0, 0.0, -7.5);
  vec3 rd = normalize(vec3(uv.x, uv.y, 1.0));

  float dO = 0.0;
  vec3 p;
  for (int i = 0; i < MAX_STEPS; i++) {
    p = ro + rd * dO;
    float dS = getDistance(p);
    dO += dS;
    if (dO > MAX_DIST || abs(dS) < SURF_DIST) break;
  }

  vec3 col = u_bg;

  if (dO < MAX_DIST) {
    vec3 n = getNormal(p);
    vec3 lightDir = normalize(vec3(1.0, 2.0, -1.0));

    float r2 = dot(p, p);
    vec4 s3 = vec4(2.0 * p, r2 - 1.0) / (r2 + 1.0);

    float mix1 = smoothstep(-1.0, 1.0, n.x);
    float mix2 = smoothstep(-1.0, 1.0, n.y);
    vec3 baseColor = mix(u_color1, u_color2, mix1);
    baseColor = mix(baseColor, u_color3, mix2);

    float dif = max(dot(n, lightDir), 0.0);
    float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 4.0);
    float sss = smoothstep(0.0, 0.15, getDistance(p + lightDir * 0.2)) * 0.8;

    float fiberPhase = atan(s3.y, s3.x) - atan(s3.w, s3.z);
    float flow = sin(fiberPhase * 15.0 - u_time * 8.0);
    float glowingLines = smoothstep(0.95, 1.0, flow) * 2.0;

    col = baseColor * (dif + 0.15)
      + (u_color1 * fresnel * 0.8)
      + (u_color2 * sss)
      + (u_color3 * glowingLines);
  } else {
    col -= length(uv) * 0.15;
  }

  gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
`;

export function createHopfArtifactThreeJSSketch(containerEl) {
  const SIZE = 800; // Fixed size, same as gyroid.js
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
  const TIME_SCALE = prefersReducedMotion ? 0.35 : 1.0;

  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  renderer.setSize(SIZE, SIZE, false);
  containerEl.appendChild(renderer.domElement);

  renderer.domElement.setAttribute('aria-label', 'Hopf Artifact. Move cursor to rotate. Click or press P to change palette.');
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.tabIndex = 0;

  const uniforms = {
    u_time: { value: 0 },
    u_resolution: { value: new THREE.Vector2(SIZE * renderer.getPixelRatio(), SIZE * renderer.getPixelRatio()) },
    u_rotation: { value: new THREE.Vector2(0, 0) },
    u_bg: { value: new THREE.Vector3() },
    u_color1: { value: new THREE.Vector3() },
    u_color2: { value: new THREE.Vector3() },
    u_color3: { value: new THREE.Vector3() },
  };

  const material = new THREE.ShaderMaterial({
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    uniforms,
    depthWrite: false,
    depthTest: false,
    transparent: true,
  });

  const geometry = new THREE.PlaneGeometry(2, 2);
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  let rotX = 0;
  let rotY = 0;
  let mx = 0;
  let my = 0;
  let rafId = 0;

  const clock = new THREE.Clock();

  function loadPalette() {
    const activePalette = PALETTES[Math.floor(Math.random() * PALETTES.length)];
    uniforms.u_bg.value.copy(normalizeColor(activePalette.bg));
    uniforms.u_color1.value.copy(normalizeColor(activePalette.colors[0]));
    uniforms.u_color2.value.copy(normalizeColor(activePalette.colors[1]));
    uniforms.u_color3.value.copy(normalizeColor(activePalette.colors[2]));
  }

  loadPalette();

  function onPointerMove(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    my = ((e.clientY - rect.top) / rect.height) * 2 - 1;
  }

  function onClick() {
    loadPalette();
  }

  function onKeyDown(e) {
    if (e.key === 'p' || e.key === 'P') {
      e.preventDefault();
      loadPalette();
    }
  }

  renderer.domElement.addEventListener('pointermove', onPointerMove, { passive: true });
  renderer.domElement.addEventListener('click', onClick);
  renderer.domElement.addEventListener('keydown', onKeyDown);

  function animate() {
    rafId = requestAnimationFrame(animate);

    const t = clock.getElapsedTime() * TIME_SCALE;
    uniforms.u_time.value = t;

    // Direct cursor mapping matching gyroid artifact behavior
    rotX = my * Math.PI + t * 0.1;
    rotY = mx * Math.PI + t * 0.05;

    uniforms.u_rotation.value.set(rotX, rotY);
    renderer.render(scene, camera);
  }

  animate();

  return {
    cleanup() {
      cancelAnimationFrame(rafId);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('click', onClick);
      renderer.domElement.removeEventListener('keydown', onKeyDown);

      try { containerEl.removeChild(renderer.domElement); } catch {}
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    }
  };
}
