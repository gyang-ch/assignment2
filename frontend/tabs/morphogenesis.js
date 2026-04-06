// Morphogenesis: reaction-diffusion surface rendered via p5.js.
// Move cursor to rotate. Click to cycle colour palette.

import p5 from 'p5';

const PALETTES = [
  { bg: [2, 5, 12], colors: [[0, 255, 180], [0, 100, 255], [200, 255, 255]] },
  { bg: [8, 2, 5], colors: [[255, 50, 100], [255, 150, 50], [255, 200, 200]] },
  { bg: [10, 8, 12], colors: [[255, 100, 200], [255, 200, 50], [100, 50, 255]] },
  { bg: [5, 5, 5], colors: [[100, 150, 255], [200, 200, 255], [255, 255, 255]] },
  { bg: [5, 5, 5], colors: [[167, 122, 75], [236, 198, 162], [164, 48, 32]] },
  { bg: [5, 5, 5], colors: [[74, 113, 105], [190, 181, 156], [115, 82, 49]] },
  { bg: [5, 5, 5], colors: [[86, 141, 75], [213, 187, 86], [210, 106, 27]] },
  { bg: [5, 5, 5], colors: [[182, 102, 54], [84, 122, 86], [189, 174, 91]] },
  { bg: [5, 5, 5], colors: [[45, 68, 114], [110, 99, 82], [217, 204, 172]] }
];

function normalizeColor(rgbArr) {
  return [rgbArr[0] / 255, rgbArr[1] / 255, rgbArr[2] / 255];
}

function getTargetSize(containerEl) {
  const w = Math.max(1, containerEl.clientWidth || 800);
  return Math.round(Math.min(900, Math.max(360, w)));
}

const VERTEX_SHADER = `
precision highp float;
attribute vec3 aPosition;
attribute vec2 aTexCoord;
varying vec2 vUv;

void main() {
  vUv = aTexCoord;
  gl_Position = vec4(aPosition, 1.0);
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

#define MAX_STEPS 150
#define MAX_DIST 25.0
#define SURF_DIST 0.005

mat2 rot(float a) {
  float s = sin(a), c = cos(a);
  return mat2(c, -s, s, c);
}

float getDistance(vec3 p) {
  float bound = length(p) - 3.8;
  if (bound > 1.0) return bound;

  p.yz *= rot(u_rotation.x);
  p.xz *= rot(u_rotation.y);

  vec3 q = p;
  float d = length(q * vec3(1.0, 1.2, 1.0)) - 2.0;

  float freq = 2.0;
  float amp = 0.6;

  for (int i = 0; i < 4; i++) {
    q.xy *= rot(1.23);
    q.yz *= rot(1.67);
    q.zx *= rot(0.89);

    float buckling = abs(sin(q.x * freq + sin(q.y * freq) + u_time * 0.3)) - 0.5;
    d += amp * buckling;

    amp *= 0.45;
    freq *= 2.0;
  }

  return max(d * 0.3, bound);
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

  vec3 ro = vec3(0.0, 0.0, -6.0);
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

    float distFromCore = length(p);
    float mix1 = smoothstep(1.5, 3.0, distFromCore);
    float mix2 = smoothstep(-1.0, 1.0, n.y);

    vec3 baseColor = mix(u_color1, u_color2, mix1);
    baseColor = mix(baseColor, u_color3, mix2 * 0.5);

    float dif = max(dot(n, lightDir), 0.0) * 0.8 + 0.2;
    float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    float sss = smoothstep(0.0, 0.25, getDistance(p + lightDir * 0.4)) * 0.9;

    col = baseColor * dif
      + (u_color1 * fresnel * 1.2)
      + (u_color2 * sss);
  } else {
    col -= length(uv) * 0.2;
  }

  gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
`;

export function createMorphogenesisArtifactP5Sketch(containerEl) {
  const cleanupFuncs = [];

  const myP5 = new p5((p) => {
    let theShader;
    let activePalette = PALETTES[0];
    let resizeObserver = null;
    let onWindowResize = null;

    const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
    const TIME_SCALE = prefersReducedMotion ? 0.35 : 1.0;

    function loadPalette() {
      activePalette = PALETTES[Math.floor(Math.random() * PALETTES.length)];
    }

    p.setup = () => {
      const size = getTargetSize(containerEl);
      p.pixelDensity(Math.min(window.devicePixelRatio || 1, 2));
      p.createCanvas(size, size, p.WEBGL);
      p.noStroke();

      p.canvas.setAttribute('aria-label', 'Morphogenesis Artifact. Move cursor to rotate. Click or press P to change palette.');
      p.canvas.setAttribute('role', 'img');
      p.canvas.tabIndex = 0;

      theShader = p.createShader(VERTEX_SHADER, FRAGMENT_SHADER);
      loadPalette();

      if (typeof ResizeObserver !== 'undefined') {
        resizeObserver = new ResizeObserver(() => {
          const next = getTargetSize(containerEl);
          p.resizeCanvas(next, next, false);
        });
        resizeObserver.observe(containerEl);
      } else {
        onWindowResize = () => {
          const next = getTargetSize(containerEl);
          p.resizeCanvas(next, next, false);
        };
        window.addEventListener('resize', onWindowResize, { passive: true });
      }
    };

    p.draw = () => {
      const t = p.millis() * 0.001 * TIME_SCALE;
      const pd = p.pixelDensity();
      const mx = p.width > 0 ? (p.mouseX / p.width) * 2 - 1 : 0;
      const my = p.height > 0 ? (p.mouseY / p.height) * 2 - 1 : 0;
      const rotX = my * Math.PI + t * 0.1;
      const rotY = mx * Math.PI + t * 0.05;

      p.shader(theShader);
      theShader.setUniform('u_time', t);
      theShader.setUniform('u_resolution', [p.width * pd, p.height * pd]);
      theShader.setUniform('u_rotation', [rotX, rotY]);
      theShader.setUniform('u_bg', normalizeColor(activePalette.bg));
      theShader.setUniform('u_color1', normalizeColor(activePalette.colors[0]));
      theShader.setUniform('u_color2', normalizeColor(activePalette.colors[1]));
      theShader.setUniform('u_color3', normalizeColor(activePalette.colors[2]));

      p.quad(-1, -1, 1, -1, 1, 1, -1, 1);
    };

    p.mousePressed = () => {
      if (p.mouseX < 0 || p.mouseX > p.width || p.mouseY < 0 || p.mouseY > p.height) return;
      loadPalette();
    };

    p.keyPressed = () => {
      if (p.key !== 'p' && p.key !== 'P') return;
      loadPalette();
    };

    p.remove = ((originalRemove) => () => {
      resizeObserver?.disconnect();
      if (onWindowResize) window.removeEventListener('resize', onWindowResize);
      originalRemove.call(p);
    })(p.remove);
  }, containerEl);

  cleanupFuncs.push(() => myP5.remove());

  return {
    cleanup() {
      cleanupFuncs.forEach((fn) => fn());
      cleanupFuncs.length = 0;
    },
    remove() {
      cleanupFuncs.forEach((fn) => fn());
      cleanupFuncs.length = 0;
    },
  };
}

export function createMorphogenesisArtifactThreeJSSketch(containerEl) {
  return createMorphogenesisArtifactP5Sketch(containerEl);
}
