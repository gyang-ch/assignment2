// Gyroid Artifact — a gyroid minimal surface rendered via GLSL raymarching.
// Mouse controls rotation. Click to cycle colour palette.

import p5 from 'p5';

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

// Convert hex palettes to [R,G,B] arrays for the GLSL shader uniforms
const PALETTES = rawColorSchemes
  .filter((scheme) => scheme.colors.length >= 4)
  .map((scheme) => ({
    bg: [5, 5, 5],
    colors: [hexToRgb(scheme.colors[1]), hexToRgb(scheme.colors[2]), hexToRgb(scheme.colors[3])],
  }));

// Define where to draw the artefact 
const VERT = `
precision highp float;

attribute vec3 aPosition;
attribute vec2 aTexCoord;
varying vec2 vTexCoord;

void main() {
  vTexCoord = aTexCoord;
  vec4 positionVec4 = vec4(aPosition, 1.0);
  positionVec4.xy = positionVec4.xy * 2.0 - 1.0;
  gl_Position = positionVec4;
}
`;

// Calculate colour of every single pixel to draw the Gyroid.
const FRAG = `
precision highp float;

varying vec2 vTexCoord;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_color1;
uniform vec3 u_color2;
uniform vec3 u_color3;
uniform vec3 u_bg;
uniform vec2 u_mouse;

#define MAX_STEPS 100
#define MAX_DIST 10.0
#define SURF_DIST 0.001

mat2 rot(float a) {
  float s = sin(a), c = cos(a);
  return mat2(c, -s, s, c);
}

float getDistance(vec3 p) {
  p.yz *= rot(u_mouse.y * 3.14 + u_time * 0.1);
  p.xz *= rot(u_mouse.x * 3.14 + u_time * 0.05);

  float scale = 4.0;
  vec3 q = p * scale;

  float d = dot(sin(q + u_time * 0.3), cos(q.yzx - u_time * 0.2));
  float gyroid = abs(d) / scale - 0.04;

  float sphere = length(p) - 1.2;
  return max(gyroid, sphere);
}

vec3 getNormal(vec3 p) {
  float d = getDistance(p);
  vec2 e = vec2(0.001, 0.0);
  vec3 n = d - vec3(
    getDistance(p - e.xyy),
    getDistance(p - e.yxy),
    getDistance(p - e.yyx)
  );
  return normalize(n);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;

  vec3 ro = vec3(0.0, 0.0, -3.0);
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

    float mix1 = smoothstep(-1.0, 1.0, n.x);
    float mix2 = smoothstep(-1.0, 1.0, n.y);
    vec3 baseColor = mix(u_color1, u_color2, mix1);
    baseColor = mix(baseColor, u_color3, mix2);

    float dif = max(dot(n, lightDir), 0.0);
    float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    float sss = smoothstep(0.0, 0.2, getDistance(p + lightDir * 0.1)) * 0.5;

    col = baseColor * (dif + 0.2)
      + (u_color1 * fresnel)
      + (u_color2 * sss);
  } else {
    col -= length(uv) * 0.15;
  }

  gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
`;

function normalizeColor(rgbArr) {
  return [rgbArr[0] / 255, rgbArr[1] / 255, rgbArr[2] / 255];
}

export function createGyroidArtifactSketch(containerEl) {
  return new p5((p) => {
    const SIZE = 800;

    let activePalette;
    let gyroidShader;

    const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
    const TARGET_FPS = prefersReducedMotion ? 24 : 60;
    const TIME_SCALE = prefersReducedMotion ? 0.35 : 1.0;

    function loadPalette() {
      activePalette = PALETTES[Math.floor(p.random(PALETTES.length))];
    }
    
    // Create the 3D canvas
    p.setup = () => {
      p.createCanvas(SIZE, SIZE, p.WEBGL);
      p.pixelDensity(1);
      p.frameRate(TARGET_FPS);

      gyroidShader = p.createShader(VERT, FRAG);
      loadPalette();

      p.noStroke();
      p.rectMode(p.CENTER);
    };

    // Send data to the Shader
    p.draw = () => {
      p.shader(gyroidShader);

      gyroidShader.setUniform('u_resolution', [p.width * p.pixelDensity(), p.height * p.pixelDensity()]);
      gyroidShader.setUniform('u_time', (p.millis() / 1000.0) * TIME_SCALE);

      const mx = p.map(p.mouseX, 0, p.width, -1, 1);
      const my = p.map(p.mouseY, 0, p.height, -1, 1);
      gyroidShader.setUniform('u_mouse', [mx, my]);

      gyroidShader.setUniform('u_bg', normalizeColor(activePalette.bg));
      gyroidShader.setUniform('u_color1', normalizeColor(activePalette.colors[0]));
      gyroidShader.setUniform('u_color2', normalizeColor(activePalette.colors[1]));
      gyroidShader.setUniform('u_color3', normalizeColor(activePalette.colors[2]));

      p.rect(0, 0, p.width, p.height);
    };

    p.mousePressed = () => {
      loadPalette();
    };
  }, containerEl);
}
