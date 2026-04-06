// Circuit Growth Grid — walkers branch and draw glowing circuit-like trails across a 3×3 grid.
// Click to regenerate with a new colour palette.

import p5 from 'p5';

const COLOR_SCHEMES = [
  ['#4B3A51', '#A77A4B', '#ECC6A2', '#A43020', '#722D24'],
  ['#204035', '#4A7169', '#BEB59C', '#735231', '#49271B'],
  ['#293757', '#568D4B', '#D5BB56', '#D26A1B', '#A41D1A'],
  ['#27403D', '#48725C', '#212412', '#F3E4C2', '#D88F2E'],
  ['#1E1D20', '#B66636', '#547A56', '#BDAE5B', '#515A7C'],
  ['#202221', '#661E2A', '#AB381B', '#EAD4A3', '#E3DED8'],
  ['#1F284C', '#2D4472', '#6E6352', '#D9CCAC', '#ECE2C6'],
  ['#023059', '#459DBF', '#87BF60', '#D9D16A', '#F2F2F2'],
  ['#121510', '#6D8325', '#D6CFB7', '#E5AD4F', '#BD5630'],
  ['#333333', '#D1B817', '#2A2996', '#B34325', '#C8CCC6'],
  ['#151817', '#001A56', '#197C3F', '#D4A821', '#C74C25'],
  ['#0E2523', '#324028', '#C26B61', '#5A788D', '#DE7944'],
  ['#1D2025', '#45312A', '#7E2F28', '#202938', '#D58E40'],
  ['#184430', '#548150', '#DEB738', '#734321', '#852419'],
  ['#442327', '#C0BC98', '#A6885D', '#8A3230', '#973B2B'],
  ['#0E122D', '#182044', '#51628E', '#91A1BA', '#AFD0C9'],
  ['#1A3431', '#2B41A7', '#6283C8', '#CCC776', '#C7AD24'],
  ['#0C0B10', '#707DA6', '#CCAD9D', '#B08E4A', '#863B34'],
  ['#0d0d0d', '#c73131', '#1b8be7', '#b7bcc4', '#f8e744']
];

export function createCircuitGridSketch(containerEl) {
  return new p5((p) => {
    const SIZE = 1000;

    let activePalette;
    let pg;
    let agents = [];

    let maxAgents = 300;

    const cols = 3;
    const rows = 3;
    let cellW;
    let cellH;

    const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
    const TARGET_FPS = prefersReducedMotion ? 12 : 30;

    function pickScheme() {
      activePalette = p.random(COLOR_SCHEMES);
    }

    function generateGaussianColor(base, scale) {
      const h = p.hue(base) + p.randomGaussian() * scale;
      const s = p.saturation(base) + p.randomGaussian() * scale;
      const b = p.brightness(base) + p.randomGaussian() * scale;
      return p.color(((h % 360) + 360) % 360, p.constrain(s, 0, 100), p.constrain(b, 0, 100), 100);
    }

    class CircuitAgent {
      constructor(x, y, angle, weight, col, bounds) {
        this.x = x;
        this.y = y;
        this.angle = angle;
        this.w = weight;
        this.baseColor = col;
        this.bounds = bounds;

        this.active = true;
        this.stepSize = p.random(2, 5);
        this.gColor = generateGaussianColor(this.baseColor, 5);

        this.oldX = x;
        this.oldY = y;
      }

      update() {
        if (!this.active) return;

        this.oldX = this.x;
        this.oldY = this.y;

        this.x += p.cos(this.angle) * this.stepSize;
        this.y += p.sin(this.angle) * this.stepSize;

        this.w *= 0.985;

        if (this.w < 0.3 || this.isOutOfBounds()) {
          this.active = false;
        }

        if (p.random() < 0.05 && agents.length < maxAgents) {
          const turnAngle = p.random([45, 90, -45, -90]);
          const newWeight = this.w * p.random(0.8, 1.2);

          agents.push(new CircuitAgent(
            this.x,
            this.y,
            this.angle + turnAngle,
            newWeight,
            this.baseColor,
            this.bounds,
          ));
        }
      }

      display(buffer) {
        if (!this.active) return;

        // Draw a wide dim stroke first, then a narrow bright one to fake a glow
        const h = p.hue(this.gColor);
        const s = p.saturation(this.gColor);
        const b = p.brightness(this.gColor);

        buffer.strokeWeight(this.w * 3.5);
        buffer.stroke(h, s, b, 10);
        buffer.line(this.oldX, this.oldY, this.x, this.y);

        buffer.strokeWeight(this.w);
        buffer.stroke(h, s, b, 80);
        buffer.line(this.oldX, this.oldY, this.x, this.y);

        if (p.random() < 0.15) this.drawSand(buffer);
      }

      drawSand(buffer) {
        buffer.strokeWeight(1.5);
        buffer.stroke(this.gColor);

        const numGrains = p.floor(p.random(1, 3));
        const spread = this.w * 3;

        for (let i = 0; i < numGrains; i++) {
          const pAngle = this.angle + 90;
          const dist = p.random(-spread, spread);
          const px = this.x + p.cos(pAngle) * dist;
          const py = this.y + p.sin(pAngle) * dist;
          buffer.point(px, py);
        }
      }

      isOutOfBounds() {
        const margin = 10;
        return (
          this.x < this.bounds.x + margin ||
          this.x > this.bounds.x + this.bounds.w - margin ||
          this.y < this.bounds.y + margin ||
          this.y > this.bounds.y + this.bounds.h - margin
        );
      }
    }

    function drawStructuralGrid() {
      p.stroke(0, 0, 100, 10);
      p.strokeWeight(1);
      p.noFill();

      for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
          const cx = c * cellW + cellW / 2;
          const cy = r * cellH + cellH / 2;
          p.rect(c * cellW, r * cellH, cellW, cellH);
          p.circle(cx, cy, cellW * 0.8);
          p.circle(cx, cy, cellW * 0.2);
        }
      }
    }

    function generateArtwork() {
      pickScheme();

      pg.clear();
      pg.blendMode(p.ADD);

      agents = [];
      cellW = p.width / cols;
      cellH = p.height / rows;

      for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
          const centerX = c * cellW + cellW / 2;
          const centerY = r * cellH + cellH / 2;

          const numSeeds = p.floor(p.random(6, 16));
          for (let i = 0; i < numSeeds; i++) {
            const angle = (360 / numSeeds) * i;
            const baseCol = p.color(p.random(activePalette));

            agents.push(new CircuitAgent(
              centerX,
              centerY,
              angle,
              p.random(4, 8),
              baseCol,
              { x: c * cellW, y: r * cellH, w: cellW, h: cellH },
            ));
          }
        }
      }

      p.loop();
    }

    p.setup = () => {
      p.createCanvas(SIZE, SIZE);
      p.pixelDensity(1);
      p.frameRate(TARGET_FPS);

      p.colorMode(p.HSB, 360, 100, 100, 100);
      p.angleMode(p.DEGREES);

      pg = p.createGraphics(SIZE, SIZE);
      pg.pixelDensity(1);
      pg.colorMode(p.HSB, 360, 100, 100, 100);
      pg.angleMode(p.DEGREES);

      generateArtwork();
    };

    p.draw = () => {
      for (let i = agents.length - 1; i >= 0; i--) {
        agents[i].update();
        agents[i].display(pg);
        if (!agents[i].active) agents.splice(i, 1);
      }

      p.background(5, 5, 10);
      drawStructuralGrid();
      p.image(pg, 0, 0);

      if (agents.length === 0 && p.frameCount > 10) {
        p.noLoop();
        console.log('Growth Complete. Safe to save.');
      }
    };

    p.mousePressed = () => {
      generateArtwork();
    };
  }, containerEl);
}