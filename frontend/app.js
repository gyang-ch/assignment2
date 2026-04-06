// Main app: handles tab switching and lazy-loads each tab's module on first visit.

const tabs        = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

// Maps each tab name to its canvas container id, module loader, and exported function name
const sketchMap = {
  circuitgrid:          { id: 'generative-sketch-2',  load: () => import('./tabs/grid.js'),            fn: 'createCircuitGridSketch' },
  gyroid:               { id: 'generative-sketch-3',  load: () => import('./tabs/gyroid.js'),          fn: 'createGyroidArtifactSketch' },
  hopfthree:            { id: 'generative-sketch-8',  load: () => import('./tabs/hopf.js'),            fn: 'createHopfArtifactThreeJSSketch' },
  morphogenesisthree:   { id: 'generative-sketch-9',  load: () => import('./tabs/morphogenesis.js'),   fn: 'createMorphogenesisArtifactThreeJSSketch' },
  ripple:               { id: 'generative-sketch-10', load: () => import('./tabs/ripple.js'),          fn: 'createEchoesSketch' },
  chromaticfugue:       { id: 'generative-sketch-11', load: () => import('./tabs/chromaticfugue.js'),  fn: 'createStyleEntropySketch' },
  driftatlas:           { id: 'generative-sketch-13', load: () => import('./tabs/driftatlas.js'),      fn: 'createStyleEntropyThreeFlowFieldSketch' },
};

const sketchInited    = {};
const sketchInstances = {};
let   activeSketchTab = null;

function switchTab(target) {
  // Tear down heavy sketches when leaving them
  if (activeSketchTab && activeSketchTab !== target) {
    if (activeSketchTab === 'hopfthree')          teardownSketch('hopfthree');
    if (activeSketchTab === 'morphogenesisthree') teardownSketch('morphogenesisthree');
    if (activeSketchTab === 'ripple')             teardownSketch('ripple');
    if (activeSketchTab === 'chromaticfugue')     teardownSketch('chromaticfugue');
    if (activeSketchTab === 'driftatlas')         teardownSketch('driftatlas');
  }

  tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === target));
  tabContents.forEach(tc => {
    const match = tc.id === `tab-${target}`;
    tc.classList.toggle('active', match);
    tc.hidden = !match;
  });

  if (sketchMap[target]) {
    initSketch(target);
    activeSketchTab = target;
  } else {
    activeSketchTab = null;
    if (target === 'probe') import('./tabs/probe.js').then(m => m.mount());
  }
}

tabs.forEach(tab => tab.addEventListener('click', () => switchTab(tab.dataset.tab)));

// Load and start a sketch for the given tab name
async function initSketch(name) {
  if (sketchInited[name]) return;
  sketchInited[name] = true;
  const { id, load, fn } = sketchMap[name];
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '';
  try {
    const mod = await load();
    sketchInstances[name] = mod[fn](el);
  } catch (err) {
    el.style.cssText = 'display:flex;align-items:center;justify-content:center;color:#555;font-family:monospace;font-size:12px;';
    el.textContent = `Failed to load sketch: ${err.message}`;
    sketchInited[name] = false; // allow retry on next click
  }
}

// Stop and remove a running sketch
function teardownSketch(name) {
  const inst = sketchInstances[name];
  if (!inst) return;
  try { inst.cleanup?.(); } catch {}
  try { inst.remove?.(); }  catch {}
  delete sketchInstances[name];
  sketchInited[name] = false;
  const el = document.getElementById(sketchMap[name]?.id);
  if (el) el.innerHTML = '';
}

// Start on the first tab
switchTab('circuitgrid');
