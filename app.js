// ==========================================
// THE GRADIENT PUZZLE — FIXED VERSION
// ==========================================

const SIZE = 16;
const SHAPE = [1, SIZE, SIZE, 1];

const CONFIG = {
  lr: 0.01,
  autoSpeed: 30
};

let state = {
  step: 0,
  auto: false,
  x: null,
  baseline: null,
  student: null,
  optBase: null,
  optStudent: null
};

// ==========================================
// LOSS FUNCTIONS
// ==========================================

function mse(a, b) {
  return tf.mean(tf.square(a.sub(b)));
}

// Histogram preservation (Sorted MSE)
function sortedMSE(a, b) {
  const aSorted = tf.sort(a.reshape([SIZE * SIZE]));
  const bSorted = tf.sort(b.reshape([SIZE * SIZE]));
  return mse(aSorted, bSorted);
}

// Smoothness (Total Variation X only)
function smoothness(y) {
  const dx = y.slice([0, 0, 0, 0], [-1, -1, SIZE - 1, -1])
    .sub(y.slice([0, 0, 1, 0], [-1, -1, SIZE - 1, -1]));
  return tf.mean(tf.square(dx));
}

// Direction: left dark → right bright
function directionX(y) {
  const mask = tf.linspace(-1, 1, SIZE)
    .reshape([1, 1, SIZE, 1]);
  return tf.mean(y.mul(mask)).mul(-1);
}

// ==========================================
// MODELS
// ==========================================

function createModel(type) {

  const m = tf.sequential();
  m.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));

  if (type === "compression") {
    m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  }
  else if (type === "transformation") {
    m.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  }
  else if (type === "expansion") {
    m.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  }

  m.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  m.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));

  return m;
}

// ==========================================
// STUDENT LOSS
// ==========================================

function studentLoss(x, y) {

  const λ1 = parseFloat(document.getElementById("smoothRange").value);
  const λ2 = parseFloat(document.getElementById("dirRange").value);

  const lossHist = sortedMSE(x, y);
  const lossSmooth = smoothness(y).mul(λ1);
  const lossDir = directionX(y).mul(λ2);

  return lossHist.add(lossSmooth).add(lossDir);
}

// ==========================================
// TRAINING
// ==========================================

function trainStep() {

  state.step++;

  tf.tidy(() => {

    // Baseline
    const baseGrads = tf.variableGrads(() => {
      const y = state.baseline.predict(state.x);
      return mse(state.x, y);
    });
    state.optBase.applyGradients(baseGrads.grads);

    // Student
    const studentGrads = tf.variableGrads(() => {
      const y = state.student.predict(state.x);
      return studentLoss(state.x, y);
    });
    state.optStudent.applyGradients(studentGrads.grads);

  });

  if (state.step % 5 === 0) {
    render();
    log("Step " + state.step);
  }
}

// ==========================================
// CORRECT PIXEL SCALING (16x16 → 320x320)
// ==========================================

async function drawTensorScaled(tensor, canvas) {

  const temp = document.createElement("canvas");
  temp.width = SIZE;
  temp.height = SIZE;

  await tf.browser.toPixels(tensor.squeeze(), temp);

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.drawImage(
    temp,
    0, 0, SIZE, SIZE,
    0, 0, canvas.width, canvas.height
  );
}

async function render() {

  const base = state.baseline.predict(state.x);
  const stud = state.student.predict(state.x);

  await drawTensorScaled(state.x, document.getElementById("inputCanvas"));
  await drawTensorScaled(base, document.getElementById("baseCanvas"));
  await drawTensorScaled(stud, document.getElementById("studentCanvas"));

  base.dispose();
  stud.dispose();
}

// ==========================================
// UI
// ==========================================

function log(msg) {
  const el = document.getElementById("log");
  el.innerHTML = "> " + msg + "<br>" + el.innerHTML;
}

function reset() {

  if (state.baseline) state.baseline.dispose();
  if (state.student) state.student.dispose();

  state.x = tf.randomUniform(SHAPE);

  state.baseline = createModel("compression");
  state.student = createModel(
    document.getElementById("archSelect").value
  );

  state.optBase = tf.train.adam(CONFIG.lr);
  state.optStudent = tf.train.adam(CONFIG.lr);

  state.step = 0;
  render();
  log("Reset.");
}

function loop() {
  if (!state.auto) return;
  trainStep();
  setTimeout(loop, CONFIG.autoSpeed);
}

function init() {

  document.getElementById("btnTrain").onclick = trainStep;

  document.getElementById("btnAuto").onclick = () => {
    state.auto = !state.auto;
    document.getElementById("btnAuto").innerText =
      state.auto ? "Auto Train (Stop)" : "Auto Train (Start)";
    loop();
  };

  document.getElementById("btnReset").onclick = reset;
  document.getElementById("archSelect").onchange = reset;

  document.getElementById("smoothRange").oninput = e =>
    document.getElementById("valSmooth").innerText = e.target.value;

  document.getElementById("dirRange").oninput = e =>
    document.getElementById("valDir").innerText = e.target.value;

  reset();
}

init();
