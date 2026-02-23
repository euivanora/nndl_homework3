// ==========================================
// GRADIENT PUZZLE — LEFT → RIGHT VERSION
// ==========================================

const SIZE = 16;
const SHAPE = [1, SIZE, SIZE, 1];

const CONFIG = {
  lr: 0.02,
  autoSpeed: 20
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
// BASIC LOSS
// ==========================================

function mse(a, b) {
  return tf.mean(tf.square(a.sub(b)));
}

// Preserve distribution (mean + variance)
function distributionLoss(x, y) {
  const meanLoss = tf.square(tf.mean(x).sub(tf.mean(y)));
  const varLoss = tf.square(
    tf.moments(x).variance.sub(tf.moments(y).variance)
  );
  return meanLoss.add(varLoss);
}

// ==========================================
// SHAPE CONSTRAINTS
// ==========================================

// Total Variation (horizontal smoothness)
function smoothness(y) {
  const dx = y.slice([0, 0, 0, 0], [-1, -1, SIZE - 1, -1])
    .sub(y.slice([0, 0, 1, 0], [-1, -1, SIZE - 1, -1]));

  return tf.mean(tf.square(dx));
}

// Direction: right brighter than left
function directionX(y) {

  const left = y.slice([0, 0, 0, 0], [-1, -1, SIZE / 2, -1]);
  const right = y.slice([0, 0, SIZE / 2, 0], [-1, -1, SIZE / 2, -1]);

  const meanLeft = tf.mean(left);
  const meanRight = tf.mean(right);

  // Loss decreases when right > left
  return meanLeft.sub(meanRight);
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

  const lossDist = distributionLoss(x, y);
  const lossSmooth = smoothness(y).mul(λ1);
  const lossDir = directionX(y).mul(λ2);

  return lossDist.add(lossSmooth).add(lossDir);
}

// ==========================================
// TRAIN STEP
// ==========================================

function trainStep() {

  state.step++;

  tf.tidy(() => {

    // Baseline autoencoder
    state.optBase.minimize(() => {
      const y = state.baseline.predict(state.x);
      return mse(state.x, y);
    }, false, state.baseline.trainableWeights.map(w => w.val));

    // Student with shape constraints
    state.optStudent.minimize(() => {
      const y = state.student.predict(state.x);
      return studentLoss(state.x, y);
    }, false, state.student.trainableWeights.map(w => w.val));

  });

  if (state.step % 5 === 0) {
    render();
    log("Step " + state.step);
  }
}

// ==========================================
// RENDER (16 → 320 scaling)
// ==========================================

async function drawTensorScaled(tensor, canvas) {

  const temp = document.createElement("canvas");
  temp.width = SIZE;
  temp.height = SIZE;

  await tf.browser.toPixels(tensor.squeeze(), temp);

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.drawImage(temp, 0, 0, SIZE, SIZE,
                0, 0, canvas.width, canvas.height);
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
