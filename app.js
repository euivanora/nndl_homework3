// ==========================================
// CONFIG
// ==========================================

const SIZE = 16;
const SHAPE_DATA = [1, SIZE, SIZE, 1];
const SHAPE_MODEL = [SIZE, SIZE, 1];

const CONFIG = {
  lr: 0.03,
  autoSpeed: 40
};

let state = {
  step: 0,
  auto: false,
  xInput: null,
  baseline: null,
  student: null,
  optBase: null,
  optStudent: null
};

// ==========================================
// LOSS COMPONENTS
// ==========================================

function mse(a, b) {
  return tf.losses.meanSquaredError(a, b).mean();
}

// --- Level 2: Sorted MSE (Histogram constraint)
function sortedMSE(a, b) {
  const aFlat = a.reshape([SIZE * SIZE]);
  const bFlat = b.reshape([SIZE * SIZE]);

  const aSorted = tf.sort(aFlat);
  const bSorted = tf.sort(bFlat);

  return mse(aSorted, bSorted);
}

// --- Level 3: Smoothness (Total Variation)
function smoothness(y) {
  const dx = y.slice([0,0,0,0],[-1,-1,SIZE-1,-1])
    .sub(y.slice([0,0,1,0],[-1,-1,SIZE-1,-1]));

  const dy = y.slice([0,0,0,0],[-1,SIZE-1,-1,-1])
    .sub(y.slice([0,1,0,0],[-1,SIZE-1,-1,-1]));

  return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
}

// --- Direction: left dark, right bright
function directionX(y) {
  const mask = tf.linspace(-1,1,SIZE)
    .reshape([1,1,SIZE,1]);

  return tf.mean(y.mul(mask)).mul(-1);
}

// ==========================================
// MODELS
// ==========================================

function createBaseline() {
  const m = tf.sequential();
  m.add(tf.layers.flatten({inputShape: SHAPE_MODEL}));
  m.add(tf.layers.dense({units:64, activation:'relu'}));
  m.add(tf.layers.dense({units:256, activation:'sigmoid'}));
  m.add(tf.layers.reshape({targetShape:[SIZE,SIZE,1]}));
  return m;
}

function createStudent(type) {
  const m = tf.sequential();
  m.add(tf.layers.flatten({inputShape: SHAPE_MODEL}));

  // TODO-A: Projection Types

  if(type==="compression"){
    m.add(tf.layers.dense({units:64, activation:'relu'}));
    m.add(tf.layers.dense({units:256, activation:'sigmoid'}));
  }
  else if(type==="transformation"){
    // 1:1 projection (Mission mode)
    m.add(tf.layers.dense({units:256, activation:'relu'}));
    m.add(tf.layers.dense({units:256, activation:'sigmoid'}));
  }
  else if(type==="expansion"){
    m.add(tf.layers.dense({units:512, activation:'relu'}));
    m.add(tf.layers.dense({units:256, activation:'sigmoid'}));
  }

  m.add(tf.layers.reshape({targetShape:[SIZE,SIZE,1]}));
  return m;
}

// ==========================================
// STUDENT LOSS (Intent Architect)
// ==========================================

function studentLoss(x, yPred) {
  return tf.tidy(()=>{

    // Level 2 constraint
    const lossSorted = sortedMSE(x, yPred);

    const l1 = parseFloat(document.getElementById("smoothRange").value);
    const l2 = parseFloat(document.getElementById("dirRange").value);

    const lossSmooth = smoothness(yPred).mul(l1);
    const lossDir = directionX(yPred).mul(l2);

    return lossSorted.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// TRAINING
// ==========================================

function trainStep(){

  state.step++;

  const baseLoss = tf.tidy(()=>{
    const {value, grads} = tf.variableGrads(()=>{
      const y = state.baseline.predict(state.xInput);
      return mse(state.xInput,y);
    });
    state.optBase.applyGradients(grads);
    return value.dataSync()[0];
  });

  const studentLossVal = tf.tidy(()=>{
    const {value, grads} = tf.variableGrads(()=>{
      const y = state.student.predict(state.xInput);
      return studentLoss(state.xInput,y);
    });
    state.optStudent.applyGradients(grads);
    return value.dataSync()[0];
  });

  log(`Step ${state.step} | Base=${baseLoss.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`);

  if(state.step % 5===0) render();
}

// ==========================================
// RENDER
// ==========================================

async function render(){
  const base = state.baseline.predict(state.xInput);
  const stud = state.student.predict(state.xInput);

  await tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("inputCanvas"));
  await tf.browser.toPixels(base.squeeze(), document.getElementById("baseCanvas"));
  await tf.browser.toPixels(stud.squeeze(), document.getElementById("studentCanvas"));

  base.dispose();
  stud.dispose();
}

// ==========================================
// UI
// ==========================================

function log(msg){
  const el=document.getElementById("log");
  el.innerHTML = "> "+msg+"<br>"+el.innerHTML;
}

function init(){

  state.xInput = tf.randomUniform(SHAPE_DATA);

  resetModels();

  document.getElementById("btnTrain").onclick=trainStep;

  document.getElementById("btnAuto").onclick=()=>{
    state.auto=!state.auto;
    document.getElementById("btnAuto").innerText=
      state.auto?"Auto Train (Stop)":"Auto Train (Start)";
    loop();
  };

  document.getElementById("btnReset").onclick=resetModels;

  document.getElementById("archSelect").onchange=resetModels;

  document.getElementById("smoothRange").oninput=e=>{
    document.getElementById("valSmooth").innerText=e.target.value;
  };

  document.getElementById("dirRange").oninput=e=>{
    document.getElementById("valDir").innerText=e.target.value;
  };

  render();
}

function resetModels(){

  if(state.baseline) state.baseline.dispose();
  if(state.student) state.student.dispose();

  state.baseline=createBaseline();
  state.student=createStudent(document.getElementById("archSelect").value);

  state.optBase=tf.train.adam(CONFIG.lr);
  state.optStudent=tf.train.adam(CONFIG.lr);

  state.step=0;
  log("Models Reset.");
  render();
}

function loop(){
  if(!state.auto) return;
  trainStep();
  setTimeout(loop, CONFIG.autoSpeed);
}

init();
