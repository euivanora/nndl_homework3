/**
 * Neural Network Design: The Gradient Puzzle
 * 
 * CORE CONCEPT:
 * Standard MSE forces the network to output pixel (x,y) = input pixel (x,y).
 * Sorted MSE forces the network to output the SAME SET of values, but allows 
 * them to be in DIFFERENT positions.
 * 
 * By adding Smoothness and Direction losses, we force the network to organize 
 * those values into a specific shape (a gradient).
 */

// --- Configuration ---
const IMG_SIZE = 16;
const CHANNELS = 1;
const LEARNING_RATE = 0.05;

// --- State ---
let inputTensor = null;
let baselineModel = null;
let studentModel = null;
let optimizerBaseline = null;
let optimizerStudent = null;
let isAutoTraining = false;
let stepCount = 0;

// UI Elements
const ui = {
    log: document.getElementById('logBox'),
    canvasInput: document.getElementById('canvasInput'),
    canvasBaseline: document.getElementById('canvasBaseline'),
    canvasStudent: document.getElementById('canvasStudent'),
    btnTrain: document.getElementById('btnTrainStep'),
    btnAuto: document.getElementById('btnAutoTrain'),
    btnReset: document.getElementById('btnReset'),
    selectArch: document.getElementById('selectArch'),
    rangeSmooth: document.getElementById('rangeSmooth'),
    rangeDir: document.getElementById('rangeDir'),
    valSmooth: document.getElementById('valSmooth'),
    valDir: document.getElementById('valDir')
};

// --- Initialization ---
async function init() {
    log("Initializing TensorFlow.js...");
    await tf.ready();
    
    generateFixedNoise();
    createModels();
    
    // Initial Render
    drawTensor(ui.canvasInput, inputTensor);
    drawTensor(ui.canvasBaseline, baselineModel.predict(inputTensor));
    drawTensor(ui.canvasStudent, studentModel.predict(inputTensor));

    log("Ready. Select architecture and start training.");
}

// --- Data Generation ---
function generateFixedNoise() {
    // Create random noise between 0 and 1
    // Shape: [1, 16, 16, 1]
    inputTensor = tf.randomUniform([1, IMG_SIZE, IMG_SIZE, CHANNELS]);
    log("Generated fixed noise input.");
}

// --- Model Architectures ---

/**
 * Creates the baseline model. 
 * This is a standard Autoencoder that tries to copy input to output.
 */
function createBaselineModel() {
    const model = tf.sequential();
    // Simple compression/expansion
    model.add(tf.layers.flatten({inputShape: [IMG_SIZE, IMG_SIZE, CHANNELS]}));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: IMG_SIZE * IMG_SIZE * CHANNELS, activation: 'sigmoid'}));
    model.add(tf.layers.reshape({targetShape: [IMG_SIZE, IMG_SIZE, CHANNELS]}));
    return model;
}

/**
 * TODO-A: ARCHITECTURE SELECTION
 * Students must implement the logic to change the bottleneck size based on selection.
 * 
 * 1. Compression: Hidden layer < Input size (e.g., 32 units)
 * 2. Transformation: Hidden layer == Input size (e.g., 256 units)
 * 3. Expansion: Hidden layer > Input size (e.g., 512 units)
 */
function createStudentModel(archType) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [IMG_SIZE, IMG_SIZE, CHANNELS]}));
    
    const inputSize = IMG_SIZE * IMG_SIZE * CHANNELS; // 256
    let hiddenUnits = 64; // Default

    if (archType === 'compression') {
        hiddenUnits = 32; // Squeeze information
    } else if (archType === 'transformation') {
        // TODO: Set hiddenUnits to match inputSize (256)
        throw new Error("TODO-A: Implement Transformation Architecture");
    } else if (archType === 'expansion') {
        // TODO: Set hiddenUnits to be larger than inputSize (e.g., 512)
        throw new Error("TODO-A: Implement Expansion Architecture");
    }

    model.add(tf.layers.dense({units: hiddenUnits, activation: 'relu'}));
    model.add(tf.layers.dense({units: hiddenUnits, activation: 'relu'}));
    model.add(tf.layers.dense({units: inputSize, activation: 'sigmoid'}));
    model.add(tf.layers.reshape({targetShape: [IMG_SIZE, IMG_SIZE, CHANNELS]}));
    
    return model;
}

function createModels() {
    if (baselineModel) baselineModel.dispose();
    if (studentModel) studentModel.dispose();
    if (optimizerBaseline) optimizerBaseline.dispose();
    if (optimizerStudent) optimizerStudent.dispose();

    const archType = ui.selectArch.value;

    baselineModel = createBaselineModel();
    studentModel = createStudentModel(archType);

    optimizerBaseline = tf.train.adam(LEARNING_RATE);
    optimizerStudent = tf.train.adam(LEARNING_RATE);

    stepCount = 0;
    log(`Models created. Arch: ${archType}`);
}

// --- Loss Functions (The Core Lesson) ---

/**
 * Level 1: Standard Mean Squared Error
 * Forces Output[x,y] == Input[x,y]
 */
function mseLoss(yTrue, yPred) {
    return tf.losses.meanSquaredError(yTrue, yPred);
}

/**
 * Level 2: Sorted MSE (Distribution Matching)
 * 
 * CONCEPT: We don't care WHERE the pixels are, only THAT they exist.
 * We sort both tensors and compare them. This allows the network to "move" pixels.
 * 
 * Math: MSE( Sort(Flatten(Input)), Sort(Flatten(Output)) )
 */
function sortedMseLoss(yTrue, yPred) {
    // 1. Flatten tensors to 1D arrays of values
    const flatTrue = yTrue.flatten();
    const flatPred = yPred.flatten();

    // 2. Sort both arrays
    // Note: tf.sort is differentiable in modern TF.js
    const sortedTrue = tf.sort(flatTrue, 'asc');
    const sortedPred = tf.sort(flatPred, 'asc');

    // 3. Calculate MSE on the sorted values
    const loss = tf.losses.meanSquaredError(sortedTrue, sortedPred);

    // Cleanup intermediate tensors
    flatTrue.dispose();
    flatPred.dispose();
    sortedTrue.dispose();
    sortedPred.dispose();

    return loss;
}

/**
 * Level 3: Smoothness (Total Variation Loss)
 * Encourages neighboring pixels to have similar values.
 * Prevents the "salt and pepper" noise look.
 */
function smoothnessLoss(yPred) {
    // Shift tensor right (pad left with 0)
    const shiftedRight = tf.pad(yPred, [[0,0], [0,0], [1,0], [0,0]], 'constant').slice([0,0,0,0], [-1,-1,-1,-1]);
    // Shift tensor down (pad top with 0)
    const shiftedDown = tf.pad(yPred, [[0,0], [1,0], [0,0], [0,0]], 'constant').slice([0,0,0,0], [-1,-1,-1,-1]);

    // Difference between neighbors
    const diffX = tf.sub(yPred, shiftedRight);
    const diffY = tf.sub(yPred, shiftedDown);

    // Sum of squared differences
    const tv = tf.square(diffX).add(tf.square(diffY)).mean();

    shiftedRight.dispose();
    shiftedDown.dispose();
    diffX.dispose();
    diffY.dispose();

    return tv;
}

/**
 * Level 3: Directional Loss
 * Encourages brightness to increase from Left to Right.
 * We create a mask that goes from -1 (left) to +1 (right).
 * Loss = -Mean(Output * Mask). Minimizing negative mean = Maximizing positive mean.
 */
function directionLoss(yPred) {
    // Create a gradient mask [-1 ... +1] along the width
    // Shape: [1, 16, 16, 1]
    const maskData = new Float32Array(IMG_SIZE * IMG_SIZE);
    for (let y = 0; y < IMG_SIZE; y++) {
        for (let x = 0; x < IMG_SIZE; x++) {
            // Normalize x from 0..15 to -1..1
            const val = (x / (IMG_SIZE - 1)) * 2 - 1;
            maskData[y * IMG_SIZE + x] = val;
        }
    }
    
    const mask = tf.tensor4d(maskData, [1, IMG_SIZE, IMG_SIZE, 1]);
    
    // Element wise multiply
    const weighted = tf.mul(yPred, mask);
    
    // We want high values on the right (positive mask) and low on left (negative mask)
    // So we want the sum of (Output * Mask) to be HIGH.
    // To minimize loss, we return negative mean.
    const loss = tf.neg(weighted.mean());

    mask.dispose();
    weighted.dispose();
    return loss;
}

/**
 * TODO-B: CUSTOM LOSS AGGREGATION
 * Combine the losses based on UI sliders.
 * 
 * Formula: L_total = L_sortedMSE + (λ₁ * L_smooth) + (λ₂ * L_dir)
 */
function calculateStudentLoss(yTrue, yPred) {
    const lambdaSmooth = parseFloat(ui.rangeSmooth.value);
    const lambdaDir = parseFloat(ui.rangeDir.value);

    // Base constraint: Must use the same colors (Inventory)
    let loss = sortedMseLoss(yTrue, yPred);

    if (lambdaSmooth > 0) {
        const smoothTerm = smoothnessLoss(yPred);
        // TODO: Add smoothTerm to loss weighted by lambdaSmooth
        throw new Error("TODO-B: Implement Smoothness Loss Addition");
    }

    if (lambdaDir > 0) {
        const dirTerm = directionLoss(yPred);
        // TODO: Add dirTerm to loss weighted by lambdaDir
        throw new Error("TODO-B: Implement Direction Loss Addition");
    }

    return loss;
}

// --- Training Loop ---

async function trainStep() {
    tf.tidy(() => {
        // 1. Train Baseline (Standard MSE)
        optimizerBaseline.minimize(() => {
            const pred = baselineModel.predict(inputTensor);
            return mseLoss(inputTensor, pred);
        }, baselineModel.trainableWeights);

        // 2. Train Student (Custom Loss)
        optimizerStudent.minimize(() => {
            const pred = studentModel.predict(inputTensor);
            return calculateStudentLoss(inputTensor, pred);
        }, studentModel.trainableWeights);
    });

    stepCount++;
    if (stepCount % 10 === 0) {
        updateVisuals();
    }
}

function updateVisuals() {
    tf.tidy(() => {
        const baseOut = baselineModel.predict(inputTensor);
        const studOut = studentModel.predict(inputTensor);
        
        drawTensor(ui.canvasBaseline, baseOut);
        drawTensor(ui.canvasStudent, studOut);

        // Calculate current losses for logging
        const baseLoss = mseLoss(inputTensor, baseOut).dataSync()[0].toFixed(4);
        // Note: Calculating student loss here is expensive, doing it roughly every 10 steps is fine
        const studLoss = calculateStudentLoss(inputTensor, studOut).dataSync()[0].toFixed(4);

        log(`Step ${stepCount} | Base Loss: ${baseLoss} | Student Loss: ${studLoss}`);
    });
}

// --- Rendering Helpers ---

function drawTensor(canvas, tensor) {
    // Tensor is [1, 16, 16, 1]. We need to get data out.
    // We clone because tensor.data() is async or sync but we want to be safe with disposal
    const data = tensor.dataSync(); 
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(IMG_SIZE, IMG_SIZE);
    
    for (let i = 0; i < data.length; i++) {
        const val = Math.floor(data[i] * 255);
        const idx = i * 4;
        imgData.data[idx] = val;     // R
        imgData.data[idx + 1] = val; // G
        imgData.data[idx + 2] = val; // B
        imgData.data[idx + 3] = 255; // Alpha
    }
    
    // Scale up the small 16x16 image to fit the canvas (160x160)
    // We draw to an offscreen canvas first or use drawImage scaling
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = IMG_SIZE;
    tempCanvas.height = IMG_SIZE;
    tempCanvas.getContext('2d').putImageData(imgData, 0, 0);
    
    ctx.imageSmoothingEnabled = false; // Keep it pixelated
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
}

function log(msg) {
    ui.log.innerHTML += `> ${msg}<br>`;
    ui.log.scrollTop = ui.log.scrollHeight;
}

// --- Event Listeners ---

ui.btnTrain.addEventListener('click', () => {
    trainStep();
    updateVisuals();
});

ui.btnAuto.addEventListener('click', () => {
    isAutoTraining = !isAutoTraining;
    ui.btnAuto.textContent = isAutoTraining ? "Auto Train (Stop)" : "Auto Train (Start)";
    ui.btnAuto.classList.toggle('danger', isAutoTraining);
    
    if (isAutoTraining) {
        const loop = () => {
            if (!isAutoTraining) return;
            trainStep();
            // Throttle slightly for visual effect
            if (stepCount % 5 === 0) updateVisuals(); 
            requestAnimationFrame(loop);
        };
        loop();
    }
});

ui.btnReset.addEventListener('click', () => {
    generateFixedNoise();
    createModels();
    drawTensor(ui.canvasInput, inputTensor);
    drawTensor(ui.canvasBaseline, baselineModel.predict(inputTensor));
    drawTensor(ui.canvasStudent, studentModel.predict(inputTensor));
    log("Weights and Input Reset.");
});

ui.selectArch.addEventListener('change', () => {
    createModels();
    log(`Architecture changed to ${ui.selectArch.value}. Weights reset.`);
    // Clear canvases visually
    updateVisuals();
});

ui.rangeSmooth.addEventListener('input', (e) => ui.valSmooth.textContent = e.target.value);
ui.rangeDir.addEventListener('input', (e) => ui.valDir.textContent = e.target.value);

// Start
init();