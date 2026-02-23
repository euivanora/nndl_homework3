// Конфигурация
const IMG_SIZE = 16;
const LEARNING_RATE = 0.01;

// Глобальные переменные
let inputTensor = null;
let baselineModel = null;
let studentModel = null;
let optimizerBaseline = null;
let optimizerStudent = null;
let isAutoTraining = false;
let stepCount = 0;
let autoTrainId = null;

// UI элементы
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

// Логирование
function log(msg) {
    ui.log.innerHTML += `> ${msg}<br>`;
    ui.log.scrollTop = ui.log.scrollHeight;
    console.log(msg);
}

// Генерация шума
function generateFixedNoise() {
    if (inputTensor) inputTensor.dispose();
    inputTensor = tf.randomUniform([1, IMG_SIZE, IMG_SIZE, 1]);
    log("Noise generated");
}

// Создание моделей
function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [IMG_SIZE, IMG_SIZE, 1]}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: 128, activation: 'relu'}));
    model.add(tf.layers.dense({units: IMG_SIZE * IMG_SIZE, activation: 'sigmoid'}));
    model.add(tf.layers.reshape({targetShape: [IMG_SIZE, IMG_SIZE, 1]}));
    return model;
}

function createStudentModel(archType) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({inputShape: [IMG_SIZE, IMG_SIZE, 1]}));
    
    let hiddenUnits = 128;
    const inputSize = IMG_SIZE * IMG_SIZE;
    
    if (archType === 'compression') {
        hiddenUnits = 32;
    } else if (archType === 'transformation') {
        hiddenUnits = inputSize;
    } else if (archType === 'expansion') {
        hiddenUnits = 512;
    }
    
    model.add(tf.layers.dense({units: hiddenUnits, activation: 'relu'}));
    model.add(tf.layers.dense({units: hiddenUnits, activation: 'relu'}));
    model.add(tf.layers.dense({units: inputSize, activation: 'sigmoid'}));
    model.add(tf.layers.reshape({targetShape: [IMG_SIZE, IMG_SIZE, 1]}));
    
    return model;
}

function createModels() {
    if (baselineModel) baselineModel.dispose();
    if (studentModel) studentModel.dispose();
    if (optimizerBaseline) optimizerBaseline.dispose();
    if (optimizerStudent) optimizerStudent.dispose();
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel(ui.selectArch.value);
    optimizerBaseline = tf.train.adam(LEARNING_RATE);
    optimizerStudent = tf.train.adam(LEARNING_RATE);
    stepCount = 0;
    
    log(`Models created: ${ui.selectArch.value}`);
}

// Функции потерь
function mseLoss(yTrue, yPred) {
    return tf.losses.meanSquaredError(yTrue, yPred);
}

function sortedMseLoss(yTrue, yPred) {
    const flatTrue = yTrue.flatten();
    const flatPred = yPred.flatten();
    const sortedTrue = tf.sort(flatTrue, 'asc');
    const sortedPred = tf.sort(flatPred, 'asc');
    const loss = tf.losses.meanSquaredError(sortedTrue, sortedPred);
    flatTrue.dispose();
    flatPred.dispose();
    sortedTrue.dispose();
    sortedPred.dispose();
    return loss;
}

function smoothnessLoss(yPred) {
    const shiftedRight = tf.pad(yPred, [[0,0], [0,0], [1,0], [0,0]], 'constant').slice([0,0,0,0], [-1,-1,-1,-1]);
    const shiftedDown = tf.pad(yPred, [[0,0], [1,0], [0,0], [0,0]], 'constant').slice([0,0,0,0], [-1,-1,-1,-1]);
    const diffX = tf.sub(yPred, shiftedRight);
    const diffY = tf.sub(yPred, shiftedDown);
    const tv = tf.square(diffX).add(tf.square(diffY)).mean();
    shiftedRight.dispose();
    shiftedDown.dispose();
    diffX.dispose();
    diffY.dispose();
    return tv;
}

function directionLoss(yPred) {
    const maskData = new Float32Array(IMG_SIZE * IMG_SIZE);
    for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
        const x = i % IMG_SIZE;
        maskData[i] = (x / (IMG_SIZE - 1)) * 2 - 1;
    }
    const mask = tf.tensor4d(maskData, [1, IMG_SIZE, IMG_SIZE, 1]);
    const weighted = tf.mul(yPred, mask);
    const loss = tf.neg(weighted.mean());
    mask.dispose();
    weighted.dispose();
    return loss;
}

function calculateStudentLoss(yTrue, yPred) {
    const lambdaSmooth = parseFloat(ui.rangeSmooth.value);
    const lambdaDir = parseFloat(ui.rangeDir.value);
    
    let loss = sortedMseLoss(yTrue, yPred);
    
    if (lambdaSmooth > 0) {
        const smoothTerm = smoothnessLoss(yPred);
        loss = tf.add(loss, tf.mul(smoothTerm, lambdaSmooth));
    }
    
    if (lambdaDir > 0) {
        const dirTerm = directionLoss(yPred);
        loss = tf.add(loss, tf.mul(dirTerm, lambdaDir));
    }
    
    return loss;
}

// Отрисовка
function drawTensor(canvas, tensor) {
    const data = tensor.dataSync();
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(IMG_SIZE, IMG_SIZE);
    
    for (let i = 0; i < data.length; i++) {
        const val = Math.floor(data[i] * 255);
        const idx = i * 4;
        imgData.data[idx] = val;
        imgData.data[idx + 1] = val;
        imgData.data[idx + 2] = val;
        imgData.data[idx + 3] = 255;
    }
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = IMG_SIZE;
    tempCanvas.height = IMG_SIZE;
    tempCanvas.getContext('2d').putImageData(imgData, 0, 0);
    
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
}

// Обучение
async function trainStep() {
    try {
        tf.tidy(() => {
            optimizerBaseline.minimize(() => {
                const pred = baselineModel.predict(inputTensor);
                return mseLoss(inputTensor, pred);
            }, baselineModel.trainableWeights);
            
            optimizerStudent.minimize(() => {
                const pred = studentModel.predict(inputTensor);
                return calculateStudentLoss(inputTensor, pred);
            }, studentModel.trainableWeights);
        });
        
        stepCount++;
        if (stepCount % 10 === 0) {
            updateVisuals();
        }
    } catch (err) {
        log(`ERROR: ${err.message}`);
        console.error(err);
        stopAutoTrain();
    }
}

function updateVisuals() {
    try {
        const baseOut = baselineModel.predict(inputTensor);
        const studOut = studentModel.predict(inputTensor);
        
        drawTensor(ui.canvasBaseline, baseOut);
        drawTensor(ui.canvasStudent, studOut);
        
        const baseLoss = mseLoss(inputTensor, baseOut).dataSync()[0].toFixed(4);
        const studLoss = calculateStudentLoss(inputTensor, studOut).dataSync()[0].toFixed(4);
        
        log(`Step ${stepCount} | Base: ${baseLoss} | Student: ${studLoss}`);
        
        baseOut.dispose();
        studOut.dispose();
    } catch (err) {
        log(`Visual ERROR: ${err.message}`);
    }
}

function stopAutoTrain() {
    isAutoTraining = false;
    ui.btnAuto.textContent = "Auto Train (Start)";
    ui.btnAuto.classList.remove('danger');
    if (autoTrainId) {
        cancelAnimationFrame(autoTrainId);
        autoTrainId = null;
    }
}

function startAutoTrain() {
    isAutoTraining = true;
    ui.btnAuto.textContent = "Auto Train (Stop)";
    ui.btnAuto.classList.add('danger');
    
    function loop() {
        if (!isAutoTraining) return;
        trainStep();
        autoTrainId = requestAnimationFrame(loop);
    }
    loop();
}

// Инициализация
async function init() {
    try {
        log("Loading TensorFlow.js...");
        await tf.ready();
        log(`TF.js version: ${tf.version.tfjs}`);
        
        generateFixedNoise();
        createModels();
        
        drawTensor(ui.canvasInput, inputTensor);
        drawTensor(ui.canvasBaseline, baselineModel.predict(inputTensor));
        drawTensor(ui.canvasStudent, studentModel.predict(inputTensor));
        
        log("✓ Ready! Click Train to start");
    } catch (err) {
        log(`FATAL ERROR: ${err.message}`);
        console.error(err);
    }
}

// Обработчики событий
ui.btnTrain.addEventListener('click', () => {
    trainStep();
    updateVisuals();
});

ui.btnAuto.addEventListener('click', () => {
    if (isAutoTraining) {
        stopAutoTrain();
    } else {
        startAutoTrain();
    }
});

ui.btnReset.addEventListener('click', () => {
    stopAutoTrain();
    generateFixedNoise();
    createModels();
    drawTensor(ui.canvasInput, inputTensor);
    drawTensor(ui.canvasBaseline, baselineModel.predict(inputTensor));
    drawTensor(ui.canvasStudent, studentModel.predict(inputTensor));
    log("Reset complete");
});

ui.selectArch.addEventListener('change', () => {
    stopAutoTrain();
    createModels();
    updateVisuals();
});

ui.rangeSmooth.addEventListener('input', (e) => ui.valSmooth.textContent = e.target.value);
ui.rangeDir.addEventListener('input', (e) => ui.valDir.textContent = e.target.value);

// Запуск
init();
