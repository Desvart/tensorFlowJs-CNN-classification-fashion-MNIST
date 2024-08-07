// import {TRAINING_DATA as TRAINING_DATA_0} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js';

// console.log(`Dimensions of Fashion MNIST data: ${getdim(TRAINING_DATA.inputs)}`);
// console.log(`Max value of Fashion MNIST data: ${tf.max(tf.tensor2d(TRAINING_DATA.inputs))}`);
//
// console.log(`Dimensions of Fashion MNIST data: ${getdim(TRAINING_DATA_0.inputs)}`);
// console.log(`Max value of historical MNIST data: ${tf.max(tf.tensor2d(TRAINING_DATA_0.inputs))}`);
// debugger;


// Define the initial speed of the animation.
let interval = 2000;
// Define the classes for the Fashion MNIST dataset.
const FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

// Get HTML elements.
const PREDICTION_ELEMENT = document.getElementById('prediction');
const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');
const RANGER = document.getElementById('ranger');
const DOM_SPEED = document.getElementById('domSpeed');

// When user drags slider update interval.
RANGER.addEventListener('input', function(e) {
    interval = this.value;
    DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms';
});

// Grab a reference to the MNIST input (pixel data) and output values.
const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Convert the inputs and outputs to tensors.
const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Create the model architecture
const model = tf.sequential();

model.add(tf.layers.conv2d({
    filters: 16,
    kernelSize: 3, // Square Filter of 3 by 3. Could also specify rectangle eg [2, 3].
    strides: 1,
    padding: 'same',
    activation: 'relu',
    inputShape: [28, 28, 1]
}));

model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    strides: 1,
    padding: 'same',
    activation: 'relu'
}));

model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({units: 128, activation: 'relu'}));

model.add(tf.layers.dense({units: 10, activation: 'softmax'}))

model.summary();

await train();
evaluate();


// FUNCTIONS ----------------------------------------------------------------------------------------------------------

function normalize(tensor, min, max) {
    const result = tf.tidy(function() {
        const MIN_VALUES = tf.scalar(min);
        const MAX_VALUES = tf.scalar(max);

        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

        return NORMALIZED_VALUES;
    });
    return result;
}

async function train() {

    // Compile the model with the defined optimizer and specify our loss function to use.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]);

    let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {
        shuffle: true,
        validationSplit: 0.15,
        batchSize: 256,
        epochs: 30,
        callbacks: {onEpochEnd: logProgress}
    });

    RESHAPED_INPUTS.dispose();
    OUTPUTS_TENSOR.dispose();
    INPUTS_TENSOR.dispose();
}

function logProgress(epoch, logs) {
    // console.log(`Epoch: ${epoch} - loss: ${Math.sqrt(logs.loss)} - accuracy: ${logs.acc}`);
    console.log(`Epoch: ${epoch} - `, logs);
}

function evaluate() {
    const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs.

    let answer = tf.tidy(function() {
        // let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]));
        let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255);
        let output = model.predict(newInput.reshape([1, 28, 28, 1]));
        output.print();
        return output.squeeze().argMax();
    });

    answer.array().then(function(index) {
        PREDICTION_ELEMENT.innerText = FASHION_MNIST_CLASSES[index];
        PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');
        answer.dispose();
        drawImage(INPUTS[OFFSET]);
    });
}

function drawImage(digit) {
    var imageData = CTX.getImageData(0, 0, 28, 28);

    for (let i = 0; i < digit.length; i++) {
        imageData.data[i * 4] = digit[i];      // Red Channel.
        imageData.data[i * 4 + 1] = digit[i];  // Green Channel.
        imageData.data[i * 4 + 2] = digit[i];  // Blue Channel.
        imageData.data[i * 4 + 3] = 255;             // Alpha Channel.
    }

    // Render the updated array of data to the canvas itself.
    CTX.putImageData(imageData, 0, 0);

    // Perform a new classification after a certain interval.
    window.requestAnimationFrame(function() {
        setTimeout(evaluate, interval);
    });
}

function drawImage2(digit, interval) {
    digit = tf.tensor(digit, [28, 28]);
    tf.browser.toPixels(digit, CANVAS);

    // Perform a new classification after a certain interval.
    setTimeout(evaluate, interval);
}

function array_equals(a, b)
{
    return a.length === b.length && a.every(function(value, index) {
        return value === b[index];
    });
}

function getdim(arr)
{
    if (/*!(arr instanceof Array) || */!arr.length) {
        return []; // current array has no dimension
    }
    var dim = arr.reduce(function(result, current) {
        // check each element of arr against the first element
        // to make sure it has the same dimensions
        return array_equals(result, getdim(current)) ? result : false;
    }, getdim(arr[0]));

    // dim is either false or an array
    return dim && [arr.length].concat(dim);
}