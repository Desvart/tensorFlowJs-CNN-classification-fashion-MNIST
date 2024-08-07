// import {TRAINING_DATA as TRAINING_DATA_0} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';
import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js';

// console.log(`Dimensions of Fashion MNIST data: ${getdim(TRAINING_DATA.inputs)}`);
// console.log(`Max value of Fashion MNIST data: ${tf.max(tf.tensor2d(TRAINING_DATA.inputs))}`);
//
// console.log(`Dimensions of Fashion MNIST data: ${getdim(TRAINING_DATA_0.inputs)}`);
// console.log(`Max value of historical MNIST data: ${tf.max(tf.tensor2d(TRAINING_DATA_0.inputs))}`);
// debugger;

// Get HTML elements.
const PREDICTION_ELEMENT = document.getElementById('prediction');
const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');

// Define the classes for the Fashion MNIST dataset.
const FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];

// Grab a reference to the MNIST input (pixel data) and output values.
const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// Convert the inputs and outputs to tensors.
const INPUTS_TENSOR = tf.tensor2d(INPUTS);
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Normalize the data set.
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
INPUTS_TENSOR.dispose();
console.log(`Normalized data set min values: ${FEATURE_RESULTS.MIN_VALUES}`);
console.log(`Normalized data set max values: ${FEATURE_RESULTS.MAX_VALUES}`);

// Create the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 100, activation: 'relu'}));
model.add(tf.layers.dense({units: 20, activation: 'relu'}));
model.add(tf.layers.dense({units: 50, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

model.summary();

await train();
evaluate();


// FUNCTIONS ----------------------------------------------------------------------------------------------------------

function normalize(tensor, min, max) {
    const result = tf.tidy(function() {
        const MIN_VALUES = min || tf.min(tensor, 0);
        const MAX_VALUES = max || tf.max(tensor, 0);

        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

        return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES};
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

    let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
        shuffle: true,        // Ensure data is shuffled again before using each epoch.
        validationSplit: 0.1,
        batchSize: 128,       // Update weights after every 512 examples.
        epochs: 100,           // Go over the data 50 times!
        callbacks: {onEpochEnd: logProgress}
    });

    OUTPUTS_TENSOR.dispose();
    FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));

    // evaluate(); // Once trained we can evaluate the model.
}

function logProgress(epoch, logs) {
    console.log(`Epoch: ${epoch} - loss: ${Math.sqrt(logs.loss)} - accuracy: ${logs.acc}`);
}

function evaluate() {
    const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs.

    let answer = tf.tidy(function() {
        // let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]));
        let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]).expandDims(), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);
        let output = model.predict(newInput.NORMALIZED_VALUES);
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
    setTimeout(evaluate, 2000);
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