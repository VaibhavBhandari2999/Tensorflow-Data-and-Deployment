import {MnistData} from './data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;
	
function getModel() {
    
    //Defining a CNN for image classification
     
    //Its mostly similar to python, not fully
	model = tf.sequential();
	model.add(tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}));                                                             //First layer is 2D convolution which is defined as layers.conv2d. We are training on 28x28 monochrome MNIST image dataset, so input shape is [28,28,1] (1 because of monochrome)(RGB would have been 3). We use kernelSize property to define the size of the convolutional filter. kernelSize:3 specifies we want to use 3x3 filters. See Conv layers on the net.
    //filters:8 is basically a set of 8 filters we will attempt to learn convolution from.
    //Our activation function is ReLU to filter out values less than 0
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));                         //Next layer is maxPooling2D layer and its a 2x2 pool. See more about this on net
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));//Another convolutional layer, this time with 16 filters
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));//Another maxPool layer
	model.add(tf.layers.flatten());//Our data will be in chunks of 5500 images at a time, loaded in as an array of 5500 28x28 images with 10 labels. This array isnt the same shape as needed for training, so flatten() will take 28x28 pixels for each and flatten them out
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));//The 28x28 pixels flattened will be fed into the dense layer with 128 neurons, which will attempt to figure out a pattern between the learnt convolutions and the labels
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));//Ouput layer of 10 units, activated by Softmax that will provide the output classification into the 10 lables of the dataset

	model.compile({optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy']}); //Parameters to compile() function are passed in as a JS dictionary, not multiple parameters

    
	return model;
}

async function train(model, data) {
    
    //Declare metrics and container used in tfvis.show.fitCallbacks
	const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];//The metrics that we want it to track
	const container = { name: 'Model Training', styles: { height: '640px' } };//Should just contain a name and any required styles and the vis library will create the DOM elements to render the details.
    
    //FitCallbacks is an object.
    //A container is passed where it will render the feedbacks and a set of metrics it will track
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    //Now, when you're training, the callback will create a container in which it will draw the feedback depending on the metrics you select
    
    
  
	const BATCH_SIZE = 512;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

    
    //We want to create an array containing trainXs and trainYs, so function handles that. 
    //this function definition basically says that function should return array of trainXs and trainYs
	const [trainXs, trainYs] = tf.tidy(() => {
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);   //It gets the nextTraining batch from data source. IN MNIST, by default, train data size is 5500, so its basically getting 5500 lines of 784 bytes
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),//It them reshapes the data into a 4d tensor with 5500 in first dimension, then 28x28 representing the image, and 1 representing the color depth (monochrome for MNIST). It will return this as thbe first element in the array mapping to trainXs. 
            //As the labels are already One Hot Encoded, function will return them as the second element in the array
			d.labels
		];
	});
    //But all this is also under a tf.tidy() clause. Tensorflow apps tend to use a lot of memory(here we have allocated 5500*28*28 tensor).
    //So tf.tid(), once the execution is done, it cleans up all the intermediate tensors, except those it returns. So in this case, d gets cleaned up after we're done, and that saves us a lot of memory
    
    

	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE);
		return [
			d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

    //Training the model is done with the fit() method, where it is passed training data(trainXs) and labels(trainYs) along with a dictionary of parameters
	return model.fit(trainXs, trainYs, {
		batchSize: BATCH_SIZE,                    //Instead of flooding the model with all the data at once, it trains using a subset at a time. In browser its especially useful, as you dont lock up the browser itself
		validationData: [testXs, testYs], //If you want the model to validate as its training in order to report back an accuracy
		epochs: 20,
		shuffle: true,                //Shuffle the data to randomize it for trainin, preventing potential overfitting if multiple similar classes are in the same batch
		callbacks: fitCallbacks          //specify callbacks so we can update the user on training status amongst other things ar various events in the training cycle.
        
        //callbacks are also used as inputs to the tensorflow visualization library to track the progress as graphs
	});
}

function setPosition(e){
	pos.x = e.clientX-100;
	pos.y = e.clientY-100;
}
    
function draw(e) {//Happens when we are drawing on th canvas itself. it contains code for managing moving he mouse around and drawing on the canvas. But as soon as you finish drawing, the function copies the contents of the canvas as a PNG. rawimage is a variable pointing to the img tag in the mnist.html file . The raw image is hidden so the user doesn't actually see it
	if(e.buttons!=1) return;
	ctx.beginPath();
	ctx.lineWidth = 24;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'white';
	ctx.moveTo(pos.x, pos.y);
	setPosition(e);
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
}
    
function save() {//Does the inference for us
	var raw = tf.browser.fromPixels(rawImage,1);//Passing the rawimage(which is hidden) saved from the canvas and saying i want only 1 color channel, as we know its B&W
	var resized = tf.image.resizeBilinear(raw, [28,28]);//As the canvas was 280x280(to make it easier to draw), we have to resize it, so the rawimage is resized into a 28x28 array
	var tensor = resized.expandDims(0); //We resize the tensor to add another dimension as its a 4d tensor that we apass in. The first dimension is the number of images, second two dimensions are Xs and Ys of the images, last dimension is the color depth (monochrome here)
    
    //Right now, we have a 28 by 28 by one, but it's only a three-dimensional Tensor. So we add that fourth dimension to it, and it's just going to be a one in here by default. So when we do the actual prediction, we're saying, I'm giving you one image, that image is 28 by 28 and it's got a color depth of one. 
    
    var prediction = model.predict(tensor);//The tensor is passed , result is also a tensor which is going to have 10 values in it and those are the values of the 10 neurons of classification(1 for each label).
    //Right now, we have a 28 by 28 by one, but it's only a three-dimensional Tensor. So we add that fourth dimension to it, and it's just going to be a one in here by default. So when we do the actual prediction, we're saying, I'm giving you one image, that image is 28 by 28 and it's got a color depth of one. 
    var pIndex = tf.argMax(prediction, 1).dataSync();
    
	alert(pIndex);
}
    
function init() {
	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');
	ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
	canvas.addEventListener("mousemove", draw);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);        //The eventlistener for save will call the save function
	clearButton = document.getElementById('cb');
	clearButton.addEventListener("click", erase);
}


async function run() {  
	const data = new MnistData();//Creates an object called 'data' which is a new instance of MNISTData class 
 	await data.load();//run function will then call data.load which will download the sprite sheet, download the images and get them into memory. Its an sync function so we wait till its done fully, as we need th efull dataset at the same time
	const model = getModel();//Creates a new NN model
	tfvis.show.modelSummary({name: 'Model Architecture'}, model);//Show the model architecture 
	await train(model, data);//Train the model, its an sync function so we wait till it completes fully
	init();//Sets up the UI, which ahs the canvas we are going to draw on and the methods for that (mousemove,mouseup,mousedown,mouseenter) and has the save and clear buttons
	alert("Training is done, try classifying your handwriting!");
}

document.addEventListener('DOMContentLoaded', run);//As soon as the HTML document is loaded, it will call the run function



    
