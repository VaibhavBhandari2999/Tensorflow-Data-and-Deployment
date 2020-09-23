let mobilenet;
let model;          //Declares the mobilenet and model variable(similar to global variables) so that they could be shared with other functions in the script
const webcam = new Webcam(document.getElementById('wc'));       //This line creates a const for a webcam object, stored in webcam.js, initializing it by pointing it at the video element in the hosting page that we call wc for webcam. The webcam class whose object we have just created is in Webcam.js file

const dataset = new RPSDataset();//Declares an object of the RPSDataset class present in rps-dataset.js file. This object is used in all the functions of this index.js file. This is like initializing the class RPSDataset with its constructor

var rockSamples=0, paperSamples=0, scissorsSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');//Here, as before, we'll load the JSON model from its hosted URL and use tf.loaLayersModel to load it into an object.
  const layer = mobilenet.getLayer('conv_pw_13_relu');              //From here, we can now get one of the output layers from the preloaded mobilenet. We're using transfer learning here, where we pick a desired layer and retrain everything under that. Here, a layer called conv_pw_13 is being selected as the one above everything which will be frozen
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    //We'll then use the tf model class to make a new model and its constructor can take inputs and outputs, which we will set to take the mobilenet inputs, namely the top of the mobilenet and then conv pw 13 relu as output. So that everything beneath that layer will be ignored when we connect a new set of layers to this model.
}

//Code for the new Deep NN that we'll use to classify with Transfer Learning
//******************************Unlike python, its not added/bolted onto the original model. Its an entirely seperate one which takes as its input the output of the previous model*************************
//As its a seperate model, we'll define it as we define any other model.
async function train() {
    
    //When we train, we use the y's. So first, we'll set the y's to null, and then we'll call encode labels passing it a three because we have three labels. It will then one-hot encode for us and put the results into dataset.ys.
  dataset.ys = null;
  dataset.encodeLabels(3);//We One hot encode the labels in the dataset as NNs cant operate on strings. If we make the strings as 0,1,2 respectively, the NN will give more weight to the 2 rather than the 0 which is wrong. Thats why we encode the strings like 001,010,100 which is called One Hot Encoding
    
    //Here, we define the layers for our model. The inputs of this will be the results of the output of the truncated mobile net, so we specify our input shape accordingly. 
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),//The first layer will be the flattened output from the mobile net model that we created earlier by truncating the full mobile net.
      tf.layers.dense({ units: 100, activation: 'relu'}),//A 100 unit hidden layer is created next
      tf.layers.dense({ units: 3, activation: 'softmax'})//Output layer will be 3 units as we'll try to transfer learning from the 1,000 class mobile net model to handle three specific classes that we need it to do.
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'}); //We will compile our model with the AdamOptimizer and categorical cross entropy as it is training on multiple categories.
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {//For model.fit we'll give it the dataset x's and dataset y's training for ten epochs. On each batch end we'll console.log the loss. 
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


//handleButton(this) is called in the retrain.html file to capture samples of the 3 classes(rock/paper/scissors). This will capture a frame from the camera for training
//Each button had an ID; zero, one, or two depending on the class. When the handleButton() function is called in retrain.html file, a 'this' is passed as arguement, which is a reference to the button or HTML element.
//So the parameter here ie elem which takes the 'this' value passed to it.
function handleButton(elem){
	switch(elem.id){
		case "0"://For the element with ID zero, we will run this code, where we increment the number of rock samples, and then find the div called rock samples and update its texts to read rock samples followed by the number of samples
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
	}
	label = parseInt(elem.id);//We'll then extract the label from the ID by converting it into int
	const img = webcam.capture();//We'll capture the content from our webcam so we can extract the features. This is the main steo where we take in the image, before this we j=have just identified for which class we're taking in the image
	dataset.addExample(mobilenet.predict(img), label);//The addExample function is in the dataset class present in rps-dataset.js file. We dont add the image captured from webcam to the dataset. Instead, we add the prediction of that image from the MobileNet. We add a example to the dataset, here an example is the image, and its classification, which we give by clicking one of the 3 buttons while capturing a frame

    //*******************************We were doing the transfer learning by removing the bottom layers from the MobileNet, truncating it, so that we just want its output to be the features learned at a higher level. If we then predict on the truncated one, then that's the output that we'll get. So we can train another neural network on those features instead of the raw webcam data and we'll effectively/basically have transfer learning. We then also pass the label to the dataset. Also, the label is a zero, a one, or a two. It's not one-hot encoded.
}


//So the first thing it will do is get the prediction by reading a frame from the webcam and classifying it. It will then evaluate the prediction and update the UI with, "I see rock," "I see paper," "I see scissors," etc. Finally, it will clean up so that it's a good browser citizen and saves space/time..
async function predict() {
    
    //Step 1: Gets Predictions
  while (isPredicting) {//The isPredicting variable is taken from the startPredicting and stopPredicting methods. When the isPredicting variable is True, the while loop is run completely. When isPredicting becomes false (by clicking on the stop predicting button), the the while loop is immediately broken
      
    //The while loop contains the code to read a frame from the webcam, use Mobilenet to get the activation, and then get a prediction from that with our retrained model. It then argmax's this and returns and returns it as 1D tensor containing the prediction
    const predictedClass = tf.tidy(() => {//As a lot of memory and tensors are being used frequently , we should tidy up to prevent memory leaks and this is done by tf.tidy()
      const img = webcam.capture();//We call webcam.capture() and pass the results to img variable
        
        //We take a set of embeddings by calling predict.mobilenet, passing in the image. We then take these embeddings and pass them to the new model to get a prediction photo. That means that when we train this model, we'll be training on the embeddings that sd gathered from moblienet.
        //We had a truncated MobileNet with the bottom layers removed so we could just see the activated convolutions, we'll pass the image to get it's set of activations
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);//Now, we can pass that to the model which was trained on these activations for rock paper and scissors classes to get a prediction back. This will then be one heart encoded with results for each of the three classes as probabilities.
      return predictions.as1D().argMax();//The predictions will be outputed as One Hot Encoded values, so we have to turn them back into 0,1,2 values. So we argmax predictions to turn it back into 0,1,2 values
    });
      
      //Step 2: Evaluate Prediction and update 
      
    const classId = (await predictedClass.data())[0];///The above function(predictedClass) is called so we can get a class ID from what the webcam sees. This class ID is fed into a switch statement which generates text for, "I see rock," "I see paper," "I see scissors."
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;//After getiing the predictedText, the text is dipslayed in the 'predictin' div in the HTML.
			
    //Step 3: Cleanup
    predictedClass.dispose();//We tidy up by disposing of the predicted class, which in turn runs the tf.tidy() function in the preidcted class function above
    await tf.nextFrame();//We call tf.nextFrame, which is a TensorFlow function that prevents UI thread from locking up so that our page can stay responsive.
  }
}

//The doTraining button will call this function which then calls the train function to train the dataset
function doTraining(){
	train();
}

//We'll create the start predicting method which simply turns on a boolean called is predicting, and that lets us to continual predictions and calls the predict function. The isPredicting variable is used in the predict() method.
function startPredicting(){
	isPredicting = true;
	predict();
}

//Similar to startPredicting() but here we put boolean as False and calls the predict function. The isPredicting variable is used in the predict() method.
function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){                      //Initialization function which sets up the webcam by calling webcam.setup() aynchronously
	await webcam.setup();
	mobilenet = await loadMobilenet();       //Called in order to get our model and initialize it
	tf.tidy(() => mobilenet.predict(webcam.capture()));
    //The first time around we can take a little time to load all the weights, etc. And so that we don't experience a lag when we want to start training or classifying, we do a webcam.capture to get a tensor and to ask mobilenet to predict what it sees in that. We don't need to do anything with this, but it does warm up the model for us. The tf.tidy and throws away any unneeded tensors so that they don't hang around taking up memory.   
		
}



init();
