let mobilenet;
let model;          //Declares the mobilenet and model variable(similar to global variables) so that they could be shared with other functions in the script
const webcam = new Webcam(document.getElementById('wc'));       //This line creates a const for a webcam object, stored in webcam.js, initializing it by pointing it at the video element in the hosting page that we call wc for webcam. The webcam class whose object we have just created is in Webcam.js file
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');//Here, as before, we'll load the JSON model from its hosted URL and use tf.loaLayersModel to load it into an object.
  const layer = mobilenet.getLayer('conv_pw_13_relu');              //From here, we can now get one of the output layers from the preloaded mobilenet. We're using transfer learning here, where we pick a desired layer and retrain everything under that. Here, a layer called conv_pw_13 is being selected as the one above everything which will be frozen
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    //We'll then use the tf model class to make a new model and its constructor can take inputs and outputs, which we will set to take the mobilenet inputs, namely the top of the mobilenet and then conv pw 13 relu as output. So that everything beneath that layer will be ignored when we connect a new set of layers to this model.
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(3);
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 3, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        }
      }
   });
}


function handleButton(elem){
	switch(elem.id){
		case "0":
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
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(mobilenet.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
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
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){                      //Initialization function which sets up the webcam by calling webcam.setup() aynchronously
	await webcam.setup();
	mobilenet = await loadMobilenet();
	tf.tidy(() => mobilenet.predict(webcam.capture()));
		
}



init();
