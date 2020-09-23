class RPSDataset {
  constructor() {//As we add new examples to the dataset, we keep tracks of their labels. That's what the labels array does. So when we initialize the class in the beginning of the index.js file, we initialize this array to empty
    this.labels = []
  }

  addExample(example, label) {//Here 'example' is the output of the prediction of the image from the truncated MobileNet. The 'lable' is the value 0,1,2 for rock,paper,scissors respectively
    if (this.xs == null) {//For the first sample, the xs is null. So we set the xs to be the tf.keep for the example, and we push the label into the labels array.
      this.xs = tf.keep(example);//This is the oppsite of tf.tidy(). In tf.tidy() we wanted to discard the tensors not being used to save memory, in tf.keep(), we want to tensors to stay. With tf.keep(), we tell Tensorflow that we want to keep this tensor, wo dont throw it away in a tf.tidy(). It basically grants the tensor an exception, that even if a tf.tidy() is called, the tf.tidy() will discard other tensors but not this own
      this.labels.push(label);
    } else {//For all subsequent samples, we just append the new samples to the old.
        //We do this by creating a temp variable for the old set of xs called oldX, and we then tf.keep, then concat of the new example to that, and then dispose of the oldX.
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));
      this.labels.push(label);//We also push the label to the array
      oldX.dispose();
    }
  }
  
//encodeLabels() takes the array of labels and one hot encodes it for training. As one hot encoding is very memory inefficient, we keep a list of labels and only create the much larger list of one hot encoded ones before we train
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        this.ys = tf.keep(tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
      } else {
        const y = tf.tidy(
            () => {return tf.oneHot(
                tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
        const oldY = this.ys;
        this.ys = tf.keep(oldY.concat(y, 0));
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
