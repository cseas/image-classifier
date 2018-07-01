# Image Classifier

In this project, I trained an image classifier with TensorFlow for Poets, starting from just a directory of images. TensorFlow is an open source library for numerical computation, specializing in machine learning applications. 

I created a classifier to tell the difference between five types of flowers: roses, sunflowers, tulips, dandelions, and daisies.

I've used transfer learning, which means I started with a model that had been already trained on another problem. I then retrained it on a similar problem. Deep learning from scratch can take days, but transfer learning can be done in short order.

I've used a model trained on the [ImageNet](http://image-net.org/) Large Visual Recognition Challenge [dataset](http://www.image-net.org/challenges/LSVRC/2012/). These models can differentiate between 1,000 different classes, like Dalmatian or dishwasher. We have a choice of model architectures, so we can determine the right tradeoff between speed, size and accuracy for our problem. I've used this same model, but retrained it to tell apart a small number of classes based on my own examples. I've used a [dataset](http://download.tensorflow.org/example_images/flower_photos.tgz) of creative-commons licensed flower photos.

## Configure MobileNet

I've retrained a [MobileNet](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html). MobileNet is a small efficient convolutional neural network. "Convolutional" just means that the same calculations are performed at each location in the image.

The MobileNet is configurable in two ways:

    Input image resolution: 128,160,192, or 224px. Unsurprisingly, feeding in a higher resolution image takes more processing time, but results in better classification accuracy.
    The relative size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.50, or 0.25.

I've used 224 0.5 for this project.

With the recommended settings, it typically takes only a couple of minutes to retrain on a laptop. I've passed the settings inside Linux shell variables.

## Start TensorBoard

Before starting the training, launch tensorboard in the background. TensorBoard is a monitoring and inspection tool included with tensorflow. You will use it to monitor the training progress.

```shell
sh tensorboard.sh
```

This command will fail with the following error if you already have a tensorboard process running:

```
ERROR:tensorflow:TensorBoard attempted to bind to port 6006, but it was already in use
```

You can kill all existing TensorBoard instances with:

```shell
pkill -f "tensorboard"
```

## Investigate the retraining script

The retrain script is from the TensorFlow Hub repo, but it is not installed as part of the pip package. So for simplicity I've included it in the codelab repository. You can run the script using the python command. Take a minute to skim its "help".

```shell
python3 -m scripts.retrain -h
```

## Run the training

As noted in the introduction, ImageNet models are networks with millions of parameters that can differentiate a large number of classes. We're only training the final layer of that network, so training will end in a reasonable amount of time.

Start your retraining with the following command (note the --summaries_dir option, sending training progress reports to the directory that tensorboard is monitoring):

```shell
sh training.sh
```

Note that this step will take a while.
This script downloads the pre-trained model, adds a new final layer, and trains that layer on the flower photos.

ImageNet does not include any of these flower species we're training on here. However, the kinds of information that make it possible for ImageNet to differentiate among 1,000 classes are also useful for distinguishing other objects. By using this pre-trained network, we are using that information as input to the final classification layer that distinguishes our flower classes.