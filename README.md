# Image Classifier

In this project, I trained an image classifier with TensorFlow for Poets, starting from just a directory of images. TensorFlow is an open source library for numerical computation, specializing in machine learning applications. 

I created a classifier to tell the difference between five types of flowers: roses, sunflowers, tulips, dandelions, and daisies. To use this classifier to detect another class of images, create a directory inside flower_photos (specified in training.sh) with the label name and provide about a 100 images to start with.

I've used Transfer Learning (Retraining), which means I started with a model that had been already trained on another problem. I then retrained it on a similar problem. Deep learning from scratch can take days, but transfer learning can be done in short order. Retraining saves a lot of time, and leverages prior work.

I've used a model trained on the [ImageNet](http://image-net.org/) Large Visual Recognition Challenge [dataset](http://www.image-net.org/challenges/LSVRC/2012/). These models can differentiate between 1,000 different classes, like Dalmatian or dishwasher. We have a choice of model architectures, so we can determine the right tradeoff between speed, size and accuracy for our problem. I've used this same model, but retrained it to tell apart a small number of classes based on my own examples. I've used a [dataset](http://download.tensorflow.org/example_images/flower_photos.tgz) of creative-commons licensed flower photos.

Under the hood, TensorFlow Poets starts with an existing classifier called Inception. Inception is one of Google's best image classifiers and is open source. It is trained on 1.2 million images from a thousand different categories. Training Inception took about 2 weeks on a fast desktops with 8 GPUs.

I concluded that Deep Learning has a major advantage when working with images. You don't need to extract features manually for classification. Instead, you can use the raw pixels of the image as features and the classifier will do the rest.

## Configure MobileNet

I've retrained a [MobileNet](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html). MobileNet is a small efficient convolutional neural network. "Convolutional" just means that the same calculations are performed at each location in the image.

The MobileNet is configurable in two ways:

    Input image resolution: 128,160,192, or 224px.  
    Unsurprisingly, feeding in a higher resolution image takes more processing time, but results in better classification accuracy.  
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

The retrain script is from the TensorFlow Hub repo, but it is not installed as part of the pip package. So for simplicity I've included it in the codelab repository. You can run the script using the python command.

## Run the training

As noted in the introduction, ImageNet models are networks with millions of parameters that can differentiate a large number of classes. We're only training the final layer of that network, so training will end in a reasonable amount of time.

Start your retraining with the following command (note the --summaries_dir option, sending training progress reports to the directory that tensorboard is monitoring):

```shell
sh training.sh
```

Note that this step will take a while.
This script downloads the pre-trained model, adds a new final layer, and trains that layer on the flower photos.

ImageNet does not include any of these flower species we're training on here. However, the kinds of information that make it possible for ImageNet to differentiate among 1,000 classes are also useful for distinguishing other objects. By using this pre-trained network, we are using that information as input to the final classification layer that distinguishes our flower classes.

## Using the Retrained Model

The retraining script writes data to the following two files:

* `tf_files/retrained_graph.pb`, which contains a version of the selected network with a final layer retrained on your categories.
* `tf_files/retrained_labels.txt`, which is a text file containing labels.

## Classifying an image

label_image.py script can be used to classify images. Take a minute to read the help for this script:
```shell
python -m scripts.label_image -h
```

Test a daisy image from the dataset using the following code:
```shell
python3 -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```

If you chose a mobilenet that takes a smaller input size, then be sure to set the --input_size flag using the shell variable you set earlier.

```shell
--input_size=${IMAGE_SIZE}
``` 

Test a downloaded image as follows:
```shell
python3 -m scripts.label_image --image=test.jpeg
```

## Trying Other Hyperparameters

The retraining script has several other command line options you can use. You can read about these options in the help for the retrain script:

```shell
python3 -m scripts.retrain -h
```

The `--learning_rate` parameter controls the magnitude of the updates to the final layer during training. So far we have left it out, so the program has used the default `learning_rate` value of `0.01`. If you specify a small `learning_rate`, like `0.005`, the training will take longer, but the overall precision might increase. Higher values of `learning_rate`, like `1.0`, could train faster, but typically reduces precision, or even makes training unstable.

You may want to set the following two options together, so your results are clearly labeled:

```shell
--learning_rate=0.5
--summaries_dir=training_summaries/LR_0.5
```

## Training on Your Own Categories

You can teach the network to recognize different categories.

In theory, all you need to do is run the tool, specifying a particular set of sub-folders. Each sub-folder is named after one of your categories and contains only images from that category.

If you complete this step and pass the root folder of the subdirectories as the argument for the `--image_dir` parameter, the script should train the images that you've provided, just like it did for the flowers.

The classification script uses the folder names as label names, and the images inside each folder should be pictures that correspond to that label, as you can see in the flower archive.
