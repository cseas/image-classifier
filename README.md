# Image Classifier

In this project, I trained an image classifier with TensorFlow for Poets, starting from just a directory of images. TensorFlow is an open source library for numerical computation, specializing in machine learning applications. 

I created a classifier to tell the difference between five types of flowers: roses, sunflowers, tulips, dandelions, and daisies.

I've used transfer learning, which means I started with a model that had been already trained on another problem. I then retrained it on a similar problem. Deep learning from scratch can take days, but transfer learning can be done in short order.

I've used a model trained on the [ImageNet](http://image-net.org/) Large Visual Recognition Challenge [dataset](http://www.image-net.org/challenges/LSVRC/2012/). These models can differentiate between 1,000 different classes, like Dalmatian or dishwasher. We have a choice of model architectures, so we can determine the right tradeoff between speed, size and accuracy for our problem. I've used this same model, but retrained it to tell apart a small number of classes based on my own examples. I've used a [dataset](http://download.tensorflow.org/example_images/flower_photos.tgz) of creative-commons licensed flower photos to use initially.