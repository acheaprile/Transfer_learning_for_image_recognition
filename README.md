# Transfer Learning for Image recognition
A ML model for image recognition using transfer learning.

## General info
The aim of this repository was to create a simple image recognition model using the weights of a pre-trained model (using weights learned on [ImageNet](http://www.image-net.org/) for this specific example but any of the [Keras Applications Models](https://keras.io/api/applications/) can be used here). This model is composed of a pre-trained model and a model on top of it composed by six Dense layers. Five of these layers use a Relu function as their activation function and the last one uses a Softmax as activaction function in order to assign probabilities to the model output. Please, note that the 5th layer of the vanilla Neural Network on top of the pre-trained model has been named as "extraction", in order to facilitate the image feature extraction from this model in any other future task (e.g. Image captioning using seq2seq).


## Project structure

This repository contains only one file which must be executed in the following order:

1. **Pre-trained model and New model on top (*mainmodel.py*):**

## Libraries

The following Python libraries were used in this repository:
- Tensorflow Keras and TF Keras Preprocessing
- Os

## Contact

Please, contact me on my email danielgarciache@gmail.com for any suggestion or question!
Thank you for visiting my GitHub! :)
