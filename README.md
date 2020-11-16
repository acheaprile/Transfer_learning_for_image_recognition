# Transfer Learning for Image recognition
A ML model for image recognition using transfer learning.

## General info
The aim of this repository was to create a simple image recognition model using the weights of a pre-trained model as starting point for our model (pre-trained model weights were learned on [ImageNet](http://www.image-net.org/) for this specific example but, any of the pre-trained [Keras Applications Models](https://keras.io/api/applications/) can be used here). The weights from the pre-trained model are not trainable in this example (initial layers are frozen with this purpose). This model is composed of a pre-trained set of layers with an additional model on top of them, which is composed by six Dense layers. Five of these layers use a Relu function as their activation function and the last one uses a Softmax as activaction function in order to assign probabilities to the model output. Please, note that the 5th layer of the vanilla Neural Network on top of the pre-trained model has been named as "extraction", in order to facilitate feature extraction from this model for any other task (e.g. Image captioning using seq2seq).

![TL1](https://i.ibb.co/gwV7MDg/tl1.jpg)

## Project structure

This repository contains only one file which must be executed in the following order:

1. **Pre-trained model and New model on top (*mainmodel.py*):**

## Libraries

The following Python libraries were used in this repository:
- Tensorflow Keras and TF Keras Preprocessing
- Os

## Contact

Please, feel free to contact me on my email danielgarciache@gmail.com for any suggestion or question!
Thank you for visiting my GitHub! :)
