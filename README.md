# machin-learning-mini-project

This repository includes my side projects of machine learning. I use these pet projects to practice the machine learning models and frameworks. 

Any comments or suggestions to improve either the model or data-processing are highly welcome! 😉 

### * Iris Classification

This project tries to classify different Iris samples into three categories. I use training set to build a simple model, and test out its performance on testing dataset. Please see file "Iris/Iris-classification-kaggle.ipynb" for implementation.

### * Student-Admission problem:
This project predicts if the student will be admitted to a graduate school based on features, including GPA, GRE score, and undergraduate school ranks. We practice basic operations and techniques for general machine learning solutions..

There are two solutions:
* single layer model: there is only one output layer without any intermediate hidden layer. 

	-  see "student_admission_single_layer_model.ipynb"

* multiple layer model: there is one hidden layer and one output layer.
	- see "student_admission_multiple_layer_model.ipynb"


### * Mini TensorFlow 11/20/2017
This project implements a prototype of tensorflow with basic operations including forward propagation and backward propagation. It use sigmoid function to facilitate the calculation of derivative of activation function. Also, it use basic stochastic gradient descent to minimize the loss function and tune the weights in the network. Please see "MiniFlow/miniFlow.py"

The other file "MiniFlow/main.py" is a driver function to predict the housing price using the miniFlow prototype. 

This project tries to implement mini-tensorflow from the scratch to understand the mechanism under the hood in depth. "

### * Basic Image Classification using Tensorflow 11/23/2017

"tf_mnist_classificatio"

This project makes use of basic tensorflow techniques to classify the images in MNIST dataset. This tiny project has most elements of a typical machine learning flow, which is good study case for beginner.  

### * Basic Practice Project using Keras

"Keras first cnn"

This project build a baby CNN using Keras framework. It is much concise and efficient to build CNN from scratch.