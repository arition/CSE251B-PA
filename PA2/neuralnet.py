################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2021
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    return inp


def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    return labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    TODO: Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    raise NotImplementedError("Softmax not implemented")


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1/(1+(np.exp((-x))))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(0,x)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        raise np.where(x > 0, x, x * 0.01)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        raise NotImplementedError("tanh gradient not implemented")

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        raise NotImplementedError("ReLU gradient not implemented")

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        raise NotImplementedError("leakyReLU gradient not implemented")


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = None    # Declare the Weight matrix
        self.b = None    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        raise NotImplementedError("Layer forward pass not implemented.")

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        raise NotImplementedError("Backprop for Layer not implemented.")


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        raise NotImplementedError("Backprop not implemented for NeuralNetwork")


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """

    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # TODO: Create splits for validation data here.
    # x_val, y_val = ...

    # TODO: train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)

    # TODO: Plots
    # plt.plot(...)
