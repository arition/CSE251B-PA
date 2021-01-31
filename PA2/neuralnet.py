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

import os
import gzip
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    Normalize your inputs here to have 0 mean and unit variance.
    """
    return (inp - inp.mean()) / inp.std()


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    encoded_labels = np.array([[1 if label == i else 0 for i in range(num_classes)] for label in labels])
    return encoded_labels


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
    one_hot_labels = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    return (np.exp(x) / np.sum(np.exp(x), axis=0)).T


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
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
        self.x = a
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
        grad = None

        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        elif self.activation_type == "leakyReLU":
            grad = self.grad_leakyReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return np.maximum(0, x)

    def leakyReLU(self, x):
        """
        Implement leaky ReLU here.
        """
        return np.maximum(0.1 * x, x)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - np.power(self.tanh(self.x), 2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return 1 if self.x > 0 else 0

    def grad_leakyReLU(self):
        """
        Compute the gradient for leaky ReLU here.
        """
        return 1 if self.x > 0 else 0.1


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

        self.delta_w_old = 0 # Save delta w
        self.delta_b_old = 0 # Save delta b

        self.w_min = self.w # Store the weight matrix
        self.b_min = self.b # Store the bias

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        self.x = x
        self.a = np.dot(x, self.w) + self.b
        return self.a

    def backward(self, delta, l2_penalty = 0):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        Num = self.x.shape[0]

        self.d_x = delta.dot(self.w.T)
        self.d_w = self.x.T.dot(delta) / Num - l2_penalty * self.w
        self.d_b = delta.sum(axis = 0) / Num

        return self.d_x

    def update_para(self, lr, momentum_En = False, gamma = None):
        if momentum_En:
            w_delta = lr * self.d_w + gamma * self.delta_w_old
            b_delta = lr * self.d_b + gamma * self.delta_b_old

            self.w += w_delta
            self.b += b_delta

            self.delta_w_old = w_delta
            self.delta_b_old = b_delta

        else:
            self.w += lr * self.d_w
            self.b += lr * self.d_b

    def store_para(self):
        self.w_min = self.w
        self.b_min = self.b

    def load_para(self):
        self.w = self.w_min
        self.b = self.b_min


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
        self.l2_penalty = None # For l2 penalty

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets

        # Forward Path
        Input = x
        for layer in self.layers:
            Input = layer.forward(Input)

        # Softmax Activation
        self.y = softmax(Input)

        # Loss
        loss = self.loss(self.y, targets)

        # l2 penalty
        if l2_penalty:
            for layer in layers:
                if isinstance(layer, Layer):
                    loss += (np.sum(layer.w ** 2)) * l2_penalty / 2

        return loss

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        Num = targets.shape[0]

        return - np.sum(np.multiply(targets, np.log(logits))) / Num

    def backward(self, l2_penalty = 0):
        '''
        Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                delta = layer.backward(delta, l2_penalty)
            else:
                delta = layer.backward(delta)

    def updata_para(self, lr, momentum_En = False, gamma = None):
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                layer.update_para(lr, momentum_En, gamma)

    def store_para(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.store_para()

    def load_para(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.load_para()

    def predict(self, x, targets):
        Input = x
        for layer in self.layers:
            Input = layer.forward(Input)

        predictions = np.argmax(softmax(Input))
        targets = np.argmax(targets)

        return np.mean(predictions == targets)

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    raise NotImplementedError("Train method not implemented")


def test(model, x_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    model.predict(x_test, y_test)


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test, y_test = load_data(path="./", mode="t10k")

    # TODO: Create splits for validation data here.
    # x_val, y_val = ...

    # TODO: train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)

    # TODO: Plots
    # plt.plot(...)
