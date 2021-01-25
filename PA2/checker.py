################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import neuralnet
import numpy as np
import yaml
import pickle


def get_data(path):
    """
    Load the sanity data to verify your implementation.
    """
    return pickle.load(open(path + 'sanity.pkl', 'rb'))


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def check_error(error, msg):
    """
    Verify that error is below the threshold.
    """
    if error < 1e-6:
        print(f"{msg} is CORRECT")
    else:
        print(f"{msg} is WRONG")


def sanity_layers(data):
    """
    Check implementation of the forward and backward pass for all activations.
    """

    # Set the seed to reproduce results.
    np.random.seed(42)

    # Pseudo-input.
    random_input = np.random.randn(1, 50)

    # Get the activations.
    act_sigmoid = neuralnet.Activation('sigmoid')
    act_tanh    = neuralnet.Activation('tanh')
    act_ReLU    = neuralnet.Activation('ReLU')
    act_leakyReLU = neuralnet.Activation('leakyReLU')

    # Get the outputs for forward-pass.
    out_sigmoid = act_sigmoid(random_input)
    out_tanh    = act_tanh(random_input)
    out_ReLU    = act_ReLU(random_input)
    out_leakyReLU = act_leakyReLU(random_input)

    # Compute the errors.
    err_sigmoid = np.sum(np.abs(data['out_sigmoid'] - out_sigmoid))
    err_tanh    = np.sum(np.abs(data['out_tanh'] - out_tanh))
    err_ReLU    = np.sum(np.abs(data['out_ReLU'] - out_ReLU))
    err_leakyReLU = np.sum(np.abs(data['out_leakyReLU'] - out_leakyReLU))

    # Check the errors.
    check_error(err_sigmoid, "Sigmoid Forward Pass")
    check_error(err_tanh,    "Tanh Forward Pass")
    check_error(err_ReLU,    "ReLU Forward Pass")
    check_error(err_leakyReLU,    "leakyReLU Forward Pass")

    print(20 * "-", "\n")

    # Compute the gradients.
    grad_sigmoid = act_sigmoid.backward(1.0)
    grad_tanh    = act_tanh.backward(1.0)
    grad_ReLU    = act_ReLU.backward(1.0)
    grad_leakyReLU = act_leakyReLU.backward(1.0)

    # Compute the errors.
    err_sigmoid_grad = np.sum(np.abs(data['grad_sigmoid'] - grad_sigmoid))
    err_tanh_grad    = np.sum(np.abs(data['grad_tanh'] - grad_tanh))
    err_ReLU_grad    = np.sum(np.abs(data['grad_ReLU'] - grad_ReLU))
    err_leakyReLU_grad = np.sum(np.abs(data['grad_leakyReLU'] - grad_leakyReLU))

    # Check the errors.
    check_error(err_sigmoid_grad, "Sigmoid Gradient")
    check_error(err_tanh_grad,    "Tanh Gradient")
    check_error(err_ReLU_grad,    "ReLU Gradient")
    check_error(err_leakyReLU_grad, "leakyReLU Gradient")

    print(20 * "-", "\n")


def sanity_network(data, default_config):
    """
    Check implementation of the neural network's forward pass and backward pass.
    """    
    # Set seed to reproduce results.
    np.random.seed(42)

    # Random input for our network.
    random_image = np.random.randn(1, 784)

    # Initialize the network using the default configuration
    nnet = neuralnet.Neuralnetwork(default_config)

    # Compute the forward pass.
    nnet(random_image, targets = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    # Compute the backward pass.
    nnet.backward()

    layer_no = 0
    for layer_idx, layer in enumerate(nnet.layers):
        if isinstance(layer, neuralnet.Layer):
            layer_no += 1
            error_x   = np.sum(np.abs(data['nnet'].layers[layer_idx].x   - layer.x))
            error_w   = np.sum(np.abs(data['nnet'].layers[layer_idx].w   - layer.w))
            error_b   = np.sum(np.abs(data['nnet'].layers[layer_idx].b   - layer.b))
            error_d_w = np.sum(np.abs(data['nnet'].layers[layer_idx].d_w - layer.d_w))
            error_d_b = np.sum(np.abs(data['nnet'].layers[layer_idx].d_b - layer.d_b))

            check_error(error_x,   f"Layer{layer_no}: Input")
            check_error(error_w,   f"Layer{layer_no}: Weights")
            check_error(error_b,   f"Layer{layer_no}: Biases")
            check_error(error_d_w, f"Layer{layer_no}: Weight Gradient")
            check_error(error_d_b, f"Layer{layer_no}: Bias Gradient")

    print(20 * "-", "\n")


if __name__ == '__main__':
    # Load the data and configuration.
    sanity_data    = get_data("./")
    default_config = load_config("./")

    # Run Sanity.
    sanity_layers(sanity_data)
    sanity_network(sanity_data, default_config)
