#!/usr/bin/env python
# coding: utf-8

# # PA2 of CSE 251B, WI 21

# ## (a) Load training and testing data then create a validation split from the training data.

# In[3]:


from neuralnet import *
import numpy as np

# Load the configuration.
config = load_config("./")

# Create the model
model  = Neuralnetwork(config)

 # Load the data
x_train, y_train = load_data(path="./", mode="train")
x_test, y_test = load_data(path="./", mode="t10k")

x_train = normalize_data(x_train)
# Y_train = one_hot_encoding(labels=Y_train)
x_test = normalize_data(x_test)
# y_test = one_hot_encoding(labels=y_test)

# Create splits for validation data here.
x_train, y_train, x_valid, y_valid = data_spliter(x_train, y_train, percentage=0.1)


# ## Part (b) Estimation of bias weight and weight

# In[6]:

print('Part b:')
print(' ')
print(' ')


from neuralnet import *
from random import shuffle
import numpy as np

# Functions
def check_d_b(model, layer, eps, output_idx):
    layer.b[0][output_idx] += eps # w+eps
    loss_1 = model.forward(np.array(xtrain), np.array(ytrain))[1]
    layer.b[0][output_idx] -= 2*eps # w-eps
    loss_2 = model.forward(np.array(xtrain), np.array(ytrain))[1]
    d_b_get = (loss_1 - loss_2) / (2 * eps) # Numerical estimation
    layer.b[0][output_idx] += eps # back to original para
    return d_b_get

def check_d_w(model, layer, eps, input_idx, output_idx):
    layer.w[input_idx][output_idx] += eps # w+eps
    loss_1 = model.forward(np.array(xtrain), np.array(ytrain))[1]
    layer.w[input_idx][output_idx] -= 2*eps # w-eps
    loss_2 = model.forward(np.array(xtrain), np.array(ytrain))[1]
    d_w_get = (loss_1 - loss_2) / (2 * eps) # Numerical estimation
    layer.w[input_idx][output_idx] += eps # back to original para
    return d_w_get

# para
eps = 0.01

# Data loading and spliting
x_b, y_b = load_data(path="./", mode="train")

class_num = list(range(10))
xtrain, ytrain = [], []
for idx in range(y_b.shape[0]):
    if len(class_num) == 0:
        break
    if np.argmax(y_b[idx]) in class_num:
        xtrain.append(x_b[idx])
        ytrain.append(y_b[idx])
        class_num.remove(np.argmax(y_b[idx]))
        
# Load model para
config_b= yaml.load(open('./partb.yaml', 'r'), Loader=yaml.SafeLoader)

model = Neuralnetwork(config_b)
model.forward(np.array(xtrain), np.array(ytrain))
model.backward()

# Calaulation
d_b, d_b_estimate, d_w, d_w_estimate = [], [], [], []
for layer in model.layers:
    if isinstance(layer, Layer):
        d_b_estimate.append(check_d_b(model=model, layer=layer, eps=eps, output_idx=1))
        d_w_estimate.append([check_d_w(model=model, layer=layer, eps=eps, input_idx=0, output_idx=1), check_d_w(model=model, layer=layer, eps = eps, input_idx=0, output_idx=2)])
        d_b.append(layer.d_b[1] * 10) # multiply by the scaling factor
        d_w.append([np.multiply(layer.d_w[0][1], 10) ,np.multiply(layer.d_w[0][2], 10)]) # multiply by the scaling factor
print('Real b: {}'.format(d_b))
print('Estimate b: {}'.format(d_b_estimate))
print('Real w: {}'.format(d_w))
print('Estimate w: {}'.format( d_w_estimate))
        





# ## (c) Cross Validation

# In[ ]:

print('Part c:')
print(' ')
print(' ')

# load model para
config_c = yaml.load(open('./partc.yaml', 'r'), Loader=yaml.SafeLoader)

# Train the model with cross validation
max_test_accu = 0

model_c = Neuralnetwork(config_c)
recording = train(model_c, x_train, y_train, x_valid, y_valid, config_c)

# Recall parameters with minimum validation loss
model_c.load_para()
test_accuracy = test(model_c, x_test, y_test)
max_test_accu = max(max_test_accu, test_accuracy)
        
print('max test accuracy: {:.4f}'.format(max_test_accu))


# ## (c) training and validation accuracy / loss vs number of training epochs

# In[ ]:


# Plots
plt.figure(1)
plt.plot(recording['epoches'], recording['train_loss'], label='train')
plt.plot(recording['epoches'], recording['valid_loss'], label='validation')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Training and validation loss vs number of training epochs')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(recording['epoches'], recording['train_accuracy'], label='train')
plt.plot(recording['epoches'], recording['valid_accuracy'], label='validation')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy vs number of training epochs')
plt.legend()
plt.show()


# ## (d) Experiment with Regularization

# In[4]:
print('Part d:')
print(' ')
print(' ')

# load model para
config_d = yaml.load(open('./partd.yaml', 'r'), Loader=yaml.SafeLoader)

recordings = []

for l2_penalty in [1e-2, 1e-3, 1e-4]:
    config_d['L2_penalty'] = l2_penalty

    # Train the model with cross validation
    max_test_accu = 0

    model_d = Neuralnetwork(config_d)
    recording = train(model_d, x_train, y_train, x_valid, y_valid, config_d)
    recordings.append(recording)

    # Recall parameters with minimum validation loss
    model_d.load_para()
    test_accuracy = test(model_d, x_test, y_test)
    max_test_accu = max(max_test_accu, test_accuracy)
            
    print(f'l2_penalty: {l2_penalty}, max test accuracy: {max_test_accu:.4f}')


# In[12]:


for i, l2_penalty in enumerate([1e-2, 1e-3, 1e-4]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(recordings[i]['epoches'], recordings[i]['train_loss'], label='train')
    ax1.plot(recordings[i]['epoches'], recordings[i]['valid_loss'], label='validation')
    ax1.set_xlabel('Epoches')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss with L2_penalty={l2_penalty}')
    ax1.legend()

    ax2.plot(recordings[i]['epoches'], recordings[i]['train_accuracy'], label='train')
    ax2.plot(recordings[i]['epoches'], recordings[i]['valid_accuracy'], label='validation')
    ax2.set_xlabel('Epoches')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy with L2_penalty={l2_penalty}')
    ax2.legend()
    fig.show()


# ## (e) Experiment with Activations.

# In[5]:

print('Part e:')
print(' ')
print(' ')

# load model para
config_e = yaml.load(open('./parte.yaml', 'r'), Loader=yaml.SafeLoader)

recordings_e = []

for activation in ['sigmoid', 'tanh', 'ReLU', 'leakyReLU']:
    config_e['activation'] = activation

    # Train the model with cross validation
    max_test_accu = 0

    model_e = Neuralnetwork(config_e)
    recording = train(model_e, x_train, y_train, x_valid, y_valid, config_e)
    recordings_e.append(recording)

    # Recall parameters with minimum validation loss
    model_e.load_para()
    test_accuracy = test(model_e, x_test, y_test)
    max_test_accu = max(max_test_accu, test_accuracy)
            
    print(f'activation: {activation}, max test accuracy: {max_test_accu:.4f}')


# In[13]:


for i, activation in enumerate(['sigmoid', 'tanh', 'ReLU', 'leakyReLU']):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(recordings_e[i]['epoches'], recordings_e[i]['train_loss'], label='train')
    ax1.plot(recordings_e[i]['epoches'], recordings_e[i]['valid_loss'], label='validation')
    ax1.set_xlabel('Epoches')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss with activation={activation}')
    ax1.legend()

    ax2.plot(recordings_e[i]['epoches'], recordings_e[i]['train_accuracy'], label='train')
    ax2.plot(recordings_e[i]['epoches'], recordings_e[i]['valid_accuracy'], label='validation')
    ax2.set_xlabel('Epoches')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy with activation={activation}')
    ax2.legend()
    fig.show()



# ## Part (f) Experiment with Network Topology

# ### (i) Different number of hidden layer

# In[2]:
print('Part f:')
print(' ')
print(' ')

# Layer Specs: 50  100, 100  50, 25  50, and 50  25
Hidden_Layer = [[50, 100], [100, 50], [25, 50], [50, 25]]
config = yaml.load(open('./parte.yaml', 'r'), Loader=yaml.SafeLoader)

# # Load the data
# x_train, y_train = load_data(path="./", mode="train")
# x_test, y_test = load_data(path="./", mode="t10k")

# x_train = normalize_data(x_train)
# # Y_train = one_hot_encoding(labels=Y_train)
# x_test = normalize_data(x_test)
# # y_test = one_hot_encoding(labels=y_test)

# # Create splits for validation data here.
# x_train, y_train, x_valid, y_valid = data_spliter(x_train, y_train, percentage=0.2)

for hidden in Hidden_Layer:
    config['layer_specs'][1:3] = hidden
    print(config['layer_specs'])
    model  = Neuralnetwork(config)


    # train the model
    recording = train(model, x_train, y_train, x_valid, y_valid, config)

    # Recall parameters with minimum validation loss
    model.load_para()

    test_accuracy = test(model, x_test, y_test)

    print('Test_accuracy: {}'.format(test_accuracy))

    # Plots
    plt.figure
    plt.plot(recording['epoches'], recording['train_loss'], label='train')
    plt.plot(recording['epoches'], recording['valid_loss'], label='validation')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Hidden Layer number: {} X {}'.format(hidden[0],hidden[1]))
    plt.show()

    plt.figure
    plt.plot(recording['epoches'], recording['train_accuracy'], label='train')
    plt.plot(recording['epoches'], recording['valid_accuracy'], label='validation')
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Hidden Layer number: {} X {}'.format(hidden[0],hidden[1]))
    plt.show()



# ### (ii) Three hidden layers

# In[4]:


config = yaml.load(open('./parte.yaml', 'r'), Loader=yaml.SafeLoader)
config['layer_specs'] = [784, 50, 30, 30, 10]
model  = Neuralnetwork(config)

# # Load the data
# x_train, y_train = load_data(path="./", mode="train")
# x_test, y_test = load_data(path="./", mode="t10k")

# x_train = normalize_data(x_train)
# # Y_train = one_hot_encoding(labels=Y_train)
# x_test = normalize_data(x_test)
# # y_test = one_hot_encoding(labels=y_test)

# # Create splits for validation data here.
# x_train, y_train, x_valid, y_valid = data_spliter(x_train, y_train, percentage=0.2)

# train the model
recording = train(model, x_train, y_train, x_valid, y_valid, config)

# Recall parameters with minimum validation loss
model.load_para()

test_accuracy = test(model, x_test, y_test)

print('Test_accuracy: {}'.format(test_accuracy))

# Plots
plt.figure
plt.plot(recording['epoches'], recording['train_loss'], label='train')
plt.plot(recording['epoches'], recording['valid_loss'], label='validation')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.title('Three Hidden Layer with: {} X {} X {}'.format(config['layer_specs'][1], config['layer_specs'][2], config['layer_specs'][3]))
plt.show()

plt.figure
plt.plot(recording['epoches'], recording['train_accuracy'], label='train')
plt.plot(recording['epoches'], recording['valid_accuracy'], label='validation')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Three Hidden Layer with: {} X {} X {}'.format(config['layer_specs'][1], config['layer_specs'][2], config['layer_specs'][3]))
plt.show()


# ### (iii) One hidden layer

# In[7]:

config = yaml.load(open('./parte.yaml', 'r'), Loader=yaml.SafeLoader)
config['layer_specs'] = [784, 100, 10]
model  = Neuralnetwork(config)

# # Load the data
# x_train, y_train = load_data(path="./", mode="train")
# x_test, y_test = load_data(path="./", mode="t10k")

# x_train = normalize_data(x_train)
# # Y_train = one_hot_encoding(labels=Y_train)
# x_test = normalize_data(x_test)
# # y_test = one_hot_encoding(labels=y_test)

# # Create splits for validation data here.
# x_train, y_train, x_valid, y_valid = data_spliter(x_train, y_train, percentage=0.2)

# train the model
recording = train(model, x_train, y_train, x_valid, y_valid, config)

# Recall parameters with minimum validation loss
model.load_para()

test_accuracy = test(model, x_test, y_test)

print('Test_accuracy: {}'.format(test_accuracy))

# Plots
plt.figure
plt.plot(recording['epoches'], recording['train_loss'], label='train')
plt.plot(recording['epoches'], recording['valid_loss'], label='validation')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.title('One Hidden Layer with: 100')
plt.show()

plt.figure
plt.plot(recording['epoches'], recording['train_accuracy'], label='train')
plt.plot(recording['epoches'], recording['valid_accuracy'], label='validation')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.title('One Hidden Layer with: 100')
plt.show()



# In[ ]:




