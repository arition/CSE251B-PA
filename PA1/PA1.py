#!python
# -*- coding: utf-8 -*-
# Fangzhou Ai, Yue Qiao @ ECE, UCSD
from dataloader import load_data
from PCA import PCA
import numpy as np
import os, random, sys
import matplotlib.pyplot as plt
from PIL import Image

# Data load and preprocess and Cross Validation Procedure

def K_split(images, cnt, k):
    '''split the indices
    Args: 
        images, cnt, k fold
    Returns:
        K_split indices
    '''
    K_split={}
    for name, idx in zip(images, cnt):
        division = cnt[idx] / k
        index = list(range(cnt[idx]))
        random.shuffle(index)
        K_split[name] = np.array([index[round(division * i):round(division * (i + 1))] for i in range(k)], dtype = object)
    return K_split

def Cross_Validation_Procedure(images, K_split, NameList, indices):
    '''Get train, validation and test set
    Args: 
        images, cnt, K_split, NameList
    Returns: 
        train_data, train_label, validation_data,
        validation_label, test_data, test_label
    '''
    train_data = []
    valid_data = []
    test_data = []
    train_label = []
    valid_label = []
    test_label = []
    for name in NameList:
        valid_indices = K_split[name][indices[0]]
        test_indices = K_split[name][indices[1]]
        train_indices = K_split[name][indices[2:]]
        train_indices = [item for sublist in train_indices for item in sublist]
        valid_data.append([images[name][i] for i in valid_indices])
        valid_label.append([name] * len(valid_indices))
        test_data.append([images[name][i] for i in test_indices])
        test_label.append([name] * len(test_indices))
        train_data.append([images[name][i] for i in train_indices])
        train_label.append([name] * len(train_indices))
    #flatten the list
    train_data = np.array([item for sublist in train_data for item in sublist])
    valid_data = np.array([item for sublist in valid_data for item in sublist])
    test_data = np.array([item for sublist in test_data for item in sublist])
    train_label = np.array([item for sublist in train_label for item in sublist])
    valid_label = np.array([item for sublist in valid_label for item in sublist])
    test_label = np.array([item for sublist in test_label for item in sublist])
    #convert shape
    train_data = train_data.reshape((len(train_data), -1))
    valid_data = valid_data.reshape((len(valid_data), -1))
    test_data = test_data.reshape((len(test_data), -1))
    #label encoding
    category = np.array(range(len(NameList)))
    resvec = np.zeros((category.size, category.max()+1))
    resvec[np.arange(category.size),category] = 1
    LabelEncoding = dict(zip(NameList, resvec))
    train_label = np.array([x if x not in LabelEncoding else LabelEncoding[x] for x in train_label])
    valid_label = np.array([x if x not in LabelEncoding else LabelEncoding[x] for x in valid_label])
    test_label = np.array([x if x not in LabelEncoding else LabelEncoding[x] for x in test_label])
    # Shuffle the train set
    p = np.random.permutation(len(train_data))
    return train_data[p], train_label[p], valid_data, valid_label, test_data, test_label
    
def simple_logistic_model(w, input):
    ''' logistic model withou hidden layer
    Args:
        input, which dimention is M * (1 + d), means M pics each pixel number is d, appnded by 1
        w, parameters, dimention of d + 1 * 1
    Returns:
        x, dimention of M * 1
        
    '''
    x = np.dot(input, w)
    x = 1/(1 + np.exp(-x)) 
    return x

def simple_logistic_model_loss(model_output, input_vec, true_label):
    loss = np.sum(true_label * np.log(model_output) \
                  + (1 - true_label) * np.log(1 - model_output)) * -1
    return loss.T


def simple_logistic_model_gradient_descent(model_output, input_vec, true_label):
    '''
    Args:
        model_output : calculated result
        input_vec : input image
        true_label : true category

    Returns:
        dw :gradient descent
    '''
    N = model_output.shape[0]
    # convert true label vector into binary
    tn = true_label.reshape((len(model_output), 1))
    #calculate the gradient
    dw = np.sum(np.dot((tn - model_output).T, input_vec), axis = 0, keepdims= True) / N
    return dw.T




def Data_preprocess_PCA(k = 10, n_components = 40):
    # Init
    images, cnt = load_data(data_dir="./resized/")
    
    
    
    # PCA verification
    k_split = K_split(images, cnt, k)
    indices = np.array(range(0, k))
    # Get train, valid and test set
    train_data, train_label, valid_data, valid_label, test_data, test_label = \
        Cross_Validation_Procedure(images, k_split, images.keys(), indices)
    # Apply the PCA
    projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors = \
        PCA(train_data, n_components)
    print('The mean and std of projected training set is', np.mean(projected), 'and',\
          np.std(projected) * np.sqrt(projected.shape[0]))
    # Project the valid and test set
    valid_data = np.dot((valid_data - mean_image), top_eigen_vectors) / top_sqrt_eigen_values
    print('The mean and std of projected validation set is', np.mean(valid_data), 'and',\
          np.std(valid_data) * np.sqrt(projected.shape[0]))
    test_data = np.dot((test_data - mean_image), top_eigen_vectors) / top_sqrt_eigen_values
    print('The mean and std of projected test set is', np.mean(test_data), 'and',\
      np.std(test_data) * np.sqrt(projected.shape[0]))



def Logistic_regression_b(k = 10, n_components = 40):

    # Split data into k-fold
    images, cnt = load_data(data_dir="./resized/")
    
    k_split = K_split(images, cnt, k)
    indices = np.array(range(0, k))
    NameList = ['Convertible', 'Minivan']
    learning_rate = 0.5
    w = np.random.rand(n_components + 1, 1)
    epoch_lim = 300 
    
    for i in range(1): # No need to run k times as stated in the problem
        # Get train, valid and test set
        train_data, train_label, valid_data, valid_label, test_data, test_label = \
            Cross_Validation_Procedure(images, k_split, NameList, indices)
        # Apply the PCA
        train_data, mean_image, top_sqrt_eigen_values, top_eigen_vectors = \
            PCA(train_data, n_components)
        # Project the valid and test set
        valid_data = np.dot((valid_data - mean_image), top_eigen_vectors) / top_sqrt_eigen_values
        test_data = np.dot((test_data - mean_image), top_eigen_vectors) / top_sqrt_eigen_values
        # append the 1 colume to fit the bias term
        train_data = np.append(np.ones((train_data.shape[0],1)),train_data, axis = 1)
        true_train = np.array([0.0 if i[0] == 1.0 else 1.0 for i in train_label])
        
        valid_data = np.append(np.ones((valid_data.shape[0],1)),valid_data, axis = 1)
        true_valid = np.array([0.0 if i[0] == 1.0 else 1.0 for i in valid_label])
        
        test_data = np.append(np.ones((test_data.shape[0],1)),test_data, axis = 1)
        true_test = np.array([0.0 if i[0] == 1.0 else 1.0 for i in test_label])
        
        
        loss_train = []
        loss_valid = []
        min_loss_valid = 100000
        best_w = w
        # Train the model  
        for epoch in range(epoch_lim):
            #Logistic model
            model_output = simple_logistic_model(w, train_data)
            # Cross entropy loss
            loss_train.append(simple_logistic_model_loss(model_output, train_data, true_train))
            # Get the gradient
            dw = simple_logistic_model_gradient_descent\
                (model_output, train_data, true_train)
            #update the weight
            w = w + learning_rate * dw
            model_valid_output = simple_logistic_model(w, valid_data)
            loss_valid.append(simple_logistic_model_loss(model_valid_output, valid_data, true_valid))
            if(loss_valid[-1] < min_loss_valid):
                best_w = w
                min_loss_valid = loss_valid[-1]
            
        # Check accuracy on test set
        model_test_output = simple_logistic_model(best_w, test_data)
        pred_test = np.array([0 if i < 0.5 else 1 for i in model_test_output])           
        accu_test = np.sum(pred_test == true_test) / len(pred_test)
        print('Accuracy on test set is {:.4f}'.format(accu_test))

def Logistic_regression_c(k = 10, n_components = 40):
    
    # Split data into k-fold
    images, cnt = load_data(data_dir="./aligned/")
    
    k_split = K_split(images, cnt, k)
    indices = np.array(range(0, k))
    NameList = ['Convertible', 'Minivan']
    learning_rate = 0.5
    w = np.random.rand(n_components + 1, 1)
    epoch_lim = 300 
    
    for i in range(k): # No need to run k times as stated in the problem
        # Get train, valid and test set
        train_data, train_label, valid_data, valid_label, test_data, test_label = \
            Cross_Validation_Procedure(images, k_split, NameList, indices)
        # Apply the PCA
        train_data, mean_image, top_sqrt_eigen_values, top_eigen_vectors = \
            PCA(train_data, n_components)
        # Project the valid and test set
        valid_data = np.dot((valid_data - mean_image), top_eigen_vectors) / top_sqrt_eigen_values
        test_data = np.dot((test_data - mean_image), top_eigen_vectors) / top_sqrt_eigen_values
        # append the 1 colume to fit the bias term
        train_data = np.append(np.ones((train_data.shape[0],1)),train_data, axis = 1)
        true_train = np.array([0.0 if i[0] == 1.0 else 1.0 for i in train_label])
        
        valid_data = np.append(np.ones((valid_data.shape[0],1)),valid_data, axis = 1)
        true_valid = np.array([0.0 if i[0] == 1.0 else 1.0 for i in valid_label])
        
        test_data = np.append(np.ones((test_data.shape[0],1)),test_data, axis = 1)
        true_test = np.array([0.0 if i[0] == 1.0 else 1.0 for i in test_label])
        
        
        loss_train = []
        loss_valid = []
        min_loss_valid = 100000
        best_w = w
        # Train the model  
        for epoch in range(epoch_lim):
            #Logistic model
            model_output = simple_logistic_model(w, train_data)
            # Cross entropy loss
            loss_train.append(simple_logistic_model_loss(model_output, train_data, true_train))
            # Get the gradient
            dw = simple_logistic_model_gradient_descent\
                (model_output, train_data, true_train)
            #update the weight
            w = w + learning_rate * dw
            model_valid_output = simple_logistic_model(w, valid_data)
            loss_valid.append(simple_logistic_model_loss(model_valid_output, valid_data, true_valid))
            if(loss_valid[-1] < min_loss_valid):
                best_w = w
                min_loss_valid = loss_valid[-1]
            
        # Check accuracy on test set
        model_test_output = simple_logistic_model(best_w, test_data)
        pred_test = np.array([0 if i < 0.5 else 1 for i in model_test_output])           
        accu_test = np.sum(pred_test == true_test) / len(pred_test)
        print('Accuracy on test set is {:.4f}'.format(accu_test))
    
                        
        # Rotate the indices to ensure each set would 
        # be selected as the valid and test set once
        indices = (indices + 1) % k

def main():
    Logistic_regression_b(10,40)
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
