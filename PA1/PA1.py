#!python
# -*- coding: utf-8 -*-
# Fangzhou Ai, Yue Qiao @ ECE, UCSD
from dataloader import load_data
from PCA import PCA
import numpy as np
import os, random
import matplotlib.pyplot as plt
from PIL import Image


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
    train_data = train_data.reshape((len(train_data), -1))
    valid_data = valid_data.reshape((len(valid_data), -1))
    test_data = test_data.reshape((len(test_data), -1))
    p = np.random.permutation(len(train_data))
    return train_data[p], train_label[p], valid_data, valid_label, test_data, test_label
    


k = 10 # choose the number you want
n_components = 4 # choose the number you want
images, cnt = load_data()
#split data into k-fold
K_split = K_split(images, cnt, k)
indices = np.array(range(0, k))
for i in range(k):
    # Get train, valid and test set
    train_data, train_label, valid_data, valid_label, test_data, test_label = \
        Cross_Validation_Procedure(images, K_split, images.keys(), indices)
    # Apply the PCA
    projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors = \
        PCA(train_data, n_components)
    # Normalize the projected
    projected = np.divide(projected, top_sqrt_eigen_values)
        
        
        
        
        
        
        
    # Rotate the indices to ensure each set would 
    # be selected as the valid and test set once
    indices = (indices + 1) % k
    