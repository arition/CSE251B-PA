from itertools import product

import matplotlib.pyplot as plt
import numpy as np


def accuracy_01(y, y_true):
    '''
    calculate the accuracy for logistic regression (0-1 loss)
    '''
    y_round = np.round(y)
    correct = np.sum(y_round == y_true)
    accuracy = correct / y.shape[0]
    return accuracy


def accuracy_one_hot(y, y_true):
    '''
    calculate the accuracy for softmax regression (one hot encoding)
    '''
    y_max = np.argmax(y, axis=1)
    y_true_max = np.argmax(y_true, axis=1)
    correct = np.sum(y_max == y_true_max)
    accuracy = correct / y.shape[0]
    return accuracy


def plot(title, train, holdout, train_std=None, holdout_std=None, train_label='Training set', holdout_label='Holdout set'):
    '''
    plot the traning loss and holdout loss
    '''
    if train_std is not None and holdout_std is not None:
        for i in range(len(train_std) // 50):
            train_std[i * 50:(i + 1) * 50 - 1] = None
            holdout_std[i * 50:(i + 1) * 50 - 1] = None
        plt.errorbar(range(len(train)), train, yerr=train_std, label=train_label)
        plt.errorbar(range(len(holdout)), holdout, yerr=holdout_std, label=holdout_label)
    else:
        plt.plot(range(len(train)), train, label=train_label)
        plt.plot(range(len(holdout)), holdout, label=holdout_label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.title(title)
    plt.show()


def plot_lr(title, train, train2, train3, train_std, train2_std, train3_std):
    '''
    plot different learning rate
    '''
    for i in range(len(train_std) // 50):
        train_std[i * 50:(i + 1) * 50 - 1] = None
        train2_std[i * 50:(i + 1) * 50 - 1] = None
        train3_std[i * 50:(i + 1) * 50 - 1] = None
    plt.errorbar(range(len(train)), train, yerr=train_std, label='Learning rate right')
    plt.errorbar(range(len(train2)), train2, yerr=train_std, label='Learning rate too high')
    plt.errorbar(range(len(train3)), train3, yerr=train_std, label='Learning rate too low')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.title(title)
    plt.show()


def plot_confusion_matrix(y, y_true):
    '''
    plot confusion matrix
    '''
    cm = np.zeros((y.shape[1], y_true.shape[1]))

    for y_, y_true_ in zip(y, y_true):
        cm[np.argmax(y_), np.argmax(y_true_)] += 1

    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(ticks=range(y_true.shape[1]), labels=['Convertible', 'Minivan', 'Pickup', 'Sedan'])
    plt.yticks(ticks=range(y.shape[1]), labels=['Convertible', 'Minivan', 'Pickup', 'Sedan'])

    threshold = (cm.max() + cm.min()) / 2.0
    cmap = plt.get_cmap()

    for i, j in product(range(y.shape[1]), range(y_true.shape[1])):
        color = cmap(256) if cm[i, j] < threshold else cmap(0)
        plt.text(j, i, str(cm[i, j]), color=color)

    plt.title('confusion matrix')

    plt.show()
