import numpy as np
import matplotlib.pyplot as plt

def accuracy_01(y, y_true):
    y_round = np.round(y)
    correct = np.sum(y_round == y_true)
    accuracy = correct / y.shape[0]
    return accuracy

def accuracy_one_hot(y, y_true):
    y_max = np.argmax(y, axis=1)
    y_true_max = np.argmax(y_true, axis=1)
    correct = np.sum(y_max == y_true_max)
    accuracy = correct / y.shape[0]
    return accuracy

def plot(title, train, holdout, train_std=None, holdout_std=None):
    plt.plot(range(len(train)), train, label='Training set')
    plt.plot(range(len(holdout)), holdout, label='Holdout set')
    if train_std is not None and holdout_std is not None:
        errorbar_x = (np.arange(int(len(train) / 50)) + 1) * 50 - 1
        plt.errorbar(errorbar_x, train[errorbar_x], yerr=train_std[errorbar_x], fmt='o')
        #plt.errorbar(range(300), train, yerr=train_std, fmt='o')
        plt.errorbar(errorbar_x, holdout[errorbar_x], yerr=holdout_std[errorbar_x], fmt='o')
        #plt.errorbar(range(300), holdout, yerr=holdout_std, fmt='o')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epochs')
    plt.show()