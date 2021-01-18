import numpy as np


class SoftmaxRegression():
    '''
    Softmax Regression
    '''

    def __init__(self, size, class_size=4, lr=1e-3):
        '''
        Args

        size: the size of input, should be n_components
        class_size: number of classes, should be 4
        lr: learning rate
        '''
        self.w = np.random.rand(class_size, size + 1)
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        ''' 
        softmax model forward

        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        Returns

        y: dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        x = self.w @ x.T
        return (np.exp(x) / np.sum(np.exp(x), axis=0)).T

    def loss(self, y: np.ndarray, true_y: np.ndarray) -> int:
        '''
        calculate the loss using cross-entropy cost function

        args

        y: forward result, dimention of (M, class_size)

        true_y: true result, dimention of (M, class_size)

        Returns

        loss: loss value
        '''

        error = np.sum(np.multiply(true_y, np.log(y)), axis=1)
        return -np.mean(error)

    def backward(self, x: np.ndarray, y: np.ndarray, true_y: np.ndarray) -> None:
        '''
        softmax model backward

        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        y: forward result, dimention of (M, class_size)

        true_y: true result, dimention of (M, class_size)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)

        gradient = (true_y - y).T @ x
        self.w += self.lr * gradient
