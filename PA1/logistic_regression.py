import numpy as np


class LogisticRegression():
    '''
    Logistic Regression
    '''

    def __init__(self, size, lr=0.01):
        '''
        Args

        size: the size of input, should be n_components
        lr: learning rate
        '''
        self.w = np.random.rand(size + 1)
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        ''' 
        logistic model forward

        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        Returns

        y: dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        x = self.w @ x.T
        return 1 / (1 + np.exp(-x))

    def loss(self, y: np.ndarray, true_y: np.ndarray) -> int:
        '''
        calculate the loss using cross-entropy cost function

        args

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)

        Returns

        loss: loss value
        '''

        return -np.mean(true_y * np.log(y) + (1 - true_y) * np.log(1 - y))

    def backward(self, x: np.ndarray, y: np.ndarray, true_y: np.ndarray) -> None:
        '''
        logistic model backward

        Args

        x: input data, which dimention is (M, d), means M pics each pixel number is d

        y: forward result, dimention of (M, 1)

        true_y: true result, dimention of (M, 1)
        '''
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)

        gradient = (true_y - y) @ x
        self.w += self.lr * gradient
