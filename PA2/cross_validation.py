import numpy as np


class Cross_Validation():
    def __init__(self, x, y, k):
        self.k = k
        shuffle_index = np.random.permutation(x.shape[0])
        self.datas = np.array_split(x[shuffle_index], k)
        self.labels = np.array_split(y[shuffle_index], k)

    def data(self):
        for i in range(self.k):
            datas = list(self.datas)
            labels = list(self.labels)
            x_val = datas.pop(i)
            y_val = labels.pop(i)
            x_train = np.concatenate(datas, axis=0)
            y_train = np.concatenate(labels, axis=0)

            yield x_train, y_train, x_val, y_val
