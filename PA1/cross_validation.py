from typing import Dict, List, Optional

import numpy as np


class CrossValidationDataset():
    '''
    Generate k-fold cross validation dataset
    '''

    def __init__(self, dataset: Dict, categories: Optional[List[str]] = None, k: int = 10) -> None:
        '''
        args

        dataset: dataset from dataloader

        categories: The categories to include in the dataset. if empty or None, it will include all categories.

        k: k-fold

        When categories > 2, it will use one-hot encoding
        '''

        super().__init__()
        labels = []
        datas = []

        if categories is None or len(categories) == 0:
            categories = list(dataset.keys())

        for index, label in enumerate(categories):
            datas += dataset[label]
            if len(categories) <= 2:
                labels += [index for _ in range(len(dataset[label]))]
            else:
                labels += [[1 if index == i else 0 for i in range(len(categories))] for _ in range(len(dataset[label]))]

        assert len(labels) == len(datas)

        datas = np.array(datas)
        labels = np.array(labels)

        datas = datas.reshape((datas.shape[0], -1))

        shuffle_index = np.random.permutation(len(labels))
        self.datas = np.split(datas[shuffle_index], k)
        self.labels = np.split(labels[shuffle_index], k)
        self.k = k

    def data(self):
        '''
        yield return the train set, test set and holdout set

        return: train_datas, train_labels, test_datas, test_labels, holdout_datas, holdout_labels
        '''

        for i in range(self.k):
            datas = list(self.datas)
            labels = list(self.labels)
            test_datas = datas.pop(i)
            test_labels = labels.pop(i)
            holdout_datas = datas.pop(i % (self.k - 1))
            holdout_labels = labels.pop(i % (self.k - 1))
            train_datas = np.concatenate(datas, axis=0)
            train_labels = np.concatenate(labels, axis=0)

            yield train_datas, train_labels, test_datas, test_labels, holdout_datas, holdout_labels
