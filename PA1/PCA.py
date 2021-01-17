import numpy as np


class PCA():
    '''
    Principal Components Analysis
    '''

    def __init__(self, x: np.ndarray, n_components: int):
        '''
        Args:

        x: has shape Mxd where M is the number of images and d is the dimension of each image

        n_components: The number of components you want to project your image onto.
        '''

        mean_image = np.average(x, axis=0)

        msd = x - mean_image  # M x d

        smart_cov_matrix = np.matmul(msd, msd.T)
        eigen_values, smart_eigen_vectors = np.linalg.eig(smart_cov_matrix)

        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        smart_eigen_vectors = smart_eigen_vectors[:, idx]

        eigen_vectors = (np.matmul(msd.T, smart_eigen_vectors)).T  # M x d

        row_norm = np.sum(np.abs(eigen_vectors)**2, axis=-1)**(1. / 2)  # M

        normalized_eigen_vectors = eigen_vectors / (row_norm.reshape(-1, 1))  # M x d

        top_eigen_vectors = normalized_eigen_vectors[:n_components].T
        top_sqrt_eigen_values = np.sqrt(eigen_values[:n_components])

        self.mean_image = mean_image
        self.top_sqrt_eigen_values = top_sqrt_eigen_values
        self.top_eigen_vectors = top_eigen_vectors

        result = np.matmul(x - self.mean_image, self.top_eigen_vectors) / self.top_sqrt_eigen_values
        assert np.mean(result) < 0.0001
        assert np.std(result) - 1 < 0.0001

    def apply(self, x: np.ndarray) -> np.ndarray:
        '''
        Apply PCA to data

        Args:

        x: has shape Mxd where M is the number of images and d is the dimension of each image
        '''

        return np.matmul(x - self.mean_image, self.top_eigen_vectors) / self.top_sqrt_eigen_values
