import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


class PCA_QT_Layer:

    def __init__(
        self,
        n_quantiles=100,
        output_distribution='uniform'
    ):
        # subsample shoul be None, or the transformation is not invertible
        self.qt = QuantileTransformer(n_quantiles=n_quantiles,
                                      output_distribution=output_distribution)
        self.pca = PCA()

    def fit_transform(self, X):
        X_1 = self.qt.fit_transform(X)
        return self.pca.fit_transform(X_1)

    def fit(self, X):
        X_1 = self.qt.fit_transform(X)
        self.pca.fit(X_1)

    def transform(self, X):
        X_1 = self.qt.transform(X)
        return self.pca.transform(X_1)

    def inverse_transform(self, X):
        X_1 = self.pca.inverse_transform(X)
        return self.qt.inverse_transform(X_1)


class PCA_QT_Transformer:

    def __init__(
        self,
        n_quantiles=100,
        output_distribution='uniform',
        n_layers=100,
    ):
        self.layers = [
            PCA_QT_Layer(n_quantiles=n_quantiles, output_distribution=output_distribution) for _ in range(n_layers)
        ]

    def fit_transform(self, X, y=None):
        for layer in self.layers:
            X = layer.fit_transform(X)
        return X

    def fit(self, X, y=None):
        for layer in self.layers:
            X = layer.fit_transform(X)

    def transform(self, X):
        for layer in self.layers:
            X = layer.transform(X)
        return X

    def inverse_transform(self, X):
        for layer in self.layers[::-1]:
            X = layer.inverse_transform(X)
        return X

    def transform_each_layer(self, X):
        result = []
        for layer in self.layers:
            X = layer.transform(X)
            result.append(X)
        return result

    def inverse_transform_each_layer(self, X):
        result = []
        for layer in self.layers[::-1]:
            X = layer.inverse_transform(X)
            result.append(X)
        return result


class SphericalCoordinatesTransformer:

    def __init__(self):
        self.dimension = None

    def fit(self, X, y=None):
        self.dimension = X.shape[1]

    def fit_transform(self, X, y=None):
        self.dimension = X.shape[1]
        return self.transform(X)

    def inverse_transform(self, X_trans):
        """Convert spherical coordinates to Cartesian coordinates."""

        if self.dimension is None:
            raise ValueError("The transformer has not been fitted yet.")

        r, angles = X_trans[:, 0], X_trans[:, 1:]

        coords = []
        k = r
        for i in range(angles.shape[1]):
            angle = angles[:, i]
            coords.append(k * np.cos(angle))
            if i == angles.shape[1] - 1:
                coords.append(k * np.sin(angle))
            else:
                k = k * np.sin(angle)

        return np.array(coords).T

    def transform(self, X):
        """Convert Cartesian coordinates to spherical coordinates."""

        if self.dimension is None:
            raise ValueError("The transformer has not been fitted yet.")

        r = np.linalg.norm(X, axis=1)
        k = r
        angles = []
        for i in range(0, self.dimension - 1):
            theta_i, k = spherical_coordinates_trans(X[:, i], k)
            angles.append(theta_i)
        angles[-1] *= np.sign(X[:, -1])
        return np.c_[r, np.array(angles).T]

    def jacobian(self, r, angles):
        """Calculate the Jacobian matrix of the spherical to Cartesian transformation."""
        # This is a placeholder for the actual Jacobian computation
        # The Jacobian for spherical coordinates in N dimensions is complex and
        # involves products and sums of sines and cosines of the angles
        return np.identity(self.dimension)  # Placeholder


def spherical_coordinates_trans(x, k):
    eps = 1e-32
    is_k_zero = k == 0
    k = np.where(is_k_zero, eps, k)
    theta = np.where(~is_k_zero, np.arccos(x / k), 0)
    r_new = k * np.sin(theta)
    return theta, r_new
