import numpy as np


class Activation:
    """Abstract class. Activation function used in a MLP neural network."""

    def __init__(self):
        pass

    def forward(self, A):
        """Computes forward activation function."""
        raise NotImplementedError()

    def back(self, X):
        """Computes derivative of loss function activation function w.r.t.
        activation function for a particular layer.
        """
        raise NotImplementedError()


class Sigmoid(Activation):
    """Sigmoid activation function used in a MLP neural network."""

    def __init__(self):
        pass

    def forward(self, A):
        """Computes activation of a single layer.

        Parameters
        ----------
        A : np.ndarray, shape (N, M[i])
            Input into activation func. at layer i: (Z.dot(W)+b).

        Returns
        -------
        np.ndarray, shape (N, M[i])
        """
        _Z = 1 / (1 + np.exp(-A))
        return _Z

    def back(self, _Z):
        """Computes partial derivative of loss function w.r.t. activation
        function for a particular layer.

        Parameters
        ----------
        Z : np.ndarray, shape(N, M[i])
            Output of layer i.

        Returns
        -------
        np.ndarray, shape(N, M[i])
            Partial derivative of loss function w.r.t. activation function
            at layer i.
        """
        dZ = _Z*(1-_Z)
        return dZ


class ReLU(Activation):
    """ReLU activation function used in a MLP neural network."""

    def __init__(self):
        pass

    def forward(self, A):
        """Computes activation of a single layer.

        Parameters
        ----------
        A : np.ndarray, shape (N, M[i])
            Input into activation func. at layer i: (Z.dot(W)+b).

        Returns
        -------
        np.ndarray, shape (N, M[i])
        """
        _Z = A * (A > 0)
        assert not np.any(np.isnan(_Z))
        return _Z

    def back(self, _Z):
        """Computes partial derivative of loss function w.r.t. activation
        function for a particular layer.

        Parameters
        ----------
        Z : np.ndarray, shape(N, M[i])
            Output of layer i.

        Returns
        -------
        np.ndarray, shape(N, M[i])
            Partial derivative of loss function w.r.t. activation function
            at layer i.
        """
        dZ = np.piecewise(_Z, [_Z < 0, _Z >= 0], [0, 1])
        return dZ
