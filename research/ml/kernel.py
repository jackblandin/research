import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


class Kernel:
    """Abstract class for kernel functions."""

    def __init__(self, pairwise):
        raise NotImplementedError()

    def transform(self, xi, xj):
        raise NotImplementedError()


class GaussianKernel(Kernel):
    """Gaussian kernel: exp[-||xi-xj||**2 / radius**2)]

    Useful for complex decision boundaries. Decrease radius to fit more complex
    boundaries.

    Parameters
    ----------
    pairwise : bool
        If True, transform() returns a matrix of pairwise distances. Otherwise,
        returns scalar distance between the two input vectors.
    radius : int, default .5
        Radius (sigma). Must be greater than zero. Lower values allow for more
        flexible classifiers, at the risk of overfitting. Large radius values
        gradually reduce the kernel to a continuous function, thereby limitting
        the ability of the kernel to fit complex boundaries.
    """

    def __init__(self, pairwise, radius=.5):
        if radius <= 0:
            raise ValueError('radius must be greater than zero.')
        self.pairwise = pairwise
        self.radius = radius
        self.name = 'gaussian'

    def transform(self, xi, xj, return_similarity=False):
        """Computes Gaussian distance.

        Parameters
        ----------
        xi : array-like, shape(m)
            Input.
        xj : array-like, shape(m)
            Input.
        return_similarity : bool, default False
            If True, returns 1 - distance instead of distance.

        Returns
        -------
        float or np.ndarray
            Gaussian distance.
        """
        if self.pairwise:
            sqdist = (np.sum(xi**2, 1).reshape(-1, 1) + np.sum(xj**2, 1)
                      - 2*np.dot(xi, xj.T))
            dist = np.exp(-.5 * sqdist / self.radius)
            if return_similarity:
                return np.ones_like(dist) - dist
            else:
                return dist
        else:
            norm = np.linalg.norm(xi-xj)
            return np.exp(-norm**2 / self.radius**2)


class LinearKernel(Kernel):
    """Linear kernel: <xi,xj>"""

    def __init__(self, pairwise):
        self.name = 'linear'
        self.pairwise = pairwise
        pass

    def transform(self, xi, xj):
        """Computes dot product of input vectors.

        Parameters
        ----------
        xi : array-like, shape(m)
            Input.
        xj : array-like, shape(m)
            Input.

        Returns
        -------
        float
            Dot product of input vectors.
        """
        return xi.dot(xj)


class PolynomialKernel(Kernel):
    """Polynomial kernel: (1 + <xi,xj>)**d.

    Parameters
    ----------
    degree : int, default 2
        Polynomial degree. Higher values allow for more complex decision
        boundaries to be fitted. Cannot learn disjoint boundaries.
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.name = 'polynomial'

    def transform(self, xi, xj):
        """Computes polynomail distance.

        Parameters
        ----------
        xi : array-like, shape(m)
            Input.
        xj : array-like, shape(m)
            Input.

        Returns
        -------
        float
            Polynomial distance.
        """
        return (1+xi.dot(xj))**self.degree


class SigmoidKernel(Kernel):
    """Sigmoid kernel: 1 / (1 + np.exp(-A))"""

    def __init__(self, radius=1):
        self.name = 'sigmoid'

    def transform(self, xi, xj):
        """Computes Sigmoid distance.

        Parameters
        ----------
        xi : array-like, shape(m)
            Input.
        xj : array-like, shape(m)
            Input.

        Returns
        -------
        float
            Sigmoid distance.
        """
        return 1 / (1 + np.exp(-xi.dot(xj)))


class TanHKernel(Kernel):
    """Hyperbolic Tangent kernel: tanh(xi*xj)"""

    def __init__(self, radius=1):
        self.name = 'tanh'

    def transform(self, xi, xj):
        """Computes TanH distance.

        Parameters
        ----------
        xi : array-like, shape(m)
            Input.
        xj : array-like, shape(m)
            Input.

        Returns
        -------
        float
            TanH distance.
        """
        return np.tanh(xi.dot(xj))


KERNEL_MAP = {
    'gaussian': GaussianKernel,
    'linear': LinearKernel,
    'polynomial': PolynomialKernel,
    'sigmoid': SigmoidKernel,
    'tanh': TanHKernel,
}
