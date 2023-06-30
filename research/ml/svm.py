import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from tabulate import tabulate

from research.ml.kernel import KERNEL_MAP


class SVM(BaseEstimator, ClassifierMixin):
    """SVM classifier implementing dual formulation.

    Parameters
    ----------
    kernel : str or function, default 'linear'
        Kernel function.
        If str, must be 'gaussian', 'linear', 'polynomial', 'sigmoid', 'tanh'.
        If function, format is function(x, y) -> float
    positive_weights_only : bool, Optional, default False
        If True, only positive weights allowed.
    **kernel_args
        Additional arguments to pass into kernel functions.

    Attributes
    ----------
    opt_result_ : scipy.optimize.optimize.OptimizeResult
        Optimization result.
    sup_idx_ : array-like<int>, len(n_support_vectors)
        Indices of the support vectors.
    sup_X_ : np.ndarray, shape(n_support_vectors, m)
        X values that are support vectors.
    sup_y_ : np.array, shape(n_support_vectors)
        Target values of support vectors.
    sup_alphas_ : np.array, shape(n_support_vectors)
        alpha values of support vectors.
    offset_ : float
        Offset (theta) used in discriminant.

    Examples
    --------
    Linear Example
    >>> # Generate random x1, x2 values over [0, 1]^2 range
    >>> X = np.random.rand(100, 2)
    >>>
    >>> # Make a nice looking boundary gap
    >>> X = X[(X[:, 0] + X[:, 1] > 1.1) | (X[:, 0] + X[:, 1] < .9)]
    >>>
    >>> # Label each dataset with target values as x1 + x2 > 1
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>>
    >>> # Fit SVM and plot decision boundary.
    >>> svm = SVM()
    >>> svm.fit(X, y)
    >>> svm.plot_discriminants(X, y)

    Nonlinear Example
    >>> # Generate random x1, x2 values over [0, 1]^2 range
    >>> X = np.random.rand(50, 2)
    >>>
    >>> # Make a nice looking boundary gap
    >>> X = X[((X[:, 0]-.5)**2 + (X[:, 1]-.5)**2 < .09) |
    >>>       ((X[:, 0]-.5)**2 + (X[:, 1]-.5)**2 > .11)]
    >>>
    >>> # Label each dataset with target values as x1 + x2 > 1
    >>> y = ((X[:, 0]-.5)**2 + (X[:, 1]-.5)**2 < .1).astype(int)
    >>>
    >>> # Fit SVM and plot decision boundary.
    >>> svm = SVM(kernel='polynomial', degree=2)
    >>> svm.fit(X, y)
    >>> svm.plot_decision_boundary(X, y)
    >>> svm.plot_discriminants(X, y)
    """
    def __init__(
            self, kernel='linear', positive_weights_only=False, **kernel_args,
    ):
        if isinstance(kernel, str):
            kernel = KERNEL_MAP.get(kernel)(**kernel_args)
        elif not callable(kernel):
            raise ValueError('Kernel must be a string or callable.')

        if kernel is None:
            msg = 'Invalid kernel. Must be in {}'.format(KERNEL_MAP.keys())
            raise ValueError(msg)

        self.kernel = kernel
        self.positive_weights_only = positive_weights_only

    def fit(self, X, y, vectorized=None):
        """Fits SVM classifer.

        Parameters
        ----------
        X : np.ndarray, shape (-1, n)
            Input.
        y : np.array, shape (n)
            Targets
        vectorized : bool, default None
            Whether to use the vectorized/non-vectorized loss function. If
            using nonlinear kernel, then this must be false (until I fix it).
            If None, then vectorized will default to True if kernel is linear,
            and False if kernel is nonlinear.

        Returns
        -------
        """
        # Don't mess with teh original copies.
        X = X.copy()
        y = y.copy()

        # My Input validation
        if self.kernel.name != 'linear' and vectorized:
            msg = 'Vectorized loss only works with linear kernel right now.'
            raise ValueError(msg)

        if vectorized is None:
            if self.kernel.name == 'linear':
                vectorized = True
            else:
                vectorized = False

        # Sklearn input validation
        X, y = check_X_y(X, y)  # Check that X and y have correct shape
        self.classes_ = unique_labels(y)  # Store the classes seen during fit

        if vectorized:
            loss = self._vectorized_loss
        else:
            loss = self._loss

        # SVM needs 1s and -1s
        y[y == 0] = -1

        initial_alphas = np.random.rand(len(X))

        # Define constraints
        #
        # Our constraints:
        #     1. sum_i(ai*yi)=0
        #     2. ai >= 0
        #     3. Positive weights only constraint:
        #        for all j:
        #            sum_i( X[i][j]*y[i] * a[i] >= 0 )
        #
        # Scipy LinearConstraint format:
        #    lb <= A.dot(x) <= ub
        #
        # Therefore:
        #     Constraint 1:
        #         A = di
        #         lb = 0
        #         ub = 0
        #     Constraint 2:
        #         A = 1
        #         lb = 0
        #         ub = np.inf
        #     Positive weights only constraint:
        #         A = X[:, j] * y
        #         lb = zero
        #         ub = np.inf
        #
        con1 = optimize.LinearConstraint(y, 0, 0)
        con2 = {'type': 'ineq', 'fun': lambda a: a}

        if not self.positive_weights_only:
            constraints = (con1, con2)
        else:
            # Force weights to be positive
            pos_weight_constraints = []

            for j in range(X.shape[1]):
                con = optimize.LinearConstraint(
                    (
                        (X[:,j].reshape(len(X), 1) * y.reshape(len(y), 1))
                        .reshape(len(X))
                    ),
                    0,
                    np.inf,
                )
                pos_weight_constraints.append(con)

            constraints = (con1, con2) + tuple(pos_weight_constraints)
        display(constraints)

        self.opt_result_ = optimize.minimize(loss, initial_alphas,
                                             constraints=constraints,
                                             args=(X, y))
        # Find indices of support vectors
        sup_idx = np.where(self.opt_result_.x > 0.001)
        self.sup_idx_ = sup_idx
        self.sup_X_ = X[sup_idx]
        self.sup_y_ = y[sup_idx]
        self.sup_alphas_ = self.opt_result_.x[sup_idx]

        self.offset_ = self._compute_offset()

        return self

    def predict(self, X, threshold=0):
        """Predicts classes for each row of input X.

        Parameters
        ----------
        X : np.ndarray, shape (-1, n)
            Input.
        threshold : float, range [-1, 1)
            If discriminant is greater than or equal to threshold, predict one,
            otherwise predict zero.

        Returns
        -------
        np.array<int>, shape (X.shape[0])
            Predicted target (0 or 1) values.
        """
        check_is_fitted(self, ['opt_result_', 'sup_idx_', 'sup_X_', 'sup_y_',
                               'sup_alphas_', 'offset_'])
        X = check_array(X)

        g = self._compute_discriminant(X)

        yhat = (g >= 0).astype(int)

        return yhat

    def predict_proba(self, X):
        """Predicts probabilities by computing the discriminants.

        Parameters
        ----------
        X : np.ndarray, shape (-1, n)
            Input.

        Returns
        -------
        np.array<float>, range [0,1], shape (X.shape[0])
            Predicted probability (discriminant normalized betwen zero and
            one).
        """
        check_is_fitted(self, ['opt_result_', 'sup_idx_', 'sup_X_', 'sup_y_',
                               'sup_alphas_', 'offset_'])
        X = check_array(X)

        g = self._compute_discriminant(X)

        prob = (g + 1) / 2

        return prob

    def weights(self, norm='l1'):
        """
        Computes the weights for each input feature. Used in IRL to infer the
        feature expectation weights. Requires that the model be fitted.

        Parameters
        ----------
        norm : str, default 'l1'
            The norm to feed into sklearn.preprocessing.normalize to normalize
            the weights.

        Returns
        -------
        weights : np.array<float>, len(n_features)
        """
        check_is_fitted(self, ['opt_result_', 'sup_idx_', 'sup_X_', 'sup_y_',
                               'sup_alphas_', 'offset_'])
        sup_X_times_sup_y = ( np.array(self.sup_X_) * self.sup_y_.reshape(len(self.sup_y_), 1)
        )
        weights = np.dot(self.sup_alphas_.T, sup_X_times_sup_y)

        # Normalize the weights
        weights = normalize([weights], norm=norm)[0]

        return weights

    def plot_decision_boundary(self, X, y):
        """Plots H, H+, H-, as well as support vectors.

        Parameters
        ----------
        X : np.ndarray, shape(n, m)
            Inputs.
        y : np.array, shape(n)
            Targets.

        Returns
        -------
        None
        """
        # Compute decision boundary
        y[y == 0] = -1
        _X = np.random.rand(75_000, self.sup_X_.shape[1])
        g = self._compute_discriminant(_X)
        TOL = .03
        H = _X[np.where(np.abs(g) < TOL)]
        Hpos = _X[np.where((np.abs(g) < 1 + TOL) & (np.abs(g) > 1 - TOL))]
        Hneg = _X[np.where((np.abs(g) > -(1 + TOL))
                           & (np.abs(g) < (-1 + TOL)))]
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        C1 = X[np.where(y == 1)]
        C2 = X[np.where(y == -1)]
        ax.scatter(C1[:, 0], C1[:, 1], label='C1', marker='x')
        ax.scatter(C2[:, 0], C2[:, 1], label='C2', marker='o')
        sv = self.sup_X_
        ax.scatter(sv[:, 0], sv[:, 1], label='SV', marker='*', s=300)
        ax.scatter(H[:, 0], H[:, 1], label='H', s=5)
        ax.scatter(Hpos[:, 0], Hpos[:, 1], label='H+', s=5)
        ax.scatter(Hneg[:, 0], Hneg[:, 1], label='H-', s=5)
        ax.legend()
        ax.set_title('SVM Decision boundary')
        ax.set_xlim([-.1, 1])
        ax.set_ylim([-.1, 1.2])
        plt.show()
        return None

    def plot_discriminants(self, X, y):
        """Plots values of g as "heat" map.

        Parameters
        ----------
        X : np.ndarray, shape(n, m)
            Inputs.
        y : np.array, shape(n)
            Targets.

        Returns
        -------
        None
        """
        y[y == 0] = -1
        _X = np.random.rand(75_000, self.sup_X_.shape[1])
        g = self._compute_discriminant(_X)
        TOL = .03
        H = _X[np.where(np.abs(g) < TOL)]
        Hpos = _X[np.where((np.abs(g) < 1 + TOL) & (np.abs(g) > 1 - TOL))]
        Hneg = _X[np.where((np.abs(g) > -(1 + TOL))
                           & (np.abs(g) < (-1 + TOL)))]
        # Plot
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 15))

        C1_idx = np.where(y == 1)
        C2_idx = np.where(y == -1)

        C1g = g[C1_idx]
        C2g = g[C2_idx]

        sns.kdeplot(g, ax=ax0, label='g(x)')
        sns.kdeplot(C1g, ax=ax0, label='C1')
        sns.kdeplot(C2g, ax=ax0, label='C2')
        ax0.set_title('Distribution of g(x) values')

        sv = self.sup_X_
        ax1.scatter(sv[:, 0], sv[:, 1], label='SV', marker='*', s=300)
        ax1.scatter(H[:, 0], H[:, 1], label='H', s=20)
        ax1.scatter(Hpos[:, 0], Hpos[:, 1], label='H+', s=10)
        ax1.scatter(Hneg[:, 0], Hneg[:, 1], label='H-', s=10)
        ax1.scatter(_X[:, 0], _X[:, 1], c=g, label='g(x)', s=1, alpha=.3)
        ax1.legend()
        ax1.set_title('Discriminant value')
        ax1.set_xlim([-.1, 1])
        ax1.set_ylim([-.1, 1.2])
        return None

    def _loss(self, alphas, _X, y, verbose=False):
        """Dual optimization loss function.

        Parameters
        ----------
        _X : np.ndarray, shape (n, m)
            Inputs. Underscore required because scipy.optimize.minimize()'s
            first parameter is actually `X`.
        y : np.array, shape (n)
            Targets.
        verbose: bool, default False
            If True, print debugging info

        Returns
        -------
        float
            Total loss.
        """
        X = _X
        left_sum = alphas.sum()
        right_sum = 0

        terms = []

        for i, xi in enumerate(X):
            ai = alphas[i]
            yi = y[i]
            for j, xj in enumerate(X):
                aj = alphas[j]
                yj = y[j]
                term = ai*aj*yi*yj*self.kernel.transform(xi, xj)
                if term != 0:
                    terms.append([xi, xj, ai, aj, yi, yj,
                                  self.kernel.transform(xi, xj),
                                  term])
                right_sum += term

        if verbose:
            print(tabulate(terms, headers=['xi', 'xj', 'ai', 'aj', 'yi', 'yj',
                                           'kernel(xi, xj)', 'rhs_sum']))
            print('\nleft_sum: {:.3f}'.format(left_sum))
            print('right_sum: {:.3f}'.format(right_sum))

        total_loss = left_sum - .5*right_sum

        # Use -1 since we need to minimize
        return -1 * total_loss

    def _vectorized_loss(self, alphas, _X, y, verbose=False):
        """Vectorized implementation of dual optimization loss function.

        Parameters
        ----------
        _X : np.ndarray, shape (n, m)
            Inputs. Underscore required because scipy.optimize.minimize()'s
            first parameter is actually `X`.
        y : np.array, shape (n)
            Targets.
        verbose: bool, default False
            If True, print debugging info

        Returns
        -------
        float
            Total loss.
        """
        X = _X
        left_sum = alphas.sum()
        # make right term -.5*a.T*H*a
        # Where H = yi*yj*xi.dot(xj.T)
        # X_ = y*X
        y_ = y.reshape(-1, 1)
        # y shape(n, 1)
        # X shape (n, 2)
        X_ = y_ * X
        # X_ shape (n, 2)
        # H is (n, n)
        H = self.kernel.transform(X_, X_.T)
        # a shape (n,)
        # a.T shape (n,)
        right_sum = alphas.T.dot(H).dot(alphas)
        total_loss = left_sum - .5*right_sum

        if verbose:
            print(tabulate([[left_sum, right_sum]],
                           headers=['left_sum', 'right_sum']))

        return -1 * total_loss

    def _compute_offset(self):
        """Compute offset (theta) from a support vector.

        Returns
        -------
        float
            Offset (theta).
        """
        # Uses first support vector, although any can be used.
        xk = self.sup_X_[0]
        yk = self.sup_y_[0]

        _sum = 0
        for xi, yi, ai in zip(self.sup_X_, self.sup_y_, self.sup_alphas_):
            _sum += ai * yi * self.kernel.transform(xi, xk)

        offset = yk - _sum
        return offset

    def _compute_discriminant(self, X):
        """Computes discriminant g(x) = wTx + theta.

        Parameters
        ----------
        X : np.ndarray, shape (-1, m)
            Input.

        Returns
        -------
        np.array<float>, shape (X.shape[0])
            Values computed by discriminant g(x).
        """
        g = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            _sum = 0
            for xi, yi, ai in zip(self.sup_X_, self.sup_y_, self.sup_alphas_):
                _sum += ai * yi * self.kernel.transform(xi, x)
            g[i] = _sum + self.offset_

        return g
