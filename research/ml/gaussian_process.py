import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class GaussianProcess(BaseEstimator):
    """
    Fits a Gaussian Process regressor.

    Parameters
    ----------
    kernel : Gaus

    Attributes
    ----------
    kernel : ml.Kernel
        Kernel function used to compute covariance matrix.

    Examples
    --------

    Note
    ----
    Most of code is taken from the following tutorial:
    https://katbailey.github.io/post/gaussian-processes-for-dummies/.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.Xtrain_ = None
        self.ytrain_ = None

    def fit(self, X, y):
        """
        Computes the Xtrain variance and stores as attribute. Also stores
        Xtrain and ytrain as attributes.

        Parameters
        ----------
        X : np.ndarray, shape (-1, n)
            Input.
        y : np.array, shape (n)
            Targets

        Returns
        -------
        None

        Note
        ----
        Note the K matrix is:
            K_11 K_21
            K_21 K_22
        """
        self.Xtrain_ = X
        self.ytrain_ = y

        # Compute Xtrain/Xtrain elements of covariance matrix (Xtrain variance)
        K_11 = self.kernel.transform(self.Xtrain_, self.Xtrain_)
        self.L_11_ = np.linalg.cholesky(K_11
                                        + 0.00005*np.eye(len(self.Xtrain_)))

    def predict(self, Xtest, n_samples=1):
        """
        Returns predictions for input data by returning the posterior mean (at
        the test points) of the joint distribution of the training data Xtrain
        and the test data Xtest.

        High-level Intuition
        --------------------
        * Goal of is to learn distribution over possible "functions" f(x) = y.
        * Compute the "difference"  between the Xtrain data and the Xtest data.
        * Compute the Xtrain covariance "feature weights" cov_fw s.t.

              XtrainCovMatrix • cov_fw = ytrain

        * Compute post. mean by mult. cov_fw by the Xtrain/Xtest "difference":

              mu = cov_fw • XtrainXtestCovDiff

        Parameters
        ----------
        Xtest : np.array
            Input data.

        Returns
        -------
        np.array, length len(Xtest)
            Predictions which are the posterior mean of the joint distribution
            of the training data and the test data.

        Note
        ----
        Note the K matrix is:
            K_11 K_21
            K_21 K_22
        """

        '''Compute the posterior mean at test points.'''
        if not self._is_fitted():
            raise NotFittedError()

        mu, L_12 = self._compute_mean_and_non_diag_covariance(Xtest)

        return mu

    def sample(self, Xtest, n_samples=1, use_prior=False):
        """
        Returns predictions for input data by returning samples from the either
        the prior or the posterior of the joint distribution of the training
        data Xtrain and the test data Xtest.

        If the model is not yet fitted or use_prior=True, then samples from
        prior are returned. Otherwise, samples are taken from the posterior.

        Parameters
        ----------
        Xtest : np.array
            Input data.
        n_samples : int, default 1
            Number of samples (predictions) to return.
        use_prior : bool, default False
            Whether or not to sample from the prior distribution. If true,
            posterior is used.

        Returns
        -------
        np.ndarray, shape (len(Xtest), n_samples)
            Predictions which are samples drawn from the joint distribution of
            the training data and the test data.
        """
        ntest = len(Xtest)

        # Compute Xtest covariance and its decomposition (sqroot)
        K_22 = self.kernel.transform(Xtest, Xtest)
        L_22 = np.linalg.cholesky(K_22 + 1e-15*np.eye(ntest))

        if use_prior or not self._is_fitted():
            # Sample n_samples sets of standard normals for our test points,
            # then multiply them by the square root of the Xtest covariance.
            f_prior = np.dot(L_22, np.random.normal(size=(ntest, n_samples)))

            return f_prior

        # Compute mean and non-diagonal (Xtrain/Xtest) elements of cov. matrix
        mu, L_12 = self._compute_mean_and_non_diag_covariance(Xtest)

        # Compute sqroot of entire covariance matrix
        L = np.linalg.cholesky(K_22 + 1e-6*np.eye(ntest) - np.dot(L_12.T,
                                                                  L_12))

        # Sample n_samples sets of standard normals for our test points, then
        # multiply them by the square root of the covariance matrix.
        f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(
            size=(ntest, n_samples)))

        return f_post

    def plot_prior_samples(self, Xtest, n_samples=1):
        """
        Plots samples of the prior (defined by kernel) at the test points.

        Parameters
        ----------
        Xtest : np.array
            Input data.
        n_samples : int, default 1
            Number of samples (predictions) to return.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Plotted figure
        axes : tuple<matplotlib.axes._subplots.AxesSubplot>
            Axes used for plotting.
        """
        f_prior = self.sample(Xtest, n_samples, use_prior=True)

        # Now let's plot the sampled functions.
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))

        # Sort values for better plotting
        sort_idx = np.argsort(Xtest.flatten())
        Xtest_sorted = np.take_along_axis(Xtest.flatten(), sort_idx, axis=0)

        for sample in range(n_samples):
            f_prior_sorted = np.take_along_axis(f_prior[:, sample], sort_idx,
                                                axis=0)
            ax.plot(Xtest_sorted, f_prior_sorted,
                    label='Prior Sample {} (predictions)'.format(sample))

        ax.set_title('{} samples from the GP prior'.format(n_samples))
        ax.legend()
        plt.show()

        return fig, (ax)

    def plot_posterior_samples(self, Xtest, n_samples=1):
        """
        Plots samples of the posterior at the test points.

        Parameters
        ----------
        Xtest : np.array
            Input data.
        n_samples : int, default 1
            Number of samples (predictions) to return.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Plotted figure
        axes : tuple<matplotlib.axes._subplots.AxesSubplot>
            Axes used for plotting.

        Note
        ----
        Model instance must be fitted prior to calling this method.
        """
        # Compute mean and non-diagonal (Xtrain/Xtest) elements of cov. matrix
        mu, L_12 = self._compute_mean_and_non_diag_covariance(Xtest)

        # Compute Xtest covariance
        K_22 = self.kernel.transform(Xtest, Xtest)

        # Compute the standard deviation so we can plot it
        s2 = np.diag(K_22) - np.sum(L_12**2, axis=0)
        stdv = np.sqrt(s2)

        # Create figure and axis for plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Sort values for better plotting
        sort_idx = np.argsort(Xtest.flatten())
        Xtest_sorted = np.take_along_axis(Xtest.flatten(), sort_idx, axis=0)
        mu_sorted = np.take_along_axis(mu, sort_idx, axis=0)

        # Plot training data points
        ax.plot(self.Xtrain_, self.ytrain_, 'bs', ms=8, label='Train')

        # Plot posterior mean
        ax.plot(Xtest_sorted, mu_sorted, '--r',
                label='Posterior $\mu$')  # noqa

        # Sample from posterior
        f_post = self.sample(Xtest, n_samples)

        # Plot sampled functions
        for sample in range(n_samples):
            f_post_sorted = np.take_along_axis(f_post[:, sample], sort_idx,
                                               axis=0)
            ax.plot(Xtest_sorted, f_post_sorted,
                    label='Posterior Sample {} (predictions)'.format(sample))

        # Plot standard deviation
        plt.gca().fill_between(Xtest_sorted, mu_sorted-2*stdv,
                               mu_sorted+2*stdv, color='#999999', alpha=.4)
        ax.set_title('{} samples from the GP posterior'.format(n_samples))
        ax.legend()
        plt.show()

        return fig, (ax)

    def _is_fitted(self):
        """
        Helper method to check if instance is fitted or not.

        Parameters
        ----------
        N/A

        Returns
        -------
        bool
            True if model is fitted, otherwise false.
        """
        try:
            check_is_fitted(self, ['Xtrain_', 'ytrain_', 'L_11_'])
            return True
        except NotFittedError:
            return False

    def _compute_mean_and_non_diag_covariance(self, Xtest):
        """
        Computes Xtrain/Xtest covariance and the mean of the joint train/test
        posterior.

        Parameters
        ----------
        Xtest : np.array
            Input data

        Returns
        -------
        np.array, shape (len(Xtest))
            Posterior mean.
        np.ndarray, shape TODO
            Xtrain/Xtest covariance.
        """
        ntest = len(Xtest)

        # Compute Xtrain/Xtest elements of covariance metrix
        K_12 = self.kernel.transform(self.Xtrain_, Xtest)

        # Compute the "difference" between the Xtrain data and the Xtest data.
        #     L_11_ is the sqroot of Xtrain Covariance.
        #     K_12 is the covariance of Xtrain and Xtest
        #     Therefore, L_12 is the matrix that solves: (L_11)(L_12) = K_12.
        L_12 = np.linalg.solve(self.L_11_, K_12)

        # Compute the Xtrain covariance "feature weighs".
        #     np.linalg.solve returns x in Ax=B, where A = L_11 and B = y_train
        #     We can interpret x as the feature weights. In other words, this
        #     step returns the feature weights where the inputs is the
        #     Xtrain/Xtrain covariance matrix elements.
        cov_fw = np.linalg.solve(self.L_11_, self.ytrain_).reshape(ntest,)

        # Obtain the posterior mean by multiplying the cov_fw by the
        # "difference" b/w Xtrain and Xtest.
        #       L12 is the "difference" b/w Xtrain/Xtest covariances.
        #       cov_fw are the weights that produce ytrain when multiplied by
        #       Xtrain.
        mu = np.dot(L_12.T, cov_fw)

        return mu, L_12
