import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from copy import deepcopy


class MLP:
    """Multip Layer Perceptron neural network implemented with NumPy.

    Uses gradient ascent.

    Parameters
    ----------
    D : int
        Input dimension.
    K : int
        Number of output classes.
    hidden_layer_sizes : array-like
        Hidden layer sizes. An array of integers.
    Z : Activation
        Activation function.
    optimizer : Optimizer
        Optimizer.

    Attributes
    ----------
    n_layers, int
        Number of hidden layers, defined by length of hidden_layer_sizes.
    M : list, length (n_layers+2)
        Hidden layer sizes, with M[0] set to D and M[-1] set to K. This makes
        computing recursive forward and back propagation easier.
    W : list, shape (len(M)+1)
        List of weight matrices.
    b : list, shape (len(M)+1)
        List of bias vectors.

    Examples
    --------
    >>> D = X_train.shape[1]
    >>> K = Y_train.shape[0]
    >>> hidden_layer_sizes = [5, 5]
    >>> Z = Sigmoid()
    >>> opt = MomentumOptimizer(hidden_layer_sizes, lr=.001, mu=.5)
    >>> model = MLP(D, hidden_layer_sizes, K, Z, opt)
    >>> model.fit(X_train, Y_train)
    >>> preds, _ = model.forward(X_test)
    """

    def __init__(self, D, K, hidden_layer_sizes, Z, optimizer):
        self.D = D
        self.K = K
        self.Z = Z
        self.optimizer = optimizer
        self.n_layers_ = len(hidden_layer_sizes)
        self.M = [D] + hidden_layer_sizes + [K]
        n_layers = self.n_layers_
        M = self.M
        self.W = [None for i in range(n_layers+1)]
        self.b = [None for i in range(n_layers+1)]

        # Randomly initialize weights
        # Normalize with 1/sqrt(D)
        norm = 1/np.sqrt(D)
        for i in range(n_layers+1):
            self.W[i] = norm*np.random.randn(M[i], M[i+1])
            self.b[i] = norm*np.random.randn(M[i+1])

    def __copy__(self):
        """Returns a copy of self.

        Returns
        -------
        MLP
            Copy of self.
        """
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__.copy())
        return obj

    def __deepcopy__(self, memo):
        """Returns a deep copy of self.

        Useful in DQNs when copying the main network to the target network.

        Parameters
        ----------
        memo : dict
            Object's memo dict.

        Returns
        -------
        MLP
            Copy of self.
        """
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__.copy())
        self.W = deepcopy(self.W, memo)
        self.b = deepcopy(self.b, memo)
        return obj

    def forward(self, X):
        """Runs one forward pass through all layers.

        Final layer inputs are normalized prior to passing through softmax.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Input matrix.

        Returns
        -------
        Y : np.ndarray, shape(N, K)
            Outputs as indicator matrix.
        Z : list, shape (n_layers+1)
            Z values indexed by layer. Used for backpropagation. Note that Z[0]
            is X and Z[-1] is Y.
        """
        n_layers = self.n_layers_
        W = self.W
        b = self.b

        # Collect Z at each layer so we can use them in backprop.
        Z = [None for i in range(n_layers+2)]
        Z[0] = X.copy()

        for i in range(1, self.n_layers_+1):
            Z[i] = self._forward_single_layer(Z[i-1], i)
            assert not np.any(np.isnan(Z[i]))

        final_A = Z[-2].dot(W[n_layers]) + b[n_layers]
        assert not np.any(np.isnan(final_A))

        # Compute final layer outputs
        Y = self._forward_final_layer(final_A)

        Z[-1] = Y
        assert len(Z) == n_layers + 2
        assert not np.any(np.isnan(Z[-1]))

        return Y, Z

    def fit(self, X, Y, epochs=1000, batch_size=1000):
        """Fits the neural network using backpropagation.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, D)
            Input matrix.
        Y : array-like, shape (N,)
            Vector of targets. This will be transformed into a onehot encoded
            indicator matrix, called T.
        batch_size : int, default=1000
            Training batch size. If batch_size > len(X), then the batch size
            will be changed to len(X).
        epochs : int
            Number of epochs.

        Returns
        -------
        None
        """
        T = self._transform_targets(Y)

        if batch_size > len(X):
            print('WARNING: Batch size > len(X). Setting batch size to len(X)')
            batch_size = len(X)
        n_batches = int(len(X) / batch_size)
        losses = []
        performance_metrics = []

        for epoch in range(epochs):
            Output = np.empty_like(T)  # Aggregate Output for entire epoch
            grad_max = 0  # For debugging

            for b in range(n_batches):
                # Select batch
                lower_idx = b*batch_size
                upper_idx = (b+1)*batch_size
                if upper_idx > len(X):
                    upper_idx = len(X) - 1
                X_batch = X[lower_idx:upper_idx, :]

                # Predict output for batch
                Output_batch, Z_batch = self.forward(X_batch)
                Output[lower_idx:upper_idx] = Output_batch

                # Update weights and biases at each layer
                T_batch = T[lower_idx:upper_idx]
                _grad_max = self.update(T_batch, Output_batch, Z_batch)
                grad_max = max([grad_max, _grad_max])

            if epoch % (epochs / 10) == 0:
                loss = self._loss(T, Output)
                P = self._transform_output(Output)
                pm = self._performance_metric(Y, P)
                out = 'epoch: {}'.format(epoch)
                out += ', loss: {:.2f}'.format(loss)
                out += ', performance_metric: {:.4f}'.format(pm)
                w_max = max([w.max() for w in self.W])
                out += ', max weight: {:.3f}'.format(w_max)
                out += ', max weight gradient: {}'.format(grad_max)
                print(out)
                losses.append(loss)
                performance_metrics.append(pm)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
        ax0.plot(losses[1:])
        ax0.set_title('Loss')
        ax1.plot(performance_metrics[1:])
        ax1.set_title('Performance Metric')
        plt.show()

        return None

    def predict(self, X):
        """Wrapper method around forward() and _transform_output.

        This method allows MLPs to conform to scikit-learn's fit/predict
        schema.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Input matrix.

        Returns
        -------
        ?
            Whatever `transformed_output()` returns.
        """
        Output, Z = self.forward(X)
        transformed_Output = self._transform_output(Output)
        return transformed_Output

    def update(self, T, Output, _Z):
        """Performs one backprop update, updating weights and biases for each
        layer.

        Parameters
        ----------
        T : np.ndarray, shape (K,)
            Targets
        Output : np.ndarray, shape (K,)
            Predictions
        _Z : list<np.ndarray>, shape (n_layer+1)
            Outputs at each layer
        Returns
        -------
        numeric
            Max gradient update (for debugging during training).
        """
        opt = self.optimizer
        n_layers = self.n_layers_

        # For each layer, iterating backwards from final layer, compute
        # deltas, weight partials, and update weights/biases.
        Delta = [None for i in range(n_layers+1)]
        grad_max = 0
        for i in range(n_layers, -1, -1):
            if i == n_layers:
                # Compute delta at final layer
                Delta[i] = self._backprop_delta_final_layer(T, Output)
            else:
                # Compute deltas at middle layer
                Delta[i] = self._backprop_delta(Delta[i+1], self.W[i+1],
                                                _Z[i+1], i)

            # Compute weight and bias partial gradients
            dJdW_i = self._dJdW(_Z[i], Delta[i], i)
            dJdb_i = self._dJdb(Delta[i], i)

            w_update, b_update, grad_max = opt.layer_update(self, i, dJdW_i,
                                                            dJdb_i, grad_max)

            # Update weights
            self.W[i] -= w_update
            self.b[i] -= b_update

        return grad_max

    def _forward_single_layer(self, prev_Z, i):
        """Runs one forward pass through a single layer.

        Parameters
        ----------
        prev_Z : numpy.ndarray, shape(N, M[i-1])
            Output of layer i-1.
        i : int
            Index of layer.

        Returns
        -------
        tuple
            Y : np.ndarray, shape (N, M[i])
                Output of layer i.
        """
        N = len(prev_Z)
        M = self.M
        W = self.W
        b = self.b
        Z = self.Z
        assert prev_Z.shape == (N, M[i-1])
        A = prev_Z.dot(W[i-1]) + b[i-1]
        assert A.shape == (N, M[i])
        _Z = Z.forward(A)
        assert _Z.shape == (N, M[i])
        assert not np.any(np.isnan(_Z))
        return _Z

    def _forward_final_layer(self, final_A):
        """Abstract method. Computes the forward pass of the final layer.

        This is different between classification and regression.

        Parameters
        ----------
        final_A, np.ndarray, shape (N, K)
            Final layer Wz+b

        Returns
        -------
        np.ndarray, shape (N, K)
            Final layer outputs.
        """
        raise NotImplementedError()

    def _transform_output(self, Output):
        """Abstract method. Transforms final layer output into predictions.

        Parameters
        ----------
        np.ndarray, shape (N, K)

        Returns
        -------
        ?
            Implementation-specific
        """
        raise NotImplementedError()

    def _transform_targets(self, Y):
        """Abstract method. Transforms targets.

        Differs b/w regression and classification.

        Parameters
        ----------
        Y : array-like, shape(N)

        Returns
        -------
        ?
            Transformed targets. Implementation is specific to MLP type.
        """
        raise NotImplementedError()

    def _to_indicator_matrix(self, x, n_distinct):
        """Turns a vector of integers into an indicator matrix.

        Parameters
        ----------
        x : array-like
            Vector of integers.
        n_distinct_values, int
            The number of distinct integers in the vector.

        Returns
        -------
        numpy.ndarray, shape (len(x), n_distinct)
            Indicator matrix
        """
        n = len(x)
        X = np.zeros((n, n_distinct))
        for i in range(n):
            X[i, x[i]] = 1
        return X

    def _loss(self, T, Y):
        """
        Abstract method. Computes the loss of the predictions.

        Parameters
        ----------
        T : np.ndarray, shape(N, K)
            Targets as indicator matrix.
        Y : np.ndarray, shape(N, K)
            Predictions as indicator matrix.

        Returns
        -------
        float
            Total loss.
        """
        raise NotImplementedError()

    def _performance_metric(self, Y, P):
        """Abstract method. Computes performance metric.

        Parameters
        ----------
        Y : array-like, shape (N,)
            Targets.
        P : array-like, shape (N,)
            Predictions.

        Returns
        -------
        float
            ?
        """
        raise NotImplementedError()

    def _backprop_delta_final_layer(self, T, Output):
        """Abstract method.

        Compute the backprop delta for the final layer. This is different for
        regression and classification, since classification uses softmax.

        Parameters
        ----------
        T : np.ndarray, shape (N[lower_idx:upper_idx], K)
            Batch targets as indicator matrix.
        Output : np.ndarray, shape (N[lower_idx:upper_idx], K)
            Batch output.

        Returns
        -------
        np.ndarray, shape (N, K)
            Delta of final layer.
        """
        raise NotImplementedError()

    def _backprop_delta(self, subs_Delta, subs_W, _Z, i):
        """Computes the delta for computing weight update.

        Parameters
        ----------
        subs_Delta : np.ndarray, shape (N, M[i+2])
            Delta at subsequent layer (i+1).
        subs_W : np.ndarray, shape (M[i+1], M[i+2])
            Weights at subsequent layer (i+1).
        _Z : np.ndarray, shape (N, M[i+1])
            Layer outputs at current layer. Note that _Z[i] comes BEFORE W[i],
            since we set _Z[0] to X. Therefore, the Z index for the ith layer
            is Z[i+1].
        i : int
            Index of hidden layer.

        Returns
        -------
        np.ndarray, shape (N, M[i+1])
            Delta of layer i.
        """
        M = self.M
        N = len(_Z)
        Z = self.Z
        assert subs_Delta.shape == (N, M[i+2])
        assert subs_W.shape == (M[i+1], M[i+2])
        assert _Z.shape == (N, M[i+1])
        dJdZ = Z.back(_Z)
        ret = (subs_Delta.dot(subs_W.T))*dJdZ
        assert ret.shape == (N, M[i+1])
        return ret

    def _dJdW(self, prev_Z, Delta, i):
        """Computes partial derivative of loss function w.r.t. weights at layer i.

        Parameters
        ----------
        prev_Z : np.ndarray, shape (N, M[i])
            Output of previous layer.
        Delta : np.ndarray, shape(N, M[i+1])
            Delta of current layer.
        i : int
            Hidden layer index.

        Returns
        -------
        np.ndarray, shape(M[i], M[i+1])
            Partial derivatives of loss function w.r.t. weights at layer i.
        """
        M = self.M
        N = len(prev_Z)
        assert prev_Z.shape == (N, M[i])
        assert Delta.shape == (N, M[i+1])
        ret = prev_Z.T.dot(Delta)
        assert ret.shape == (M[i], M[i+1])
        return ret

    def _dJdb(self, Delta, i):
        """
        Computes partial derivative of loss function w.r.t. biases at layer i.

        Parameters
        ----------
        Delta : np.ndarray, shape(N, M[i+1])
            Delta of current layer.
        i : int
            Hidden layer index.

        Returns
        -------
        np.ndarray, shape(M[i+1])
            Partial derivatives of loss function w.r.t. biases at layer i.
        """
        M = self.M
        N = Delta.shape[0]
        assert Delta.shape == (N, M[i+1])
        ret = np.sum(Delta, axis=0)
        assert ret.shape == (M[i+1],)
        return ret


class MLPClassifier(MLP):

    def _transform_targets(self, Y):
        """Transforms targets into indicator matrix.

        Parameters
        ----------
        Y : array-like, shape(N)

        Returns
        -------
        np.ndarray, shape(N, K)
            Transformed targets as indicator matrix.
        """
        K = self.K
        return self._to_indicator_matrix(Y, K)

    def _forward_final_layer(self, final_A):
        """Computes the forward pass of the final layer.

        For classificaiton, this normalizes the inputs and computes softmax.

        Parameters
        ----------
        final_A, np.ndarray, shape (N, K)
            Final layer Wz+b

        Returns
        -------
        np.ndarray, shape (N, K)
            Final layer outputs.
        """
        K = self.K
        N = final_A.shape[0]

        # Normalize inputs
        final_A_mean = np.stack((final_A.mean(axis=1),) * K, axis=-1)
        final_A_std = np.stack((final_A.std(axis=1),) * K, axis=-1)
        final_A = (final_A - final_A_mean) / final_A_std

        # Compute softmax
        expA = np.exp(final_A)
        assert not np.any(np.isnan(expA))
        Y = expA / expA.sum(axis=1, keepdims=True)
        assert Y.shape == (N, K)
        assert not np.any(np.isnan(Y))

        return Y

    def _transform_output(self, Output):
        """Transforms final layer output into predictions, which are an array
        of predicted classes.

        Parameters
        ----------
        np.ndarray, shape (N, K)

        Returns
        -------
        np.ndarray, shape (N,)
            Array of predicted classes.
        """
        P = np.argmax(Output, axis=1)
        return P

    def _loss(self, T, Y):
        """Computes the loss of the predictions using T*log(Y).

        Parameters
        ----------
        T : np.ndarray, shape(N, K)
            Targets as indicator matrix.
        Y : np.ndarray, shape(N, K)
            Predictions as indicator matrix.

        Returns
        -------
        float
            Total cost (error).
        """
        loss = -T*np.log(Y)
        loss = loss.sum()
        assert not np.any(np.isnan(loss))
        return loss

    def _performance_metric(self, Y, P):
        """Determines the classification rate, n_correct / n_total.

        Parameters
        ----------
        Y : array-like, shape (N,)
            Targets.
        P : array-like, shape (N,)
            Predictions.

        Returns
        -------
        float
            n_correct / n_total
        """
        assert np.shape(Y) == np.shape(P)
        n_correct = 0
        n_total = 0
        for i in range(len(Y)):
            n_total += 1
            if Y[i] == P[i]:
                n_correct += 1
        return float(n_correct) / n_total

    def _backprop_delta_final_layer(self, T, Output):
        """Compute the backprop delta for the final layer.

        For classification, this factors in the softmax derivative.

        Parameters
        ----------
        T : np.ndarray, shape (N[lower_idx:upper_idx], K)
            Batch targets as indicator matrix.
        Output : np.ndarray, shape (N[lower_idx:upper_idx], K)
            Batch output.

        Returns
        -------
        np.ndarray, shape (N, K)
            Delta of final layer.
        """
        return -1*(T - Output)


class MLPRegressor(MLP):

    def _transform_targets(self, Y):
        """Returns targets as is.

        Parameters
        ----------
        Y : array-like, shape(N)

        Returns
        -------
        array-like, shape (N)
            Targets as is.
        """
        return Y

    def _forward_final_layer(self, final_A):
        """Computes the forward pass of the final layer.

        For regression, this is just the final Wz+b.

        Parameters
        ----------
        final_A, np.ndarray, shape (N, K)
            Final layer Wz+b

        Returns
        -------
        np.ndarray, shape (N, K)
            Final layer outputs.
        """
        return final_A

    def _transform_output(self, Output):
        """Returns Output as is.

        Parameters
        ----------
        np.ndarray, shape (N, K)

        Returns
        -------
        np.array, shape (N,)
            Array of Outputs.
        """
        return Output

    def _loss(self, T, Y):
        """Computes the loss of the predictions using MSE.

        Note that this is actually the reward since we're doing gradient
        ascent.

        Parameters
        ----------
        T : np.ndarray, shape(N, K)
            Targets as indicator matrix.
        Y : np.ndarray, shape(N, K)
            Predictions as indicator matrix.

        Returns
        -------
        float
            Total loss.
        """
        loss = .5*((T-Y)**2).mean()
        assert not np.any(np.isnan(loss))
        return loss

    def _performance_metric(self, Y, P):
        """Computes r2.

        Parameters
        ----------
        Y : array-like, shape (N,)
            Targets.
        P : array-like, shape (N,)
            Predictions.

        Returns
        -------
        float
            r2
        """
        return r2_score(Y, P)

    def _backprop_delta_final_layer(self, T, Output):
        """Compute the backprop delta for the final layer. For regression, this
        is just dJ/dY, where J is MSE, or `.5*((T-Y)**2).mean()`.

        Parameters
        ----------
        T : np.ndarray, shape (N[lower_idx:upper_idx], K)
            Batch targets as indicator matrix.
        Output : np.ndarray, shape (N[lower_idx:upper_idx], K)
            Batch output.

        Returns
        -------
        np.ndarray, shape (N, K)
            Delta of final layer.
        """
        return -1*(T-Output)
