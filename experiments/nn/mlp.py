import numpy as np
import matplotlib.pyplot as plt


class Activation:

    def __init__(self):
        """
        Activation function used in a MLP neural network.
        """
        pass

    def forward(self, A):
        """
        Computes forward activation function.
        """
        raise NotImplementedError()

    def back(self, X):
        """
        Computes derivative of loss function activation function w.r.t.
        activation function for a particular layer.
        """
        raise NotImplementedError()


class Sigmoid(Activation):

    def __init__(self):
        """
        Sigmoid activation function used in a MLP neural network.
        """
        pass

    def forward(self, A):
        """
        Computes activation of a single layer.

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
        """
        Computes partial derivative of loss function w.r.t. activation
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

    def __init__(self):
        """
        ReLU activation function used in a MLP neural network.
        """
        pass

    def forward(self, A):
        """
        Computes activation of a single layer.

        Parameters
        ----------
        A : np.ndarray, shape (N, M[i])
            Input into activation func. at layer i: (Z.dot(W)+b).

        Returns
        -------
        np.ndarray, shape (N, M[i])
        """
        _Z = A * (A > .5)
        assert not np.any(np.isnan(_Z))
        return _Z

    def back(self, _Z):
        """
        Computes partial derivative of loss function w.r.t. activation
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
        dZ = np.piecewise(_Z, [_Z < .5, _Z >= .5], [0, 1])
        return dZ


class MLP:

    def __init__(self, D, hidden_layer_sizes, K, Z, learning_rate=1e-5, reg=.01):
        """
        Multip Layer Perceptron neural network implemented with NumPy. Uses gradient ascent.

        Parameters
        ----------
        D : int
            Input dimension.
        hidden_layer_sizes : array-like
            Hidden layer sizes. An array of integers.
        K : int
            Number of output classes.
        Z : Activation
            Activation function.
        learning_rate : numeric
            Learning rate.
        reg : numeric, default .01
            Regularization parameter.

        Attributes
        ----------
        n_layers, int
            Number of hidden layers, defined by length of hidden_layer_sizes.
        M : list, length (n_layers+2)
            Hidden layer sizes, with M[0] set to D and M[-1] set to K. This
            makes computing
            recursive forward and back propagation easier.
        W : numpy.array, shape (len(M)+1)
            Array of weight matrices.
        b : numpy.array, shape (len(M)+1)
            Array of bias matrices.

        Examples
        --------
        >>> D = X_train.shape[1]
        >>> K = Y_train.shape[0]
        >>> hidden_layer_sizes = [5, 5]
        >>> Z = Sigmoid()
        >>> model = FeedForward(D, hidden_layer_sizes, K, Z)
        >>> model.fit(X_train, Y_train)
        >>> preds, _ = model.forward(X_test)
        """
        self.D = D
        self.K = K
        self.Z = Z
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_layers_ = len(hidden_layer_sizes)
        self.M = [D] + hidden_layer_sizes + [K]
        n_layers = self.n_layers_
        M = self.M
        self.W = [None for i in range(n_layers+1)]
        self.b = [None for i in range(n_layers+1)]

        # randomly initialize weights
        for i in range(n_layers+1):
            self.W[i] = np.random.randn(M[i], M[i+1])
            self.b[i] = np.random.randn(M[i+1])

    def forward(self, X):
        """
        Runs one forward pass through all layers. Final layer inputs are normalized
        prior to passing through softmax.

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
        K = self.K
        N = X.shape[0]

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
        """
        Fits the neural network using backpropagation.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, D)
            Input matrix.
        Y : array-like, shape (N,)
            Vector of targets. This will be transformed into a onehot encoded
            indicator matrix, called T.
        batch_size : int, default=1000
            Training batch size. If batch_size > len(X), then the batch size will be changed to len(X).
        epochs : int
            Number of epochs.

        Returns
        -------
        None
        """
        n_layers = self.n_layers_
        K = self.K
        learning_rate = self.learning_rate
        reg = self.reg

        T = self._transform_targets(Y)

        if batch_size > len(X):
            print('WARNING: Batch size > len(X). Setting batch size to len(X)')
            batch_size = len(X)
        n_batches = int(len(X) / batch_size)
        losses = []
        performance_metrics = []

        for epoch in range(epochs):
            _Output = np.empty_like(T)  #  Aggregate Output for entire epoch

            for b in range(n_batches):
                lower_idx = b*batch_size
                upper_idx = (b+1)*batch_size
                if upper_idx > len(X):
                    upper_idx = len(X) - 1
                X_batch = X[lower_idx:upper_idx, :]
                Output, _Z = self.forward(X_batch)
                _Output[lower_idx:upper_idx] = Output

                # this is gradient ASCENT, not DESCENT
                Delta = [None for i in range(n_layers+1)]
                for i in range(n_layers, -1, -1):  # Iterate backwards
                    W = self.W
                    b = self.b

                    if i == n_layers:
                        Delta[i] = self._backprop_delta_final_layer(T[lower_idx:upper_idx, :], Output)
                    else:
                        Delta[i] = self._backprop_delta(Delta[i+1], W[i+1], _Z[i+1],
                                                        i)

                    dJdW_i = self._dJdW(_Z[i], Delta[i], i)
                    dJdb_i = self._dJdb(Delta[i], i)

                    W[i] -= learning_rate * dJdW_i + reg*W[i]
                    b[i] -= learning_rate * dJdb_i + reg*b[i]

                    self.W = W
                    self.b = b
            if epoch % (epochs / 5) == 0:
                l = self._loss(T, _Output)
                P = self._transform_output(Output)
                pm = self._performance_metric(Y, P)
                print("epoch:", epoch, "loss:", l, "performance_metric:", pm)
                losses.append(l)
                performance_metrics.append(pm)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
        ax0.plot(losses[1:])
        ax0.set_title('Loss')
        ax1.plot(performance_metrics[1:])
        ax1.set_title('Performance Metric')
        plt.show()

        return None

    def predict(self, X):
        """
        Wrapper method around forward() and _transform_output. This method allows
        MLPs to conform to scikit-learn's fit/predict schema.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Input matrix.

        Returns
        -------
        ?
            Whatever `transformed_output()` returns.
        """
        Output = self.forward(X)
        transformed_Output = self._transform_output(Output)
        return transformed_Output

    def _forward_single_layer(self, prev_Z, i):
        """
        Runs one forward pass through a single layer.

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
        """
        Abstract method. Computes the forward pass of the final layer. This is different between
        classifcation and regression.

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
        """
        Abstract method. Transforms final layer output into predictions.

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
        """
        Abstract method. Transforms targets, differs b/w regression and classification.

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
        """
        Turns a vector of integers into an indicator matrix.

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
        """
        Abstract method. Computes performance metric.

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
        """
        Abstract method. Compute the backprop delta for the final layer. This is different for regression and
        classification, since classification uses softmax.

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
        """
        Computes the delta for computing weight update.

        Parameters
        ----------
        subs_Delta : np.ndarray, shape (N, M[i+2])
            Delta at subsequent layer (i+1).
        subs_W : np.ndarray, shape (M[i+1], M[i+2])
            Weights at subsequent layer (i+1).
        _Z : np.ndarray, shape (N, M[i+1])
            Layer outputs at current layer. Note that _Z[i] comes BEFORE W[i],
            since we set _Z[0] to X. Therefore, the Z index for the ith layer is
            Z[i+1].
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
        """
        Computes partial derivative of loss function w.r.t. weights at layer i.

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
        """
        Transforms targets into indicator matrix.

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
        """
        Computes the forward pass of the final layer. For classificaiton, this normalizes the
        inputs and computes softmax.

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
        """
        Transforms final layer output into predictions, which are an array of predicted classes.

        Parameters
        ----------
        np.ndarray, shape (N, K)

        Returns
        -------
        np.ndarray, shape (N,)
            Array of predicted classes.
        """
        P = np.argmax(_Output, axis=1)
        return P

    def _loss(self, T, Y):
        """
        Computes the loss of the predictions using T*log(Y).

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
        """
        Determines the classification rate, n_correct / n_total.

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
        """
        Compute the backprop delta for the final layer. For classification, this factors in
        the softmax derivative.

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


from sklearn.metrics import r2_score

class MLPRegressor(MLP):

    def _transform_targets(self, Y):
        """
        Returns targets as is.

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
        """
        Computes the forward pass of the final layer. For regression, this is just the final Wz+b.

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
        """
        Returns Output as is.

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
        """
        Computes the loss of the predictions using MSE. Note that this is actually the reward since we're doing gradient ascent.

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
        """
        Computes r2.

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
#         Y_mean = Y.mean()
#         ssreg = ((P - Y_mean)**2).sum()
#         sstot = ((Y - Y_mean)**2).sum()
#         r2 = ssreg / sstot
#         return r2
        return r2_score(Y, P)

    def _backprop_delta_final_layer(self, T, Output):
        """
        Compute the backprop delta for the final layer. For regression, this is
        just dJ/dY, where J is MSE, or `.5*((T-Y)**2).mean()`.

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
