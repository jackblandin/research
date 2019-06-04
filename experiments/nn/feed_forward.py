import numpy as np
import matplotlib.pyplot as plt


class FeedForward:

    def __init__(self, D, hidden_layer_sizes, K, learning_rate=1e-3):
        """
        Feed forward neural network implemented with NumPy. Uses tanh as the
        activation function. Uses gradient ascent.

        Parameters
        ----------
        D : int
            Input dimension.
        hidden_layer_sizes : array-like
            Hidden layer sizes. An array of integers.
        K : int
            Number of output classes.
        learning_rate : numeric
            Learning rate.

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
        >>> model = FeedForward(D, hidden_layer_sizes, K)
        >>> model.fit(X_train, Y_train)
        >>> preds, _ = model.forward(X_test)
        """
        self.D = D
        self.K = K
        self.learning_rate = learning_rate
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
        Runs one forward pass through all layers.

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
        final_A = Z[-2].dot(W[n_layers]) + b[n_layers]
        # Compute softmax
        expA = np.exp(final_A)
        Y = expA / expA.sum(axis=1, keepdims=True)
        Z[-1] = Y
        N = X.shape[0]
        assert Y.shape == (N, self.K)
        assert len(Z) == n_layers + 2
        return Y, Z

    def fit(self, X, Y, epochs=1000):
        """
        Fits the neural network using backpropagation.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, D)
            Input matrix.
        Y : array-like, shape (N,)
            Vector of targets. This will be transformed into a onehot encoded
            indicator matrix, called T.
        epochs : int
            Number of epochs.

        Returns
        -------
        None
        """
        n_layers = self.n_layers_
        K = self.K
        learning_rate = self.learning_rate

        T = self._to_indicator_matrix(Y, K)

        costs = []
        for epoch in range(epochs):
            Output, Z = self.forward(X)
            if epoch % (epochs / 10) == 0:
                c = self._cost(T, Output)
                P = np.argmax(Output, axis=1)
                r = self._classification_rate(Y, P)
                print("epoch:", epoch, "cost:", c, "classification_rate:", r)
                costs.append(c)

            # this is gradient ASCENT, not DESCENT
            Delta = [None for i in range(n_layers+1)]
            for i in range(n_layers, -1, -1):  # Iterate backwards
                W = self.W
                b = self.b

                if i == n_layers:
                    Delta[i] = T - Output
                else:
                    Delta[i] = self._backprop_delta(Delta[i+1], W[i+1], Z[i+1],
                                                    i)

                dJdW_i = self._dJdW(Z[i], Delta[i], i)
                dJdb_i = self._dJdb(Delta[i], i)

                W[i] += learning_rate * dJdW_i
                b[i] += learning_rate * dJdb_i

                self.W = W
                self.b = b

        plt.plot(costs)
        plt.show()

        return None

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
        assert prev_Z.shape == (N, M[i-1])
        A = prev_Z.dot(W[i-1]) + b[i-1]
        assert A.shape == (N, M[i])
        Z = self._activation(A, i)
        assert Z.shape == (N, M[i])
        return Z

    def _activation(self, A, i):
        """
        Computes activation (tanh) of a single layer.

        Parameters
        ----------
        A : np.ndarray, shape (N, M[i])
            Input into activation func. at layer i: (Z.dot(W)+b).
        i : int
            Index of hidden layer. Note that X is the zeroth hidden layer.

        Returns
        -------
        np.ndarray, shape (N, M[i])
        """
        N = len(A)
        M = self.M
        assert A.shape == (N, M[i])
        Z = 1 / (1 + np.exp(-A))
        assert Z.shape == (N, M[i])
        return Z

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

    def _cost(self, T, Y):
        """
        Computes the cost (error) of the predictions using T*log(Y).

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
        tot = T * np.log(Y)
        return tot.sum()

    def _classification_rate(self, Y, P):
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

    def _backprop_delta(self, subs_Delta, subs_W, Z, i):
        """
        Computes the delta for computing weight update.

        Parameters
        ----------
        subs_Delta : np.ndarray, shape (N, M[i+2])
            Delta at subsequent layer (i+1).
        subs_W : np.ndarray, shape (M[i+1], M[i+2])
            Weights at subsequent layer (i+1).
        Z : np.ndarray, shape (N, M[i+1])
            Layer outputs at current layer. Note that Z[i] comes BEFORE W[i],
            since we set Z[0] to X. Therefore, the Z index for the ith layer is
            Z[i+1].
        i : int
            Index of hidden layer.

        Returns
        -------
        np.ndarray, shape (N, M[i+1])
            Delta of layer i.
        """
        M = self.M
        N = len(Z)
        assert subs_Delta.shape == (N, M[i+2])
        assert subs_W.shape == (M[i+1], M[i+2])
        assert Z.shape == (N, M[i+1])
        dJdZ = self._dJdZ(Z, i)
        ret = (subs_Delta.dot(subs_W.T))*dJdZ
        assert ret.shape == (N, M[i+1])
        return ret

    def _dJdZ(self, Z, i):
        """
        Computes partial derivative of loss function w.r.t. Z at layer i.

        Parameters
        ----------
        Z : np.ndarray, shape(N, M[i])
            Output of layer i.
        i : int
            Hidden layer index.

        Returns
        -------
        np.ndarray, shape(N, M[i])
            Partial derivative of loss function w.r.t. Z at layer i.

        NOTE
        ----
        For now, this assumes activation function is tanh.

        TODO
        ----
        Generalize for all types of activation functions.
        """
        M = self.M
        N = len(Z)
        assert Z.shape == (N, M[i+1])
        ret = Z*(1-Z)
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
