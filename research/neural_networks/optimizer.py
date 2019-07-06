import numpy as np


class Optimizer:
    """Abstract class. Strategy for updating weights and biases of MLP."""

    def __init__(self):
        pass

    def update(self):
        raise NotImplementedError()


class MomentumOptimizer(Optimizer):
    """Optimzier strategy that uses momentum.

    Other options are regularization and gradient-clipping.

    Parameters
    ----------
    D : int
        MLP input dimension.
    K : int
        MLP number of output classes.
    hidden_layer_sizes : array-like
        Hidden layer sizes. An array of integers.
        momentums are the same shape as the weights.
    mu : float, default 0
        Momentum parameter.
    reg : float, default 0
        Regularization parameter.
    clip_thresh : float, default np.inf
       Maximum weight/bias gradient allowed. If gradient is larger than
       clip_thresh, the gradient is replaced with clip_thresh.

    Attributes
    ----------
    vW_ : list, shape (len(M)+1)
        List of weight momentums. Same shapes as MLP's matrices in W.
    vb_ : list, shape (len(M)+1)
        List of bias momentums. Same shapes as MLP's bias vectors.
    """

    def __init__(self, D, K, hidden_layer_sizes, mu=0, reg=0,
                 clip_thresh=np.inf):
        self.mu = mu
        self.reg = reg
        self.clip_thresh = clip_thresh
        self.vW_ = [None for i in range(len(hidden_layer_sizes)+1)]
        self.vb_ = [None for i in range(len(hidden_layer_sizes)+1)]
        M = [D] + hidden_layer_sizes + [K]
        for i in range(len(hidden_layer_sizes)+1):
            self.vW_[i] = np.zeros((M[i], M[i+1]))
            self.vb_[i] = np.zeros((M[i+1]))

    def update(self, mlp, T, Output, _Z):
        """Updates the mlp's weights using optimzer strategy.

        Parameters
        ----------
        mlp : MLP
            MLP to be updated.
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
        n_layers = mlp.n_layers_
        learning_rate = mlp.learning_rate
        reg = self.reg
        mu = self.mu

        # For each layer, iterating backwards from final layer, compute
        # deltas, weight partials, and update weights/biases.
        Delta = [None for i in range(n_layers+1)]
        grad_max = 0
        for i in range(n_layers, -1, -1):
            clip_thresh = self.clip_thresh
            W = mlp.W
            b = mlp.b
            vW_i = self.vW_[i]
            vb_i = self.vb_[i]

            if i == n_layers:
                # Compute delta at final layer
                Delta[i] = mlp._backprop_delta_final_layer(T, Output)
            else:
                # Compute deltas at middle layer
                Delta[i] = mlp._backprop_delta(Delta[i+1], W[i+1], _Z[i+1], i)

            # Compute weight and bias partial gradients
            dJdW_i = mlp._dJdW(_Z[i], Delta[i], i)
            dJdb_i = mlp._dJdb(Delta[i], i)

            # Clip gradients that are too large
            dJdW_i[dJdW_i > clip_thresh] = clip_thresh
            dJdW_i[dJdW_i < -1*clip_thresh] = -1*clip_thresh
            dJdb_i[dJdb_i > clip_thresh] = clip_thresh
            dJdb_i[dJdb_i < -1*clip_thresh] = -1*clip_thresh

            # Compute max gradient update (for debugging)
            grad_max = max((grad_max, max(dJdW_i.max(), dJdW_i.min(),
                                          key=abs)))

            # Adjust gradients with regularization
            dJdW_i = dJdW_i - reg*W[i]
            dJdb_i = dJdb_i - reg*b[i]

            # Update momentums (velocities)
            self.vW_[i] = mu*vW_i + learning_rate*dJdW_i
            self.vb_[i] = mu*vb_i + learning_rate*dJdb_i

            # Update weights
            mlp.W[i] -= self.vW_[i]
            mlp.b[i] -= self.vb_[i]

        return grad_max
