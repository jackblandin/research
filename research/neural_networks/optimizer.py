import numpy as np


class Optimizer:
    """Abstract class. Strategy for updating weights and biases of MLP."""

    def __init__(self):
        pass

    def _layer_update(self, mlp, i, dJdW_i, dJdb_i, grad_max):
        """Returns weight+bias updates for a single layer, i."""
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

    def layer_update(self, mlp, i, dJdW_i, dJdb_i, grad_max):
        """Returns weight+bias updates for a single layer, i.

        Parameters
        ----------
        mlp : MLP
            MLP instance.
        i : int
            Layer index.
        dJdW_i : np.ndarray
            Weight gradients at layer i.
        dJdb_i : np.array
            Bias gradients at layer i.
        grad_max : numeric
            Current maximum gradient for any weight.

        Returns
        -------
        tuple
            np.ndarray
                Weight updates.
            np.ndarray
                Bias updates.
            float
                Max gradient of any weight (for debugging).
        """
        learning_rate = mlp.learning_rate
        reg = self.reg
        mu = self.mu
        clip_thresh = self.clip_thresh
        W_i = mlp.W[i]
        b_i = mlp.b[i]

        # Clip gradients that are too large
        dJdW_i[dJdW_i > clip_thresh] = clip_thresh
        dJdW_i[dJdW_i < -1*clip_thresh] = -1*clip_thresh
        dJdb_i[dJdb_i > clip_thresh] = clip_thresh
        dJdb_i[dJdb_i < -1*clip_thresh] = -1*clip_thresh

        # Compute max gradient update (for debugging)
        grad_max = max((grad_max, max(dJdW_i.max(), dJdW_i.min(),
                                      key=abs)))

        # Adjust gradients with regularization
        dJdW_i = dJdW_i - reg*W_i
        dJdb_i = dJdb_i - reg*b_i

        # Update momentums (velocities)
        self.vW_[i] = mu*self.vW_[i] + learning_rate*dJdW_i
        self.vb_[i] = mu*self.vb_[i] + learning_rate*dJdb_i

        w_update = self.vW_[i]
        b_update = self.vb_[i]

        return w_update, b_update, grad_max
