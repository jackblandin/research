class ObsActSeqFeatureTransformer:
    """Transforms observation-action sequences into string representations.

    Attributes
    ----------
    idx_lookup_ : dict
        Converts string repr to an integer index.
    next_idx_ : int
        Next index to use when encountering a new string representation.
    inverse_lookup_ : dict
        Converts integer index back into original observation.

    Examples
    --------
    >>> ft = ObsActSeqFeatureTransformer()
    >>> ft.transform([[[0], 1], [[2], 0]])
    0
    >>> ft.idx_lookup_
    {'[1]-1, [0]-0': 0}
    >>> ft.next_idx_
    1
    """

    def __init__(self):
        self.idx_lookup_ = {}
        self.next_idx_ = 0
        self.inverse_lookup_ = {}

    def transform(self, X):
        """Transforms observation-inputs into string representations.

        Parameters
        ----------
        X : array<array-like>
            Array of observation-action inputs. Each inner array is an array of
            length 2 with first element as last action and second element as
            last observation.

        Returns
        -------
        int or None
            Index of string representation of observation-action sequence.
        """
        X_t = ''
        for i, x in enumerate(X):
            if i > 0:
                X_t += ', '
            obs = x[0]
            act = x[1]
            X_t += '['
            X_t += ','.join([str(i) for i in obs])
            X_t += ']-' + str(act)
        if X_t not in self.idx_lookup_:
            self.idx_lookup_[X_t] = self.next_idx_
            self.next_idx_ += 1
            self.inverse_lookup_[self.idx_lookup_[X_t]] = X
        return self.idx_lookup_[X_t]

    def inverse_transform(self, idx):
        """Untransforms integer index into original observaion-action array.

        Parameters
        ----------
        idx : int
            Index of string representation of observation-action sequence.

        Returns
        -------
        list or None
            Original observation-action array.
        """
        if idx not in self.inverse_lookup_:
            return None
        else:
            return self.inverse_lookup_[idx]
