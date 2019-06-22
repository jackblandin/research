class NullFeatureTransformer:
    """Transformer that just returns input as is. """

    def __init__(self):
        pass

    def transform(self, X):
      """Returns input unchanged.

      Parameters
      ----------
      X : any
        Raw observation/state input.

      Returns
      -------
      <X.__class__>
        Input `X` unchanged.
      """
      return X
