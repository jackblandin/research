class ObservationAsStatesTransformer:

    def __init__(self, env):
        """
        Parameters
        ----------
        env : gym.Env
        OpenAI Gym environment.

        Maps
            [0] -> 0
            [1] -> 1
            [2] -> 2
        """
        pass

    def transform(self, o):
        """
        Maps
            [0] -> 0
            [1] -> 1
            [2] -> 2

        Parameters
        ----------
        o : list or array-like
            A single observation.

        Returns
        -------
        int
            Discrete state value (one of 0, 1, or 2).
        """
        if o == [0]:
            return 0
        elif o == [1]:
            return 1
        elif o == [2]:
            return 2
        else:
            raise ValueError('Invalid observation: '.format(o))
