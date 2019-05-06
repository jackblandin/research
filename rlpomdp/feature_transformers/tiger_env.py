import gym
import numpy as np


class ObservationAsStatesTransformer:

    def __init__(self, env):
        """
        Parameters
        ----------
        env : gym.Env
        OpenAI Gym environment.

        Maps
            [1, 0, 0, 0] -> 0
            [0, 1, 0, 0] -> 1
            [0, 0, 1, 0] -> 2
            [0, 0, 0, 1] -> 3
        """
        pass

    def transform(self, o):
        """
        Maps
            [1, 0, 0, 0] -> 0
            [0, 1, 0, 0] -> 1
            [0, 0, 1, 0] -> 2
            [0, 0, 0, 1] -> 3

        Parameters
        ----------
        o : list or array-like
            A single observation.

        Returns
        -------
        int
            Discrete state value (one of 0, 1, 2, or 3).
        """
        if o == [1, 0, 0, 0]:
            return 0
        elif o == [0, 1, 0, 0]:
            return 1
        elif o == [0, 0, 1, 0]:
            return 2
        elif o == [0, 0, 0, 1]:
            return 3
        else:
            raise ValueError('Invalid observation: '.format(o))


class SeqArrayToSortedStringTransformer:

    def __init__(self, env, seq_len=3):
        """
        Takes all possible combinations of observation sequences of length
        seq_len and transforms them into a flattened string, then stores these
        strings into a sorted array. The position index in the array
        corresponds to the state value.

        E.g. if obs space was [0,1], [1,0], then:
            [0,0], [0,0], [0,0] -> '000000' -> 0
            [0,0], [0,0], [1,0] -> '000010' -> 2
            [0,0], [0,0], [0,1] -> '000001' -> 3
            [0,0], [1,0], [0,0] -> '001000' -> 1
            ...

        These string to sorted-index are stored in a dict as key and indexes.
        For the transform function, the observation sequence is transformed
        into a string, and this string is used as the lookup in the dict to get
        the index value.

        Parameters
        ----------
        env : gym.Env
            OpenAI Gym environment.
        seq_len : int
            Number of sequential observations to use as the state.
        """
        if not isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            raise ValueError('SeqArrayToSortedStringTransformer only works \
                             for Discrete spaces.')
        # generate all possible combos of length seq_len
        all_combos = _all_combos(env.observation_space.n, seq_len)  # TODO
        all_combos = all_combos.sort()
        self.lookup = {all_combos[i]: i for i in range(len(all_combos))}

    def transform(self, observations_seq):
        """
        Input observations equence is flattened into a string, which is used
        as the key in the lookup dict to obtain the discrete state value.

        Parameters
        ----------
        observations_seq : numpy 2d-array

        Returns
        -------
        int
            Discrete state value.
        """
        seq_str = ''.join(map(lambda f: str(int(f)), np.ndarray.flatten(
            observations_seq)))
        return self.lookup[seq_str]


def _all_combos(num_obs, seq_len):
    """
    Parameters
    ----------
    num_obs : int
        'n' in observation space.
    seq_len : int
        Number of sequential observations to use as the state.

    Returns
    -------
    list
        Array of strings where each string represents a distinct observation
        sequence.
    """
    # TODO
