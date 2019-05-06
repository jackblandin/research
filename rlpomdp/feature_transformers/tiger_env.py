import gym
import numpy as np
from itertools import product


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

    def __init__(self, env, seq_len):
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


        Attributes
        ----------
        _lookup : dict
            Mapping between raw state sequence to an int representation.
        _reverse_lookup : dict
            Inverse of _lookup. Used for verbose output.
        """
        if not isinstance(env.observation_space, gym.spaces.discrete.Discrete):
            raise ValueError('SeqArrayToSortedStringTransformer only works \
                             for Discrete spaces.')
        # generate all possible permutations of length seq_len
        all_perms = _all_perms(env.observation_space.n, seq_len)
        all_perms.sort()
        self._lookup = {all_perms[i]: i for i in range(len(all_perms))}
        self._reverse_lookup = {i: all_perms[i] for i in range(len(all_perms))}

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
        observations_seq = np.asarray(
            [np.asarray(o) for o in observations_seq])
        seq_str = ''.join(map(lambda f: str(int(f)), np.ndarray.flatten(
            np.asarray(observations_seq))))
        return self._lookup[seq_str]


def _all_perms(num_obs, seq_len):
    """
    Generates all possible permuation subsequences (with replacement) of length
    1 to seq_len of in the range (0, num_obs-1).

    Examples
    --------
    >>> _all_perms(3, 2)
    array(['', '100', '010', '001', '100100', '100010', '100001', '010100',
        '010010', '010001', '001100', '001010', '001001'], dtype='<U6')

    Parameters
    ----------
    num_obs : int
        'n' in observation space.
    seq_len : int
        Number of sequential observations to use as the state.

    Returns
    -------
    array of arrays (numpy)
        Arary of arrays of strings where each string represents a distinct
        observation sequence.
    """
    permutations = []
    all_uniq_obs = []
    for i in range(num_obs):
        uniq_obs = np.asarray([0 for j in range(num_obs)])
        uniq_obs[i] = 1
        all_uniq_obs.append(uniq_obs)

    all_uniq_obs = np.asarray(all_uniq_obs)

    for L in range(0, seq_len+1):
        for subset in product(all_uniq_obs, repeat=L):
            flattened_seq = np.ndarray.flatten(np.array(subset))
            seq_str = ','.join(flattened_seq.astype(str)).replace(',', '')
            permutations.append(seq_str)

    permutations = np.asarray(permutations)

    return permutations
