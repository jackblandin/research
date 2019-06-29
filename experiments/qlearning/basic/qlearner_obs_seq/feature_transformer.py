import gym
import numpy as np
from itertools import product


class SeqFeatureTransformer:
    """Transforms observations into string representations.

    Attributes
    ----------
    idx_lookup_ : dict
        Converts string repr to an integer index.
    next_idx_ : int
        Next index to use when encountering a new string representation.

    Examples
    --------
    >>> sft = SeqFeatureTransformer()
    >>> sft.transform([[1,2,3], [4,5,6]])
    0
    >>> sft.idx_lookup_
    {'123456': 0}
    >>> sft.next_idx_
    1
    """

    def __init__(self):
        self.idx_lookup_ = {}
        self.next_idx_ = 0
        self.inverse_lookup_ = {}

    def transform(self, X):
        """Transforms observations into string representations.

        Parameters
        ----------
        X : array<array-like>
            Array of state/observation inputs. Each inner array is one state or
            observation.

        Returns
        -------
        int
            Index of string representation of state/observation sequence.
        """
        X_t = ''
        for x in X:
            X_t += ''.join([str(i) for i in x])
        if X_t not in self.idx_lookup_:
            self.idx_lookup_[X_t] = self.next_idx_
            self.next_idx_ += 1
            self.inverse_lookup_[self.idx_lookup_[X_t]] = X
        return self.idx_lookup_[X_t]

    def inverse_transform(self, idx):
        """Untransforms integer index into original observaion sequence.

        Parameters
        ----------
        idx : int
            Index of string representation of observation sequence.

        Returns
        -------
        list or None
            Original observation array.
        """
        if idx not in self.inverse_lookup_:
            return None
        else:
            return self.inverse_lookup_[idx]


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
