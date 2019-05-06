# 3rd party
import numpy as np

# Local
from ..feature_transformers.tiger_env import (
    SeqArrayToSortedStringTransformer, ObservationAsStatesTransformer)


class QLearner:

    def __init__(self, env, alpha=.1, gamma=.9):
        """
        Simple Q Learner with just observations as states. The action is
        associated with the last observation.

        Parameters
        ----------
        env : gym.Env
            OpenAI Gym environment
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.feature_transformer = ObservationAsStatesTransformer(env)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        # axis0 is transformed observation, axis1 is action, value is Q value
        self.Q = np.random.uniform(low=-1, high=1,
                                   size=(num_states, num_actions))

    def predict(self, o):
        """
        Parameters
        ----------
        o : list or array-like
            A single observation.

        Returns
        -------
        numpy array
            Each index is an action, and values are Q values of taking that
            action, given the transformed state.
        """
        o_trans = self.feature_transformer.transform(o)
        return self.Q[o_trans]

    def update(self, otm1, atm1, r, ot, at):
        """
        Performs TD(0) update on the model.

        Parameters
        ----------
        otm1 : list or array-like
            Previous observation (o t "minus" 1)
        atm1 : int
            Previous action
        r : float
            Reward of taking previous action given previous observation
        ot : list or array-like
            Current observation.
        at : int
            Action chosen to update the model. Usually this is the best action.

        Returns
        -------
        None
        """
        otm1_trans = self.feature_transformer.transform(otm1)
        ot_trans = self.feature_transformer.transform(ot)
        G = r + self.gamma*self.Q[ot_trans, at]
        self.Q[otm1_trans, atm1] += self.alpha*(G - self.Q[otm1_trans, atm1])

    def sample_action(self, o, eps):
        """
        Uses epsilon greedy strategy to either sample a random action, or
        select the best action.

        Parameters
        ----------
        o : list or array-like
            A single observation.
        eps : float
            Epsilon used in epsilon-greedy.

        Returns
        -------
        int
            An action.
        """
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return self.best_action(o)

    def best_action(self, o):
        """
        Returns the action with the highest q value, given an observation.

        Parameters
        ----------
        o : object
            Observation.

        Returns
        -------
        int
            Best action given observation.
        """
        return np.argmax(self.predict(o))


class QLearnerSeq:

    def __init__(self, env, alpha=.1, gamma=.9, seq_len=3):
        """
        Started 05/04/2019
        Q Learner with all combinations of seq_len observations as state space.

        Parameters
        ----------
        env : gym.Env
            OpenAI Gym environment.
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        seq_len : int
            Number of sequential observations to use as the state.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.seq_len = seq_len
        self.feature_transformer = SeqArrayToSortedStringTransformer(
            env, seq_len=seq_len)
        num_obs = env.observation_space.n
        exp = seq_len
        num_states = 1
        while exp > 0:
            num_states += num_obs**exp
            exp = exp - 1
        num_actions = env.action_space.n
        # axis0 is transformed observation, axis1 is action, value is Q value
        self.Q = np.random.uniform(low=0, high=0,
                                   size=(num_states, num_actions))
        # self.Q[:, 2] = 1  # Optimistic value for listening
        self.last_n_obs = []

    def predict(self, o):
        """
        Uses the input observation (o) and the last <seq_len-1> observations as
        the input to the Q matrix.

        Parameters
        ----------
        o : list or array-like <int>
            Single observation

        Returns
        -------
        An array where index is an action, and values are the Q values
        of taking that action.
        """
        last_n_obs = self.last_n_obs.copy()
        if len(last_n_obs) == self.seq_len:
            last_n_obs.pop(0)
        last_n_obs.append(o)
        last_n_obs_trans = self.feature_transformer.transform(last_n_obs)
        return self.Q[last_n_obs_trans]

    def update(self, otm1, atm1, r, ot, at):
        """
        Performs TD(0) update on the model using sequences of observations as
        states.

        Parameters
        ----------
        otm1 : list or array-like
            Previous observation (o t "minus" 1)
        atm1 : int
            Previous action
        r : float
            Reward of taking previous action given previous observation
        ot : list or array-like
            Current observation.
        at : int
            Action chosen to update the model. Usually this is the best action.

        Returns
        -------
        None
        """
        # Append new observation to last two observations
        if len(self.last_n_obs) == self.seq_len:
            self.last_n_obs.pop(0)
        self.last_n_obs.append(otm1)
        last_n_obs_trans = self.feature_transformer.transform(self.last_n_obs)

        # Create the future rewards observation sequence (ot)
        g_last_n_obs = self.last_n_obs.copy()
        if len(g_last_n_obs) == self.seq_len:
            g_last_n_obs.pop(0)
        g_last_n_obs.append(ot)
        g_last_n_obs_trans = self.feature_transformer.transform(g_last_n_obs)

        # TD(0) update using observation sequences as states
        G = r + self.gamma*self.Q[g_last_n_obs_trans, at]
        self.Q[last_n_obs_trans, atm1] += self.alpha*(
            G - self.Q[last_n_obs_trans, atm1])

    def sample_action(self, o, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return self.best_action(o)

    def best_action(self, o):
        """
        Returns the action with the highest q value, given an observation.

        Parameters
        ----------
        o : list or array-like <int>
            A single observation.

        Returns
        -------
        int
            Best action given observation.
        """
        return np.argmax(self.predict(o))

    def __str__(self):
        """
        String representation of the model.

        Returns
        -------
        str
            A string showing the Q values of each state/action.
        """
        s = '\n'
        s += '\n{:<47}   OPEN LEFT | OPEN RIGHT | LISTEN'.format(' ')
        s += '\n\t\t{} | --------- | ----------  | ------'.format('-'*43)
        o_seq_as_strs = []
        for o_seq_idx in range(self.Q.shape[0]):
            seq_str = self.feature_transformer._reverse_lookup[o_seq_idx]
            grouped = zip(*(iter(seq_str),) * self.env.observation_space.n)
            grouped = np.asarray(list(grouped)).astype(int)
            grouped = [self.env.translate_obs(g) for g in grouped]
            o_seq_as_strs.append(grouped)
        for idx, o_str in enumerate(o_seq_as_strs):
            action_qs = self.Q[idx]
            if not any([q != 0 for q in action_qs]):
                continue
            o_seq_as_strs
            s += '\n{:<47} {:>11} | {:>10} | {:>6}'.format(
                str(o_str), action_qs[0].round(1), action_qs[1].round(1),
                action_qs[2].round(1))
        s += '\n\n'
        return s
