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
        return np.argmax(self.predict(o))


class QLearnerSeq:

    def __init__(self, env, alpha=.1, gamma=.9, seq_len=3):
        """
        Started 05/04/2019
        Q Learner with all combinations of three observations as state space.

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
        self.feature_transformer = SeqArrayToSortedStringTransformer(env)
        num_states = env.observation_space.n**seq_len
        num_actions = env.action_space.n
        # axis0 is transformed observation, axis1 is action, value is Q value
        self.Q = np.random.uniform(low=-1, high=1,
                                   size=(num_states, num_actions))

    def predict(self, o_seq):
        """
        Parameters
        ----------
        o_seq : array
            Sequence of observations.

        Returns
        -------
        An array where index is an action, and values are the Q values
        of taking that action.
        """
        o_seq_trans = self.feature_transformer.transform(o_seq)
        return self.Q[o_seq_trans]

    def update(self, otm1_seq, atm1, r, ot_seq, at):
        """
        Performs TD(0) update on the model.

        Parameters
        ----------
        otm1_seq : 2D numpy array
            Previous observation sequence (o t "minus" 1)
        atm1 : int
            Previous action
        r : float
            Reward of taking previous action given previous observation seq.
        ot : 2D numpy array
            Current observation sequence.
        at : int
            Action chosen to update the model. Usually this is the best action.

        Returns
        -------
        None
        """
        otm1_seq_trans = self.feature_transformer.transform(otm1_seq)
        ot_seq_trans = self.feature_transformer.transform(ot_seq)
        G = r + self.gamma*self.Q[ot_seq_trans, at]
        self.Q[otm1_seq_trans, atm1] += self.alpha*(G - self.Q[otm1_seq_trans,
                                                               atm1])

    def sample_action(self, o, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return self.best_action(o)

    def best_action(self, o_seq):
        return np.argmax(self.predict(o_seq))
