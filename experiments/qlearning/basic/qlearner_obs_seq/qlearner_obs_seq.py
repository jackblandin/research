import numpy as np

from .feature_transformer import SeqArrayToSortedStringTransformer
from ....utils.tiger_env_utils import env_translate_obs, env_translate_action


class QLearnerObsSeq:

    def __init__(self, env, initial_alpha=.1, gamma=.9, alpha_decay=0,
                 seq_len=3):
        """
        Started 05/04/2019
        Q Learner with all combinations of seq_len observations as state space.

        Parameters
        ----------
        env : gym.Env
            OpenAI Gym environment.
        initial_alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        alpha_decay : float, default 0
            Learning rate alpha will decay at 1/_n_updates**_alpha_decay.
        seq_len : int
            Number of sequential observations to use as the state.

        Attributes
        ----------
        feature_transformer : object
            Transforms raw state into representation, usually a reduced one.
        Q : 2D numpy array
            Q <state,value> matrix where
                - axis0 index is transformed observation
                - axis1 index is action
                - value is Q value.
            E.g. Q[1][2] represents the Q value for taking action 2 in
            (transformed) state 1.
        _n_updates : int
            Number of updates made to Q matrix.
        last_n_obs : list
            A sequence of the most recent <seq_len> raw observations.
            TODO 05/09/2019 - Could improve this by only storing *transformed*
            observations.
        train_obs_seq_counts : dict<str, int>
            Dict with stringified observation sequences as keys and values as
            the number of times this sequence was predicted on. The
            observations and actions are referring to the sets that are sampled
            from experience replay and used to update the DQN. Currently this
            is only used for
            debuggging.
        train_obs_seq_action_counts : dict<str, int>
            Dict with stringified observation sequence + action as keys and
            values as the count of taking that action on that observation
            sequence. The observations and actions are referring to the sets
            that are sampled from experience replay and used to update the DQN.
            Currently this is only used for debugging.
        """
        self.env = env
        self.initial_alpha = initial_alpha
        self.gamma = gamma
        self._alpha_decay = alpha_decay
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
        self._n_updates = 0
        self.last_n_obs = []
        self.train_obs_seq_counts = {}
        self.train_obs_seq_action_counts = {}

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

        # Compute alpha (if there's nonzero alpha decay)
        alpha = self.initial_alpha / (self._n_updates+1)**self._alpha_decay

        # TD(0) update using observation sequences as states
        G = r + self.gamma*self.Q[g_last_n_obs_trans, at]
        self.Q[last_n_obs_trans, atm1] += alpha*(
            G - self.Q[last_n_obs_trans, atm1])
        self._n_updates += 1

        # Update training observation sequence counts
        transl_obs = [env_translate_obs(o) for o in self.last_n_obs]
        transl_obs = ', '.join(transl_obs)
        if transl_obs in self.train_obs_seq_counts:
            self.train_obs_seq_counts[transl_obs] += 1
        else:
            self.train_obs_seq_counts[transl_obs] = 1

        ##
        # Update training observation sequence + action counts
        ##
        transl_act = env_translate_action(at)
        obs_seq_act = ' => '.join([transl_obs, transl_act])
        if obs_seq_act in self.train_obs_seq_action_counts:
            self.train_obs_seq_action_counts[obs_seq_act] += 1
        else:
            self.train_obs_seq_action_counts[obs_seq_act] = 1

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
        s += '\n{:<47} | OPEN LEFT | OPEN RIGHT | LISTEN'.format(' ')
        s += '\n{} | --------- | ---------- | ------'.format('-'*47)
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
